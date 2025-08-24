import io
import re
import sqlite3
import json
import asyncio
from typing import List, Dict, Any, Optional

import streamlit as st
from docx import Document as DocxDocument
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PyPDF2 import PdfReader
from jinja2 import Environment, FileSystemLoader
from playwright.async_api import async_playwright

import requests
from openai import OpenAI
import google.generativeai as genai
from config import DATABASE_FILE, GEMINI_MODEL, OPENAI_MODEL, ANTHROPIC_MODEL, OLLAMA_MODEL

_ANTHROPIC_OK = False
try:
    import anthropic
    _ANTHROPIC_OK = True
except ImportError:
    pass

# --- Database & File Reading ---
def init_db():
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, job_title TEXT, resume_name TEXT, match_score INTEGER, ats_score INTEGER, tailored_resume_text TEXT, cover_letter_text TEXT)")
    except sqlite3.Error as e: st.warning(f"Database init error: {e}")

def log_analysis(job_title, resume_name, match_score, ats_score, tailored_resume_text, cover_letter_text):
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO history (job_title, resume_name, match_score, ats_score, tailored_resume_text, cover_letter_text) VALUES (?, ?, ?, ?, ?, ?)", (job_title, resume_name, int(match_score), int(ats_score), tailored_resume_text, cover_letter_text))
    except sqlite3.Error as e: st.warning(f"Database error while logging: {e}")

def get_history() -> List:
    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT id, timestamp, job_title, resume_name, match_score FROM history ORDER BY timestamp DESC LIMIT 20")
            return c.fetchall()
    except sqlite3.Error as e:
        st.warning(f"Database error while fetching history: {e}"); return []

def clean_linkedin_jd(text: str):
    return re.sub(r"Seniority level\s*.*?Employment type", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

def read_file_to_text(upload):
    try:
        name = upload.name.lower()
        bytes_data = upload.getvalue()
        if name.endswith(".txt"): return bytes_data.decode("utf-8", errors="ignore")
        if name.endswith(".docx"): return "\n".join(p.text for p in DocxDocument(io.BytesIO(bytes_data)).paragraphs)
        if name.endswith(".pdf"): return "".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(bytes_data)).pages)
    except Exception as e: st.error(f"Error reading file {upload.name}: {e}")
    return ""

# --- Centralized LLM Calling Logic ---
def _call_llm_api(backend: str, prompt: str, keys: Dict[str, str], expect_json: bool) -> Optional[str]:
    try:
        if backend == "Google Gemini" and keys.get("gemini"):
            genai.configure(api_key=keys["gemini"]); model = genai.GenerativeModel(GEMINI_MODEL)
            return model.generate_content(prompt).text
        elif backend == "OpenAI GPT" and keys.get("openai"):
            client = OpenAI(api_key=keys["openai"])
            messages = [{"role": "user", "content": prompt}]
            if expect_json: messages.insert(0, {"role": "system", "content": "Return ONLY valid JSON."})
            resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.1, response_format={"type": "json_object"} if expect_json else {"type": "text"})
            return resp.choices[0].message.content
        elif backend == "Anthropic Claude" and keys.get("anthropic") and _ANTHROPIC_OK:
            client = anthropic.Anthropic(api_key=keys["anthropic"])
            msg = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=4096, system="Return ONLY valid JSON." if expect_json else "", messages=[{"role": "user", "content": prompt}])
            return "".join(block.text for block in msg.content if hasattr(block, "text"))
        elif backend == "Ollama (local)":
            messages = [{"role": "user", "content": prompt}]
            if expect_json: messages.insert(0, {"role": "system", "content": "Return ONLY valid JSON."})
            resp = requests.post("http://localhost:11434/api/chat", json={"model": OLLAMA_MODEL, "messages": messages, "stream": False, "format": "json" if expect_json else "text"}, timeout=120)
            resp.raise_for_status(); return resp.json().get("message", {}).get("content", "")
    except Exception as e:
        if any(k in str(e).lower() for k in ["permission", "authentication", "api key"]): st.warning(f"{backend} failed: Invalid API Key.")
        else: st.warning(f"{backend} call failed: {e}")
    return None

def _execute_llm_call(prompt: str, primary: str, keys: Dict[str, str], expect_json: bool) -> str:
    order = [primary] + [b for b in ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Ollama (local)"] if b != primary]
    for backend in order:
        if response := _call_llm_api(backend, prompt, keys, expect_json): return response.strip()
    st.error(f"All AI backends failed."); return "{}" if expect_json else ""

def call_llm_json(prompt: str, primary: str, keys: Dict[str, str]) -> Dict[str, Any]:
    raw_response = _execute_llm_call(prompt, primary, keys, expect_json=True)
    if match := re.search(r"\{[\s\S]*\}", raw_response):
        try: return json.loads(match.group(0))
        except json.JSONDecodeError: pass
    st.error("AI returned invalid JSON."); return {}

def call_llm_text(prompt: str, primary: str, keys: Dict[str, str]) -> str:
    return _execute_llm_call(prompt, primary, keys, expect_json=False)

# --- AI-Powered Resume Parser ---
def ai_parse_resume(content: str, primary_backend: str, keys: Dict[str, str]) -> Dict[str, Any]:
    """Uses an LLM to parse raw resume text into a structured dictionary for templating."""
    prompt = f"""
    You are an expert resume parser. Convert the following resume text into a structured JSON object that conforms to the template's needs.
    The JSON object must have these exact keys: "name", "contact", and "sections".
    - "name": A string with the candidate's full name.
    - "contact": A single string containing all contact info (email, phone, LinkedIn), separated by " | ".
    - "sections": A JSON object where each key is a section title (e.g., "Profile Summary", "Work Experience").
        - For "Work Experience", the value should be a list of objects, where each object has "company", "role", "date", and "details" (a list of strings).
        - For "Technical Skills", the value should be an object of key-value pairs (e.g., "Cloud Computing": "AWS, ...").
        - For all other sections, the value should be a list of strings.

    Here is the resume text:
    ---
    {content}
    """
    with st.spinner("AI is parsing the resume structure..."):
        parsed_data = call_llm_json(prompt, primary_backend, keys)
    
    # Basic validation and fallback
    if not parsed_data or "name" not in parsed_data or "sections" not in parsed_data:
        st.warning("AI parsing failed. Falling back to basic structure.")
        lines = content.split('\n')
        return {
            "name": lines[0] if lines else "Resume",
            "contact": lines[1] if len(lines) > 1 else "",
            "sections": {"Full Content": lines[2:]}
        }
    return parsed_data

# --- Document Exporters ---
async def _generate_pdf_async(html_content: str) -> bytes:
    """Async helper to render HTML and generate PDF bytes using Playwright."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_content)
        pdf_bytes = await page.pdf(format='A4', print_background=True)
        await browser.close()
        return pdf_bytes

def format_pdf_stable(content: str, primary_backend: str, keys: Dict[str, str]) -> io.BytesIO:
    """Builds a professional PDF using an AI parser and an HTML template."""
    if not content.strip(): return io.BytesIO()
    
    try:
        resume_data = ai_parse_resume(content, primary_backend, keys)
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template('template.html')
        html_out = template.render(resume_data)
        pdf_bytes = asyncio.run(_generate_pdf_async(html_out))
        return io.BytesIO(pdf_bytes)
    except Exception as e:
        st.error(f"Failed to generate PDF: {e}")
        return io.BytesIO()

def format_docx_stable(content: str, primary_backend: str, keys: Dict[str, str]) -> io.BytesIO:
    """Builds a professional DOCX using an AI parser."""
    if not content.strip(): return io.BytesIO()

    data = ai_parse_resume(content, primary_backend, keys)
    doc = DocxDocument()
    for section in doc.sections:
        section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = Inches(0.8)
    
    # Header
    p = doc.add_paragraph(); p.add_run(data.get("name", "Unnamed")).bold = True
    p.runs[0].font.size = Pt(22); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p = doc.add_paragraph(data.get("contact", ""))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(10)
    
    # Sections
    for title, section_content in data.get("sections", {}).items():
        p = doc.add_paragraph(); p.add_run(title.upper()).bold = True
        p.runs[0].font.size = Pt(11)
        p.paragraph_format.space_before = Pt(8); p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.border_bottom = Pt(0.75)
        
        if "EXPERIENCE" in title.upper() and isinstance(section_content, list):
            for entry in section_content:
                p = doc.add_paragraph(); p.add_run(entry.get("company", "")).bold = True
                p.add_run(f"\t{entry.get('date', '')}")
                p.paragraph_format.tab_stops.add_tab_stop(Inches(7), alignment=WD_ALIGN_PARAGRAPH.RIGHT)
                if entry.get("role"): doc.add_paragraph(entry.get("role"), style='Intense Quote')
                for detail in entry.get("details", []): doc.add_paragraph(detail, style='List Bullet')
        elif "SKILLS" in title.upper() and isinstance(section_content, dict):
            for cat, val in section_content.items():
                p = doc.add_paragraph(); p.add_run(f"{cat}: ").bold = True
                p.add_run(val)
        elif isinstance(section_content, list):
            doc.add_paragraph("\n".join(section_content))

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio