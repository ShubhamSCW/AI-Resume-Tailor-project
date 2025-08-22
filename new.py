# app_ultimate.py
# -----------------------------------------------------------------------------------
# Ultimate AI Resume Suite (All-In-One)
# Features:
# - Resume Parsing & Preprocessing
#   * Smarter python-docx styling (bold keywords, consistent spacing)
#   * NLP via spaCy (fallback to NLTK) + language detection
#   * Auto section detection (regex + optional AI assist)
#   * Basic multilingual parsing support (langdetect)
# - AI Enhancements
#   * Multiple backends: Gemini, OpenAI, Anthropic, Ollama (local), HuggingFace (optional)
#   * Automatic fallback chain if a backend fails
#   * Strict JSON outputs for robust parsing
#   * Readability (textstat) & complexity flag
# - UI/UX
#   * Drag & drop multi-upload with preview
#   * Multi-resume comparison
#   * Interactive charts: keyword heatmap; word clouds for strengths/weaknesses
#   * Session caching + optional SQLite persistence
#   * Light/Dark theme toggle
# - Exporting & Customization
#   * Exports: DOCX, PDF, TXT, JSON
#   * Templates/styles (Minimalist, Modern, Creative)
#   * Dynamic highlight of missing keywords in tailored resume
# -----------------------------------------------------------------------------------

import os
import re
import io
import json
import time
import base64
import sqlite3
import tempfile
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict

import streamlit as st

# File parsing / formatting
from docx import Document as DocxDocument
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PyPDF2 import PdfReader

# PDF export (ReportLab)
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor

# NLP
import textstat
from langdetect import detect, LangDetectException

# Prefer spaCy; fallback to NLTK
_SPACY_OK = False
try:
    import spacy
    _SPACY_OK = True
except Exception:
    _SPACY_OK = False

_NLTK_OK = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    _NLTK_OK = True
except Exception:
    _NLTK_OK = False

# ML & viz
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Backends (optional installs)
import google.generativeai as genai
from openai import OpenAI
try:
    import anthropic
    _ANTHROPIC_OK = True
except Exception:
    _ANTHROPIC_OK = False

import requests

try:
    from transformers import pipeline
    _HF_OK = True
except Exception:
    _HF_OK = False


# ==============================================
# Helpers: text, parsing, NLP
# ==============================================

SECTION_PATTERNS = [
    r'^\s*(summary|professional summary|profile)\s*$',
    r'^\s*(experience|work experience|professional experience)\s*$',
    r'^\s*(education|academic background)\s*$',
    r'^\s*(skills|technical skills|core competencies)\s*$',
    r'^\s*(projects)\s*$',
    r'^\s*(certifications)\s*$',
    r'^\s*(publications)\s*$',
    r'^\s*(awards|achievements)\s*$',
    r'^\s*(volunteer|volunteering)\s*$',
]
SECTION_REGEX = re.compile("|".join(SECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# SpaCy pipeline selection
@st.cache_resource
def load_spacy_model(lang_code: str = "en"):
    if not _SPACY_OK:
        return None
    # minimal mapping; extend as necessary
    lang_map = {
        "en": "en_core_web_sm",
        "es": "es_core_news_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "it": "it_core_news_sm",
        "pt": "pt_core_news_sm",
        "xx": "xx_ent_wiki_sm"  # multilingual (if installed)
    }
    model_name = lang_map.get(lang_code, "en_core_web_sm")
    try:
        return spacy.load(model_name)
    except Exception:
        # fallback to multilingual if available
        try:
            return spacy.load("xx_ent_wiki_sm")
        except Exception:
            return None

def sent_segment(text: str, lang_code: str = "en") -> List[str]:
    text = normalize_whitespace(text)
    if _SPACY_OK:
        nlp = load_spacy_model(lang_code)
        if nlp is not None:
            try:
                doc = nlp(text)
                return [s.text.strip() for s in doc.sents if s.text.strip()]
            except Exception:
                pass
    # fallback to NLTK
    if _NLTK_OK:
        try:
            # ensure punkt
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    # worst-case fallback
    return [s.strip() for s in text.split("\n") if s.strip()]

def tokenize_words(text: str) -> List[str]:
    if _SPACY_OK:
        nlp = load_spacy_model("en")
        if nlp is not None:
            try:
                return [t.text for t in nlp(text)]
            except Exception:
                pass
    if _NLTK_OK:
        try:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", text)

def read_file_to_text(upload) -> str:
    """Extract text from TXT/DOCX/PDF uploads."""
    name = upload.name.lower()
    try:
        if name.endswith('.txt'):
            return upload.read().decode('utf-8', errors='ignore')
        elif name.endswith('.docx'):
            doc = DocxDocument(upload)
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith('.pdf'):
            text = []
            reader = PdfReader(upload)
            for page in reader.pages:
                try:
                    text.append(page.extract_text() or "")
                except Exception:
                    continue
            return "\n".join(text)
        else:
            return ""
    except Exception:
        return ""

def auto_detect_sections(text: str) -> Dict[str, List[str]]:
    """
    Heuristic section splitter. Returns dict of section -> lines.
    Lines not matched go to 'Other'.
    """
    blocks = defaultdict(list)
    current = "Other"
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if re.match(SECTION_REGEX, line):
            current = re.sub(r'\s+', ' ', line.title())
            blocks[current]  # ensure key present
        else:
            blocks[current].append(line)
    return dict(blocks)

def extract_keywords(jd_text: str, top_k: int = 30) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{1,}", jd_text)
    tokens = [t.lower() for t in tokens if len(t) > 2]
    counter = Counter(tokens)
    common = [w for w, _ in counter.most_common(top_k)]
    return common

def compute_similarity(text_a: str, text_b: str) -> float:
    vect = TfidfVectorizer(stop_words='english')
    tfidf = vect.fit_transform([text_a, text_b])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])

def keyword_coverage(resume_text: str, jd_keywords: List[str]) -> Dict[str, int]:
    res_lc = resume_text.lower()
    return {k: int(k in res_lc) for k in jd_keywords}

def generate_wordcloud(words: List[str], title: str = ""):
    text = " ".join(words) if words else ""
    wc = WordCloud(width=800, height=400, background_color=None, mode="RGBA")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc.generate(text), interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


# ==============================================
# LLM Backends (strict JSON where needed)
# ==============================================

def safe_json_extract(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}

def call_gemini_json(prompt: str, api_key: str) -> Dict[str, Any]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return safe_json_extract(getattr(resp, "text", "") or "")

def call_openai_json(prompt: str, api_key: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Return ONLY valid JSON."},
                  {"role": "user", "content": prompt}],
        temperature=0.2
    )
    return safe_json_extract(resp.choices[0].message.content)

def call_anthropic_json(prompt: str, api_key: str) -> Dict[str, Any]:
    if not _ANTHROPIC_OK:
        raise RuntimeError("Anthropic SDK not installed.")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
        system="Return ONLY valid JSON."
    )
    # Anthropics returns list of content blocks
    text = ""
    try:
        for blk in msg.content:
            if blk.type == "text":
                text += blk.text
    except Exception:
        text = str(msg)
    return safe_json_extract(text)

def call_ollama_json(prompt: str) -> Dict[str, Any]:
    # Requires Ollama running locally: https://github.com/ollama/ollama
    # e.g., model "llama3.1"
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3.1",
        "messages": [{"role": "system", "content": "Return ONLY valid JSON."},
                     {"role": "user", "content": prompt}],
        "stream": False
    }
    r = requests.post(url, json=data, timeout=120)
    r.raise_for_status()
    out = r.json()
    content = out.get("message", {}).get("content", "")
    return safe_json_extract(content)

def call_huggingface_json(prompt: str) -> Dict[str, Any]:
    if not _HF_OK:
        raise RuntimeError("transformers not installed.")
    gen = pipeline("text-generation", model="gpt2")  # placeholder small model
    text = gen(prompt, max_length=512, num_return_sequences=1)[0]["generated_text"]
    return safe_json_extract(text)

def call_llm_json(prompt: str, primary: str, keys: Dict[str, str]) -> Dict[str, Any]:
    """
    primary: one of ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Ollama (local)", "HuggingFace (local)"]
    keys: {"gemini": "...", "openai": "...", "anthropic": "..."} (as available)
    Fallback order tries others automatically.
    """
    backends = ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Ollama (local)", "HuggingFace (local)"]
    # rotate so primary is first
    order = [primary] + [b for b in backends if b != primary]

    last_err = None
    for b in order:
        try:
            if b == "Google Gemini" and keys.get("gemini"):
                return call_gemini_json(prompt, keys["gemini"])
            if b == "OpenAI GPT" and keys.get("openai"):
                return call_openai_json(prompt, keys["openai"])
            if b == "Anthropic Claude" and keys.get("anthropic"):
                return call_anthropic_json(prompt, keys["anthropic"])
            if b == "Ollama (local)":
                return call_ollama_json(prompt)
            if b == "HuggingFace (local)":
                return call_huggingface_json(prompt)
        except Exception as e:
            last_err = e
            st.warning(f"{b} failed: {e}")
            continue
    st.error(f"All backends failed. Last error: {last_err}")
    return {}

def call_llm_text(prompt: str, primary: str, keys: Dict[str, str]) -> str:
    """Text (non-JSON). Uses same fallback chain."""
    json_prompt = f"Return ONLY plain text.\n\n{prompt}"
    # Reuse JSON callers but read raw string when possible
    # Simpler: use OpenAI/Gemini first; else Ollama
    order = [primary] + [b for b in ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Ollama (local)", "HuggingFace (local)"] if b != primary]
    for b in order:
        try:
            if b == "Google Gemini" and keys.get("gemini"):
                genai.configure(api_key=keys["gemini"])
                model = genai.GenerativeModel("gemini-1.5-flash")
                return (model.generate_content(prompt).text or "").strip()
            if b == "OpenAI GPT" and keys.get("openai"):
                client = OpenAI(api_key=keys["openai"])
                r = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2
                )
                return r.choices[0].message.content.strip()
            if b == "Anthropic Claude" and keys.get("anthropic") and _ANTHROPIC_OK:
                client = anthropic.Anthropic(api_key=keys["anthropic"])
                msg = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1500,
                    messages=[{"role":"user","content":prompt}]
                )
                text = ""
                for blk in msg.content:
                    if getattr(blk, "type", "") == "text":
                        text += blk.text
                return text.strip()
            if b == "Ollama (local)":
                url = "http://localhost:11434/api/chat"
                data = {"model": "llama3.1", "messages":[{"role":"user","content":prompt}], "stream": False}
                r = requests.post(url, json=data, timeout=120)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "").strip()
            if b == "HuggingFace (local)" and _HF_OK:
                gen = pipeline("text-generation", model="gpt2")
                return gen(prompt, max_length=512, num_return_sequences=1)[0]["generated_text"].strip()
        except Exception as e:
            st.warning(f"{b} failed (text): {e}")
            continue
    return ""


# ==============================================
# Formatting & Export
# ==============================================

def style_docx_paragraph(run, bold=False, size=11, color=None):
    run.bold = bold
    run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color

def format_docx(content: str, file_path: str, template: str = "Minimalist", keywords_to_bold: Optional[List[str]] = None):
    """
    - Bold headings (ALL CAPS short lines)
    - Bullet for lines starting with -, •, *
    - Bold given keywords dynamically
    - Templates: Minimalist, Modern, Creative
    """
    try:
        doc = DocxDocument()
        normal = doc.styles['Normal']
        normal.font.name = "Calibri" if template == "Minimalist" else ("Helvetica" if template == "Modern" else "Georgia")
        normal.font.size = Pt(11)

        accent = RGBColor(0x00, 0x33, 0x66) if template in ("Modern", "Creative") else None

        def add_header(text):
            p = doc.add_paragraph()
            run = p.add_run(text)
            style_docx_paragraph(run, bold=True, size=13, color=accent)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)

        kws = [k.lower() for k in (keywords_to_bold or [])]

        for raw in content.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.isupper() and len(line.split()) < 6:
                add_header(line)
            elif line.startswith(("-", "•", "*")):
                p = doc.add_paragraph(style="List Bullet")
                text = line[1:].strip()
                # keyword bolding within bullet
                if kws:
                    pos = 0
                    lower = text.lower()
                    # simple span highlighting by splitting on keywords
                    tokens = re.split(r'(\W+)', text)  # keep punctuation
                    for tok in tokens:
                        if tok.lower() in kws:
                            style_docx_paragraph(p.add_run(tok), bold=True)
                        else:
                            p.add_run(tok)
                else:
                    p.add_run(text)
            else:
                p = doc.add_paragraph()
                if kws:
                    tokens = re.split(r'(\W+)', line)
                    for tok in tokens:
                        if tok.lower() in kws:
                            style_docx_paragraph(p.add_run(tok), bold=True)
                        else:
                            p.add_run(tok)
                else:
                    p.add_run(line)
        doc.save(file_path)
    except Exception as e:
        st.error(f"DOCX export error: {e}")

def format_pdf(content: str, file_path: str):
    try:
        doc = SimpleDocTemplate(
            file_path, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch
        )
        story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Header', fontName='Helvetica-Bold', fontSize=12,
                                  spaceBefore=12, spaceAfter=6, textColor=HexColor('#003366')))
        styles.add(ParagraphStyle(name='CustomBullet', parent=styles['Normal'], leftIndent=20, firstLineIndent=0, spaceBefore=2))

        for raw in content.split("\n"):
            line = raw.strip()
            if not line:
                continue
            if line.isupper() and len(line.split()) < 6:
                p = Paragraph(line, styles['Header'])
            elif line.startswith(("-", "•", "*")):
                bullet_text = f"<bullet>&bull;</bullet>{line[1:].strip()}"
                p = Paragraph(bullet_text, styles['CustomBullet'])
            else:
                p = Paragraph(line, styles['Normal'])
            story.append(p)
        doc.build(story)
    except Exception as e:
        st.error(f"PDF export error: {e}")

def export_txt(content: str, file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path

def export_json(data: Dict[str, Any], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return file_path

def highlight_missing_keywords_html(text: str, missing: List[str]) -> str:
    if not missing:
        return f"<div>{text}</div>"
    def repl(match):
        token = match.group(0)
        if token.lower() in missing:
            return f'<span style="background-color:#ffe4e1;border-radius:3px;padding:1px 2px">{token}</span>'
        return token
    pattern = r"[A-Za-z][A-Za-z0-9+.#-]{1,}"
    return re.sub(pattern, repl, text, flags=re.IGNORECASE)


# ==============================================
# Optional persistence (SQLite)
# ==============================================

@st.cache_resource
def get_db(path="resume_suite.db"):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS analyses(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts INTEGER,
      backend TEXT,
      score INTEGER,
      grade TEXT,
      payload TEXT
    )
    """)
    return conn

def persist_analysis(conn, backend: str, score: int, grade: str, payload: Dict[str, Any]):
    conn.execute("INSERT INTO analyses(ts, backend, score, grade, payload) VALUES(?,?,?,?,?)",
                 (int(time.time()), backend, score, grade, json.dumps(payload)))
    conn.commit()


# ==============================================
# Streamlit App
# ==============================================

st.set_page_config(page_title="Ultimate AI Resume Suite", page_icon="🚀", layout="wide")

# Theme toggle (basic CSS)
theme = st.sidebar.toggle("🌙 Dark mode", value=False)
if theme:
    st.markdown("""
        <style>
        .stApp { background: #0e1117; color: #e0e0e0; }
        .stMarkdown, .stText, .stCaption { color: #e0e0e0 !important; }
        </style>
    """, unsafe_allow_html=True)

st.title("🚀 Ultimate AI Resume Suite")
st.caption("Analyze, compare, tailor, and export ATS-friendly resumes with multi-model AI and rich visuals.")

# AI Config
st.sidebar.header("⚙️ AI Configuration")
primary_backend = st.sidebar.selectbox(
    "Primary Backend",
    ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Ollama (local)", "HuggingFace (local)"],
    index=0
)
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="google-generativeai")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password", help="openai")
anthropic_key = st.sidebar.text_input("Anthropic API Key", type="password", help="anthropic (optional)")

keys = {"gemini": gemini_key, "openai": openai_key, "anthropic": anthropic_key}


st.sidebar.header("🎨 Output Options")
resume_tone = st.sidebar.selectbox("Resume Tone", ["Professional", "Creative", "Technical", "Enthusiastic"])
docx_template = st.sidebar.selectbox("DOCX Template", ["Minimalist", "Modern", "Creative"])
persist_to_sqlite = st.sidebar.checkbox("Persist analyses to SQLite", value=False)

# Upload + JD
st.subheader("1) Upload one or more resumes & paste the job description")
c1, c2 = st.columns([1, 1])
with c1:
    resumes = st.file_uploader("Drag & drop resumes here (Multi-upload OK)", type=["txt", "docx", "pdf"], accept_multiple_files=True)
with c2:
    jd_text = st.text_area("Paste Job Description", height=220, placeholder="Paste JD here...")

# Language detection & preview
if resumes:
    st.markdown("#### Quick Preview")
    pv_cols = st.columns(min(3, len(resumes)))
    for i, f in enumerate(resumes[:3]):
        with pv_cols[i]:
            rt = read_file_to_text(f)
            lang = detect_language(rt[:1000]) if rt else "unknown"
            st.caption(f"**{f.name}** (lang: {lang})")
            st.text_area("Preview", value=(rt[:400] + ("…" if len(rt) > 400 else "")), height=150, key=f"pv_{i}")

if st.button("✨ Analyze & Generate", type="primary", use_container_width=True):
    if not resumes or not jd_text:
        st.warning("Please upload at least one resume and paste the JD.")
        st.stop()

    # Prepare JD keywords (multilingual agnostic simple approach)
    jd_keywords = extract_keywords(jd_text, top_k=25)

    # Analyze all resumes
    all_results = []
    for rf in resumes:
        text = read_file_to_text(rf)
        text_norm = normalize_whitespace(text)
        sim = compute_similarity(text_norm, jd_text)
        coverage = keyword_coverage(text_norm, jd_keywords)

        # JSON grading via LLM (with fallback chain)
        grading_prompt = f"""
Return STRICT JSON with keys EXACTLY:
"SCORE" (integer 0-100),
"GRADE" (one of "A","B","C","D","F"),
"SUMMARY" (string),
"STRENGTHS" (array of strings),
"WEAKNESSES" (array of strings).

Evaluate the resume vs the job description from an ATS + recruiter perspective.
Be precise and use content present.

Job Description:
---
{jd_text}
---

Resume:
---
{text_norm}
---
"""
        result = call_llm_json(grading_prompt, primary_backend, keys) or {}
        score = int(result.get("SCORE", round(sim * 100)))
        grade = result.get("GRADE", "?")
        strengths = result.get("STRENGTHS", [])
        weaknesses = result.get("WEAKNESSES", [])
        summary = result.get("SUMMARY", "")

        if persist_to_sqlite:
            try:
                conn = get_db()
                persist_analysis(conn, primary_backend, score, grade, {
                    "name": rf.name,
                    "summary": summary,
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "coverage": coverage
                })
            except Exception as e:
                st.warning(f"SQLite persist failed: {e}")

        all_results.append({
            "name": rf.name,
            "text": text_norm,
            "score": score,
            "grade": grade,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "coverage": coverage
        })

    # Rank by score
    all_results.sort(key=lambda x: x["score"], reverse=True)

    st.success("Analysis complete.")
    tabA, tabB, tabC = st.tabs(["📊 Comparison & Insights", "✍️ Tailoring & Highlights", "💾 Export"])

    # -------- TAB A: charts --------
    with tabA:
        # Heatmap: resume vs keywords
        st.subheader("Keyword Coverage Heatmap")
        heat_df = pd.DataFrame(
            [{**r["coverage"], "Resume": r["name"]} for r in all_results]
        ).set_index("Resume")
        fig_heat = px.imshow(
            heat_df.values,
            labels=dict(x="Keyword", y="Resume", color="Present"),
            x=list(heat_df.columns),
            y=list(heat_df.index),
            aspect="auto",
            title="JD Keyword Presence (1=present, 0=absent)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Gauge for top resume
        best = all_results[0]
        st.subheader(f"Top Match: {best['name']}")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best["score"],
            title={'text': "Match Score"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Word clouds
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            st.markdown("**Strengths Word Cloud**")
            generate_wordcloud([w for r in all_results for w in r["strengths"]], title="Strengths")
        with col_wc2:
            st.markdown("**Weaknesses Word Cloud**")
            generate_wordcloud([w for r in all_results for w in r["weaknesses"]], title="Weaknesses")

        # Detail expanders
        st.subheader("Per-Resume Details")
        for r in all_results:
            with st.expander(f"{r['name']} — Score {r['score']} / Grade {r['grade']}"):
                st.write("**Summary:**", r["summary"])
                st.write("**Strengths:**")
                for s in r["strengths"]:
                    st.markdown(f"- {s}")
                st.write("**Weaknesses:**")
                for w in r["weaknesses"]:
                    st.markdown(f"- {w}")
                st.caption(f"Readability (Flesch Reading Ease): {textstat.flesch_reading_ease(r['text']):.1f}")

    # -------- TAB B: tailoring --------
    with tabB:
        st.subheader("Tailored Resume & Highlighting")
        best = all_results[0]
        tail_prompt = f"""
You are a senior resume writer. Return ONLY plain text.
Rewrite and tailor this resume to perfectly match the job description using a '{resume_tone}' tone.
Keep ATS-friendly formatting with standard sections and action/results-driven bullets.
Integrate relevant JD keywords naturally.

Job Description:
---
{jd_text}
---

Original Resume:
---
{best['text']}
---
"""
        tailored_resume = call_llm_text(tail_prompt, primary_backend, keys)
        st.markdown("**Tailored Resume (raw text)**")
        st.text_area("Tailored Resume", value=tailored_resume, height=350, key="tailed")

        # Missing keywords highlight (show what JD has but tailored resume still misses)
        tr_lc = (tailored_resume or "").lower()
        missing = sorted([k for k in jd_keywords if k not in tr_lc])
        st.markdown("**Missing JD Keywords in Tailored Resume:** " + (", ".join(missing) if missing else "None 🎉"))
        html_highlight = highlight_missing_keywords_html(tailored_resume or "", missing)
        st.markdown("**Highlighted (inline):**", unsafe_allow_html=False)
        st.markdown(f"<div style='padding:10px;border:1px solid #ddd;border-radius:8px'>{html_highlight}</div>", unsafe_allow_html=True)

        # Optional: AI-structured sectioning (assist)
        st.markdown("---")
        st.markdown("**Auto-structured Sections (heuristic)**")
        sections = auto_detect_sections(tailored_resume or "")
        for sec, lines in sections.items():
            st.markdown(f"**{sec}**")
            for ln in lines[:8]:
                st.markdown(f"- {ln}")
            if len(lines) > 8:
                st.caption(f"... and {len(lines)-8} more")

        # Cover letter
        st.markdown("---")
        st.subheader("Cover Letter")
        user_name = st.text_input("Your Name for Cover Letter", value="")
        if st.button("Generate Cover Letter"):
            cover_prompt = f"""
Return ONLY plain text. Write a concise 3-4 paragraph cover letter for '{user_name or "Candidate"}',
based on the tailored resume and job description. Highlight 2-3 matching qualifications and end with a call to action.

Job Description:
---
{jd_text}
---

Tailored Resume:
---
{tailored_resume}
---
"""
            cover_letter = call_llm_text(cover_prompt, primary_backend, keys)
            st.text_area("Cover Letter", value=cover_letter, height=300, key="cover_letter")

    # -------- TAB C: export --------
    with tabC:
        st.subheader("Export Center")
        exp_cols = st.columns(4)
        tailored_text = st.session_state.get("tailed", tailored_resume if 'tailored_resume' in locals() else "")
        cover_text = st.session_state.get("cover_letter", "")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Exports for Tailored Resume
            tr_docx = os.path.join(tmpdir, "Tailored_Resume.docx")
            tr_pdf  = os.path.join(tmpdir, "Tailored_Resume.pdf")
            tr_txt  = os.path.join(tmpdir, "Tailored_Resume.txt")
            tr_json = os.path.join(tmpdir, "Tailored_Resume.json")

            format_docx(tailored_text, tr_docx, template=docx_template, keywords_to_bold=jd_keywords)
            format_pdf(tailored_text, tr_pdf)
            export_txt(tailored_text, tr_txt)
            export_json({"tailored_resume": tailored_text, "jd_keywords": jd_keywords}, tr_json)

            # Exports for Cover Letter (if exists)
            cl_docx = os.path.join(tmpdir, "Cover_Letter.docx")
            cl_pdf  = os.path.join(tmpdir, "Cover_Letter.pdf")
            cl_txt  = os.path.join(tmpdir, "Cover_Letter.txt")
            cl_json = os.path.join(tmpdir, "Cover_Letter.json")

            if cover_text:
                format_docx(cover_text, cl_docx, template=docx_template)
                format_pdf(cover_text, cl_pdf)
                export_txt(cover_text, cl_txt)
                export_json({"cover_letter": cover_text}, cl_json)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Tailored Resume (DOCX)", data=open(tr_docx, "rb"), file_name="Tailored_Resume.docx")
                st.download_button("⬇️ Tailored Resume (PDF)",  data=open(tr_pdf, "rb"),  file_name="Tailored_Resume.pdf")
                st.download_button("⬇️ Tailored Resume (TXT)",  data=open(tr_txt, "rb"),  file_name="Tailored_Resume.txt")
                st.download_button("⬇️ Tailored Resume (JSON)", data=open(tr_json, "rb"), file_name="Tailored_Resume.json")
            with c2:
                if cover_text:
                    st.download_button("⬇️ Cover Letter (DOCX)", data=open(cl_docx, "rb"), file_name="Cover_Letter.docx")
                    st.download_button("⬇️ Cover Letter (PDF)",  data=open(cl_pdf, "rb"),  file_name="Cover_Letter.pdf")
                    st.download_button("⬇️ Cover Letter (TXT)",  data=open(cl_txt, "rb"),  file_name="Cover_Letter.txt")
                    st.download_button("⬇️ Cover Letter (JSON)", data=open(cl_json, "rb"), file_name="Cover_Letter.json")
                else:
                    st.info("Generate a cover letter in Tab B to enable cover-letter exports.")
