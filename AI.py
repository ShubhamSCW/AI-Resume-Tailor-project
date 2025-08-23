# app_ultimate.py
# -----------------------------------------------------------------------------------
# Ultimate AI Resume Suite (All-In-One) - v6.0 (Dynamic Job Field Analysis)
# Features:
# - Resume Parsing & Preprocessing
#   * Smarter python-docx styling (bold keywords, consistent spacing)
#   * NLP via spaCy (fallback to NLTK) + language detection
#   * Auto section detection (regex)
# - AI Enhancements
#   * Dynamic Job Field Detection: Automatically identifies the job role from the JD. <<< NEW >>>
#   * Context-Aware Tailoring: AI acts as a specialist for the detected field. <<< NEW >>>
#   * Multi-backend support (Gemini, OpenAI, Anthropic, Ollama, etc.)
#   * Strict JSON outputs for granular feedback
# - Advanced Analytics
#   * ATS (Applicant Tracking System) friendliness score and detailed checklist
#   * Action Verb and Quantitative Metrics counter
# - UI/UX
#   * Drag & drop multi-upload and comparison
#   * "Quick Metrics" dashboard for the top resume
#   * Interactive charts for score comparison and keyword coverage
#   * Session caching and personalization options
# - Exporting & Customization
#   * On-demand exports in DOCX, PDF, and TXT formats
#   * Dynamic highlighting of missing keywords
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
from docx.shared import Pt, RGBColor
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
except ImportError:
    _SPACY_OK = False

_NLTK_OK = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    _NLTK_OK = True
except ImportError:
    _NLTK_OK = False

# ML & viz
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
except ImportError:
    _ANTHROPIC_OK = False

import requests

try:
    from transformers import pipeline
    _HF_OK = True
except ImportError:
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

BUSINESS_JARGON_STOP_WORDS = {
    'experience', 'work', 'responsibilities', 'skills', 'company', 'team', 'project', 'role',
    'development', 'management', 'solution', 'solutions', 'technology', 'technologies',
    'environment', 'system', 'systems', 'tools', 'process', 'processes', 'data', 'analysis',
    'business', 'requirements', 'design', 'implementation', 'support', '-','–',
    'communication', 'years', 'job', 'description', 'candidate', 'ability', 'knowledge', 'etc'
}

ACTION_VERBS = {
    'achieved', 'accelerated', 'administered', 'advised', 'advocated', 'analyzed', 'authored',
    'automated', 'built', 'calculated', 'centralized', 'chaired', 'coached', 'collaborated',
    'conceived', 'consolidated', 'constructed', 'consulted', 'converted', 'coordinated',
    'created', 'debugged', 'decreased', 'defined', 'delivered', 'designed', 'developed',
    'directed', 'documented', 'drove', 'eliminated', 'engineered', 'enhanced', 'established',
    'evaluated', 'executed', 'expanded', 'facilitated', 'founded', 'generated', 'grew',
    'guided', 'identified', 'implemented', 'improved', 'increased', 'influenced', 'initiated',
    'innovated', 'inspired', 'integrated', 'interpreted', 'invented', 'launched', 'led',
    'managed', 'mastered', 'mentored', 'modernized', 'motivated', 'negotiated', 'optimized',
    'orchestrated', 'overhauled', 'owned', 'pioneered', 'planned', 'prioritized', 'produced',
    'proposed', 'quantified', 'ran', 'rebuilt', 'reduced', 're-engineered', 'resolved',
    'restructured', 'revamped', 'saved', 'scaled', 'shipped', 'simplified', 'solved',
    'spearheaded', 'standardized', 'streamlined', 'strengthened', 'succeeded', 'supervised',
    'taught', 'trained', 'transformed', 'unified', 'won', 'wrote'
}

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

@st.cache_resource
def load_spacy_model(lang_code: str = "en"):
    if not _SPACY_OK:
        return None
    model_name = "en_core_web_sm"
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"spaCy model '{model_name}' not found. Run: python -m spacy download {model_name}")
        return None

# <<< MODIFIED >>> This function is now generic for all job types.
def extract_keywords_smarter(jd_text: str, top_k: int = 30) -> List[str]:
    if not jd_text:
        return []

    # Regex for multi-word capitalized phrases (e.g., "Power BI", "Google Analytics")
    phrase_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z0-9]+)+)\b')
    keywords = {match.group(0).lower() for match in phrase_pattern.finditer(jd_text)}
    
    nlp = load_spacy_model("en")
    if nlp is None or not _SPACY_OK:
        st.warning("spaCy not available, falling back to basic keyword extraction.")
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+.#-]{2,}", jd_text.lower())
        found_keywords = list(keywords) + [word for word in tokens if word not in BUSINESS_JARGON_STOP_WORDS and len(word) > 2]
        return list(dict.fromkeys(found_keywords))[:top_k]

    doc = nlp(jd_text)
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            keyword = token.text.lower().strip()
            if len(keyword) > 2 and keyword not in BUSINESS_JARGON_STOP_WORDS:
                keywords.add(keyword)

    text_lower = jd_text.lower()
    keyword_counts = {kw: text_lower.count(kw) for kw in keywords}
    sorted_keywords = sorted(keyword_counts.items(), key=lambda item: (item[1], len(item[0])), reverse=True)
    
    return [kw for kw, count in sorted_keywords[:top_k]]

def read_file_to_text(upload) -> str:
    name = upload.name.lower()
    try:
        bytes_data = upload.getvalue()
        if name.endswith('.txt'):
            return bytes_data.decode('utf-8', errors='ignore')
        elif name.endswith('.docx'):
            doc = DocxDocument(io.BytesIO(bytes_data))
            return "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith('.pdf'):
            reader = PdfReader(io.BytesIO(bytes_data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return ""
    except Exception as e:
        st.error(f"Error reading file {upload.name}: {e}")
        return ""

def compute_similarity(text_a: str, text_b: str) -> float:
    try:
        vect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf = vect.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        return 0.0

def keyword_coverage(resume_text: str, jd_keywords: List[str]) -> Dict[str, int]:
    res_lc = resume_text.lower()
    # Use word boundaries for more accurate matching
    return {k: int(re.search(r'\b' + re.escape(k) + r'\b', res_lc, re.IGNORECASE) is not None) for k in jd_keywords}

def analyze_ats_friendliness(text: str) -> Dict[str, Any]:
    checks = {}
    score = 100
    if not (re.search(r'[\w\.-]+@[\w\.-]+', text) and re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)):
        score -= 20
        checks["❌ Contact Info"] = "Missing email or phone number in a standard format."
    else:
        checks["✅ Contact Info"] = "Email and phone number found."

    if len([s for s in ['experience', 'education', 'skills'] if re.search(f'^{s}', text, re.I | re.M)]) < 3:
        score -= 25
        checks["❌ Standard Sections"] = "Missing standard sections like 'Experience', 'Education', or 'Skills'."
    else:
        checks["✅ Standard Sections"] = "Found essential sections."
    return {"score": max(0, score), "checks": checks}

def analyze_action_verbs_and_metrics(text: str) -> Dict[str, int]:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    action_verb_count = len([v for v in words if v in ACTION_VERBS])
    metric_count = len(re.findall(r'\b\d+(\.\d+)?%?\b|\$\d+[,.]\d*', text))
    return {"action_verbs": action_verb_count, "quantitative_metrics": metric_count}

def safe_json_extract(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

# A simplified, unified LLM call function for demonstration
def call_llm(prompt: str, api_key: str, is_json: bool = False) -> Any:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.3 if is_json else 0.5,
            response_mime_type="application/json" if is_json else "text/plain",
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        
        if is_json:
            return safe_json_extract(response.text)
        return response.text.strip()

    except Exception as e:
        st.error(f"An error occurred with the AI backend: {e}")
        return {} if is_json else ""

# <<< NEW FEATURE >>> Function to detect the job field from the JD
def detect_job_field_from_jd(jd_text: str, api_key: str) -> str:
    prompt = f"""
    Analyze the following job description and return ONLY the specific job title or field it represents (e.g., "Data Scientist", "DevOps Engineer", "Marketing Manager", "Sales Executive").
    
    Job Description:
    {jd_text[:1500]}
    """
    return call_llm(prompt, api_key, is_json=False) or "Unknown"

# ==============================================
# Streamlit App
# ==============================================

st.set_page_config(page_title="Ultimate AI Resume Suite", page_icon="🚀", layout="wide")
st.title("🚀 Ultimate AI Resume Suite")

with st.sidebar.expander("⚙️ AI Configuration", expanded=True):
    # Simplified to use Gemini as the primary example, but easily extendable
    gemini_key = st.text_input("Google Gemini API Key", type="password")

with st.sidebar.expander("🎨 Customization"):
    candidate_name = st.text_input("Your Full Name", placeholder="e.g., Alex Doe")
    resume_tone = st.selectbox("Resume Tone", ["Professional", "Dynamic", "Technical"])

c1, c2 = st.columns(2)
with c1:
    resumes = st.file_uploader("1. Upload Your Resume(s)", type=["txt", "docx", "pdf"], accept_multiple_files=True)
with c2:
    jd_text = st.text_area("2. Paste the Job Description", height=245)

if st.button("✨ Analyze & Generate", type="primary", use_container_width=True):
    if not (resumes and jd_text and gemini_key):
        st.warning("Please upload a resume, paste the job description, and enter your Gemini API key.")
        st.stop()
    
    with st.spinner("Performing deep analysis... This may take a moment."):
        
        # Step 1: Detect the job field
        detected_field = detect_job_field_from_jd(jd_text, gemini_key)
        st.info(f"Detected Job Field: **{detected_field}**")
        st.session_state.detected_field = detected_field

        # Step 2: Extract keywords
        st.session_state.jd_keywords = extract_keywords_smarter(jd_text)
        
        # Step 3: Analyze each uploaded resume
        all_results = []
        for rf in resumes:
            text = read_file_to_text(rf)
            if not text: continue
            
            text_norm = normalize_whitespace(text)
            
            all_results.append({
                "name": rf.name, 
                "text": text_norm, 
                "score": round(compute_similarity(text_norm, jd_text) * 100),
                "coverage": keyword_coverage(text_norm, st.session_state.jd_keywords),
                "ats": analyze_ats_friendliness(text_norm),
                "verbs_metrics": analyze_action_verbs_and_metrics(text_norm)
            })
        
        if not all_results:
            st.error("Could not process any of the uploaded resumes.")
            st.stop()
            
        all_results.sort(key=lambda x: x["score"], reverse=True)
        st.session_state.analysis_results = all_results
        best_resume = all_results[0]

        # Step 4: Generate the tailored resume using the new dynamic prompt
        missing_keywords = [k for k, v in best_resume['coverage'].items() if v == 0]

        tail_prompt = f"""
        You are an expert AI Talent Acquisition Specialist and Professional Resume Writer. Your mission is to analyze the provided Job Description for a '{detected_field}' role and rewrite the Original Resume to maximize its chances of passing both automated (ATS) and human reviews.

        - Identify the most critical skills, technologies, and qualifications mentioned in the job description.
        - Intelligently and naturally weave these missing elements into the 'SUMMARY', the most recent 'EXPERIENCE' section, and the 'SKILLS' section. The following keywords are likely missing and should be included: {', '.join(missing_keywords)}.
        - Rephrase existing bullet points to use stronger action verbs and align them with the responsibilities in the job description.
        - Ensure the final output is a complete, professional, and highly targeted resume in a standard, parsable format.
        - Ensure ALL SECTION HEADINGS (like 'SUMMARY', 'EXPERIENCE') are IN ALL CAPS.
        - Return ONLY the full text of the rewritten resume and nothing else.

        Job Description:
        ---
        {jd_text}
        ---

        Original Resume:
        ---
        {best_resume['text']}
        ---
        """
        
        st.session_state.tailored_resume = call_llm(tail_prompt, gemini_key, is_json=False)
        st.success("Analysis and tailoring complete!")


if 'analysis_results' in st.session_state:
    all_results = st.session_state.analysis_results
    jd_keywords = st.session_state.jd_keywords
    best_resume = all_results[0]

    st.header("Results Dashboard", divider='rainbow')

    tab_insights, tab_resume, tab_export = st.tabs(["📊 Analysis Insights", "✍️ Tailored Resume", "💾 Export Documents"])

    with tab_insights:
        st.subheader(f"🏆 Top Resume: {best_resume['name']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Similarity Score", f"{best_resume['score']}/100")
        m2.metric("ATS Score", f"{best_resume['ats']['score']}/100")
        m3.metric("Action Verbs", best_resume['verbs_metrics']['action_verbs'])
        m4.metric("Quant. Metrics", best_resume['verbs_metrics']['quantitative_metrics'])
        st.divider()

        st.subheader("Keyword Analysis")
        coverage_data = best_resume['coverage']
        found_kws = sum(coverage_data.values())
        total_kws = len(coverage_data)
        
        st.progress(found_kws / total_kws if total_kws > 0 else 0, text=f"{found_kws} of {total_kws} keywords found")
        
        missing_keywords_str = ", ".join([f"`{k}`" for k, v in coverage_data.items() if v == 0])
        if missing_keywords_str:
            st.warning(f"**Missing Keywords:** {missing_keywords_str}")
        else:
            st.success("🎉 All keywords found!")

    with tab_resume:
        st.subheader("AI-Tailored Resume")
        st.markdown("This resume has been rewritten by the AI to align with the job description. You can edit it below.")
        st.text_area("Tailored Resume Content:", value=st.session_state.get('tailored_resume', ""), height=600, key="tailored_text_area")

    with tab_export:
        st.subheader("Download Your Documents")
        tailored_text_export = st.session_state.get("tailored_text_area", "")
        if tailored_text_export:
            st.info("Click a button below to download your tailored resume.")
            c1, c2, c3 = st.columns(3)
            with c1:
                docx_data = format_docx_content(tailored_text_export, keywords_to_bold=jd_keywords)
                if docx_data:
                    st.download_button("⬇️ Download as DOCX", data=docx_data, file_name="AI_Tailored_Resume.docx", use_container_width=True)
            with c2:
                pdf_data = format_pdf_content(tailored_text_export)
                if pdf_data:
                    st.download_button("⬇️ Download as PDF", data=pdf_data, file_name="AI_Tailored_Resume.pdf", use_container_width=True)
            with c3:
                st.download_button("⬇️ Download as TXT", data=tailored_text_export.encode('utf-8'), file_name="AI_Tailored_Resume.txt", use_container_width=True)
        else:
            st.warning("Generate a tailored resume first before exporting.")