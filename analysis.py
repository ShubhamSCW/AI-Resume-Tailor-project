# ================================
# analysis.py
# ================================

import json
import re
import html
from typing import Any, Dict, List
from collections import Counter

import streamlit as st
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import ACTION_VERBS, RESUME_CLICHES
from utils import call_llm_json, call_llm_text # <-- Import from utils

# --- Basic Analysis Functions ---
def analyze_ats_friendliness(text: str) -> Dict[str, Any]:
    checks, score = {}, 100
    if not (re.search(r"[\w\.-]+@[\w\.-]+", text) and re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)):
        score -= 20; checks["❌ Contact Info"] = "Missing a clear email or phone number."
    else: checks["✅ Contact Info"] = "Email and phone number found."
    if re.search(r"linkedin\.com/in/", text, re.IGNORECASE): checks["✅ LinkedIn Profile"] = "LinkedIn profile URL found."
    else: score -= 5; checks["⚠️ LinkedIn Profile"] = "Consider adding your LinkedIn profile URL."
    found_sections = [s for s in ["experience", "education", "skills"] if re.search(fr"^\s*{s}\b", text, re.I | re.M)]
    if len(found_sections) < 3: score -= 25; checks["❌ Standard Sections"] = f"Missing standard sections."
    else: checks["✅ Standard Sections"] = "Found essential sections."
    word_count = len(text.split())
    if not (400 <= word_count <= 1000): score -= 10; checks["⚠️ Word Count"] = f"Word count is {word_count}. Ideal: 400-1000."
    else: checks["✅ Word Count"] = f"Good word count ({word_count})."
    return {"score": max(0, score), "checks": checks}

def analyze_readability(text: str) -> Dict[str, Any]:
    try:
        grade = textstat.flesch_kincaid_grade(text)
        rating = "Good" if 8 <= grade <= 12 else ("Simple" if grade < 8 else "Complex")
        return {"flesch_grade": f"{grade:.1f}", "rating": rating}
    except Exception: return {"flesch_grade": "N/A", "rating": "Error"}

def analyze_action_verbs(text: str) -> int:
    return len(set(re.findall(r"\b\w+\b", text.lower())).intersection(ACTION_VERBS))

def detect_cliches(text: str) -> List[str]:
    return [c for c in RESUME_CLICHES if re.search(r"\b" + re.escape(c) + r"\b", text, re.IGNORECASE)]

def compute_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b: return 0.0
    try:
        vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf = vect.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError: return 0.0

@st.cache_data(show_spinner=False)
def extract_keywords_smarter(jd_text: str, top_k: int = 30) -> List[str]:
    if not jd_text: return []
    try:
        import spacy
        try: nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(jd_text.lower())
        chunks = [chunk.text for chunk in doc.noun_chunks if not chunk.root.is_stop and len(chunk.text.split()) > 1]
        words = [t.lemma_ for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop and len(t.text) > 3]
        return [kw for kw, _ in Counter(chunks + words).most_common(top_k)]
    except (ImportError, OSError):
        return [kw for kw, _ in Counter(re.findall(r"\b[a-zA-Z-]{4,}\b", jd_text.lower())).most_common(top_k)]

def generate_ats_preview(resume_text: str, keywords: List[str]) -> str:
    clean_text = html.escape(re.sub(r"\s+", " ", resume_text).strip())
    if not keywords: return f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{clean_text}</pre>"
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(k) for k in sorted_keywords) + r")\b"
    highlighted = re.sub(pattern, r'<mark>\1</mark>', clean_text, flags=re.IGNORECASE)
    return f"<pre style='white-space: pre-wrap; word-wrap: break-word;'>{highlighted}</pre>"

# --- Core AI Pipeline ---
def perform_full_analysis(resume_text: str, jd_text: str, primary_backend: str, keys: Dict[str, str]):
    with st.status("Running analysis pipeline...", expanded=True) as status:
        status.update(label="Step 1/4: Running basic checks...")
        basic_scores = {"score": int(round(compute_similarity(resume_text, jd_text) * 100)), "ats": analyze_ats_friendliness(resume_text), "readability": analyze_readability(resume_text), "action_verbs": analyze_action_verbs(resume_text)}
        
        status.update(label="Step 2/4: Parsing resume with AI...")
        resume_json_prompt = f"Parse the following resume into structured JSON with keys: contact_info, summary, experience (list of objects with title, company, dates, responsibilities), education, and skills. Return ONLY valid JSON.\n\n{resume_text[:4000]}"
        resume_json = call_llm_json(resume_json_prompt, primary_backend, keys)

        status.update(label="Step 3/4: Analyzing writing performance with AI...")
        star_prompt = f"Analyze the 'responsibilities' in this JSON. Evaluate how well they follow the STAR method (Situation, Task, Action, Result). Quantified results are best. Return JSON with keys 'score' (0-100), 'summary', and 'feedback'.\n\n{json.dumps(resume_json.get('experience', []), indent=2)}"
        star_analysis = call_llm_json(star_prompt, primary_backend, keys)
        sentiment_prompt = f"Analyze the sentiment of this resume text (tone, confidence). Return JSON with 'label' and 'score' (0.0-1.0).\n\n{resume_text[:1000]}"
        performance_scores = {"star_analysis": star_analysis, "sentiment": call_llm_json(sentiment_prompt, primary_backend, keys), "cliches": detect_cliches(resume_text)}
        basic_scores["star_score"] = star_analysis.get("score", 0)

        status.update(label="Step 4/4: Extracting and analyzing keywords...")
        keywords = extract_keywords_smarter(jd_text)
        keyword_data = {k: bool(re.search(r"\b" + re.escape(k) + r"\b", resume_text.lower())) for k in keywords}
    
    return {"basic": basic_scores, "performance": performance_scores, "keywords": keyword_data, "keywords_list": keywords}