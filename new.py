import streamlit as st
import os
import tempfile
import re
from collections import Counter
import PyPDF2
from docx import Document
from docx.shared import Pt, RGBColor
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import google.generativeai as genai
from openai import OpenAI
import plotly.graph_objects as go
import textstat

# ------------------------------------------------------------------
# File Formatting & Analysis Utilities
# ------------------------------------------------------------------

def format_docx(content, file_path):
    """Formats the given text content and saves it as a DOCX file."""
    try:
        doc = Document()
        style = doc.styles['Normal']
        font = style.font
        font.name = "Calibri"
        font.size = Pt(11)

        for line in content.split("\n"):
            line = line.strip()
            if not line: continue
            if line.isupper() and len(line.split()) < 6:
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
            elif line.startswith(("-", "•", "*")):
                p = doc.add_paragraph(line[1:].strip(), style="List Bullet")
                p.paragraph_format.left_indent = Pt(36)
            else:
                doc.add_paragraph(line)
        doc.save(file_path)
    except Exception as e:
        st.error(f"Error creating DOCX file: {e}")

def format_pdf(content, file_path):
    """Formats the given text content and saves it as a PDF file using ReportLab."""
    try:
        doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
        story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Header', fontName='Helvetica-Bold', fontSize=12, spaceBefore=12, spaceAfter=6, textColor=HexColor('#003366')))
        styles.add(ParagraphStyle(name='CustomBullet', parent=styles['Normal'], leftIndent=20, firstLineIndent=0, spaceBefore=2))

        for line in content.split("\n"):
            line = line.strip()
            if not line: continue
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
        st.error(f"An unexpected error occurred during PDF creation: {e}")

def get_keywords(text):
    """Utility function to extract keywords from text."""
    stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return words - stopwords

def grade_resume(resume_text):
    """Grades the resume based on readability, action verbs, and quantifiable metrics."""
    readability_score = textstat.flesch_reading_ease(resume_text)
    action_verbs = ['managed', 'led', 'developed', 'created', 'implemented', 'achieved', 'increased', 'reduced', 'improved']
    quantifiable_metrics = len(re.findall(r'\b\d+[\%]?\b', resume_text))
    
    verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
    
    # Grade calculation
    grade = "C"
    feedback = []
    if readability_score > 60 and verb_count > 3 and quantifiable_metrics > 2:
        grade = "A"
        feedback.append("✅ Excellent use of action verbs and quantifiable results.")
    elif readability_score > 50 and verb_count > 1 and quantifiable_metrics > 0:
        grade = "B"
        feedback.append("👍 Good start, but could be improved by adding more specific metrics and stronger action verbs.")
    else:
        feedback.append("⚠️ Needs improvement. Focus on using stronger action verbs and adding numbers to show your impact (e.g., 'increased sales by 15%').")

    return grade, readability_score, verb_count, quantifiable_metrics, feedback

# ------------------------------------------------------------------
# AI Generation Logic
# ------------------------------------------------------------------

def call_ai_backend(prompt, backend, api_key):
    """Generic function to call the selected AI backend."""
    if backend == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"An error occurred with Google Gemini: {e}")
            return None
    elif backend == "OpenAI GPT":
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional career coach who only outputs clean, plain text without any markdown."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"An error occurred with OpenAI GPT: {e}")
            return None

def tailor_resume(resume_text, jd_text, tone, backend, api_key):
    """Generates the tailored resume."""
    prompt = f"""
    As an expert resume writer, rewrite and tailor the following resume to perfectly match the provided Job Description, adopting a '{tone}' tone.
    Your primary goals are:
    1.  **ATS-Friendly Formatting**: Ensure clean, professional formatting with standard section headers.
    2.  **Keyword Integration**: Naturally weave relevant keywords from the Job Description into the resume.
    3.  **Impactful Language**: Rephrase bullet points to be action-oriented and results-driven.
    Job Description:\n---\n{jd_text}\n---\nOriginal Resume:\n---\n{resume_text}\n---
    CRITICAL INSTRUCTION: Provide ONLY the rewritten resume content as plain text, with no markdown formatting (like `**` or `*`).
    """
    return call_ai_backend(prompt, backend, api_key)

def generate_cover_letter(resume_text, jd_text, user_name, backend, api_key):
    """Generates a cover letter."""
    prompt = f"""
    As an expert career coach, write a concise and professional cover letter for '{user_name}' based on their tailored resume and the job description provided.
    The cover letter should:
    1.  Be 3-4 paragraphs long.
    2.  Highlight 2-3 key qualifications from the resume that directly match the job description.
    3.  Express enthusiasm for the role and the company.
    4.  End with a clear call to action.
    Job Description:\n---\n{jd_text}\n---\nTailored Resume:\n---\n{resume_text}\n---
    CRITICAL INSTRUCTION: Provide ONLY the cover letter content as plain text, with no markdown formatting.
    """
    return call_ai_backend(prompt, backend, api_key)

# ------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------

st.set_page_config(page_title="AI Resume Suite", page_icon="🚀", layout="wide")

st.title("🚀 AI Resume Suite")
st.markdown("Your all-in-one tool to analyze, tailor, and generate career documents with the power of AI.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    backend = st.radio("Select AI Backend:", ["Google Gemini", "OpenAI GPT"])
    api_key = st.text_input("Enter Your API Key", type="password")
    st.header("🎨 Customization")
    resume_tone = st.selectbox("Select Resume Tone:", ["Professional", "Creative", "Technical", "Enthusiastic"])
    st.markdown("---")
    st.info("Your data is not stored. All processing is done in memory.")

# --- Main Content Area ---
st.subheader("1. Provide Your Documents")
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Your Resume", type=["txt", "docx", "pdf"])
with col2:
    jd_text = st.text_area("Paste the Job Description", height=200)

if st.button("✨ Analyze & Generate Documents", type="primary", use_container_width=True):
    if not api_key or not resume_file or not jd_text:
        st.warning("Please provide your API Key, Resume, and Job Description.")
    else:
        with st.spinner('Working our magic... This may take a moment.'):
            # --- Read and Analyze ---
            resume_text = ""
            if resume_file.name.endswith(".txt"):
                resume_text = resume_file.read().decode("utf-8")
            elif resume_file.name.endswith(".docx"):
                doc = Document(resume_file)
                resume_text = "\n".join([p.text for p in doc.paragraphs])
            elif resume_file.name.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(resume_file)
                for page in pdf_reader.pages:
                    resume_text += page.extract_text()

            resume_keywords = get_keywords(resume_text)
            jd_keywords = get_keywords(jd_text)
            
            if not jd_keywords:
                st.error("Could not extract keywords from the job description. Please try a different one.")
            else:
                matching_keywords = resume_keywords.intersection(jd_keywords)
                score = int((len(matching_keywords) / len(jd_keywords)) * 100)
                grade, read_score, verb_count, metrics_count, feedback = grade_resume(resume_text)

                # --- AI Generation ---
                tailored_text = tailor_resume(resume_text, jd_text, resume_tone, backend, api_key)
                
                if tailored_text:
                    st.balloons()
                    st.success("✅ Documents generated successfully!")

                    # --- UI Tabs for Output ---
                    tab1, tab2, tab3 = st.tabs(["📊 Resume Analysis", "📄 Tailored Resume", "✉️ Cover Letter Generator"])

                    with tab1:
                        st.header("Resume Analysis")
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': "Job Description Match Score"},
                            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#2a9d8f"}}))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Resume Grade")
                        st.metric(label="Overall Grade", value=grade)
                        st.write(" ".join(feedback))
                        
                        r_col1, r_col2, r_col3 = st.columns(3)
                        r_col1.metric("Readability Score", f"{read_score:.1f}")
                        r_col2.metric("Action Verbs Found", verb_count)
                        r_col3.metric("Quantifiable Metrics", metrics_count)

                    with tab2:
                        st.header("Your Tailored Resume")
                        st.text_area("Resume Content", tailored_text, height=400)
                        
                        with tempfile.TemporaryDirectory() as tmpdir:
                            docx_path = os.path.join(tmpdir, "Tailored_Resume.docx")
                            pdf_path = os.path.join(tmpdir, "Tailored_Resume.pdf")
                            format_docx(tailored_text, docx_path)
                            format_pdf(tailored_text, pdf_path)

                            dl_col1, dl_col2 = st.columns(2)
                            with dl_col1:
                                with open(docx_path, "rb") as f:
                                    st.download_button("Download DOCX", f, "Tailored_Resume.docx", use_container_width=True)
                            with dl_col2:
                                if os.path.exists(pdf_path):
                                    with open(pdf_path, "rb") as f:
                                        st.download_button("Download PDF", f, "Tailored_Resume.pdf", use_container_width=True)
                    
                    with tab3:
                        st.header("Generate a Cover Letter")
                        user_name = st.text_input("Your Full Name", placeholder="e.g., Alex Doe")
                        if st.button("Generate Cover Letter", use_container_width=True):
                            if not user_name:
                                st.warning("Please enter your name.")
                            else:
                                with st.spinner("Writing your cover letter..."):
                                    cover_letter_text = generate_cover_letter(tailored_text, jd_text, user_name, backend, api_key)
                                    if cover_letter_text:
                                        st.text_area("Cover Letter Content", cover_letter_text, height=400)
