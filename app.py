import os
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import perform_full_analysis, call_llm_text, generate_ats_preview
from config import APP_VERSION
from utils import (
    clean_linkedin_jd, get_history, init_db, log_analysis,
    format_docx_stable, format_pdf_stable, read_file_to_text
)

# --- App Setup ---
st.set_page_config(page_title="AI Resume Suite", page_icon="🚀", layout="wide")
init_db()
st.title(f"🚀 Ultimate AI Resume Suite (v{APP_VERSION})")

# --- Session State ---
for key, default in [("analysis_run", False), ("analysis_data", {}), ("jd_input", "")]:
    if key not in st.session_state: st.session_state[key] = default

# --- Helper Functions ---
def create_radar_chart(data: dict):
    categories = ['Match Score', 'ATS Score', 'STAR Score', 'Readability']
    readability_map = {'Good': 100, 'Simple': 75, 'Complex': 50, 'N/A': 0, "Error": 0}
    scores = [data.get('score', 0), data.get('ats', {}).get('score', 0), data.get('star_score', 0), readability_map.get(data.get('readability', {}).get('rating'), 0)]
    fig = go.Figure(data=go.Scatterpolar(r=scores, theta=categories, fill='toself', line=dict(color='#6272a4')))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100]), bgcolor="#282a36"), showlegend=False, paper_bgcolor="#0E1117", font_color="#FAFAFA")
    return fig

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    with st.expander("⚙️ AI & API Settings", expanded=True):
        primary_backend = st.selectbox("Primary AI Backend", ["Google Gemini", "OpenAI GPT", "Anthropic Claude", "Ollama (local)"])
        keys = {"gemini": st.text_input("Gemini API Key", type="password"), "openai": st.text_input("OpenAI API Key", type="password"), "anthropic": st.text_input("Anthropic API Key", type="password")}
    st.header("📚 Analysis History")
    for r in get_history(): st.caption(f"{r['timestamp'][:16]} · Score {r['match_score']} · {r['resume_name']}")

# --- Main Interface ---
c1, c2 = st.columns(2)
with c1: resumes = st.file_uploader("1. Upload Your Resume(s)", type=["txt", "docx", "pdf"], accept_multiple_files=True)
with c2:
    st.text_area("2. Paste the Job Description", height=245, key="jd_input")
    if st.button("🧹 Clean JD", use_container_width=True) and st.session_state.jd_input:
        st.session_state.jd_input = clean_linkedin_jd(st.session_state.jd_input)

# --- Main Workflow ---
if st.button("✨ Analyze & Tailor Resumes", type="primary", use_container_width=True):
    if not (resumes and st.session_state.jd_input): st.warning("Please upload a resume and paste a job description.")
    else:
        st.session_state.analysis_data, st.session_state.resumes_list = {}, resumes
        for rf in resumes:
            text = read_file_to_text(rf)
            if text: st.session_state.analysis_data[rf.name] = perform_full_analysis(text, st.session_state.jd_input, primary_backend, keys)
        if st.session_state.analysis_data: st.session_state.analysis_run = True; st.rerun()
        else: st.error("Could not process any uploaded resumes.")

# --- Results Dashboard ---
if st.session_state.analysis_run and st.session_state.analysis_data:
    st.markdown("---")
    st.subheader("Results Dashboard")
    
    sorted_resumes = sorted(st.session_state.analysis_data.items(), key=lambda item: item[1]["basic"]["score"], reverse=True)
    selected_name_full = st.selectbox("Select a resume:", [name for name, data in sorted_resumes])
    data = st.session_state.analysis_data[selected_name_full]
    
    # FIXED: Remove original extension from filename to prevent .pdf.pdf
    selected_name_base = os.path.splitext(selected_name_full)[0]

    original_text = ""
    for r_file in st.session_state.resumes_list:
        if r_file.name == selected_name_full: original_text = read_file_to_text(r_file); break

    tabs = st.tabs(["📊 Insights", "🚀 Performance", "📄 ATS Preview", "✍️ Tailored Resume", "✉️ Cover Letter", "🎙️ Interview Prep", "💾 Export"])
    
    with tabs[0]: # Insights
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Overall Match", f"{data['basic']['score']}%")
            st.metric("ATS Friendliness", f"{data['basic']['ats']['score']}/100")
            st.metric("Readability", data['basic']['readability'].get('rating', 'N/A'))
            st.metric("Action Verbs", data['basic']['action_verbs'])
        with c2: st.plotly_chart(create_radar_chart(data['basic']), use_container_width=True)
        present = [k for k, v in data["keywords"].items() if v]
        missing = [k for k, v in data["keywords"].items() if not v]
        st.success(f"**Present Keywords ({len(present)}):** {', '.join(present) if present else 'None'}")
        st.warning(f"**Missing Keywords ({len(missing)}):** {', '.join(missing) if missing else 'None'}")

    with tabs[1]: # Performance
        st.progress(value=data['performance']['star_analysis'].get('score', 0), text=f"**STAR Method Score: {data['performance']['star_analysis'].get('score', 0)}/100**")
        with st.expander("**AI Feedback on STAR Method**"):
            st.markdown(f"**Summary:** {data['performance']['star_analysis'].get('summary', 'N/A')}")
            st.markdown(f"**Feedback:** {data['performance']['star_analysis'].get('feedback', 'N/A')}")
        st.markdown(f"**Sentiment:** **{data['performance']['sentiment'].get('label', 'N/A')}** (Score: {data['performance']['sentiment'].get('score', 0.0):.2f})")
        if data['performance'].get("cliches"): st.info("⚠️ **Clichés Detected:** " + ", ".join(data['performance']["cliches"]))
    
    with tabs[2]: # ATS Preview
        st.subheader("📄 Applicant Tracking System (ATS) Preview")
        preview_html = generate_ats_preview(original_text, data.get("keywords_list", []))
        st.markdown(preview_html, unsafe_allow_html=True)

    # --- Generative Tabs ---
    for key_suffix, tab, title, prompt_template in [
        ("resume", tabs[3], "✍️ AI-Tailored Resume", "You are an expert resume writer. Perform a surgical, additive-only enhancement of the resume based on the job description. ONLY add missing keywords naturally. Do NOT remove content. Output ONLY the complete, plain-text resume.\n\nJD:\n{jd}\n\nRESUME:\n{resume}"),
        ("cover_letter", tabs[4], "✉️ AI-Generated Cover Letter", "You are a career coach. Write a compelling 3-4 paragraph cover letter connecting the resume strengths to the job's main requirements. Output ONLY the final cover letter text.\n\nJD:\n{jd}\n\nRESUME:\n{resume}"),
        ("interview_prep", tabs[5], "🎙️ AI-Generated Interview Prep", "You are an expert interview coach. Create a concise interview prep guide with: Top 3 technical questions, Top 3 behavioral questions, Key talking points for the candidate, and 2 questions for the interviewer. Use markdown.\n\nJD:\n{jd}\n\nRESUME:\n{resume}")
    ]:
        key = f"{key_suffix}_{selected_name_full}"
        with tab:
            st.subheader(title)
            if key not in st.session_state:
                if st.button(f"Generate {title.split(' ')[-1]}", key=f"btn_{key}", use_container_width=True):
                    with st.spinner("Generating..."):
                        prompt = prompt_template.format(jd=st.session_state.jd_input, resume=original_text)
                        st.session_state[key] = call_llm_text(prompt, primary_backend, keys)
                        st.rerun()
            if key in st.session_state:
                st.text_area("Edit the result:", value=st.session_state[key], height=400, key=f"editor_{key}")

    with tabs[6]: # Export
        st.subheader("💾 Download & Save Center")
        resume_text_export = st.session_state.get(f"resume_{selected_name_full}", original_text)
        cover_letter_export = st.session_state.get(f"cover_letter_{selected_name_full}", "")
        
        if st.button("💾 Log Analysis to History", use_container_width=True):
            log_analysis(job_title="N/A", resume_name=selected_name_full, match_score=data['basic']['score'], ats_score=data['basic']['ats']['score'], tailored_resume_text=resume_text_export, cover_letter_text=cover_letter_export)
            st.toast("Analysis logged!")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Tailored Resume")
            st.download_button("📄 Download DOCX", data=format_docx_stable(resume_text_export, primary_backend, keys), file_name=f"Resume_{selected_name_base}.docx", use_container_width=True, disabled=not resume_text_export)
            st.download_button("📑 Download PDF", data=format_pdf_stable(resume_text_export, primary_backend, keys), file_name=f"Resume_{selected_name_base}.pdf", use_container_width=True, disabled=not resume_text_export)
        with c2:
            st.markdown("#### Cover Letter")
            st.download_button("✉️ Download DOCX", data=format_docx_stable(cover_letter_export, primary_backend, keys), file_name=f"Cover_Letter_{selected_name_base}.docx", use_container_width=True, disabled=not cover_letter_export)
            st.download_button("📝 Download PDF", data=format_pdf_stable(cover_letter_export, primary_backend, keys), file_name=f"Cover_Letter_{selected_name_base}.pdf", use_container_width=True, disabled=not cover_letter_export)

else:
    st.info("👋 Welcome! Upload your resume, paste a job description, and click 'Analyze' to begin.")