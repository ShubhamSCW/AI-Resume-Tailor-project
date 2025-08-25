# client_app.py
import io
import json
import requests
import streamlit as st

st.set_page_config(page_title="ProResumeAI Client", page_icon="🧠", layout="wide")
st.title("🧠 ProResumeAI — Client UI")

# --- Backend URL ---
backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000", help="Your FastAPI server base URL")

# --- API Keys (kept in browser, sent per request only) ---
with st.sidebar:
    st.header("🔑 API Keys (sent per request)")
    gk = st.text_input("Gemini API Key", type="password")
    ok = st.text_input("OpenAI API Key", type="password")
    ck = st.text_input("Anthropic API Key", type="password")
    ollama_base = st.text_input("Ollama Base (optional)", value="")
    ollama_model = st.text_input("Ollama Model (optional)", value="")

st.subheader("1) Inputs")
col1, col2 = st.columns(2)
with col1:
    jd = st.text_area("📄 Job Description", height=260, placeholder="Paste the full JD here…")
with col2:
    resume = st.text_area("👤 Your Resume (plain text)", height=260, placeholder="Paste your resume text here…")

api_keys = {
    "gemini": gk or None,
    "openai": ok or None,
    "anthropic": ck or None,
    "ollama_base": ollama_base or None,
    "ollama_model": ollama_model or None,
}

st.markdown("---")
st.subheader("2) Run Tailoring")

c1, c2, c3 = st.columns(3)

if c1.button("🚀 Tailor (Full Pipeline)", use_container_width=True):
    if not jd.strip() or not resume.strip():
        st.warning("Please provide both JD and Resume.")
    else:
        with st.spinner("Running tailoring pipeline…"):
            try:
                r = requests.post(f"{backend_url}/api/v1/tailor", json={
                    "job_description": jd,
                    "resume_text": resume,
                    "api_keys": api_keys
                }, timeout=180)
                if r.status_code != 200:
                    st.error(f"Backend error: {r.status_code} — {r.text}")
                else:
                    data = r.json()
                    st.session_state["result"] = data
                    st.success("Done!")
            except Exception as e:
                st.error(f"Request failed: {e}")

if c2.button("🧩 Skills Gap Only", use_container_width=True):
    if not jd.strip() or not resume.strip():
        st.warning("Please provide both JD and Resume.")
    else:
        with st.spinner("Generating skills gap…"):
            try:
                r = requests.post(f"{backend_url}/api/v1/skills-gap", json={
                    "job_description": jd,
                    "resume_text": resume,
                    "api_keys": api_keys
                }, timeout=120)
                if r.status_code != 200:
                    st.error(f"Backend error: {r.status_code} — {r.text}")
                else:
                    st.session_state["skills_gap_only"] = r.json()
                    st.success("Done!")
            except Exception as e:
                st.error(f"Request failed: {e}")

if c3.button("🧪 Check API /docs", use_container_width=True):
    st.info("Open the FastAPI docs in a new tab: " + f"{backend_url}/docs")

st.markdown("---")
st.subheader("3) Results")

res = st.session_state.get("result")
if res:
    mr = res.get("match_report", {})
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Match %", mr.get("match_percentage", 0))
    k2.metric("ATS Score", mr.get("ats_score", 0))
    k3.metric("Readability", mr.get("readability", {}).get("rating", "N/A"))
    k4.metric("Flesch Grade", mr.get("readability", {}).get("flesch_grade", "N/A"))

    st.markdown("### 🔑 Keywords")
    cols = st.columns(2)
    with cols[0]:
        st.write("**Present:**")
        st.code(", ".join(mr.get("present_keywords", [])) or "—")
    with cols[1]:
        st.write("**Missing:**")
        st.code(", ".join(mr.get("missing_keywords", [])) or "—")

    st.markdown("### ✍️ Tailored Resume")
    st.text_area("Tailored Resume (editable before export)", value=res.get("tailored_resume", ""), height=420, key="tailored_text")

    st.markdown("### ✉️ Cover Letter")
    st.text_area("Cover Letter (TXT only)", value=res.get("cover_letter", ""), height=220, key="cover_text")

    st.markdown("### 🎙️ Interview Guide")
    st.markdown(res.get("interview_guide", ""))

    st.markdown("### 🧠 Skills Gap")
    st.json(res.get("skills_gap_report", {}))

    st.markdown("---")
    st.subheader("4) Export (PDF/DOCX via backend)")

    colx, coly = st.columns(2)
    with colx:
        if st.button("📄 Export PDF", use_container_width=True):
            try:
                # use current edited text for export
                exp_payload = {
                    "job_description": jd,
                    "resume_text": st.session_state.get("tailored_text") or res.get("tailored_resume", ""),
                    "api_keys": api_keys
                }
                r = requests.post(f"{backend_url}/api/v1/export/pdf", json=exp_payload, timeout=240)
                if r.status_code != 200:
                    st.error(f"Export failed: {r.status_code} — {r.text}")
                else:
                    st.success("PDF ready.")
                    st.download_button(
                        "⬇️ Download resume.pdf",
                        data=r.content,
                        file_name="resume.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"PDF export error: {e}")

    with coly:
        if st.button("📝 Export DOCX", use_container_width=True):
            try:
                exp_payload = {
                    "job_description": jd,
                    "resume_text": st.session_state.get("tailored_text") or res.get("tailored_resume", ""),
                    "api_keys": api_keys
                }
                r = requests.post(f"{backend_url}/api/v1/export/docx", json=exp_payload, timeout=240)
                if r.status_code != 200:
                    st.error(f"Export failed: {r.status_code} — {r.text}")
                else:
                    st.success("DOCX ready.")
                    st.download_button(
                        "⬇️ Download resume.docx",
                        data=r.content,
                        file_name="resume.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"DOCX export error: {e}")

sg = st.session_state.get("skills_gap_only")
if sg:
    st.markdown("### 🧠 Skills Gap (Single Endpoint)")
    st.json(sg)
