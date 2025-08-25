# app.py — Fixed state management for contact detail inputs.
from typing import Dict
import streamlit as st
from ai.providers import ProviderManager
from ai.selector import choose_order
from pipeline import ProResumePipeline
from utils import (
    file_to_json, extract_contact_info,
    format_pdf_preserve, format_docx_preserve,
    init_db, get_history, log_analysis,
)

st.set_page_config(page_title="ProResumeAI — Expert Synthesis", page_icon="✨", layout="wide")

# ---------- Theme ----------
CSS = """
<style>
:root{--bg:#f9fafb;--panel:#ffffff;--ink:#111827;--muted:#6b7280;--accent:#2563eb;--accent2:#9333ea;--border:#e5e7eb;--radius:12px;--shadow:0 2px 10px rgba(0,0,0,0.06);--font:'Inter','Segoe UI',system-ui,-apple-system,Arial,sans-serif}
html,body,[data-testid="stApp"]{background:var(--bg)!important;color:var(--ink);font-family:var(--font)}
.block-container{padding-top:1rem;padding-bottom:3rem}
.card{border:1px solid var(--border);border-radius:var(--radius);padding:16px 20px;background:var(--panel);box-shadow:var(--shadow)}
.small{color:var(--muted);font-size:13px}
.stButton>button,.stDownloadButton>button{border:none;border-radius:var(--radius);background:linear-gradient(90deg,var(--accent),var(--accent2));color:#fff;font-weight:600;padding:.6rem 1.1rem;box-shadow:var(--shadow)}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.05)}
textarea,.stTextInput input{background:#fff!important;border:1px solid var(--border)!important;color:#111827!important;border-radius:10px!important}
hr{border:none;height:1px;background:linear-gradient(90deg,transparent,#e5e7eb,transparent)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="card">'
    '<h2 style="margin:0">✨ ProResumeAI — Expert Synthesis Engine</h2>'
    '<div class="small">AI-driven resume generation with persona-based expertise.</div>'
    '</div>', unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔑 API Keys")
    keys: Dict[str, str] = {
        "openai": st.text_input("OpenAI API Key", type="password"),
        "gemini": st.text_input("Google Gemini API Key", type="password"),
        "anthropic": st.text_input("Anthropic API Key", type="password"),
        "ollama_base": st.text_input("Ollama Base (optional)", value=""),
        "ollama_model": st.text_input("Ollama Model (optional)", value=""),
    }
    st.caption("Keys are sent with each request and are not stored.")
    st.markdown("---")
    st.header("⚙️ AI Settings")
    strategy = st.selectbox("AI Model Strategy", ["Best quality", "Fastest", "Local first"])
    creativity = st.slider("Creativity", 0.0, 1.0, 0.25, 0.05)
    enforce_bullets = st.checkbox("Enforce concise metrics bullets", True)
    st.markdown("---")
    st.header("🗃️ Recent History")
    init_db()
    for r in get_history():
        st.caption(f"{r['timestamp'][:16]} · Score {r['match_score']} · {r['resume_name']}")

order = choose_order(keys, strategy=strategy)
providers = ProviderManager(order=order, temperature=creativity)
pipe = ProResumePipeline(providers, enforce_bullets=enforce_bullets)

# ---------- Session State Management ----------
defaults = {
    "result": None,
    "jd_text": "",
    "resume_text": "",
    "user_name": "",
    "user_email": "",
    "user_phone": "",
    "user_linkedin": "",
    "user_extra": "",
    "processed_jd_name": None,
    "processed_resume_name": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- Inputs ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("📄 Job Description")
    jd_file = st.file_uploader("Upload JD", type=["txt", "pdf", "docx"], key="jd_file")
    
    # FIXED: This logic now only runs when a NEW file is uploaded.
    if jd_file and jd_file.name != st.session_state.processed_jd_name:
        st.session_state.jd_text = file_to_json(jd_file)["raw_text"]
        st.session_state.processed_jd_name = jd_file.name
        
    st.text_area("JD (editable)", key="jd_text", height=300)

with c2:
    st.subheader("👤 Your Original Resume")
    r_file = st.file_uploader("Upload Resume", type=["txt", "pdf", "docx"], key="resume_file")
    
    # FIXED: This logic now only runs when a NEW file is uploaded, not on every keystroke.
    if r_file and r_file.name != st.session_state.processed_resume_name:
        st.session_state.resume_text = file_to_json(r_file)["raw_text"]
        st.session_state.processed_resume_name = r_file.name
        
        # When a new resume is uploaded, try to extract contact info just once.
        detected_info = extract_contact_info(st.session_state.resume_text)
        st.session_state.user_name = detected_info.get("name", "")
        st.session_state.user_email = detected_info.get("email", "")
        st.session_state.user_phone = detected_info.get("phone", "")
        st.session_state.user_linkedin = detected_info.get("linkedin", "")
        
    st.text_area("Original Resume (editable fact source)", key="resume_text", height=300)

st.markdown("---")

# ---------- Optional Contact Details Section ----------
st.subheader("👤 Verify or Add Contact Details (Optional)")
st.caption("We'll try to detect these from your resume. You can correct or add them here.")

col1, col2 = st.columns(2)
with col1:
    st.text_input("Full Name", key="user_name", placeholder="e.g., Jane Doe")
    st.text_input("Email Address", key="user_email", placeholder="e.g., jane.doe@email.com")
with col2:
    st.text_input("Phone Number", key="user_phone", placeholder="e.g., (123) 456-7890")
    st.text_input("LinkedIn Profile URL", key="user_linkedin", placeholder="e.g., linkedin.com/in/janedoe")

st.text_input("Extra Link (Portfolio, GitHub, etc.)", key="user_extra", placeholder="e.g., github.com/janedoe")

st.markdown("---")
st.subheader("🚀 Generate Tailored Assets")

col_full, col_clear = st.columns([3, 1])
if col_full.button("✨ Synthesize Resume & Cover Letter", use_container_width=True, type="primary"):
    if not st.session_state.jd_text.strip() or not st.session_state.resume_text.strip():
        st.warning("Please provide both a Job Description and your Original Resume.")
    elif not any(keys.values()):
        st.error("Please enter at least one API key in the sidebar.")
    else:
        with st.spinner("AI Expert is crafting your new resume... This may take a moment."):
            try:
                # --- LOGIC TO PREPEND CONTACT INFO ---
                contact_lines = []
                if st.session_state.user_name:
                    contact_lines.append(st.session_state.user_name)
                
                details_line = []
                if st.session_state.user_email: details_line.append(st.session_state.user_email)
                if st.session_state.user_phone: details_line.append(st.session_state.user_phone)
                if st.session_state.user_linkedin: details_line.append(st.session_state.user_linkedin)
                if st.session_state.user_extra: details_line.append(st.session_state.user_extra)
                
                if details_line:
                    contact_lines.append(" | ".join(details_line))

                final_resume_input = st.session_state.resume_text
                if contact_lines:
                    contact_header = "\n".join(contact_lines)
                    final_resume_input = f"{contact_header}\n\n---\n\n{st.session_state.resume_text}"
                
                res = pipe.run_all(st.session_state.jd_text, final_resume_input, keys)
                
                st.session_state.result = res
                st.success("Synthesis complete! Your new resume and assets are ready below.")
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

if col_clear.button("🧹 Clear Output", use_container_width=True):
    st.session_state.result = None
    st.rerun()

# ---------- Results & Export ----------
if st.session_state.result:
    res = st.session_state.result
    st.markdown("---")
    st.subheader("📊 Analysis of Your New Resume")
    mr = res["match_report"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("JD Match Score", f"{mr.get('match_percentage', 0)}%")
    k2.metric("ATS Score", f"{mr.get('ats_score', 0)}/100")
    k3.metric("Readability", mr.get('readability', {}).get('rating', "N/A"))
    k4.metric("Grade Level", mr.get('readability', {}).get('flesch_grade', "N/A"))

    st.markdown("### ✍️ Your New AI-Synthesized Resume")
    st.text_area(
        "Editable Resume Text",
        value=res.get("tailored_resume", ""),
        height=450,
        key="tailored_text_final"
    )

    exp1, exp2 = st.columns(2)
    final_resume_text = st.session_state.tailored_text_final
    try:
        pdf_io = format_pdf_preserve(final_resume_text)
        exp1.download_button("⬇️ Download as PDF",
                           data=pdf_io.getvalue(),
                           file_name="AI_Tailored_Resume.pdf",
                           mime="application/pdf",
                           use_container_width=True)
    except Exception as e:
        exp1.error(f"PDF export failed: {e}")

    try:
        docx_io = format_docx_preserve(final_resume_text)
        exp2.download_button("⬇️ Download as DOCX",
                           data=docx_io.getvalue(),
                           file_name="AI_Tailored_Resume.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True)
    except Exception as e:
        exp2.error(f"DOCX export failed: {e}")

    with st.expander("✉️ View Cover Letter, Interview Prep & Skills Gap"):
        st.markdown("### ✉️ Cover Letter")
        st.text_area("Cover Letter Text", value=res.get("cover_letter", ""), height=250)
        
        st.markdown("### 🎙️ Interview Prep Guide")
        st.markdown(res.get("interview_guide", "Not generated."))

        st.markdown("### 🧠 Skills Gap Report")
        st.json(res.get("skills_gap_report", {}))