import streamlit as st
import os
import tempfile
import textwrap
from docx import Document
from docx.shared import Pt, RGBColor
# Using a more robust PDF generation library: reportlab
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
import google.generativeai as genai
from openai import OpenAI

# ------------------------------------------------------------------
# File Formatting Utilities
# These functions handle the creation of DOCX and PDF files.
# ------------------------------------------------------------------

def format_docx(content, file_path):
    """Formats the given text content and saves it as a DOCX file."""
    try:
        doc = Document()
        # Set default font for the document
        style = doc.styles['Normal']
        font = style.font
        font.name = "Calibri"
        font.size = Pt(11)

        # Simple parsing to identify headers and bullet points
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Treat uppercase lines with few words as section headers
            if line.isupper() and len(line.split()) < 6:
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(0x00, 0x33, 0x66) # Dark blue color
                p.paragraph_format.space_before = Pt(12)
                p.paragraph_format.space_after = Pt(6)
            # Treat lines starting with common bullet symbols as list items
            elif line.startswith(("-", "•", "*")):
                # Remove the bullet character before adding, as the style will add its own
                p = doc.add_paragraph(line[1:].strip(), style="List Bullet")
                p.paragraph_format.left_indent = Pt(36)
            # Regular text
            else:
                doc.add_paragraph(line)

        doc.save(file_path)
    except Exception as e:
        st.error(f"Error creating DOCX file: {e}")

def format_pdf(content, file_path):
    """
    Formats the given text content and saves it as a PDF file using the robust ReportLab library.
    This permanently fixes issues with long words and text wrapping.
    """
    try:
        doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=inch, leftMargin=inch, topMargin=inch, bottomMargin=inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Add custom styles for headers and bullet points, using a unique name for the bullet style.
        styles.add(ParagraphStyle(name='Header', fontName='Helvetica-Bold', fontSize=12, spaceBefore=12, spaceAfter=6, textColor=HexColor('#003366')))
        styles.add(ParagraphStyle(name='CustomBullet', parent=styles['Normal'], leftIndent=20, firstLineIndent=0, spaceBefore=2))

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Section headers
            if line.isupper() and len(line.split()) < 6:
                p = Paragraph(line, styles['Header'])
                story.append(p)
            # Bullet points
            elif line.startswith(("-", "•", "*")):
                # ReportLab uses a <bullet> tag for proper formatting
                bullet_text = f"<bullet>&bull;</bullet>{line[1:].strip()}"
                p = Paragraph(bullet_text, styles['CustomBullet'])
                story.append(p)
            # Regular text
            else:
                p = Paragraph(line, styles['Normal'])
                story.append(p)
        
        doc.build(story)
    except Exception as e:
        st.error(f"An unexpected error occurred during PDF creation: {e}")


# ------------------------------------------------------------------
# AI Resume Tailoring Logic
# This function calls the selected AI backend to tailor the resume.
# ------------------------------------------------------------------

def tailor_resume(resume_text, jd_text, backend, api_key):
    """
    Calls the selected AI model to rewrite the resume based on the job description.
    """
    prompt = f"""
    As an expert resume writer, rewrite and tailor the following resume to perfectly match the provided Job Description.

    Your primary goals are:
    1.  **ATS-Friendly Formatting**: Ensure the output is clean, professional, and easily parsable by Applicant Tracking Systems. Use standard section headers (e.g., PROFESSIONAL SUMMARY, WORK EXPERIENCE, SKILLS, EDUCATION).
    2.  **Keyword Integration**: Naturally weave relevant keywords and phrases from the Job Description into the resume content, especially in the summary and experience sections.
    3.  **Conciseness and Impact**: Rephrase bullet points to be action-oriented and results-driven. Start each point with a strong action verb.
    4.  **Clarity**: The final output should be clear, professional, and ready to be saved as a document.

    Job Description:
    ---
    {jd_text}
    ---

    Original Resume:
    ---
    {resume_text}
    ---

    Provide ONLY the rewritten and improved resume content. Do not include any introductory phrases, markdown formatting, or extra symbols.
    """

    # --- Google Gemini Backend ---
    if backend == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error(f"An error occurred with Google Gemini: {e}")
            return None

    # --- OpenAI GPT Backend ---
    elif backend == "OpenAI GPT":
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional resume writer."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"An error occurred with OpenAI GPT: {e}")
            return None

# ------------------------------------------------------------------
# Streamlit User Interface
# ------------------------------------------------------------------

def display_resume_preview(content):
    """Displays a formatted preview of the resume in the Streamlit UI."""
    st.markdown("---")
    st.subheader("4. Preview Your Tailored Resume")
    
    # Use a container with a border for a cleaner look
    with st.container():
        # Simple conversion to Markdown for preview
        preview_content = ""
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                preview_content += "\n"
                continue
            
            if line.isupper() and len(line.split()) < 6:
                preview_content += f"#### {line}\n"
            elif line.startswith(("-", "•", "*")):
                preview_content += f"- {line[1:].strip()}\n"
            else:
                preview_content += f"{line}  \n"
        
        st.markdown(preview_content)


# --- Page Configuration ---
st.set_page_config(page_title="AI Resume Tailor", page_icon="📄", layout="wide")

# --- Header ---
st.title("📄 AI Resume Tailor")
st.markdown("Upload your resume and paste a job description to get a tailored version optimized by AI.")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    backend = st.radio("Select AI Backend:", ["Google Gemini", "OpenAI GPT"], help="Choose the AI model you want to use.")
    api_key = st.text_input("Enter Your API Key", type="password", help="Your API key is required to use the AI service.")
    st.markdown("---")
    st.info("Your data is not stored. All processing happens in memory.")


# --- Main Content Area ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Your Resume")
    resume_file = st.file_uploader("Supported formats: TXT, DOCX", type=["txt", "docx"])

with col2:
    st.subheader("2. Paste the Job Description")
    jd_text = st.text_area("Paste the full job description here", height=250)

# --- Generate Button and Logic ---
if st.button("✨ Generate Tailored Resume", type="primary", use_container_width=True):
    if not api_key:
        st.warning("Please enter your API key in the sidebar.")
    elif not resume_file:
        st.warning("Please upload your resume file.")
    elif not jd_text:
        st.warning("Please paste the job description.")
    else:
        with st.spinner('AI is tailoring your resume... Please wait.'):
            try:
                # --- Read uploaded file ---
                if resume_file.name.endswith(".txt"):
                    resume_text = resume_file.read().decode("utf-8")
                elif resume_file.name.endswith(".docx"):
                    doc = Document(resume_file)
                    resume_text = "\n".join([p.text for p in doc.paragraphs])

                # --- Call AI to tailor resume ---
                tailored_text = tailor_resume(resume_text, jd_text, backend, api_key)

                if tailored_text:
                    st.success("✅ Resume tailored successfully!")

                    # --- Create downloadable files in a temporary directory ---
                    with tempfile.TemporaryDirectory() as tmpdir:
                        docx_path = os.path.join(tmpdir, "Tailored_Resume.docx")
                        pdf_path = os.path.join(tmpdir, "Tailored_Resume.pdf")

                        format_docx(tailored_text, docx_path)
                        format_pdf(tailored_text, pdf_path)

                        # --- Display download buttons ---
                        st.subheader("3. Download Your New Resume")
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            with open(docx_path, "rb") as f_docx:
                                st.download_button(
                                    "⬇️ Download DOCX",
                                    data=f_docx,
                                    file_name="Tailored_Resume.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    use_container_width=True
                                )
                        with dl_col2:
                            # Check if PDF was created before showing download button
                            if os.path.exists(pdf_path):
                                with open(pdf_path, "rb") as f_pdf:
                                    st.download_button(
                                        "⬇️ Download PDF",
                                        data=f_pdf,
                                        file_name="Tailored_Resume.pdf",
                                        mime="application/pdf",
                                        use_container_width=True
                                    )
                            else:
                                st.error("PDF generation failed. Please check the error message above.")
                    
                    # --- Display the formatted preview ---
                    display_resume_preview(tailored_text)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
