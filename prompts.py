# ================================
# prompts.py
# ================================

def get_job_field_prompt(jd_text: str) -> str:
    """Generates a prompt to identify the job field from a job description."""
    return (
        "Analyze the following job description and return ONLY the specific job title or field "
        f"(e.g., 'Data Scientist', 'Senior Backend Engineer').\n\n---\n\n{jd_text[:1500]}"
    )

def get_resume_parsing_prompt(resume_text: str) -> str:
    """Generates a prompt to parse a resume into a structured JSON."""
    return (
        "You are an expert resume parser. Convert the following resume into a structured JSON object. "
        "The schema should include: `contact_info` (with name, location, phone, email, linkedin_url), "
        "`summary`, `experience` (a list of objects with title, company, dates, and a 'responsibilities' list), "
        "`education` (a list of objects with degree, university, year), and a `skills` list. "
        "If a section or field is not found, use a null or empty value. "
        "Return ONLY the valid JSON data.\n\n"
        f"---\n\n{resume_text[:4000]}"
    )

def get_jd_parsing_prompt(jd_text: str) -> str:
    """Generates a prompt to parse a job description into a structured JSON."""
    return (
        "You are an expert job description parser. Analyze the following job description and convert it into a structured JSON object. "
        "The JSON keys MUST be the actual section headings from the text (e.g., 'Responsibilities', 'Qualifications'). "
        "Also, ensure there is a top-level `job_title` key. "
        "Return ONLY the valid JSON data.\n\n"
        f"---\n\n{jd_text[:4000]}"
    )

def get_sentiment_analysis_prompt(resume_text: str) -> str:
    """Generates a prompt for sentiment analysis of a resume."""
    return (
        "Analyze the sentiment of the following resume text. Consider the tone, confidence, and professionalism. "
        'Return ONLY a valid JSON object with a "label" (e.g., "Highly Confident", "Professional", "Neutral") '
        'and a "score" (a float from -1.0 for negative to 1.0 for positive).\n\n'
        f'---\n\nText: "{resume_text[:1000]}"'
    )

def get_star_analysis_prompt(experience_json: str) -> str:
    """Generates a prompt to analyze the STAR method in experience bullet points."""
    return (
        "You are a career coach. Analyze the 'responsibilities' in the following JSON, which represents a resume's experience section. "
        "Evaluate how well the bullet points follow the STAR method (Situation, Task, Action, Result). "
        "A strong bullet point is concise and quantifies the result (e.g., 'Increased API performance by 30% by implementing caching'). "
        "Provide a concise summary, an overall score (0-100), and specific, actionable feedback on how to improve the bullet points. "
        'Return ONLY a valid JSON object with keys "score", "summary", and "feedback".\n\n'
        f"---\n\nExperience JSON:\n{experience_json}"
    )

def get_resume_tailoring_prompt(jd_text: str, resume_text: str, job_field: str) -> str:
    """Generates a prompt to tailor a resume for a specific job description."""
    return (
        f"You are an expert resume writer specializing in '{job_field}' roles. "
        "Perform a surgical, additive-only enhancement of the resume below based on the provided job description. "
        "Your goal is to naturally integrate missing keywords and align the resume's language with the job's requirements. "
        "Focus on the summary and experience sections. Do NOT remove or significantly alter existing content. "
        "The output should be the complete, enhanced resume in plain text format. Do NOT add commentary before or after.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text}\n\n"
        f"--- ORIGINAL RESUME ---\n{resume_text}"
    )

def get_cover_letter_prompt(jd_text: str, resume_text: str, job_field: str) -> str:
    """Generates a prompt to write a cover letter."""
    return (
        "You are a professional career coach. Write a compelling, professional, and concise 3-4 paragraph cover letter "
        f"for a '{job_field}' role. The letter should directly connect the candidate's strengths (from the resume) "
        "to the key requirements outlined in the job description. Maintain a confident and enthusiastic tone. "
        "The output should be ONLY the final cover letter text, without any introductory or concluding remarks.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text}\n\n"
        f"--- CANDIDATE'S RESUME ---\n{resume_text}"
    )

def get_interview_prep_prompt(jd_text: str, resume_text: str) -> str:
    """Generates a prompt for creating interview preparation notes."""
    return (
        "You are an expert interview coach. Based on the provided resume and job description, create a concise interview preparation guide. "
        "The guide should include:\n"
        "1. **Top 3 Technical Questions:** Likely technical questions based on the required skills.\n"
        "2. **Top 3 Behavioral Questions:** STAR-method based questions relevant to the role.\n"
        "3. **Key Talking Points:** 3-5 bullet points the candidate should emphasize to show they are a perfect fit.\n"
        "4. **Questions for the Interviewer:** Two insightful questions the candidate can ask.\n\n"
        "Format the output as clean markdown. Do NOT add any introductory text like 'Here is the guide...'.\n\n"
        f"--- JOB DESCRIPTION ---\n{jd_text}\n\n"
        f"--- CANDIDATE'S RESUME ---\n{resume_text}"
    )