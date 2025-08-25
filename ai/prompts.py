# ai/prompts.py — Advanced, persona-driven generative prompts.

def _join_list(lst):
    if not lst:
        return ""
    return ", ".join([str(x) for x in lst])

# --------- JD Analysis (Unchanged but still important) ---------
def jd_analysis_prompt(jd_text: str) -> str:
    return f"""
You are an expert recruitment analyst. Your task is to analyze the provided Job Description (JD) and return a structured JSON object.

Rules for Keyword Extraction:
- **Focus on hard skills**: Identify specific technologies (e.g., Python, React, AWS), software (e.g., Salesforce, Jira), tools, methodologies (e.g., Agile, Scrum), and tangible qualifications (e.g., CPA, PMP).
- **Create concise phrases**: Combine words into meaningful technical terms (e.g., "RESTful APIs", "data modeling").
- **Exclude generic terms**: Do NOT include soft skills (e.g., "communication") or common business jargon (e.g., "responsibilities").
- **Rank by importance**: The most critical keywords should appear first.

Return a JSON object with this schema:
{{
  "job_title": "Inferred job title",
  "keywords": ["Top keyword 1", "Top keyword 2", ...]
}}

Job Description:
---
{jd_text}
---
Ensure your output is ONLY the valid JSON object.
""".strip()

# --------- Advanced Resume Synthesis (NEW MASTER PROMPT) ---------
def generative_synthesis_prompt(jd_text: str, resume_text: str, job_field: str, keywords: list) -> str:
    """
    This is the core generative prompt. It adopts a persona and synthesizes a new resume.
    """
    kw_str = _join_list(keywords)
    return f"""
You are a world-class professional resume writer and an expert career coach specializing in the '{job_field}' field.
Your mission is to synthesize a brand-new, elite-tier resume for your client based on their original resume and the target job description.

**Core Directives:**
1.  **Synthesize, Don't Just Edit:** You will write a new resume from the ground up. Use the "Original Resume" only as a database of facts (jobs, skills, metrics, dates). Do not be constrained by its original phrasing, order, or structure.
2.  **Fact-Based Generation:** You MUST NOT invent or hallucinate facts. Every skill, project, job, date, and metric in the new resume must be derived directly from the "Original Resume".
3.  **JD-First Alignment:** The new resume's narrative, from the professional summary to the bullet points, must be laser-focused on the requirements and keywords in the "Job Description". Use the provided keyword list to ensure full alignment.
4.  **Quantify Everything:** Rephrase experience bullet points to lead with strong action verbs and focus on measurable, quantified outcomes (e.g., "Reduced latency by 30%," "Increased user engagement by 15%," "Managed a budget of $500K"). Follow the STAR (Situation, Task, Action, Result) method.
5.  **Professional Structure:** The output must be a complete resume in plain text. It MUST begin with the candidate's contact information, followed by standard, ATS-friendly section headers in ALL CAPS. The required structure is:
    - **Candidate Name** (as the very first line)
    - **Email | Phone | LinkedIn URL | Extra Link** (as the second line, separated by '|')
    - **PROFILE SUMMARY**
    - **CORE SKILLS**
    - **WORK EXPERIENCE**
    - **EDUCATION**
    - **CERTIFICATIONS** (if any)

**Inputs:**
---
**Target Job Description:**
{jd_text}
---
**Key Keywords to Emphasize:**
{kw_str}
---
**Original Resume (Fact Source):**
{resume_text}
---

Produce the complete, synthesized, and tailored resume as clean, plain text, strictly following the structure defined above. Do not include any commentary before or after the resume.
""".strip()

# --------- Cover Letter (Enhanced for Synthesis) ---------
def cover_letter_prompt(jd_text: str, tailored_resume_text: str, job_field: str) -> str:
    return f"""
Act as a professional career writer. Write a concise, compelling, and professional 200-word cover letter for the '{job_field}' role.
Use the provided tailored resume and job description to highlight 2-3 key achievements that directly match the company's needs.
The letter should be structured in three short paragraphs:
1.  Introduction: State the role you're applying for and your enthusiasm.
2.  Body: Connect your key achievements (with metrics) to the job requirements.
3.  Closing: Reiterate your interest and call to action.

**Job Description:**
---
{jd_text}
---
**Candidate's Tailored Resume:**
---
{tailored_resume_text}
---
Return ONLY the final letter text, with no extra commentary.
""".strip()

# --------- Other Prompts (Unchanged) ---------
def interview_prep_prompt(jd_text: str, resume_text: str) -> str:
    return f"""
Create a concise interview prep guide (MARKDOWN):
1) Top 3 Technical Questions (1 line each)
2) Top 3 Behavioral Questions (1 line each)
3) Key Talking Points (5 bullets based on resume)
4) 2 Insightful Questions for the Interviewer (1 line each)
Focus on the exact alignment between the JD and the resume.

JD:
---
{jd_text}
---
RESUME:
---
{resume_text}
---
""".strip()

def skills_gap_prompt(jd_text: str, resume_text: str) -> str:
    return f"""
Compare the JD and Resume to identify skill gaps for the candidate.
Return a JSON object with the schema:
{{
  "critical_gaps": ["List of essential skills the candidate is missing"],
  "nice_to_have": ["List of secondary skills the candidate is missing"],
  "learning_path": [
    {{ "topic": "Skill to learn", "resources": ["A book, course, or project idea"], "expected_outcome": "A measurable goal" }}
  ]
}}
JD:
---
{jd_text}
---
RESUME:
---
{resume_text}
---
Only return JSON, no commentary.
""".strip()