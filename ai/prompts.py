# ai/prompts.py — Advanced, persona-driven generative prompts.

def _join_list(lst):
    if not lst:
        return ""
    return ", ".join([str(x) for x in lst])

# --------- JD Analysis (tightened extraction rules) ---------
def jd_analysis_prompt(jd_text: str) -> str:
    return f"""
You are an expert recruitment analyst. Analyze the Job Description (JD) and return a compact JSON object.

Rules for Keyword Extraction:
- Focus on HARD SKILLS only: technologies, frameworks, libraries, tools, platforms, certifications, data stores, protocols, methodologies.
- Normalize to canonical short phrases (1–3 words): e.g., "REST APIs", "data modeling", "ETL", "Kubernetes".
- Merge variants and avoid duplicates: e.g., "PostgreSQL" vs "Postgres" → "PostgreSQL".
- Exclude soft skills and generic words entirely.
- Rank by hiring impact (most critical first).

Return ONLY this JSON schema:
{{
  "job_title": "Inferred job title",
  "keywords": ["Top keyword 1", "Top keyword 2", ...]
}}

Job Description:
---
{jd_text}
---
""".strip()

# --------- Advanced Resume Synthesis (stricter fact guard) ---------
def generative_synthesis_prompt(jd_text: str, resume_text: str, job_field: str, keywords: list) -> str:
    kw_str = _join_list(keywords)
    return f"""
You are a world-class professional resume writer and career coach for '{job_field}' roles.
Synthesize a new, elite-tier resume based on the ORIGINAL resume facts and the target JD.

NON-NEGOTIABLE RULES (read carefully):
- FACT-ONLY: Do NOT invent or add any skill, project, employer, title, date, or metric that is NOT clearly present in the Original Resume. No speculation, no guessing.
- REPHRASE-ONLY: You may rewrite phrasing for clarity, impact, and quantification — but every bullet must be traceable to existing facts in the Original Resume.
- JD-FIRST EMPHASIS: Prioritize relevance to the JD and the keyword list. Prefer facts that align with the JD, reorder content accordingly.
- QUANTIFY: Where numbers exist in the Original Resume, surface them early using action verbs and STAR style.
- STRUCTURE (exactly):
  Candidate Name
  Email | Phone | LinkedIn URL | Extra Link
  PROFILE SUMMARY
  CORE SKILLS
  WORK EXPERIENCE
  EDUCATION
  CERTIFICATIONS

If a JD keyword is not present in the Original Resume, DO NOT add it; instead emphasize adjacent, truthful capabilities already present.

Inputs:
---
Target Job Description:
{jd_text}
---
Key Keywords to Emphasize:
{kw_str}
---
Original Resume (Fact Source Only — do not add new facts):
{resume_text}
---

Output: A complete, ATS-friendly resume in plain text using the exact section order above. No commentary before or after.
""".strip()

# --------- Cover Letter (unchanged, concise) ---------
def cover_letter_prompt(jd_text: str, tailored_resume_text: str, job_field: str) -> str:
    return f"""
Act as a professional career writer. Write a concise, compelling 200-word cover letter for the '{job_field}' role.
Use the tailored resume and JD to highlight 2–3 achievements with metrics that match the role.
Structure: 1) intro, 2) matching achievements, 3) close with call to action.

Job Description:
---
{jd_text}
---
Candidate's Tailored Resume:
---
{tailored_resume_text}
---
Return ONLY the final letter text.
""".strip()

def interview_prep_prompt(jd_text: str, resume_text: str) -> str:
    return f"""
Create a concise interview prep guide (MARKDOWN):
1) Top 3 Technical Questions (1 line each)
2) Top 3 Behavioral Questions (1 line each)
3) Key Talking Points (5 bullets based on resume)
4) 2 Insightful Questions for the Interviewer (1 line each)
Align tightly to the JD and resume.

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
Return ONLY this JSON:
{{
  "critical_gaps": ["Essential skills missing from the resume"],
  "nice_to_have": ["Secondary skills missing"],
  "learning_path": [
    {{ "topic": "Skill to learn", "resources": ["A book, course, or project idea"], "expected_outcome": "Measurable goal" }}
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
""".strip()
