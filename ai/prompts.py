# ai/prompts.py — Persona prompts with strict, professional generation

def _join_list(lst):
    if not lst:
        return ""
    return ", ".join([str(x) for x in lst])

# --------- JD Analysis (unchanged focus) ---------
def jd_analysis_prompt(jd_text: str) -> str:
    return f"""
You are a senior tech recruiter. Read the JD carefully and return a compact JSON with the inferred role and a
PRIORITIZED list of HARD-SKILL keywords that are explicitly important in this JD.

How to read the JD:
1) Infer the role name (short, conventional; no company/team/location).
2) Harvest ONLY concrete, testable hard skills/technical phrases that are explicitly required or emphasized, especially under:
   "Requirements", "Must Have", "Minimum Qualifications", "Qualifications", "Skills",
   "Technical Requirements", "Responsibilities".
   Prefer items that are:
   - In bullet lists in those sections
   - Marked by words: must, required, minimum, strongly preferred, expertise in
   - Capitalized acronyms (SQL, ETL, SSO, SOC 2, ISO 27001, CI/CD)
   - Cloud/services, frameworks, libraries, tools, protocols (AWS Lambda, EKS, Terraform, Kafka, Airflow, Grafana, Prometheus, ArgoCD, GitOps, Docker, Kubernetes)
   - Role/domain-specific terms when present (Zero Downtime Deployment, Observability, Cloud Cost Optimization)

Do NOT include:
- Soft skills (communication, leadership, ownership)
- Generic words (team, stakeholders, business)
- Vague phrases (fast-paced, dynamic)

Normalization rules:
- Use canonical short phrases (1–3 words when sensible): e.g., "REST APIs", "Data Modeling"
- Merge variants: "Postgres"/"PostgreSQL" → "PostgreSQL"; "RESTful API"/"REST API" → "REST APIs"
- Keep acronyms as-is (CI/CD, SOC 2, ISO 27001)
- Deduplicate

Target size: comprehensive but focused (typically 20–50) reflecting what’s actually emphasized.

Return ONLY this JSON object:
{{
  "job_title": "Inferred job title",
  "keywords": ["Most important first", "Next", "..."]
}}

Job Description:
---
{jd_text}
---
""".strip()



# --------- Advanced Resume Synthesis (professional, fact-only, line-separated) ---------
def generative_synthesis_prompt(jd_text: str, resume_text: str, job_field: str, keywords: list) -> str:
    """
    AI-first synthesis: the model infers structure, emphasis, and tone from the JD and the original resume.
    We keep hard constraints (fact-only, no hallucination, no bullets/numbers) but avoid prescribing a template.
    """
    def _join_list(lst):
        return ", ".join([str(x) for x in lst]) if lst else ""
    kw_str = _join_list(keywords)

    return f"""
You are a world-class resume writer and '{job_field}' expert. Using ONLY information that already exists in the Original Resume,
create a polished, ATS-friendly resume aimed at a '{job_field}' role.

Think silently first:
- Infer seniority, specialization, and emphasis from the JD and the Original Resume.
- Infer which sections are truly valuable for this candidate and the best order for this target role.
- Infer which achievements and skills most strongly satisfy the JD and the keyword list.
- As a '{job_field}' expert, decide on the most impactful industry-specific terminology to use.
- Plan a crisp narrative and line-level phrasing. Do not reveal your planning.

Hard constraints:
- FACT-ONLY: Do not introduce any skill, project, employer, title, date, or metric that is not present in the Original Resume.
- JD ALIGNMENT: Prioritize achievements and skills that match the JD and these keywords: {kw_str}
- NO BULLETS / NO NUMBERS: Each achievement is a single sentence on its own line (plain line breaks only).
- KEEP TITLES & DATES EXACT if they appear in the Original Resume.
- If metrics exist in the Original Resume, reuse them precisely; otherwise keep lines truthful without inventing numbers.
- Avoid fluff, first-person, cliches, and vague statements.

Output expectations (AI-decided, not hard-coded):
- Start with the candidate’s name and a single-line contact header if those exist in the Original Resume; copy the contact line verbatim where possible.
- Choose sensible section labels commonly recognized by ATS (e.g., PROFILE SUMMARY, CORE SKILLS, WORK EXPERIENCE, EDUCATION, CERTIFICATIONS ETC) but include only what applies to the candidate based on the Original Resume.
- Order and density of content are up to you, based on JD fit and resume facts.
- For the CORE SKILLS section, identify key skills from the Original Resume that align with the JD. For each skill, present it on a new line, followed by a brief, one-line description that contextualizes the skill based on the candidate's actual experience. For example: "Python: Leveraged for backend development, data analysis, and automation scripts."
- For WORK EXPERIENCE, list roles in reverse-chronological order; under each role, write 3–6 crisp one-line achievements (no bullets/numbers).

Inputs
---
Target Job Description:
{jd_text}
---
Original Resume (Fact Source Only; do not add new facts):
{resume_text}
---

Now produce the final resume as clean plain text. 
Do not include analysis, notes, or any commentary — only the resume content.
""".strip()


# --------- Cover Letter (unchanged logic, concise) ---------
def cover_letter_prompt(jd_text: str, tailored_resume_text: str, job_field: str) -> str:
    return f"""
Act as a professional career writer. Write a concise, compelling 200-word cover letter for the '{job_field}' role.
Use the tailored resume and JD to highlight 2–3 achievements that match the role. Facts only.

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

# --------- Other prompts (unchanged) ---------
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
