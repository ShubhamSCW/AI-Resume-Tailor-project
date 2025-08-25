# schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class APIKeys(BaseModel):
    gemini: Optional[str] = None
    openai: Optional[str] = None
    anthropic: Optional[str] = None
    ollama_base: Optional[str] = None
    ollama_model: Optional[str] = None

class TailorRequest(BaseModel):
    job_description: str = Field(..., min_length=20)
    resume_text: str = Field(..., min_length=20)
    api_keys: Optional[APIKeys] = None  # keys supplied from browser

class MatchReport(BaseModel):
    match_percentage: int
    missing_keywords: List[str]
    present_keywords: List[str]
    ats_score: int
    readability: Dict[str, str]

class TailorResponse(BaseModel):
    status: str
    match_report: MatchReport
    tailored_resume: str
    cover_letter: str
    interview_guide: str
    skills_gap_report: Dict

class SkillsGapRequest(BaseModel):
    job_description: str
    resume_text: str
    api_keys: Optional[APIKeys] = None

class SkillsGapResponse(BaseModel):
    status: str
    skills_gap_report: Dict
