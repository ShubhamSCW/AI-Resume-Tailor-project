# pipeline.py — Advanced, AI-native resume synthesis pipeline.
from typing import Dict, List, Tuple
import json
import re
from ai.providers import ProviderManager
from ai.prompts import (
    jd_analysis_prompt,
    generative_synthesis_prompt,
    cover_letter_prompt,
    interview_prep_prompt,
    skills_gap_prompt,
)

# ----------------- Helpers -----------------
def _as_text(x) -> str:
    if isinstance(x, dict):
        return x.get("raw_text", "") or ""
    return str(x or "")

def _safe_json_extract(txt: str) -> Dict:
    if not txt: return {}
    try: return json.loads(txt)
    except Exception: pass
    try:
        s = txt.find("{"); e = txt.rfind("}")
        if s != -1 and e != -1 and e > s: return json.loads(txt[s:e+1])
    except Exception: return {}
    return {}

def _readability(text: str) -> Dict:
    try:
        import textstat
        grade = float(textstat.flesch_kincaid_grade(text or ""))
        rating = "Good" if 8 <= grade <= 12 else ("Simple" if grade < 8 else "Complex")
        return {"flesch_grade": round(grade, 1), "rating": rating}
    except Exception: return {"flesch_grade": "N/A", "rating": "N/A"}

def _ats_score(text: str) -> Tuple[int, Dict[str, str]]:
    checks, score = {}, 100
    if bool(re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text or "", re.I)) and bool(re.search(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4})", text or "")):
        checks["✅ Contact Info"] = "Email & phone detected."
    else: score -= 20; checks["❌ Contact Info"] = "Missing clear email or phone."
    if all(re.search(fr"^\s*{s}\b", text or "", re.I | re.M) for s in ["experience", "education", "skills"]):
        checks["✅ Sections"] = "Experience/Education/Skills present."
    else: score -= 20; checks["⚠️ Sections"] = "Add Experience, Education, Skills headings."
    wc = len((text or "").split())
    if 250 <= wc <= 1600: checks["✅ Length"] = f"Word count OK ({wc})."
    else: score -= 10; checks["⚠️ Length"] = f"Word count {wc}. Aim 250–1600."
    return max(0, score), checks

def _normalize_kw_list(items) -> List[str]:
    out, seen = [], set()
    for x in items or []:
        s = (str(x or "").strip())
        if len(s) < 2: 
            continue
        key = re.sub(r"[\s\-/_.]+", " ", s.lower()).strip()
        if key.endswith("s") and len(key) > 3:  # simple plural fold
            key = key[:-1]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# ---------- Keyword presence with variants ----------
def _kw_variants(kw: str) -> List[str]:
    k = kw.strip()
    parts = re.sub(r"[\s\-_]+", " ", k, flags=re.I).split()
    base = [
        k,
        re.sub(r"[\s\-_]+", "-", k),
        re.sub(r"[\s\-_]+", " ", k),
        re.sub(r"[\s\-_]+", "", k),
    ]
    # plural/singular lightweight variants
    if len(parts) == 1:
        p = parts[0]
        base += [p.rstrip("s"), p + "s", p + "es"]
    return sorted(set([b.lower() for b in base if b]))

def _contains_kw(text_low: str, kw: str) -> bool:
    for v in _kw_variants(kw):
        # allow boundaries around non-alnum; handle hyphen/space joins
        if re.search(rf"(?:^|[^a-z0-9]){re.escape(v)}(?:[^a-z0-9]|$)", text_low):
            return True
    return False

# ----------------- Post-processor -----------------
def postprocess_resume_text(text: str) -> str:
    if not text: return ""
    t = text.replace("\r\n", "\n").replace("\u00A0", " ")
    t = re.sub(r'\*\*(.*?)\*\*', r'\1', t)  # strip markdown bold
    t = re.sub(r'^\s*#+\s*(.*?)\s*$', r'\1', t, flags=re.MULTILINE)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

# ----------------- AI-Native Pipeline -----------------
class ProResumePipeline:
    def __init__(self, providers: ProviderManager, **_):
        self.providers = providers

    def _json(self, prompt: str, keys: Dict, order=None) -> Dict:
        out = self.providers.generate(prompt, json_mode=True, keys=keys, order=order)
        return _safe_json_extract(out)

    def _text(self, prompt: str, keys: Dict, order=None) -> str:
        return self.providers.generate(prompt, json_mode=False, keys=keys, order=order)

    def run_all(self, jd, resume, keys: Dict) -> Dict:
        order = self.providers.order
        jd_txt = _as_text(jd)
        rs_txt = _as_text(resume)

        # 1) JD analysis
        jd_analysis = self._json(jd_analysis_prompt(jd_txt), keys, order)
        job_title = jd_analysis.get("job_title", "the target role")
        jd_keywords = _normalize_kw_list(jd_analysis.get("keywords", []))

        # 2) Synthesis
        tailored_resume = self._text(
            generative_synthesis_prompt(
                jd_text=jd_txt,
                resume_text=rs_txt,
                job_field=job_title,
                keywords=jd_keywords,
            ),
            keys, order
        )
        final_resume = postprocess_resume_text(tailored_resume)

        # 3) Post-analysis on the synthesized resume
        present_keywords, missing_keywords = [], []
        rlow = re.sub(r"[\s\-_]+", " ", (final_resume or "").lower())
        for k in jd_keywords:
            if _contains_kw(rlow, k):
                present_keywords.append(k)
            else:
                missing_keywords.append(k)

        match_pct = int(round(100.0 * len(present_keywords) / max(1, len(jd_keywords))))
        ats_score, _ = _ats_score(final_resume)
        readability = _readability(final_resume)

        # 4) Cover letter
        cover_letter = self._text(cover_letter_prompt(jd_txt, final_resume, job_title), keys, order)

        # 5) Interview prep & skills gap (gap vs original resume)
        interview_guide = self._text(interview_prep_prompt(jd_txt, final_resume), keys, order)
        skills_gap_report = self._json(skills_gap_prompt(jd_txt, rs_txt), keys, order)

        return {
            "match_report": {
                "match_percentage": match_pct,
                "missing_keywords": missing_keywords,
                "present_keywords": present_keywords,
                "ats_score": ats_score,
                "readability": readability,
            },
            "tailored_resume": final_resume,
            "cover_letter": cover_letter.strip(),
            "interview_guide": interview_guide.strip(),
            "skills_gap_report": skills_gap_report,
        }
