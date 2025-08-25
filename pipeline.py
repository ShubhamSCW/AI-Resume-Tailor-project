# pipeline.py — AI-native resume synthesis pipeline with line-separated formatting.
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
def _is_acronym(s: str) -> bool:
    s2 = s.replace(" ", "")
    return 2 <= len(s2) <= 6 and s2.isupper() and s2.isalpha()

def _has_number(s: str) -> bool:
    return bool(re.search(r"\d", s))

def _wordish_variants(kw: str) -> List[str]:
    forms = {
        kw,
        kw.lower(),
        kw.upper(),
        re.sub(r"[\s\-_/]+", " ", kw),
        re.sub(r"[\s\-_/]+", "-", kw),
        re.sub(r"[\s\-_/]+", "", kw),
    }
    base = kw.rstrip("s")
    forms |= {base, base.lower(), base.upper()}
    return sorted(set([f for f in forms if f]))

def _salience_rank_keywords(jd_text: str, kws: List[str]) -> List[str]:
    """
    Score keywords by signals that indicate 'specially defined/important' in the JD:
    - frequency
    - presence in emphasized sections (requirements/must have/qualifications/skills/responsibilities)
    - presence on bullet lines
    - acronym bonus (SQL, ETL, CI/CD)
    - numbered standard bonus (SOC 2, ISO 27001)
    """
    text = jd_text or ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    sec_headers = re.compile(
        r"^\s*(requirements?|must[-\s]*have|minimum\s+qualifications?|qualifications?|skills?|technical\s+requirements?|responsibilities?)\s*:?$",
        re.I
    )
    bullet_re = re.compile(r"^\s*[\-\*\u2022\u2023\u25CF\u25AA\u00B7\u2013\u2014]\s+")
    emphasized_idx = set()
    in_emph = False
    for i, ln in enumerate(lines):
        if sec_headers.match(ln):
            in_emph = True
            emphasized_idx.add(i)
            continue
        if in_emph and (ln.strip() == "" or (ln.isupper() and len(ln.split()) <= 8)):
            in_emph = False
        if in_emph:
            emphasized_idx.add(i)

    scores = {}
    low_text = " " + re.sub(r"\s+", " ", text.lower()) + " "
    for kw in kws:
        s = 0.0
        # Frequency across variants
        freq = 0
        for v in _wordish_variants(kw):
            if not v: continue
            freq += len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(v.lower())}(?![A-Za-z0-9])", low_text))
        s += min(freq, 5) * 1.5

        # Emphasis section hits
        hit_emph = 0
        for i in emphasized_idx:
            ln_low = " " + lines[i].lower() + " "
            for v in _wordish_variants(kw):
                if v and re.search(rf"(?<![A-Za-z0-9]){re.escape(v.lower())}(?![A-Za-z0-9])", ln_low):
                    hit_emph += 1
                    break
        s += min(hit_emph, 5) * 2.0

        # Bullet line hits
        bullet_hits = 0
        for ln in lines:
            if bullet_re.match(ln):
                ln_low = " " + ln.lower() + " "
                for v in _wordish_variants(kw):
                    if v and re.search(rf"(?<![A-Za-z0-9]){re.escape(v.lower())}(?![A-Za-z0-9])", ln_low):
                        bullet_hits += 1
                        break
        s += min(bullet_hits, 5) * 1.2

        if _is_acronym(kw): s += 1.5
        if _has_number(kw): s += 1.2

        # If JD uses must/required/minimum and kw appears, slight global boost
        if freq > 0 and re.search(r"(must|required|min(?:imum)?)", low_text):
            s += 0.8

        scores[kw] = s

    ranked = sorted(kws, key=lambda k: (-scores.get(k, 0.0), k.lower()))
    return ranked[:60]

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
        if len(s) < 2: continue
        key = re.sub(r"[\s\-/_.]+", " ", s.lower()).strip()
        if key.endswith("s") and len(key) > 3:
            key = key[:-1]
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out

# ----------------- Formatting helpers -----------------
def _normalize_headers(line: str) -> str:
    norm_map = {
        r"^\s*professional\s+summary\s*$": "PROFILE SUMMARY",
        r"^\s*summary\s*$": "PROFILE SUMMARY",
        r"^\s*core\s+skills\s*$": "CORE SKILLS",
        r"^\s*skills\s*$": "CORE SKILLS",
        r"^\s*(work\s+experience|experience)\s*$": "WORK EXPERIENCE",
        r"^\s*education\s*$": "EDUCATION",
        r"^\s*certifications?\s*$": "CERTIFICATIONS",
    }
    for pat, repl in norm_map.items():
        if re.match(pat, line, flags=re.I):
            return repl
    return line

def _line_separated_skills_block(text: str) -> str:
    """Ensure CORE SKILLS section is line-separated (one skill per line)."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        out.append(ln)
        if re.match(r"^\s*CORE SKILLS\s*$", ln, flags=re.I):
            j = i + 1
            buf = []
            while j < len(lines):
                nxt = lines[j]
                if re.match(r"^\s*[A-Z][A-Z\s]{2,}\s*$", nxt) and nxt.strip() not in {"CORE SKILLS"}:
                    break
                if nxt.strip():
                    buf.extend([x.strip() for x in re.split(r"[,\|/;·•]", nxt) if x.strip()])
                j += 1
            # dedupe, then add each skill on its own line
            seen = set()
            for sk in buf:
                if sk.lower() not in seen:
                    seen.add(sk.lower())
                    out.append(sk)
            i = j
            continue
        i += 1
    return "\n".join(out)

def _line_separated_bullets(text: str) -> str:
    """Replace bullets/numbering with simple line-separated entries."""
    lines = []
    for ln in text.splitlines():
        # Strip bullets/numbers
        ln2 = re.sub(r"^\s*[\-\*\u2022\u2023\u25CF\u25AA\u00B7\u2013\u2014\d\.\)]+\s*", "", ln).strip()
        lines.append(ln2)
    return "\n".join(lines)

def postprocess_resume_text(text: str) -> str:
    if not text: return ""
    t = text.replace("\r\n", "\n").replace("\u00A0", " ")
    # remove markdown marks
    t = re.sub(r'\*\*(.*?)\*\*', r'\1', t)
    t = re.sub(r'^\s*#+\s*(.*?)\s*$', r'\1', t, flags=re.MULTILINE)
    # normalize headers
    lines = [_normalize_headers(ln) for ln in t.splitlines()]
    t = "\n".join(lines)
    # enforce line-separated bullets
    t = _line_separated_bullets(t)
    # enforce line-separated skills
    t = _line_separated_skills_block(t)
    # collapse extra blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)
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
        jd_keywords_raw = _normalize_kw_list(jd_analysis.get("keywords", []))
        jd_keywords = _salience_rank_keywords(jd_txt, jd_keywords_raw)

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

        # 3) Post-analysis
        present_keywords, missing_keywords = [], []
        rlow = re.sub(r"[\s\-_]+", " ", (final_resume or "").lower())
        for k in jd_keywords:
            if any(w in rlow for w in [k.lower(), k.lower().rstrip("s"), k.lower()+"s"]):
                present_keywords.append(k)
            else:
                missing_keywords.append(k)

        match_pct = int(round(100.0 * len(present_keywords) / max(1, len(jd_keywords))))
        ats_score, _ = _ats_score(final_resume)
        readability = _readability(final_resume)

        # 4) Cover letter
        cover_letter = self._text(cover_letter_prompt(jd_txt, final_resume, job_title), keys, order)

        # 5) Interview & skill gaps
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
