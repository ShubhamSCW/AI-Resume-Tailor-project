```markdown
# 🚀 ProResumeAI — The Ultimate Career Tailoring Suite (v1.2.0)

**AI-tailored resumes & cover letters — JD-first, ATS-first.**  
ProResumeAI ingests your Job Description and resume, analyzes gaps, and generates an **additive-only**, **bullet-perfect** resume, a tight cover letter, an interview prep guide, and a skills-gap learning path. It uses a **multi-model backend with automatic fallback** — you supply **API keys in the app sidebar**; nothing is stored.

---

## ✨ Why ProResumeAI?

Tailoring every application is hard. ProResumeAI automates the boring parts and upgrades the rest:

- **JD Analysis** → role inference, ranked keywords  
- **Resume Analysis** → ATS checks, readability, STAR feedback  
- **Tailoring** → additive-only, quantified bullets, enforced section headings  
- One-click **cover letter** and **interview prep**  
- **Premium PDF/DOCX export** via Playwright & DOCX exporters  

---

## 📦 Feature Highlights

### 🔍 Analysis & Matching
- **JD Reader** → role, responsibilities, ranked keywords  
- **Resume Match Report** → match %, present/missing keywords  
- **ATS Signals** → email/phone/sections/length checks  
- **Readability** → Flesch-Kincaid grade & rating  

### ✍️ Generation
- **Resume from Scratch** (optionally guided by JD)  
- **Two-pass Tailoring** → tailor ➜ polish (concise, quantified bullets)  
- **Cover Letter** (180–260 words)  
- **Interview Prep** → top technical & behavioral Qs, talking points, questions to ask  
- **Skills Gap** → critical gaps + learning path  

### 🧠 AI, Your Way
- Paste **API keys in the sidebar** (OpenAI, Gemini, Anthropic; optional Ollama)  
- **Automatic selection** of best available provider (Quality / Speed / Local-first)  
- **Graceful fallback** if a provider is unavailable or rate-limited  
- Keys are **never stored**; they’re only sent when you click actions  

### 📄 Export (Premium)
- **PDF** via Playwright + `template.html`  
- **DOCX** via python-docx  
- Exporters live in `utils.py` (unchanged semantics)  

---

## 🛠 Tech Stack

- **Frontend/UI**: Streamlit  
- **AI Providers**: Gemini / OpenAI / Anthropic / Ollama (local)  
- **NLP / Scoring**: spaCy, scikit-learn, textstat  
- **Export**: Playwright + Jinja2 (PDF), python-docx (DOCX)  
- **Database**: SQLite (local history)  
- **Security**: Keys supplied in-app (sidebar), per session, never stored  

---

## 🗺 Project Structure

```

proresumeai/
├── app.py              # Streamlit app (UI + pipeline)
├── pipeline.py         # Orchestration (JD → tailor → polish → CL → prep → gap)
├── utils.py            # File I/O, logging, exporters (PDF/DOCX)
├── config.py           # Constants (DB file, defaults)
├── template.html       # HTML template for PDF export
├── ai/
│   ├── providers.py    # AI provider abstraction (Gemini, OpenAI, Claude, Ollama)
│   ├── selector.py     # Auto provider order logic
│   └── prompts.py      # Prompts (analysis, tailoring, polish, CL, prep, gap, scratch)
├── requirements.txt    # Python dependencies
└── README.md

````

---

## 🚀 Getting Started — OS-Specific Setup

> **Prerequisites (all OS):**
> - Python **3.9+**
> - (Optional) [Ollama](https://ollama.ai/) if you want local LLMs (e.g., `ollama pull llama3.1`)
> - API keys for cloud providers (OpenAI / Gemini / Anthropic)

---

### macOS

```bash
# 1) Clone & enter
git clone https://github.com/your-username/proresumeai.git
cd proresumeai

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Playwright Chromium for PDF export (one-time)
python -m playwright install chromium

# 5) spaCy English model (one-time)
python -m spacy download en_core_web_sm

# 6) Run Streamlit app
streamlit run app.py
````

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

> **Troubleshooting (macOS):**
>
> * If Playwright errors:
>   `xattr -dr com.apple.quarantine ~/.cache/ms-playwright`
> * If you see “No module named spacy”:
>   run `pip install spacy` before step 5.

---

### Windows (PowerShell)

```powershell
REM 1) Clone & enter
git clone https://github.com/your-username/proresumeai.git
cd proresumeai

REM 2) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

REM 3) Install dependencies
pip install -r requirements.txt

REM 4) Playwright Chromium for PDF export (one-time)
python -m playwright install chromium

REM 5) spaCy English model (one-time)
python -m spacy download en_core_web_sm

REM 6) Run Streamlit app
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

> **Troubleshooting (Windows):**
>
> * If Playwright install fails, check proxy/firewall.
> * If Streamlit won’t open, copy-paste the local URL manually into your browser.

---

### Linux (Debian/Ubuntu/Fedora/Arch)

```bash
# 1) Clone & enter
git clone https://github.com/your-username/proresumeai.git
cd proresumeai

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Playwright Chromium for PDF export (one-time)
python -m playwright install chromium

# (Optional: for some distros, install OS deps)
python -m playwright install-deps

# 5) spaCy English model (one-time)
python -m spacy download en_core_web_sm

# 6) Run Streamlit app
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

> **Troubleshooting (Linux):**
>
> * If Chromium fails to launch:
>   run `python -m playwright install-deps`
> * On minimal servers, add fonts:
>   `sudo apt-get install fonts-liberation`

---

## 🔑 Using API Keys

* Paste OpenAI / Gemini / Anthropic keys in the **sidebar** (never stored).
* Optional: provide **Ollama Base** (e.g., `http://localhost:11434`) and **Model** (`llama3.1`) for local models.
* Provider selection is **automatic**, based on available keys + strategy (Best Quality / Fastest / Local-first).

**Common provider errors:**

* **OpenAI 429** → you hit quota, check billing.
* **Gemini/Anthropic auth** → invalid or expired key.
* **Ollama refused** → Ollama app not running or model not pulled.

---

## 🧪 Quick Test (No Keys)

Without cloud keys, you can still:

* Upload JD/Resume and run **ATS & Readability** analysis.
* Use **Ollama** locally if installed.

Full tailoring and cover letter need at least one cloud API key.

---

## 🤝 Contributing

PRs welcome! Ideas:

* More PDF/DOCX templates
* Extra provider support
* Visual keyword density & diff tools

---

## 📜 License

MIT License — see `LICENSE`.

```
```
