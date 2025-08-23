# 🚀 Ultimate AI Resume Suite

Tired of manually customizing your resume for every single job application? This tool is your all-in-one solution to analyze, compare, and tailor your resumes against job descriptions, helping you beat the ATS bots and land your next interview.

This Streamlit application leverages multiple AI backends (including local models) to provide deep analysis, generate tailored documents, and present the results in a rich, interactive dashboard.



---

## ✨ Key Features

The suite is packed with features designed to give you a competitive edge.

### 🔬 Deep Analysis & Comparison
* **Multi-Resume Comparison**: Upload multiple resumes to see which one is the best fit for a job.
* **Comprehensive ATS Score**: Get a detailed friendliness score (0-100) with an actionable checklist to ensure your resume is parsable by applicant tracking systems.
* **Action Verb & Metrics Analysis**: Quantifies your use of strong action verbs and measurable results to highlight your achievements.
* **Granular AI Feedback**: Receive specific, section-by-section feedback on your resume's content and structure.

### 🤖 Multi-Model AI Power
* **Multiple Backends**: Supports major AI providers (**Google Gemini**, **OpenAI GPT**, **Anthropic Claude**) for cloud-based generation.
* **Local Model Support**: Run analysis locally and for free using **Ollama** or **Hugging Face Transformers**.
* **Automatic Fallback**: If the primary AI backend fails, the app automatically tries the next one in the chain to ensure you always get a result.

### 📊 Rich Visualizations
* **Interactive Charts**: Compare overall scores and keyword coverage for all uploaded resumes with clear bar charts.
* **Quick Metrics Dashboard**: See the most important stats for your top resume at a glance.
* **Word Clouds**: Instantly visualize the key strengths and weaknesses identified by the AI.

### ✍️ Effortless Tailoring & Export
* **AI-Powered Resume Tailoring**: Automatically rewrites your best resume to perfectly match the tone and keywords of the job description.
* **Automatic Cover Letter Generation**: Creates a concise, professional cover letter based on your newly tailored resume.
* **Multiple Export Formats**: Download your final resume and cover letter as `.docx`, `.pdf`, or `.txt`.

---

## 🚀 Getting Started

Follow these steps to get the application running on your local machine.

### Prerequisites
* Python 3.8+
* An API key from either [Google AI Studio](https://ai.google.dev/) (for Gemini) or [OpenAI](https://platform.openai.com/account/api-keys) (for GPT).
* API keys for any cloud backends you wish to use (e.g., Gemini, OpenAI).

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ultimate-ai-resume-suite.git](https://github.com/your-username/ultimate-ai-resume-suite.git)
    cd ultimate-ai-resume-suite
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install all required dependencies:**
    The `requirements.txt` file should include all necessary packages. Pinning versions (`pip freeze > requirements.txt`) is recommended for reproducibility.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy NLP model:**
    This is used for advanced text processing and keyword extraction.
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Running the Application
1.  **Launch the app from your terminal:**
    ```bash
    streamlit run app_ultimate.py
    ```
2.  Open your web browser to the local URL provided (usually `http://localhost:8501`).

---

## ⚙️ Configuration

All configuration is handled in the application's sidebar.

### API Keys
* To use cloud models like **Gemini**, **OpenAI**, or **Anthropic**, expand the "AI Configuration" section in the sidebar and paste your API key into the corresponding field.

### Using Local Models
* **Ollama**: Ensure the Ollama desktop application is **running** on your machine before starting the Streamlit app. You must also have the required model pulled (e.g., `ollama pull llama3.1`).
* **Hugging Face**: This requires a local installation of PyTorch. If you haven't already, install it by running `pip install torch`.

---

## 🛠️ How to Use

1.  **Configure AI**: Select your primary AI backend and enter API keys in the sidebar.
2.  **Upload & Paste**: Drag and drop one or more resumes and paste the target job description.
3.  **Analyze**: Hit the "✨ Analyze & Generate" button to kick off the deep analysis.
4.  **Review Insights**: Explore the "Analysis Insights" tab to view the interactive dashboard with comparison charts, ATS scores, and keyword coverage.
5.  **Edit & Refine**: Navigate to the "View/Edit Tailored Resume" and "View/Edit Cover Letter" tabs to edit the AI-generated content in real-time.
6.  **Export**: Go to the "Export Documents" tab to download your polished, ATS-friendly documents.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`)
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`)
4.  Push to the Branch (`git push origin feature/NewFeature`)
5.  Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
