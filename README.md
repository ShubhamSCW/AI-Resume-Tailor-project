# 📄 AI Resume Tailor

A Streamlit web application that uses Generative AI (Google Gemini or OpenAI GPT) to tailor your resume to a specific job description, helping you create an ATS-friendly and keyword-optimized application.

## ✨ Features

-   **AI-Powered Tailoring**: Leverages large language models to rewrite and optimize your resume.
-   **Dual Backend Support**: Choose between Google Gemini and OpenAI GPT models.
-   **Multiple File Formats**: Upload your resume as a `.txt` or `.docx` file.
-   **Downloadable Output**: Get your tailored resume in both `.docx` and `.pdf` formats.
-   **Live Preview**: Instantly see a formatted preview of your new resume within the app.
-   **User-Friendly Interface**: Simple, clean, and intuitive UI built with Streamlit.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   An API key from either [Google AI Studio](https://ai.google.dev/) (for Gemini) or [OpenAI](https://platform.openai.com/account/api-keys) (for GPT).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ai-resume-tailor.git](https://github.com/your-username/ai-resume-tailor.git)
    cd ai-resume-tailor
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

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Streamlit app from your terminal:**
    ```bash
    streamlit run app.py
    ```
2.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## usage How to Use

1.  **Select AI Backend**: In the sidebar, choose either "Google Gemini" or "OpenAI GPT".
2.  **Enter API Key**: Paste your corresponding API key into the text input field in the sidebar.
3.  **Upload Resume**: In the main area, click "Browse files" to upload your current resume (`.txt` or `.docx`).
4.  **Paste Job Description**: Copy the job description you are applying for and paste it into the text area.
5.  **Generate**: Click the "✨ Generate Tailored Resume" button.
6.  **Download & Preview**: After a moment, download links for `.docx` and `.pdf` files will appear, and a formatted preview will be displayed on the page.

---


