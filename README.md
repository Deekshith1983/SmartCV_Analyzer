# Smart CV Analyzer

Smart CV Analyzer is a Streamlit application for:

1. Job Prediction from uploaded resume text using pre-trained model artifacts.
2. Resume Analysis against a Job Description with visual EDA insights and downloadable PDF report.

## View App

Live demo: https://smartcvanalyzer-t4hc5tnsxsqqsc4eb4bbuk.streamlit.app/

## Features

### Job Prediction page
- Resume upload and preview.
- Role prediction using saved model artifacts (`clf.pkl`, `tfidf.pkl`, `encoder.pkl`).
- Resume EDA summary (term-level view).
- Live job listings from Adzuna.
- YouTube preparation videos.

### Resume Analysis page
- Inputs: Resume (PDF/DOCX) + Job Description text.
- Visual EDA only (skills, keyword, education, experience, section evaluation).
- Match score and category breakdown.
- PDF download report for analysis summary.

## Tech Stack
- Python, Streamlit
- scikit-learn, joblib
- pandas, matplotlib
- PyPDF2, python-docx
- requests, google-api-python-client

## Setup

1. Clone repository.

```bash
git clone https://github.com/Deekshith1983/SmartCV_Analyzer.git
cd SmartCV_Analyzer
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Configure API keys (local only).

Copy the template and fill your real values:

```bash
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
```

Template format:

```toml
[api_keys]
youtube = "YOUR_YOUTUBE_API_KEY"
adzuna_app_id = "YOUR_ADZUNA_APP_ID"
adzuna_app_key = "YOUR_ADZUNA_APP_KEY"
```

4. Run app.

```bash
streamlit run main.py
```

## Security Notes

- Never commit real API keys to GitHub.
- `.streamlit/secrets.toml` is git-ignored.
- Keep only `.streamlit/secrets.toml.example` in the repository.
- If any key was exposed before, rotate/regenerate it immediately.

## Upload Updated Project to GitHub

Run these commands from project root:

```bash
git rm --cached .streamlit/secrets.toml
git add .
git commit -m "Update UI, EDA analysis flow, PDF report export, and secure secrets handling"
git push origin main
```

If `git rm --cached` says file is not tracked, continue with the next commands.

## Project Structure

```text
SmartCV_Analyzer/
├── main.py
├── KnnModel.ipynb
├── clf.pkl
├── tfidf.pkl
├── encoder.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```
