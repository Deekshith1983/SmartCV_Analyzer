# Smart CV Analyzer

## Overview
Smart CV Analyzer is an AI-powered web application that predicts the most suitable job category for a user based on their uploaded resume. It also provides real-time job listings and curated YouTube video resources to help users prepare for interviews and build effective resumes.

## Features
- **Resume Upload:** Upload your resume in PDF format for instant analysis.
- **Job Prediction:** Get an AI-predicted job category based on your resume content.
- **Real-Time Job Listings:** View top job openings from platforms like Internshala and Unstop, tailored to your predicted job category.
- **Interview & Resume Tips:** Watch curated YouTube videos for interview preparation and resume building, fetched in real-time.

## Tech Stack
- **Backend:** Python, scikit-learn, joblib
- **Frontend:** Streamlit
- **ML Models:** TF-IDF Vectorizer, KNN/SVM Classifier, LabelEncoder
- **APIs:**
  - YouTube Data API v3 (for fetching video tips)
  - JSearch API (for real-time job listings)
- **PDF Parsing:** PyPDF2

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Deekshith1983/SmartCV_Analyzer.git
   cd SmartCV_Analyzer
   ```
2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **API Keys:**
   - Add your YouTube Data API key and JSearch API key to `.streamlit/secrets.toml`:
     ```toml
     [api_keys]
     youtube = "YOUR_YOUTUBE_API_KEY"
     jsearch = "YOUR_JSEARCH_API_KEY"
     ```
4. **Model Files:**
   - Ensure `clf.pkl`, `tfidf.pkl`, and `encoder.pkl` are present in the project root. Train and export them if needed.
5. **Run the App:**
   ```bash
   streamlit run main.py
   ```

## Usage
1. Open the app in your browser (usually at http://localhost:8501).
2. Upload your resume (PDF).
3. Click "Analyze Resume" to get your predicted job category, job listings, and video tips.

## File Structure
```
SmartCV_Analyzer/
├── main.py
├── model.ipynb
├── clf.pkl
├── tfidf.pkl
├── encoder.pkl
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml
```

## Requirements
- Python 3.8+
- streamlit
- scikit-learn
- joblib
- PyPDF2
- google-api-python-client
- requests

## Acknowledgements
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [JSearch API](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)
- [Streamlit](https://streamlit.io/)

## License
This project is licensed under the MIT License.

## 🔗 Live Demo

🚀 **[Click here to try the deployed app](https://job-predictor.streamlit.app/)**
