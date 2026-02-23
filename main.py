# ============================= IMPORTS =============================
import streamlit as st
import joblib
import PyPDF2
import os
import random
import requests
from googleapiclient.discovery import build


# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="Resume Job Predictor",
    page_icon="🎯",
    layout="wide"
)

# ============================= VANTA BACKGROUND =============================
st.markdown("""
<style>

/* ========== HORIZONTAL JOB SCROLLER ========== */
.job-scroll-container {
    display: flex;
    gap: 1.2rem;
    overflow-x: auto;
    padding: 1rem 0;
    scroll-snap-type: x mandatory;
}

.job-scroll-container::-webkit-scrollbar {
    height: 8px;
}
.job-scroll-container::-webkit-scrollbar-thumb {
    background: #cfd8ff;
    border-radius: 10px;
}

.job-scroll-card {
    min-width: 300px;
    max-width: 320px;
    background: #ffffff;
    border-radius: 18px;
    padding: 1.2rem;
    scroll-snap-align: start;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    transition: transform 0.3s ease;
}

.job-scroll-card:hover {
    transform: translateY(-6px);
}

.job-scroll-card h4 {
    margin-bottom: 0.4rem;
    font-weight: 700;
    color: #1f2937;
}

.job-scroll-card p {
    font-size: 0.9rem;
    color: #4b5563;
}

.job-scroll-card a {
    display: inline-block;
    margin-top: 0.6rem;
    color: #4f46e5;
    font-weight: 600;
}

/* ========== UPLOAD INTERFACE (LIKE IMAGE) ========== */
section[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 20px;
    padding: 2rem;
    border: 2px dashed #d1d5db;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

section[data-testid="stFileUploader"] label {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111827;
}

section[data-testid="stFileUploader"] small {
    color: #6b7280;
}

/* ========== INPUT TEXT AREA CLEAN LOOK ========== */
textarea {
    background: #ffffff !important;
    color: #111827 !important;
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# ============================= LOAD ML MODELS =============================
BASE_DIR = os.path.dirname(__file__)
clf = joblib.load(os.path.join(BASE_DIR, "clf.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))


# ============================= YOUTUBE API =============================
def fetch_random_youtube_videos(query, fetch_count=10, display_count=3):
    youtube = build(
        "youtube",
        "v3",
        developerKey=st.secrets["api_keys"]["youtube"]
    )
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=fetch_count,
        safeSearch="none"
    )
    response = request.execute()
    videos = [
        f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        for item in response.get("items", [])
    ]
    return random.sample(videos, min(display_count, len(videos)))


# ============================= ADZUNA JOB SEARCH (REPLACED) =============================
def fetch_job_listings(query, location="India", max_results=5):
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": st.secrets["api_keys"]["adzuna_app_id"],
        "app_key": st.secrets["api_keys"]["adzuna_app_key"],
        "what": query,
        "results_per_page": max_results,
        "content-type": "application/json"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        st.error(f"Job API error: {e}")
        return []


# ============================= RESUME EXTRACTION =============================
def extract_text_from_resume(file):
    reader = PyPDF2.PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages)


# ============================= JOB PREDICTION =============================
def predict_job(resume_text):
    X_vec = tfidf.transform([resume_text])
    pred = clf.predict(X_vec)
    return pred[0] if isinstance(pred[0], str) else encoder.inverse_transform(pred.astype(int))[0]


# ============================= UI =============================
st.title("🎯 Resume Job Predictor")
st.write("Upload your resume → Predict job → Apply → Prepare")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_resume(uploaded_file)

    st.subheader("📄 Resume Content")
    st.text_area("Extracted Text", resume_text, height=350)

    if st.button("🔍 Analyze Resume"):
        result = predict_job(resume_text)
        st.success(f"✅ Predicted Job Role: **{result}**")

        # ============================= JOBS =============================
        st.markdown("## 💼 Live Job Openings")

        st.markdown('<div class="job-scroll-container">', unsafe_allow_html=True)

        for job in jobs:
            st.markdown(f"""
            <div class="job-scroll-card">
            <h4>{job.get('title','N/A')}</h4>
            <p><b>Company:</b> {job.get('company',{}).get('display_name','N/A')}</p>
            <p>📍 {job.get('location',{}).get('display_name','N/A')}</p>
            <a href="{job.get('redirect_url','#')}" target="_blank">Apply →</a>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ============================= VIDEOS =============================
        st.markdown("## 🎥 Preparation Videos")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗣️ Interview Tips")
            for url in fetch_random_youtube_videos("interview tips for freshers"):
                st.video(url)

        with col2:
            st.subheader("📝 Resume Tips")
            for url in fetch_random_youtube_videos("resume building tips", fetch_count=8, display_count=2):
                st.video(url)
