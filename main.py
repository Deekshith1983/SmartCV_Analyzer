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
<!-- ========== AURORA GLASS UI THEME ========== -->
<div class="aurora-bg"></div>

<style>

/* ---------- Background ---------- */
.stApp {
    background: radial-gradient(circle at 20% 20%, #1a2a6c, transparent 40%),
                radial-gradient(circle at 80% 30%, #00c6ff, transparent 35%),
                radial-gradient(circle at 50% 80%, #7f00ff, transparent 40%),
                #0b1020;
    animation: auroraMove 22s ease-in-out infinite;
    color: #ffffff;
}

@keyframes auroraMove {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%; }
    50%  { background-position: 100% 50%, 0% 50%, 50% 0%; }
    100% { background-position: 0% 0%, 100% 0%, 50% 100%; }
}

/* ---------- Layout ---------- */
.block-container {
    padding: 2.5rem 3rem;
}

/* ---------- Title ---------- */
h1 {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #00eaff, #9b5cff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeDown 1.2s ease;
}

@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-30px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Glass Cards ---------- */
.glass-card,
.job-card,
section[data-testid="stFileUploader"],
textarea {
    background: rgba(255, 255, 255, 0.10) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.18);
    box-shadow: 0 20px 40px rgba(0,0,0,0.35);
}

/* ---------- File Uploader ---------- */
section[data-testid="stFileUploader"] {
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* ---------- Text Area ---------- */
textarea {
    color: #ffffff !important;
    font-size: 0.95rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #00eaff, #9b5cff);
    color: #fff;
    border-radius: 32px;
    padding: 0.7rem 2.2rem;
    font-weight: 700;
    font-size: 1.1rem;
    border: none;
    transition: all 0.35s ease;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
}

/* ---------- Job Cards ---------- */
.job-card {
    padding: 1.3rem;
    margin-bottom: 1.2rem;
    animation: fadeUp 0.8s ease forwards;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(25px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---------- Links ---------- */
a {
    color: #00eaff !important;
    font-weight: 600;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* ---------- Alerts ---------- */
.stAlert {
    border-radius: 16px;
    backdrop-filter: blur(12px);
}

/* ---------- Videos ---------- */
iframe {
    border-radius: 18px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.6);
}

/* ---------- Section headers ---------- */
h2, h3 {
    font-weight: 700;
    margin-top: 2rem;
    animation: fadeUp 0.8s ease;
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
        jobs = fetch_job_listings(result)

        if jobs:
            for job in jobs:
                st.markdown(f"""
                <div class="job-card">
                    <h4>{job.get('title','N/A')}</h4>
                    <p><b>Company:</b> {job.get('company',{}).get('display_name','N/A')}</p>
                    <p>📍 {job.get('location',{}).get('display_name','N/A')}</p>
                    <a href="{job.get('redirect_url','#')}" target="_blank">🔗 Apply Now</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No jobs found")

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
