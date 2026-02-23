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
<div id="vanta-bg"></div>

<style>
#vanta-bg {
    position: fixed;
    width: 100%;
    height: 100%;
    z-index: -1;
    top: 0;
    left: 0;
}
.block-container {
    padding: 2.5rem 3rem;
}
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #ffffff;
}
h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00c6ff, #7f00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.job-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(12px);
    padding: 1.2rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    border-left: 6px solid #00c6ff;
}
.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #7f00ff);
    border-radius: 30px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1.1rem;
}
textarea {
    background: rgba(0,0,0,0.45) !important;
    color: white !important;
}
iframe {
    border-radius: 16px;
}
</style>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
<script>
VANTA.NET({
  el: "#vanta-bg",
  color: 0x00c6ff,
  backgroundColor: 0x0f2027,
  points: 12.0,
  maxDistance: 22.0,
  spacing: 18.0
})
</script>
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
