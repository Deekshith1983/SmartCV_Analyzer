# ============================= IMPORTS =============================
import streamlit as st
import joblib
import PyPDF2
import os
import random
import requests
import base64
from googleapiclient.discovery import build
from streamlit.components.v1 import html


# ============================= PAGE CONFIG =============================
st.set_page_config(
    page_title="Resume Job Predictor",
    page_icon="🎯",
    layout="wide"
)

# ============================= BACKGROUND (UNCHANGED) =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        -45deg,
        #0f2027,
        #203a43,
        #2c5364,
        #1a2980,
        #26d0ce
    );
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ===== Upload UI ===== */
section[data-testid="stFileUploader"] {
    background: #ffffff;
    border-radius: 22px;
    padding: 2.5rem;
    border: 2px dashed #d1d5db;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
}

/* ===== PDF Preview ===== */
.pdf-box {
    background: #ffffff;
    border-radius: 18px;
    padding: 0.8rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}

/* ===== Horizontal Job Scroll ===== */
.job-scroll {
    display: flex;
    flex-wrap: nowrap;
    gap: 1.2rem;
    overflow-x: auto;
    padding: 1rem 0 1.5rem 0;
}

.job-scroll::-webkit-scrollbar {
    height: 8px;
}
.job-scroll::-webkit-scrollbar-thumb {
    background: #c7d2fe;
    border-radius: 10px;
}

/* ===== Job Card ===== */
.job-card {
    flex: 0 0 320px;
    background: #ffffff;
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 14px 30px rgba(0,0,0,0.10);
    transition: transform 0.3s ease;
}

.job-card:hover {
    transform: translateY(-6px);
}

.job-card h4 {
    margin-bottom: 0.4rem;
    font-weight: 700;
    color: #1f2937;
}

.job-card p {
    font-size: 0.92rem;
    color: #4b5563;
    margin: 0.2rem 0;
}

.job-card a {
    display: inline-block;
    margin-top: 0.7rem;
    color: #4f46e5;
    font-weight: 600;
    text-decoration: none;
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


# ============================= ADZUNA JOB SEARCH =============================
def fetch_job_listings(query, max_results=5):
    url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": st.secrets["api_keys"]["adzuna_app_id"],
        "app_key": st.secrets["api_keys"]["adzuna_app_key"],
        "what": query,
        "results_per_page": max_results
    }
    return requests.get(url, params=params).json().get("results", [])


# ============================= RESUME EXTRACTION =============================
def extract_text_from_resume(file):
    reader = PyPDF2.PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages)


# ============================= JOB PREDICTION =============================
def predict_job(resume_text):
    X_vec = tfidf.transform([resume_text])
    pred = clf.predict(X_vec)
    return pred[0] if isinstance(pred[0], str) else encoder.inverse_transform(pred.astype(int))[0]


# ============================= PDF PREVIEW (CHROME SAFE) =============================
def show_pdf(file):
    pdf_bytes = file.read()
    b64 = base64.b64encode(pdf_bytes).decode()

    html(f"""
    <div class="pdf-box">
        <object
            data="data:application/pdf;base64,{b64}"
            type="application/pdf"
            width="100%"
            height="520px">
            <p>PDF preview not supported.</p>
        </object>
    </div>
    """, height=550)


# ============================= UI =============================
st.title("🎯 Resume Job Predictor")
st.write("Upload your resume → Preview → Predict → Apply")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    uploaded_file.seek(0)
    st.subheader("📄 Resume Preview")
    show_pdf(uploaded_file)

    uploaded_file.seek(0)
    resume_text = extract_text_from_resume(uploaded_file)

    if st.button("🔍 Analyze Resume"):
        result = predict_job(resume_text)
        st.success(f"✅ Predicted Job Role: **{result}**")

        # ============================= HORIZONTAL JOB CARDS =============================
        st.markdown("## 💼 Live Job Openings")

        jobs = fetch_job_listings(result)

        job_cards_html = '<div class="job-scroll">'
        for job in jobs:
            job_cards_html += f"""
            <div class="job-card">
                <h4>{job.get('title','N/A')}</h4>
                <p><b>Company:</b> {job.get('company',{}).get('display_name','N/A')}</p>
                <p>📍 {job.get('location',{}).get('display_name','N/A')}</p>
                <a href="{job.get('redirect_url','#')}" target="_blank">Apply →</a>
            </div>
            """
        job_cards_html += '</div>'

        st.markdown(job_cards_html, unsafe_allow_html=True)

        # ============================= VIDEOS =============================
        st.markdown("## 🎥 Preparation Videos")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗣️ Interview Tips")
            for url in fetch_random_youtube_videos("interview tips for freshers"):
                st.video(url)

        with col2:
            st.subheader("📝 Resume Building Tips")
            for url in fetch_random_youtube_videos("resume building tips", fetch_count=8, display_count=2):
                st.video(url)
