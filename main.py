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


# ============================= UI =============================
st.title("🎯 Resume Job Predictor")
st.write("Upload your resume → Preview → Predict job → Explore jobs")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    # ✅ PDF PREVIEW (NOW INCLUDED)
    st.subheader("📄 Resume Preview")
    st.pdf(uploaded_file)

    resume_text = extract_text_from_resume(uploaded_file)

    if st.button("🔍 Analyze Resume"):
        result = predict_job(resume_text)

        # SUCCESS MESSAGE
        st.success(f"✅ Predicted Job Role: **{result}**")

        # =========================================================
        # 🔥 JOB OPENINGS — YOUR ORIGINAL CODE (UNCHANGED)
        # =========================================================
        st.markdown("## 💼 Live Job Openings")

        jobs = fetch_job_listings(result)

        job_cards_html = """
        <style>
        .job-carousel {
            display: flex;
            flex-direction: row;
            gap: 1.5rem;
            overflow-x: auto;
            padding: 1rem 0 1.5rem 0;
            scrollbar-width: thin;
        }

        .job-carousel::-webkit-scrollbar {
            height: 8px;
        }
        .job-carousel::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 10px;
        }

        .job-card {
            min-width: 300px;
            max-width: 320px;
            background: #ffffff;
            border-radius: 20px;
            padding: 1.4rem;
            box-shadow: 0 14px 30px rgba(0,0,0,0.12);
            flex-shrink: 0;
            transition: transform 0.3s ease;
        }

        .job-card:hover {
            transform: translateY(-6px);
        }

        .job-card h4 {
            margin-bottom: 0.4rem;
            font-weight: 700;
            color: #111827;
        }

        .job-card p {
            font-size: 0.92rem;
            color: #4b5563;
            margin: 0.25rem 0;
        }

        .job-card a {
            display: inline-block;
            margin-top: 0.6rem;
            font-weight: 600;
            color: #2563eb;
            text-decoration: none;
        }
        </style>

        <div class="job-carousel">
        """

        for job in jobs:
            job_cards_html += f"""
            <div class="job-card">
                <h4>{job.get('title','N/A')}</h4>
                <p><b>Company:</b> {job.get('company',{}).get('display_name','N/A')}</p>
                <p>📍 {job.get('location',{}).get('display_name','India')}</p>
                <a href="{job.get('redirect_url','#')}" target="_blank">Apply →</a>
            </div>
            """

        job_cards_html += "</div>"

        st.markdown(job_cards_html, unsafe_allow_html=True)

        # =========================================================
        # VIDEOS (AFTER CAROUSEL)
        # =========================================================
        st.markdown("## 🎥 Preparation Videos")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗣️ Interview Tips")
            for url in fetch_random_youtube_videos("interview tips for freshers"):
                st.video(url)

        with col2:
            st.subheader("📝 Resume Building Tips")
            for url in fetch_random_youtube_videos(
                "resume building tips", fetch_count=8, display_count=2
            ):
                st.video(url)
