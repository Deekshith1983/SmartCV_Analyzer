# ============================= IMPORTS =============================
import streamlit as st
import joblib
import PyPDF2
import os
import random
import requests
from googleapiclient.discovery import build


# ============================= PAGE CONFIG (TOP MOST) =============================
st.set_page_config(
    page_title="Resume Job Predictor",
    page_icon="🎯",
    layout="wide"
)

# ============================= GLOBAL STYLING =============================
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}

.block-container {
    padding: 2rem 3rem;
}

h1 {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h2, h3 {
    color: #00c6ff;
    font-weight: 700;
}

section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px dashed #00c6ff;
}

.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 25px;
    padding: 0.6rem 1.8rem;
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    transition: all 0.3s ease-in-out;
}

.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(0,198,255,0.6);
}

textarea {
    background-color: rgba(0,0,0,0.35) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #00c6ff !important;
}

.stAlert[data-baseweb="notification"] {
    border-radius: 14px;
    font-size: 1.05rem;
}

.job-card {
    background: rgba(255,255,255,0.08);
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    border-left: 5px solid #00c6ff;
}

a {
    color: #00c6ff !important;
    font-weight: 600;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

iframe {
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}
</style>
""", unsafe_allow_html=True)


# ============================= LOAD ML MODELS =============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'clf.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

clf = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
encoder = joblib.load(ENCODER_PATH)


# ============================= YOUTUBE API =============================
def fetch_random_youtube_videos(query, fetch_count=10, display_count=3):
    api_key = st.secrets["api_keys"]["youtube"]
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=fetch_count,
        safeSearch="strict"
    )
    response = request.execute()

    all_videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        all_videos.append(f"https://www.youtube.com/watch?v={video_id}")

    return random.sample(all_videos, min(display_count, len(all_videos)))


# ============================= JOB API =============================
def fetch_job_listings(query, location="India", max_results=5):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['api_keys']['jsearch']}"
    }
    params = {
        "query": f"{query} in {location}",
        "page": "1",
        "num_pages": "2"
    }
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        return data.get("data", [])[:max_results]
    except Exception as e:
        st.error(f"Failed to fetch jobs: {e}")
        return []


# ============================= RESUME EXTRACTION =============================
def extract_text_from_resume(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# ============================= JOB PREDICTION =============================
def predict_job(resume_text):
    X_vec = tfidf.transform([resume_text])
    pred = clf.predict(X_vec)
    if isinstance(pred[0], str):
        return pred[0]
    return encoder.inverse_transform(pred.astype(int))[0]


# ============================= STREAMLIT UI =============================
st.title("🎯 Resume Job Predictor")
st.write("Upload your resume to predict a job role, explore real-time jobs, and watch interview prep videos.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file:
    uploaded_file.seek(0)
    resume_text = extract_text_from_resume(uploaded_file)

    st.subheader("📄 Extracted Resume Text")
    st.text_area("Resume Content", resume_text, height=350)

    if st.button("🔍 Analyze Resume"):
        result = predict_job(resume_text)
        st.success(f"✅ Predicted Job Category: **{result}**")

        # ============================= JOB LISTINGS =============================
        st.markdown("## 💼 Top 5 Real-Time Job Listings")
        jobs = fetch_job_listings(result)

        if jobs:
            for job in jobs:
                location_info = job.get('job_city') or job.get('job_country') or "Location not listed"
                apply_link = job.get('job_apply_link')

                st.markdown(f"""
                <div class="job-card">
                    <h4>🔹 {job.get('job_title','N/A')}</h4>
                    <p><b>Company:</b> {job.get('employer_name','N/A')}</p>
                    <p>📍 {location_info}</p>
                    {"<a href='"+apply_link+"' target='_blank'>🔗 Apply Now</a>" if apply_link else ""}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No jobs found")

        # ============================= YOUTUBE VIDEOS =============================
        st.markdown("## 🎥 Preparation Videos")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🗣️ Interview Tips")
            for url in fetch_random_youtube_videos("interview tips for freshers"):
                st.video(url)

        with col2:
            st.subheader("📝 Resume Building Tips")
            for url in fetch_random_youtube_videos("resume making tips", fetch_count=8, display_count=2):
                st.video(url)
