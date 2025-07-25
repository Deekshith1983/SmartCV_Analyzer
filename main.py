import streamlit as st
import joblib
import PyPDF2
import os
import random
import requests
from googleapiclient.discovery import build

# ==== Load ML Models ====
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'clf.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'encoder.pkl')

clf = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECTORIZER_PATH)
encoder = joblib.load(ENCODER_PATH)

# ==== YouTube API Integration ====
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
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        all_videos.append(video_url)
    return random.sample(all_videos, min(display_count, len(all_videos)))

# ==== Job API Integration ====
def fetch_job_listings(query, location="India", max_results=5):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['api_keys']['jsearch']}"
    }
    params = {
        "query": f"{query} in {location}",
        "page": "1",
        "num_pages": "2"  # Fetch up to ~20 results
    }
    try:
        resp = requests.get(url, headers=headers, params=params)
        data = resp.json()
        jobs = data.get("data", [])
        return jobs[:max_results]
    except Exception as e:
        st.error(f"Failed to fetch jobs: {e}")
        return []

# ==== Resume Text Extraction ====
def extract_text_from_resume(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ==== Job Prediction ====
def predict_job(resume_text):
    cleaned_text = [resume_text]
    X_vec = tfidf.transform(cleaned_text)
    pred = clf.predict(X_vec)
    if isinstance(pred[0], str):
        return pred[0]
    job = encoder.inverse_transform(pred.astype(int))
    return job[0]

# ==== Streamlit UI ====
st.title("🎯 Resume Job Predictor")
st.write("Upload your resume to get a job prediction, explore real-time jobs, and watch curated videos to prepare for interviews.")

uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    uploaded_file.seek(0)
    resume_text = extract_text_from_resume(uploaded_file)

    st.subheader("📄 Extracted Resume Text")
    st.text_area("Resume Content", resume_text, height=400)

    if st.button("🔍 Analyze Resume"):
        result = predict_job(resume_text)
        st.success(f"✅ Predicted Job Category: **{result}**")

        # ==== Real-Time Job Listings ====
        st.markdown("## 💼 Top 5 Real-Time Job Listings")
        jobs = fetch_job_listings(result, location="India", max_results=5)
        if jobs:
            for job in jobs:
                st.markdown(f"**🔹 {job.get('job_title', 'N/A')}** at *{job.get('employer_name', 'N/A')}*")
                location_info = job.get('job_city') or job.get('job_country') or "Location not listed"
                st.markdown(f"📍 {location_info}")
                apply_link = job.get('job_apply_link')
                if apply_link:
                    st.markdown(f"[🔗 Apply Now]({apply_link})")
                    st.markdown("---")
        else:
            st.warning("⚠️ No jobs found. Try again later or with a different keyword.")


        # ==== YouTube Tips ====
        st.markdown("## 🎥 Video Tips to Succeed")
        col1, col2 = st.columns(2)

        with col1:
            st.header("🗣️ Interview Tips")
            interview_videos = fetch_random_youtube_videos("interview tips for freshers", fetch_count=10, display_count=3)
            for url in interview_videos:
                st.video(url)

        with col2:
            st.header("📝 Resume Building Tips")
            resume_videos = fetch_random_youtube_videos("resume making tips", fetch_count=8, display_count=2)
            for url in resume_videos:
                st.video(url)
