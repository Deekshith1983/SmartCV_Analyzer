import os
import random
import re
import base64
import textwrap
from collections import Counter
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import PyPDF2
import requests
import streamlit as st
from docx import Document
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Smart Cv Analyzer", page_icon="🎯", layout="wide")

st.markdown(
    """
<style>
:root {
    --accent-1: #10b981;
    --accent-2: #3b82f6;
    --accent-3: #f59e0b;
    --accent-4: #ef4444;
}
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
.app-title {
    color: #e2e8f0;
    letter-spacing: 0.2px;
    margin-bottom: 0.4rem;
}
.sticky-nav {
    position: sticky;
    top: 0.4rem;
    z-index: 1000;
    padding: 0.6rem 0.8rem 0.5rem 0.8rem;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(226, 232, 240, 0.2);
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.72);
    backdrop-filter: blur(8px);
}
.section-title {
    color: #f8fafc;
    margin-top: 1.25rem;
    margin-bottom: 0.4rem;
    border-left: 4px solid var(--accent-1);
    border-radius: 8px;
    padding: 0.35rem 0.75rem;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.18), rgba(59, 130, 246, 0.06));
}
.section-note {
    color: #d1d5db;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
}
.block-spacer {
    height: 0.4rem;
}
div[data-testid="stRadio"] > div[role="radiogroup"] {
    gap: 0.55rem;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label {
    border: 1px solid rgba(203, 213, 225, 0.35);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    background: rgba(30, 41, 59, 0.55);
    transition: all 0.2s ease;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
    border-color: rgba(16, 185, 129, 0.8);
    transform: translateY(-1px);
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.95), rgba(59, 130, 246, 0.95));
    border-color: transparent;
    box-shadow: 0 8px 18px rgba(16, 185, 129, 0.28);
}
div[data-testid="stRadio"] > div[role="radiogroup"] label p {
    color: #e2e8f0;
    font-weight: 600;
}
div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) p {
    color: #ffffff;
}
</style>
""",
    unsafe_allow_html=True,
)


BASE_DIR = os.path.dirname(__file__)

DEFAULT_SKILLS = [
    "python",
    "java",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "excel",
    "power bi",
    "tableau",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "react",
    "node",
    "html",
    "css",
    "javascript",
    "git",
]

EDUCATION_KEYWORDS = [
    "btech",
    "b.e",
    "bachelor",
    "mtech",
    "m.e",
    "master",
    "phd",
    "mba",
    "bsc",
    "msc",
]

SECTION_HINTS = {
    "Skills Section": ["skills", "technologies", "tools"],
    "Projects": ["project", "projects"],
    "Experience": ["experience", "work history", "employment"],
    "Certifications": ["certification", "certifications", "certificate"],
}

COLORS = {
    "green": "#10b981",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "gray": "#e5e7eb",
}


@st.cache_resource
def load_artifacts():
    clf_model = joblib.load(os.path.join(BASE_DIR, "clf.pkl"))
    tfidf_model = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "encoder.pkl"))
    return clf_model, tfidf_model, label_encoder


clf, tfidf, encoder = load_artifacts()


def fetch_random_youtube_videos(query, fetch_count=10, display_count=3):
    try:
        youtube = build("youtube", "v3", developerKey=st.secrets["api_keys"]["youtube"])
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=fetch_count,
            safeSearch="none",
        )
        response = request.execute()
        videos = [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in response.get("items", [])
        ]
        return random.sample(videos, min(display_count, len(videos)))
    except Exception:
        return []


def fetch_job_listings(query, max_results=5):
    try:
        url = "https://api.adzuna.com/v1/api/jobs/in/search/1"
        params = {
            "app_id": st.secrets["api_keys"]["adzuna_app_id"],
            "app_key": st.secrets["api_keys"]["adzuna_app_key"],
            "what": query,
            "results_per_page": max_results,
        }
        response = requests.get(url, params=params, timeout=20)
        return response.json().get("results", [])
    except Exception:
        return []


def extract_text_from_resume(file):
    reader = PyPDF2.PdfReader(file)
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return " ".join(chunks).strip()


def extract_text_from_docx(file):
    doc = Document(file)
    chunks = [p.text for p in doc.paragraphs if p.text]
    return " ".join(chunks).strip()


def extract_text_from_uploaded_file(file):
    suffix = os.path.splitext(file.name.lower())[1]
    if suffix == ".pdf":
        return extract_text_from_resume(file)
    if suffix == ".docx":
        return extract_text_from_docx(file)
    return ""


def clean_tokens(text):
    tokens = re.findall(r"[a-zA-Z][a-zA-Z+#.]{1,}", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def build_eda(text, top_n=15):
    tokens = clean_tokens(text)
    counter = Counter(tokens)
    top_terms = pd.DataFrame(counter.most_common(top_n), columns=["term", "count"])

    total_words = len(tokens)
    unique_words = len(counter)
    lexical_diversity = (unique_words / total_words) if total_words else 0.0

    return {
        "total_words": total_words,
        "unique_words": unique_words,
        "chars": len(text),
        "lexical_diversity": lexical_diversity,
        "top_terms": top_terms,
        "token_set": set(tokens),
    }


def extract_years_experience(text):
    candidates = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text.lower())
    if not candidates:
        return 0.0
    return max(float(v) for v in candidates)


def extract_education_terms(text):
    lower_text = text.lower()
    return {term for term in EDUCATION_KEYWORDS if term in lower_text}


def classify_section_status(text, keywords):
    lower_text = text.lower()
    hits = sum(1 for kw in keywords if kw in lower_text)
    if hits >= 2:
        return "Strong"
    if hits == 1:
        return "Moderate"
    return "Missing"


def generate_improvement_suggestions(skill_gap, keyword_gap, exp_gap, education_gap):
    suggestions = []
    if skill_gap:
        suggestions.append(f"Add or strengthen these JD skills: {', '.join(sorted(skill_gap)[:6])}.")
    if keyword_gap:
        suggestions.append(f"Include ATS keywords naturally in projects/experience: {', '.join(sorted(keyword_gap)[:8])}.")
    if exp_gap > 0:
        suggestions.append("Show deeper project impact with measurable outcomes to compensate for experience gap.")
    if education_gap:
        suggestions.append("Highlight relevant education/certifications that match JD requirements.")
    if not suggestions:
        suggestions.append("Resume already aligns well. Improve by adding quantified achievements for stronger impact.")
    return suggestions


def predict_job_with_details(text):
    x_vec = tfidf.transform([text])
    pred = clf.predict(x_vec)
    raw_pred = pred[0]

    if isinstance(raw_pred, str):
        label = raw_pred
    else:
        label = encoder.inverse_transform(pred.astype(int))[0]

    confidence = None
    if hasattr(clf, "predict_proba"):
        try:
            confidence = float(clf.predict_proba(x_vec).max())
        except Exception:
            confidence = None

    return {
        "label": label,
        "confidence": confidence,
        "vector": x_vec,
    }


def render_similarity_pie(similarity_score):
    match_pct = max(0.0, min(1.0, similarity_score))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(
        [match_pct, 1 - match_pct],
        labels=["Similarity", "Gap"],
        startangle=90,
        colors=[COLORS["green"], COLORS["red"]],
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Resume vs Job Description Match")
    ax.axis("equal")
    _, chart_col, _ = st.columns([1, 2, 1])
    with chart_col:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_donut(score, title, color="#22c55e"):
    bounded = max(0.0, min(1.0, score))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie(
        [bounded, 1 - bounded],
        labels=["Match", "Gap"],
        startangle=90,
        colors=[color, COLORS["gray"]],
        wedgeprops={"width": 0.42, "edgecolor": "white"},
    )
    ax.set_title(title)
    ax.axis("equal")
    _, chart_col, _ = st.columns([1, 2, 1])
    with chart_col:
        st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_section_header(title, note=""):
    st.markdown(f"<h3 class='section-title'>{title}</h3>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<p class='section-note'>{note}</p>", unsafe_allow_html=True)


def render_pdf_preview(uploaded_pdf, height=700):
    if hasattr(st, "pdf"):
        try:
            st.pdf(uploaded_pdf)
            return
        except Exception:
            pass

    pdf_bytes = uploaded_pdf.getvalue()
    if not pdf_bytes:
        st.info("PDF preview unavailable for this file.")
        return

    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    iframe_html = f"""
    <iframe
        src="data:application/pdf;base64,{encoded_pdf}"
        width="100%"
        height="{height}"
        type="application/pdf"
        style="border: none; border-radius: 8px;"
    ></iframe>
    """
    st.markdown(iframe_html, unsafe_allow_html=True)


def build_analysis_pdf(
    final_match_score,
    skill_score,
    exp_score,
    edu_score,
    keyword_score,
    suggestions,
):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")

    lines = [
        "Smart Cv Analyszer - Resume Analysis Report",
        "",
        f"Final Match Score: {final_match_score * 100:.1f}%",
        f"Skills Match: {skill_score * 100:.1f}%",
        f"Experience Match: {exp_score * 100:.1f}%",
        f"Education Match: {edu_score * 100:.1f}%",
        f"Keyword Match: {keyword_score * 100:.1f}%",
        "",
        "Suggestions:",
    ]

    for tip in suggestions:
        wrapped = textwrap.wrap(f"- {tip}", width=95)
        lines.extend(wrapped if wrapped else ["- "])

    y = 0.97
    for i, line in enumerate(lines):
        fontsize = 16 if i == 0 else 11
        weight = "bold" if i in (0, 8) else "normal"
        ax.text(0.04, y, line, fontsize=fontsize, fontweight=weight, va="top", ha="left")
        y -= 0.035 if i == 0 else 0.024
        if y < 0.05:
            break

    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def render_eda_section(title, eda_data):
    st.markdown(f"### {title}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words", eda_data["total_words"])
    m2.metric("Unique Words", eda_data["unique_words"])
    m3.metric("Characters", eda_data["chars"])
    m4.metric("Lexical Diversity", f"{eda_data['lexical_diversity']:.2f}")

    if not eda_data["top_terms"].empty:
        st.bar_chart(eda_data["top_terms"].set_index("term"))
    else:
        st.info("Not enough text to plot top terms.")


def show_job_cards(jobs):
    if not jobs:
        st.info("No live jobs found right now. Check API keys or try again later.")
        return

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
    <h4>{job.get('title', 'N/A')}</h4>
    <p><b>Company:</b> {job.get('company', {}).get('display_name', 'N/A')}</p>
    <p>Location: {job.get('location', {}).get('display_name', 'India')}</p>
    <a href="{job.get('redirect_url', '#')}" target="_blank">Apply</a>
</div>
"""

    job_cards_html += "</div>"
    st.markdown(job_cards_html, unsafe_allow_html=True)


def page_prediction_with_eda():
    st.title("Job Prediction")
    st.write("Upload your resume to predict a role and view resume EDA insights.")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume_predict")
    if not uploaded_file:
        return

    st.subheader("Resume Preview")
    render_pdf_preview(uploaded_file)
    resume_text = extract_text_from_resume(uploaded_file)

    if not resume_text:
        st.warning("Could not read text from this PDF. Try another resume.")
        return

    if st.button("Analyze Resume", key="analyze_resume"):
        details = predict_job_with_details(resume_text)
        st.success(f"Predicted Job Role: {details['label']}")
        if details["confidence"] is not None:
            st.metric("Model Confidence", f"{details['confidence'] * 100:.1f}%")

        resume_eda = build_eda(resume_text)
        render_eda_section("Resume EDA", resume_eda)

        st.markdown("## Live Job Openings")
        jobs = fetch_job_listings(details["label"])
        show_job_cards(jobs)

        st.markdown("## Preparation Videos")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Interview Tips")
            videos = fetch_random_youtube_videos("interview tips for freshers")
            if videos:
                for url in videos:
                    st.video(url)
            else:
                st.info("No videos available right now.")

        with col2:
            st.subheader("Resume Building Tips")
            videos = fetch_random_youtube_videos("resume building tips", 8, 2)
            if videos:
                for url in videos:
                    st.video(url)
            else:
                st.info("No videos available right now.")


def page_resume_jd_match():
    st.title("Resume Analysis")
    st.write("Analyze resume fit against a job description with section-wise insights.")

    render_section_header("Input Section", "Upload resume and paste job description to run EDA visuals.")
    left, right = st.columns(2)
    with left:
        resume_file = st.file_uploader("Upload Resume (PDF / DOCX)", type=["pdf", "docx"], key="resume_match")
    with right:
        job_description = st.text_area("Paste Job Description", height=220, placeholder="Paste full JD text here...")

    if not resume_file or not job_description.strip():
        st.info("Upload a resume and provide a job description to run match analysis.")
        return

    if st.button("Analyze Resume", key="resume_jd_analysis"):
        resume_text = extract_text_from_uploaded_file(resume_file)
        if not resume_text:
            st.warning("Could not read text from this file.")
            return

        with st.expander("Resume Preview"):
            st.write(resume_text[:2500] + ("..." if len(resume_text) > 2500 else ""))
        st.caption(f"Resume word count: {len(clean_tokens(resume_text))}")

        resume_tokens = clean_tokens(resume_text)
        jd_tokens = clean_tokens(job_description)
        resume_set = set(resume_tokens)
        jd_set = set(jd_tokens)

        jd_skills = {skill for skill in DEFAULT_SKILLS if skill in job_description.lower()}
        matched_skills = {skill for skill in jd_skills if skill in resume_text.lower()}
        missing_skills = jd_skills - matched_skills

        skill_score = (len(matched_skills) / len(jd_skills)) if jd_skills else 0.0
        keyword_overlap = resume_set.intersection(jd_set)
        missing_keywords = jd_set - resume_set
        keyword_score = (len(keyword_overlap) / len(jd_set)) if jd_set else 0.0

        jd_exp = extract_years_experience(job_description)
        resume_exp = extract_years_experience(resume_text)
        exp_score = 1.0 if jd_exp == 0 else min(resume_exp / jd_exp, 1.0)

        jd_edu = extract_education_terms(job_description)
        resume_edu = extract_education_terms(resume_text)
        edu_score = (len(jd_edu.intersection(resume_edu)) / len(jd_edu)) if jd_edu else 1.0

        # Regular EDA-based similarity only (no model signal in EDA page)
        final_match_score = (0.4 * skill_score) + (0.2 * exp_score) + (0.2 * edu_score) + (0.2 * keyword_score)

        render_section_header("Overall Match Score")
        st.progress(final_match_score)
        col_score_1, col_score_2 = st.columns([1, 2])
        with col_score_1:
            render_donut(final_match_score, "Resume Match Score", COLORS["green"])
        with col_score_2:
            score_chart = pd.DataFrame(
                {
                    "Category": ["Skills", "Experience", "Education", "Keywords"],
                    "Score": [skill_score, exp_score, edu_score, keyword_score],
                }
            ).set_index("Category")
            st.bar_chart(score_chart)

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Skills Comparison")
        if jd_skills:
            skill_status_df = pd.DataFrame(
                {
                    "skill": sorted(jd_skills),
                }
            )
            skill_status_df["Matched"] = skill_status_df["skill"].apply(lambda x: 1 if x in matched_skills else 0)
            skill_status_df["Missing"] = 1 - skill_status_df["Matched"]
            st.bar_chart(skill_status_df.set_index("skill")[["Matched", "Missing"]])

            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.pie(
                [len(matched_skills), len(missing_skills)],
                labels=["Matched Skills", "Missing Skills"],
                startangle=90,
                colors=[COLORS["green"], COLORS["amber"]],
                wedgeprops={"linewidth": 1, "edgecolor": "white"},
            )
            ax.axis("equal")
            _, chart_col, _ = st.columns([1, 2, 1])
            with chart_col:
                st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No predefined JD skills found for visual comparison.")

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Keyword Analysis")
        jd_freq = Counter(jd_tokens)
        matched_kw_df = pd.DataFrame(
            [(k, jd_freq[k]) for k in keyword_overlap], columns=["keyword", "frequency"]
        ).sort_values("frequency", ascending=False)
        missing_kw_df = pd.DataFrame(
            [(k, jd_freq[k]) for k in missing_keywords], columns=["keyword", "frequency"]
        ).sort_values("frequency", ascending=False)

        kw_col1, kw_col2 = st.columns(2)
        with kw_col1:
            if not matched_kw_df.empty:
                st.bar_chart(matched_kw_df.head(12).set_index("keyword"))
            else:
                st.info("No matched keywords to visualize.")
        with kw_col2:
            if not missing_kw_df.empty:
                st.bar_chart(missing_kw_df.head(12).set_index("keyword"))
            else:
                st.info("No missing keywords to visualize.")

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Experience Analysis")
        exp_df = pd.DataFrame(
            {
                "type": ["Required (JD)", "Available (Resume)"],
                "years": [jd_exp if jd_exp else 0.0, resume_exp],
            }
        ).set_index("type")
        exp_col1, exp_col2 = st.columns([2, 1])
        with exp_col1:
            st.bar_chart(exp_df)
        with exp_col2:
            render_donut(exp_score, "Experience Match", COLORS["blue"])

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Section-wise Resume Evaluation")
        section_rows = []
        status_to_score = {"Strong": 1.0, "Moderate": 0.6, "Missing": 0.2}
        for section, hints in SECTION_HINTS.items():
            status = classify_section_status(resume_text, hints)
            section_rows.append(
                {
                    "Section": section,
                    "StatusScore": status_to_score[status],
                }
            )
        section_df = pd.DataFrame(section_rows).set_index("Section")
        st.bar_chart(section_df)

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Resume Improvement Suggestions")
        suggestions = generate_improvement_suggestions(
            skill_gap=missing_skills,
            keyword_gap=missing_keywords,
            exp_gap=max(jd_exp - resume_exp, 0.0),
            education_gap=jd_edu - resume_edu,
        )
        suggestion_df = pd.DataFrame(
            {
                "Area": ["Skills Gap", "Keyword Gap", "Experience Gap", "Education Gap"],
                "Priority": [
                    len(missing_skills),
                    len(missing_keywords),
                    max(jd_exp - resume_exp, 0.0),
                    len(jd_edu - resume_edu),
                ],
            }
        ).set_index("Area")
        st.bar_chart(suggestion_df)

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Resume-JD Similarity Explanation")
        render_similarity_pie(final_match_score)
        contribution_df = pd.DataFrame(
            {
                "Component": ["Skills", "Experience", "Education", "Keywords"],
                "Value": [skill_score, exp_score, edu_score, keyword_score],
            }
        ).set_index("Component")
        st.bar_chart(contribution_df)

        st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)
        render_section_header("Download Report")
        pdf_report = build_analysis_pdf(
            final_match_score=final_match_score,
            skill_score=skill_score,
            exp_score=exp_score,
            edu_score=edu_score,
            keyword_score=keyword_score,
            suggestions=suggestions,
        )
        st.download_button(
            label="Download Analysis Report",
            data=pdf_report,
            file_name="resume_analysis_report.pdf",
            mime="application/pdf",
        )


st.markdown("<div class='sticky-nav'>", unsafe_allow_html=True)
st.markdown("<h1 class='app-title'>Smart Cv Analyszer</h1>", unsafe_allow_html=True)
selected_page = st.radio(
    "",
    ["Job Prediction", "Resume Analysis"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if selected_page == "Job Prediction":
    page_prediction_with_eda()
else:
    page_resume_jd_match()
