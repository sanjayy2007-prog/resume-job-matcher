from flask import Flask, render_template, request
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    # Get uploaded resume
    resume = request.files["resume"]

    # Get job description from textarea
    job_text = request.form["job_description_text"]

    # Save resume
    resume_path = os.path.join(
        UPLOAD_FOLDER,
        resume.filename
    )

    resume.save(resume_path)

    # -----------------------------
    # Read Resume PDF
    # -----------------------------

    reader = PdfReader(resume_path)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text

    # -----------------------------
    # Clean Text
    # -----------------------------

    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    # -----------------------------
    # TF-IDF
    # -----------------------------

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_clean, job_clean]
    )

    # -----------------------------
    # Cosine Similarity
    # -----------------------------

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )

    match_score = similarity[0][0] * 100

    # -----------------------------
    # Skill Database
    # -----------------------------

    skills = [
        "python",
        "java",
        "c++",
        "sql",
        "machine learning",
        "deep learning",
        "data science",
        "data analysis",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit learn",
        "tensorflow",
        "pytorch",
        "natural language processing",
        "computer vision",
        "flask",
        "django",
        "git",
        "github",
        "excel",
        "power bi",
        "tableau",
        "statistics",
        "html",
        "css",
        "javascript"
    ]

    matched_skills = []
    missing_skills = []

    # -----------------------------
    # Compare Required Skills
    # -----------------------------

    for skill in skills:

        if skill in job_clean:

            if skill in resume_clean:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

    # -----------------------------
    # Recommendation
    # -----------------------------

    if match_score >= 75:

        recommendation = (
            "Excellent Match - "
            "You can apply for this job!"
        )

    elif match_score >= 50:

        recommendation = (
            "Good Match - "
            "Improve a few skills before applying."
        )

    else:

        recommendation = (
            "Low Match - "
            "Learn the missing skills to improve "
            "your resume."
        )

    # -----------------------------
    # Show Result Page
    # -----------------------------

    return render_template(
        "result.html",
        score=round(match_score, 2),
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)