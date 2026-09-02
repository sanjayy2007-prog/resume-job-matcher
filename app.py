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


def extract_pdf_text(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    return text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    # -----------------------------
    # Resume
    # -----------------------------

    if "resume" not in request.files:
        return "Error: Please upload a resume."

    resume = request.files["resume"]

    if resume.filename == "":
        return "Error: Please select a resume PDF."

    if not resume.filename.lower().endswith(".pdf"):
        return "Error: Resume must be a PDF file."

    resume_path = os.path.join(
        UPLOAD_FOLDER,
        resume.filename
    )

    resume.save(resume_path)

    try:
        resume_text = extract_pdf_text(resume_path)
    except Exception:
        return "Error: Unable to read the resume PDF."

    if not resume_text.strip():
        return "Error: No readable text found in the resume."


    # -----------------------------
    # Job Description
    # -----------------------------

    job_text = request.form.get(
        "job_description_text",
        ""
    ).strip()

    job_pdf = request.files.get(
        "job_description_pdf"
    )

    # If text is empty, use PDF
    if not job_text and job_pdf and job_pdf.filename:

        if not job_pdf.filename.lower().endswith(".pdf"):
            return "Error: Job Description must be a PDF file."

        job_path = os.path.join(
            UPLOAD_FOLDER,
            job_pdf.filename
        )

        job_pdf.save(job_path)

        try:
            job_text = extract_pdf_text(job_path)
        except Exception:
            return "Error: Unable to read the Job Description PDF."

    if not job_text.strip():
        return "Error: Please paste a Job Description or upload a PDF."


    # -----------------------------
    # Clean Text
    # -----------------------------

    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)


    # -----------------------------
    # TF-IDF
    # -----------------------------

    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [resume_clean, job_clean]
        )

    except Exception:
        return "Error: Unable to process the documents."


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


    # -----------------------------
    # Skill Matching
    # -----------------------------

    matched_skills = []
    missing_skills = []

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
    # Result
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