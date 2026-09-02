from flask import Flask, render_template, request, Response
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
        return (
            "Error: Please paste a Job Description "
            "or upload a PDF."
        )


    # -----------------------------
    # Clean Text
    # -----------------------------

    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)


    # -----------------------------
    # TF-IDF Similarity
    # -----------------------------

    try:

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [resume_clean, job_clean]
        )

        similarity = cosine_similarity(
            vectors[0],
            vectors[1]
        )

        text_score = similarity[0][0] * 100

    except Exception:
        return "Error: Unable to calculate similarity."


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
    # Skill Score
    # -----------------------------

    total_required_skills = (
        len(matched_skills)
        +
        len(missing_skills)
    )


    if total_required_skills > 0:

        skill_score = (
            len(matched_skills)
            /
            total_required_skills
        ) * 100

    else:

        skill_score = 0


    # -----------------------------
    # Final Score
    # -----------------------------

    final_score = (

        (text_score * 0.70)

        +

        (skill_score * 0.30)

    )


    # Keep score between 0 and 100

    final_score = max(
        0,
        min(100, final_score)
    )


    # -----------------------------
    # Recommendation
    # -----------------------------

    if final_score >= 75:

        recommendation = (
            "Excellent Match - "
            "Your resume strongly matches this job."
        )

    elif final_score >= 50:

        recommendation = (
            "Good Match - "
            "Improve the missing skills to increase "
            "your chances."
        )

    else:

        recommendation = (
            "Low Match - "
            "Consider improving the missing skills "
            "and tailoring your resume."
        )


    # -----------------------------
    # Result Page
    # -----------------------------

    return render_template(
        "result.html",

        score=round(
            final_score,
            2
        ),

        text_score=round(
            text_score,
            2
        ),

        skill_score=round(
            skill_score,
            2
        ),

        matched_skills=matched_skills,

        missing_skills=missing_skills,

        recommendation=recommendation
    )


# =================================================
# DOWNLOAD REPORT
# =================================================

@app.route(
    "/download-report",
    methods=["POST"]
)
def download_report():

    score = request.form.get(
        "score",
        "0"
    )

    text_score = request.form.get(
        "text_score",
        "0"
    )

    skill_score = request.form.get(
        "skill_score",
        "0"
    )

    recommendation = request.form.get(
        "recommendation",
        ""
    )

    matched_skills = request.form.getlist(
        "matched_skills"
    )

    missing_skills = request.form.getlist(
        "missing_skills"
    )


    # -----------------------------
    # Create Report
    # -----------------------------

    report = f"""
========================================
        RESUME JOB MATCHER
========================================

FINAL MATCH SCORE
-----------------
{score}%


TEXT SIMILARITY SCORE
---------------------
{text_score}%


SKILL MATCH SCORE
-----------------
{skill_score}%


MATCHED SKILLS
--------------
"""


    if matched_skills:

        for skill in matched_skills:

            report += f"- {skill}\n"

    else:

        report += "- No matched skills found\n"


    report += """

MISSING SKILLS
--------------
"""


    if missing_skills:

        for skill in missing_skills:

            report += f"- {skill}\n"

    else:

        report += "- No missing skills\n"


    report += f"""

RECOMMENDATION
--------------
{recommendation}


========================================
Generated by Resume Job Matcher
========================================
"""


    # -----------------------------
    # Download
    # -----------------------------

    return Response(

        report,

        mimetype="text/plain",

        headers={

            "Content-Disposition":
            "attachment; "
            "filename=resume_match_report.txt"

        }

    )


# =================================================
# RUN APPLICATION
# =================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )