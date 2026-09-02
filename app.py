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
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    resume = request.files["resume"]
    job_description = request.files["job_description"]

    resume_path = os.path.join(UPLOAD_FOLDER, resume.filename)
    job_path = os.path.join(UPLOAD_FOLDER, job_description.filename)

    resume.save(resume_path)
    job_description.save(job_path)

    # Read Resume
    reader = PdfReader(resume_path)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text

    # Read Job Description
    with open(job_path, "r", encoding="utf-8") as file:
        job_text = file.read()

    # Clean text
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    # TF-IDF
    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(
        [resume_clean, job_clean]
    )

    # Similarity
    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )

    match_score = similarity[0][0] * 100

    # Skills
    skills = [
        "python",
        "machine learning",
        "data analysis",
        "pandas",
        "numpy",
        "scikit learn",
        "sql",
        "natural language processing"
    ]

    matched_skills = []
    missing_skills = []

    for skill in skills:

        if skill in resume_clean:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # Recommendation
    if match_score >= 75:
        recommendation = "Excellent Match - You can apply for this job!"

    elif match_score >= 50:
        recommendation = "Good Match - Improve a few skills before applying."

    else:
        recommendation = "Low Match - Learn the missing skills to improve your resume."

    return f"""
    <h1>Resume Match Result</h1>

    <h2>Match Score: {round(match_score, 2)}%</h2>

    <h3>Matched Skills</h3>
    <p>{", ".join(matched_skills)}</p>

    <h3>Missing Skills</h3>
    <p>{", ".join(missing_skills)}</p>

    <h3>Recommendation</h3>
    <p>{recommendation}</p>

    <br>
    <a href="/">Analyze Another Resume</a>
    """


if __name__ == "__main__":
    app.run(debug=True)