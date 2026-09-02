from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# -----------------------------
# Clean Text
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -----------------------------
# Get Resume File
# -----------------------------
resume_file = input("Enter resume file name: ")

reader = PdfReader(resume_file)

resume_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        resume_text += text


# -----------------------------
# Get Job Description File
# -----------------------------
job_file = input("Enter job description file name: ")

with open(job_file, "r", encoding="utf-8") as file:
    job_text = file.read()


# -----------------------------
# Clean Resume and Job Text
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
# Skills
# -----------------------------
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


# -----------------------------
# Final Result
# -----------------------------
print("\n==============================")
print("     RESUME MATCH RESULT")
print("==============================")

print("\nMatch Score:",
      round(match_score, 2), "%")


print("\nMatched Skills:")

if matched_skills:

    for skill in matched_skills:
        print("✓", skill)

else:
    print("No matching skills found.")


print("\nMissing Skills:")

if missing_skills:

    for skill in missing_skills:
        print("✗", skill)

else:
    print("No major missing skills.")


# -----------------------------
# Recommendation
# -----------------------------
print("\n==============================")
print("       RECOMMENDATION")
print("==============================")


if match_score >= 75:

    print("Excellent Match - You can apply for this job!")

elif match_score >= 50:

    print("Good Match - Improve a few skills before applying.")

else:

    print("Low Match - Learn the missing skills to improve your resume.")