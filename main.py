from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read Resume
reader = PdfReader("sample_resume.pdf")
resume_text = ""

for page in reader.pages:
    resume_text += page.extract_text()

# Read Job Description
with open("job_description.txt", "r", encoding="utf-8") as file:
    job_text = file.read()

# Convert text into numbers
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform([resume_text, job_text])

# Calculate similarity
similarity = cosine_similarity(vectors[0], vectors[1])

match_score = similarity[0][0] * 100

print("Resume Match Score:", round(match_score, 2), "%")