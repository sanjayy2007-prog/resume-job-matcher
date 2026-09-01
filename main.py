from pypdf import PdfReader

reader = PdfReader("sample_resume.pdf")

print("Number of pages:", len(reader.pages))

for page in reader.pages:
    text = page.extract_text()
    print(text)