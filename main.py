from flask import Flask, request, render_template
import os
import docx2txt 
import PyPDF2 
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

@app.route("/")
def matchresume():
    return render_template('matchresume.html')
@app.route('/matcher', methods=['POST'])
def matcher():
    job_description = request.form.get('job_description')
    resume_files = request.files.getlist('resumes')

    if not job_description or len(resume_files) < 2:
        return render_template(
            'matchresume.html',
            message="Please upload at least 2 resumes at once (CTRL + select)."
        )

    resumes_text = []
    filenames = []

    for resume_file in resume_files:
        unique_name = f"{uuid.uuid4()}_{resume_file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)

        resume_file.save(filepath)
        resumes_text.append(extract_text(filepath))
        filenames.append(resume_file.filename)


    # TF-IDF + Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([job_description] + resumes_text)

    similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()
    scores = [round(s * 100, 2) for s in similarities]

    # Rank resumes
    ranked = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

    top_resumes = []
    similarity_scores = []
    status = []

    for i, (name, score) in enumerate(ranked[:5]):
        top_resumes.append(name)
        similarity_scores.append(score)

        if i == 0:
            status.append("SELECTED")
        elif i <= 2:
            status.append("SHORTLISTED")
        else:
            status.append("REJECTED")

    return render_template(
        'matchresume.html',
        message="Top matching resumes:",
        top_resumes=top_resumes,
        similarity_scores=similarity_scores,
        status=status
    )
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(port=5001, debug=True)