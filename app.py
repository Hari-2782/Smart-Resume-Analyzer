import os
import re
import fitz  
import numpy as np
import spacy
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

SECTION_HEADERS = [
    "experience", "education", "skills", "projects", "certifications", "languages", "summary", "contact", "extracurriculars"
]

# --- Utility Functions ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def extract_text_from_html(html):
    # Remove tags, keep text
    return re.sub('<[^<]+?>', '', html).replace('\n', ' ').strip()

def spacy_keywords(text, top_n=20):
    doc = nlp(text)
    # Use noun chunks and entities for keywords
    keywords = set()
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 3:
            keywords.add(chunk.lemma_.lower())
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART", "LANGUAGE"]:
            keywords.add(ent.lemma_.lower())
    # Fallback: most common nouns
    words = [token.lemma_.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop]
    for w in words:
        keywords.add(w)
    return list(keywords)[:top_n]

def check_sections_spacy(text):
    found = []
    for section in SECTION_HEADERS:
        # Use regex and spaCy to find section headers
        if re.search(rf'\b{section}\b', text, re.IGNORECASE):
            found.append(section)
        else:
            doc = nlp(text)
            for sent in doc.sents:
                if section in sent.text.lower():
                    found.append(section)
                    break
    return list(set(found))

def get_synonyms_spacy(word, job_keywords):
    # Use spaCy similarity to find related terms
    word_doc = nlp(word)
    for kw in job_keywords:
        kw_doc = nlp(kw)
        if word_doc.similarity(kw_doc) > 0.8:
            return True
    return False

def check_ats_friendly(text):
    # Penalize for tables, columns, images, graphics (very basic)
    issues = []
    if re.search(r'table|column|image|graphic', text, re.IGNORECASE):
        issues.append("Avoid tables, columns, images, or graphics for ATS-friendliness.")
    if len(text) > 20000:
        issues.append("Resume is too long. Keep it concise.")
    return issues

def compute_ats_score(resume_text, job_desc):
    # 1. Similarity score
    sim_score = compute_similarity(resume_text, job_desc)
    # 2. Section coverage
    found_sections = check_sections_spacy(resume_text)
    section_score = int(100 * len(found_sections) / len(SECTION_HEADERS))
    # 3. Keyword match (with synonyms)
    job_keywords = spacy_keywords(job_desc)
    resume_keywords = set(spacy_keywords(resume_text, top_n=50))
    matched_keywords = [kw for kw in job_keywords if kw in resume_keywords or get_synonyms_spacy(kw, resume_keywords)]
    keyword_score = int(100 * len(matched_keywords) / max(1, len(job_keywords)))
    # 4. ATS issues
    ats_issues = check_ats_friendly(resume_text)
    ats_score = 100 if not ats_issues else max(60, 100 - 20 * len(ats_issues))
    # Weighted total
    total_score = int(0.5 * sim_score + 0.2 * section_score + 0.2 * keyword_score + 0.1 * ats_score)
    return total_score, found_sections, job_keywords, matched_keywords, ats_issues

def generate_suggestions(resume_text, job_desc, found_sections, job_keywords, matched_keywords, ats_issues, score):
    suggestions = []
    # Section suggestions
    for section in SECTION_HEADERS:
        if section not in found_sections:
            suggestions.append({"category": "Section", "message": f"Add a '{section.title()}' section to your resume."})
    # Keyword suggestions (with synonyms)
    for kw in job_keywords:
        if kw not in matched_keywords:
            suggestions.append({"category": "Keyword", "message": f"Include or rephrase to match the keyword '{kw}' from the job description."})
    # ATS issues
    for issue in ats_issues:
        suggestions.append({"category": "ATS", "message": issue})
    # Score improvement
    if score < 80:
        suggestions.append({"category": "Score", "message": "Follow the above suggestions to reach an 80+ score."})
    return suggestions

# --- Old Similarity Function ---
def compute_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
    return round(similarity[0][0] * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "resume" not in request.files:
            return "No file uploaded", 400
        file = request.files["resume"]
        job_desc = request.form["job_desc"]

        if file.filename == "" or job_desc.strip() == "":
            return "Invalid input", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        resume_text = extract_text_from_pdf(filepath)
        score, found_sections, job_keywords, matched_keywords, ats_issues = compute_ats_score(resume_text, job_desc)
        suggestions = generate_suggestions(resume_text, job_desc, found_sections, job_keywords, matched_keywords, ats_issues, score)

        return render_template("index.html", score=score, suggestions=suggestions, resume_text=resume_text)

    return render_template("index.html", score=None, suggestions=None, resume_text=None)

@app.route("/editor", methods=["GET", "POST"])
def editor():
    if request.method == "POST":
        resume_html = request.form.get("resume_html", "")
        job_desc = request.form.get("job_desc", "")
        resume_text = extract_text_from_html(resume_html)
        score, found_sections, job_keywords, matched_keywords, ats_issues = compute_ats_score(resume_text, job_desc)
        suggestions = generate_suggestions(resume_text, job_desc, found_sections, job_keywords, matched_keywords, ats_issues, score)
        return jsonify({"score": score, "suggestions": suggestions, "message": "Resume scored and suggestions generated."})
    # GET: load editor with blank or sample resume
    resume_text = request.args.get('resume_text', None)
    return render_template("editor.html", resume_text=resume_text)

if __name__ == "__main__":
    app.run(debug=True)