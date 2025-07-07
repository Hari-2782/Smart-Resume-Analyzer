# Resume Score

Resume Score is a web application that analyzes and scores resumes against job descriptions using ATS (Applicant Tracking System) criteria. It provides actionable suggestions to improve your resume's match with a given job description.

## Features
- Upload your resume (PDF) and job description to receive an ATS compatibility score
- Highlights missing sections and keywords
- Provides suggestions to improve your resume
- Editor mode for direct resume text input and scoring
- Supports synonym and section detection using NLP
- Built with Flask, spaCy, and Sentence Transformers

## How It Works
1. Upload your resume and paste the job description.
2. The app extracts text from your resume and analyzes it using NLP.
3. It compares your resume to the job description for section coverage, keyword match, and ATS-friendliness.
4. You receive a score and suggestions for improvement.

## Installation
1. Clone this repository.
2. Ensure Python 3.11+ is installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or, if using poetry:
   poetry install
   ```
4. Download the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
5. Screenshots
![Screenshot 2025-07-07 153913](https://github.com/user-attachments/assets/fa5948d4-ac2e-45b3-b2f8-fa31c17697b1)
![Screenshot 2025-07-07 153939](https://github.com/user-attachments/assets/7ac1484a-52ee-4760-971f-28096ea51da8)


## Usage
Run the app locally:
```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

## Dependencies
- Flask
- sentence-transformers
- pymupdf
- scikit-learn
- nltk
- language-tool-python
- spacy

## License
MIT License

## Author
Harisanth v
