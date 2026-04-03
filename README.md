# 📄 ATS Resume Checker — End-to-End NLP Application

A complete end-to-end NLP web application that checks how well your resume matches a job description — giving you an **ATS compatibility score from 0 to 100**, just like real Applicant Tracking Systems used by recruiters.

🔗 **[Live Demo](https://ats-resume-checker-sdq7.onrender.com)** *(Deployed on Render — may take a few seconds to wake up)*

---

## 🚀 Features

- **Multi-format Resume Support** — Upload your resume in any format: PDF, DOCX, or plain text
- **Job Description Input** — Paste any job description directly into the app
- **ATS Score (0–100)** — Get an instant compatibility score based on semantic similarity between your resume and the JD
- **Custom NLP Pipeline** — Built from scratch using spaCy, not a black-box API
- **Dual Server Architecture** — Powered by both Flask & FastAPI
- **Dockerized** — Fully containerized for consistent, reproducible deployments

---

## 🧠 How It Works

The scoring engine is built on a custom NLP pipeline using **spaCy**:

### 1. Text Extraction
- Parses resume content from PDF, DOCX, or plain text formats
- Extracts raw job description from user input

### 2. Custom NLP Preprocessing (spaCy)
- **Tokenization** — Splits text into meaningful tokens
- **Lemmatization** — Reduces words to their root form *(e.g., "running" → "run")*
- **Stopword Removal** — Filters out noise words *(e.g., "the", "is", "and")*

### 3. Cosine Similarity Scoring
- Converts both cleaned texts into vector representations
- Computes **Cosine Similarity** to measure semantic closeness
- Maps similarity to a **0–100 ATS score**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| NLP Engine | spaCy (Tokenizer, Lemmatizer, Stopword Removal) |
| Similarity | Cosine Similarity |
| Backend | Flask + FastAPI |
| Containerization | Docker |
| Deployment | Render |
| Language | Python |

---

## 🐳 Docker Setup

```bash
# Pull and run the Docker image
docker build -t ats-checker .
docker run -p 5000:5000 ats-checker
```

---

## ⚙️ Local Setup

```bash
# Clone the repository
git clone https://github.com/Shivansh212/ATS-RESUME-CHECKER
cd ats-resume-checker

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
python app.py
```

---



## 💡 Why This Project?

Most candidates apply to jobs without knowing if their resume will even pass the ATS filter. This tool demystifies that process — giving instant, explainable feedback powered by real NLP, not just keyword counting.

---

## 🙌 Author

Built with 💻 and curiosity. Feel free to fork, star ⭐, or open issues!