import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------
# Download NLTK Data
# -------------------------

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))


# -------------------------
# Text Preprocessing
# -------------------------

def preprocess_text(text):

    text = str(text).lower()

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [word for word in tokens if word not in string.punctuation]

    return " ".join(tokens)


# -------------------------
# Skill Extraction (FAST)
# -------------------------

def extract_skills(text, skills_db):

    text = str(text).lower()

    found_skills = []

    for skill in skills_db:

        if skill in text:

            found_skills.append(skill)

    return list(set(found_skills))


# -------------------------
# Load Skills Database
# -------------------------

with open("skills_db.txt") as f:

    skills_db = [skill.strip().lower() for skill in f.readlines()]


# -------------------------
# Load Resume Dataset
# -------------------------

df = pd.read_csv("Resume.csv")

# Optional: limit dataset for faster testing
df = df.head(500)

print("Total Resumes:", len(df))

print("\nDataset Columns:")
print(df.columns)


# -------------------------
# Detect Resume Column
# -------------------------

resume_column = None

for col in df.columns:

    if "resume" in col.lower():

        resume_column = col

        break


if resume_column is None:

    print("ERROR: Resume column not found in dataset")
    exit()


print("\nUsing Resume Column:", resume_column)


# -------------------------
# Job Description
# -------------------------

with open("job_description.txt") as f:

    job_description = f.read()


clean_job = preprocess_text(job_description)

job_skills = extract_skills(job_description, skills_db)

print("\nJob Required Skills:", job_skills)


# -------------------------
# Process Resume Dataset
# -------------------------

resume_texts = []
resume_skills = []

for resume in df[resume_column]:

    clean_resume = preprocess_text(resume)

    skills = extract_skills(resume, skills_db)

    resume_texts.append(clean_resume)

    resume_skills.append(skills)


# -------------------------
# TF-IDF Vectorization
# -------------------------

documents = [clean_job] + resume_texts

vectorizer = TfidfVectorizer(max_features=5000)

tfidf_matrix = vectorizer.fit_transform(documents)


# -------------------------
# Cosine Similarity
# -------------------------

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()


# -------------------------
# Ranking Candidates
# -------------------------

results = []

for i in range(len(resume_texts)):

    matched = list(set(resume_skills[i]).intersection(job_skills))

    missing = list(set(job_skills) - set(resume_skills[i]))

    results.append({

        "candidate_id": i,
        "score": round(similarity[i] * 100, 2),
        "matched_skills": matched,
        "missing_skills": missing

    })


results = sorted(results, key=lambda x: x["score"], reverse=True)


# -------------------------
# Display Top Candidates
# -------------------------

print("\nTop 10 Candidates\n")

for r in results[:10]:

    print("Candidate ID:", r["candidate_id"])
    print("Match Score:", r["score"], "%")
    print("Matched Skills:", r["matched_skills"])
    print("Missing Skills:", r["missing_skills"])
    print("--------------------------------")