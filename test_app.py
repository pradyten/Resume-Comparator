from app import preprocess_text, analyze_resume_keywords

def test_stopwords_removal():
    text = "This is a sample resume with python and java skills."
    cleaned = preprocess_text(text)
    # Check basic stopword removal and lowercase
    assert "this" not in cleaned
    assert "python" in cleaned

def test_keyword_analysis():
    resume_text = "I have experience with python, java, and sql."
    job_desc = "Looking for skills in python, sql, cloud."
    missing, _ = analyze_resume_keywords(resume_text, job_desc)
    assert "cloud" in missing.get("skills", [])
