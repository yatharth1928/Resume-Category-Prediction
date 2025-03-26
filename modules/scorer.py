import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz

# ✅ Load the auto-generated keywords
with open('category_keywords.json', 'r') as f:
    category_keywords = json.load(f)

# ✅ Preprocessing function
def preprocess(text):
    #"""Clean and lowercase text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text

# ✅ Fuzzy matching function
def fuzzy_match(keyword, text, threshold=80):
    words = text.split()
    matches = [word for word in words if fuzz.partial_ratio(keyword, word) >= threshold]
    return len(matches)

# ✅ Enhanced scoring function
def calculate_resume_score(resume_text, category):
    resume_text = preprocess(resume_text)

    # ✅ Get category keywords
    keywords = category_keywords.get(category, [])
    
    if not keywords:
        return 0  # Return 0 score if no keywords found

    # ✅ TF-IDF for weighting
    tfidf = TfidfVectorizer(vocabulary=keywords)
    tfidf_matrix = tfidf.fit_transform([resume_text])
    
    # ✅ TF-IDF Score
    tfidf_score = tfidf_matrix.sum()

    # ✅ Fuzzy Matching Score
    fuzzy_score = sum(fuzzy_match(keyword, resume_text) for keyword in keywords)

    # ✅ Weighted scoring formula
    combined_score = (0.7 * tfidf_score) + (0.3 * fuzzy_score)

    # ✅ Normalize to 100-point scale
    normalized_score = min(combined_score * 10, 100)

    return round(normalized_score, 2)
