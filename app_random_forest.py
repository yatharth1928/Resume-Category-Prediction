import streamlit as st
import joblib
import re
import PyPDF2
import json
#from fuzzywuzzy import fuzz
#from modules.scorer import calculate_resume_score

# ✅ Load the model and vectorizer
model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

with open('category_keywords.json', 'r') as f:
    category_keywords = json.load(f)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + " "
    return text.strip()

# 🌟 Streamlit UI
st.title("Resume Classifier with Random Forest")

uploaded_file = st.file_uploader("Upload Resume (TXT or PDF)", type=["txt", "pdf"])

#category = st.selectbox("Select Job Category", list(category_keywords.keys()))

if uploaded_file is not None:
    # ✅ Read the resume
    text=extract_text_from_pdf(uploaded_file)
    
    # ✅ Vectorize the resume
    input_vector = vectorizer.transform([text])
    
    
    # 🔥 Predict the category
    prediction = model.predict(input_vector)[0]

    # ✅ Score the resume for the selected category
    #resume_score = calculate_resume_score(text, category)
    
    # ✅ Display the prediction
    st.success(f'Predicted Job Category: **{prediction}**')
    #st.info(f'Resume Score for {category}: **{resume_score}/100**')
#streamlit run app_random_forest.py

