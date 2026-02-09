import streamlit as st
import spacy
import re
from pdfminer.high_level import extract_text
import tempfile
import os

#Load SpaCy
nlp = spacy.load("en_core_web_sm")

#Skill Database
SKILLS_DB = [
    "machine learning", "python", "java", "sql", "communication", "teamwork",
    "leadership", "teamlead", "manager", "scrum", "sqa", "word", "access",
    "deep learning", "data analysis", "nlp", "project management", "excel",
    "pandas", "keras", "unity", "sfml", "godot", "c", "c++", "tensorflow"
]

#Degree Map
DEGREE_MAP = {
    "bachelor": "Bachelor's",
    "bs": "Bachelor's",
    "b.sc": "Bachelor's",
    "b.tech": "Bachelor's",
    "master": "Master's",
    "ms": "Master's",
    "m.sc": "Master's",
    "m.tech": "Master's",
    "phd": "PhD"
}

#PDF Text Extraction
def extract_pdf_text(pdf_path):
    return extract_text(pdf_path)

#Name Extraction
def extract_name(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines[:10]:
        if "@" in line or any(char.isdigit() for char in line):
            continue

        words = line.split()
        if 2 <= len(words) <= 4:
            return line

    return None

#Email Extraction
def extract_email(text):
    match = re.findall(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )
    return match[0] if match else None

#Phone Extraction
def extract_phone(text):
    match = re.findall(r"\+?\d[\d -]{8,}\d", text)
    return match[0] if match else None

#Skills Extraction
def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILLS_DB if skill in text]
    return sorted(set(found))

#Education Extraction
def extract_education(text):
    found = set()
    text = text.lower()

    for key, value in DEGREE_MAP.items():
        if re.search(rf"\b{key}\b", text):
            found.add(value)

    return list(found)

#Resume Parser
def parse_resume(pdf_path):
    text = extract_pdf_text(pdf_path)

    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text),
        "Education": extract_education(text)
    }

#streamlit from here

st.set_page_config(page_title="NLP Resume Parser", page_icon="📄")
st.title("📄 NLP Resume Parser")
st.write("Upload a PDF resume to extract structured information")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])


if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("Parsing resume..."):
        resume_text = extract_pdf_text(temp_path)
        parsed_data = {
            "Name": extract_name(resume_text),
            "Email": extract_email(resume_text),
            "Phone": extract_phone(resume_text),
            "Skills": extract_skills(resume_text),
            "Education": extract_education(resume_text)
        }

    os.remove(temp_path)

    st.subheader("✅ Extracted Information")
    st.json(parsed_data)

    st.markdown("---")
    st.subheader("📃 Raw Resume Text")
    st.text_area("Resume Content", resume_text, height=300)


