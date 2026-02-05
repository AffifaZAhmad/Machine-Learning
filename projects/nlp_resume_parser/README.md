**AI RESUME PARSER SYSTEM:**

This project builds an **AI-Powered Resume Parsing System** using **Natural Language Processing (NLP)** and **Pattern Matching (Regex)**. It automatically extracts structured information from **PDF resumes** such as **Name, Email, Phone Number, Skills, and Education**, similar to real-world ATS (Applicant Tracking Systems).

### **Technologies Used**

* Python (Jupyter Notebook)
* spaCy (NLP)
* pdfminer.six (PDF Text Extraction)
* Regex (Pattern Matching)
* Pandas (optional for structured output)

### **Parsing Techniques**

#### **1. PDF Text Extraction**

* Extracts raw text from PDF resumes.
* Converts unstructured resume content into readable text.

#### **2. Name Extraction (spaCy NLP)**

* Uses spaCy’s English language model.
* Identifies **PERSON entities** to extract candidate names.

#### **3. Contact Information Extraction**

* Uses regex patterns to extract:
  * Email addresses
  * Phone numbers

#### **4. Skill Extraction**

* Matches resume text against a predefined **skills database**.
* Extracts relevant technical and soft skills present in the resume.

#### **5. Education Extraction**

* Detects degree keywords such as:
  Bachelor, Master, BS, MS, PhD
* Returns identified education levels.

### **Example Output**

{
 'Name': 'Affifa Z Ahmad',
 'Email': 'affifa@email.com',
 'Phone': '+923001234567',
 'Skills': ['python', 'machine learning', 'nlp'],
 'Education': ['BACHELOR']
}

### **How to Use**

* Provide the file path of the resume PDF.
* Run the parsing function or notebook.
* The system outputs structured resume information.

Example:

resume_data = parse_resume("path/to/resume.pdf")
print(resume_data)

### **Applications**

* Resume screening systems
* Recruitment automation
* Applicant Tracking Systems (ATS)
* HR analytics tools

### **Future Enhancements**

* Experience and project extraction
* DOCX resume support
* Machine learning-based skill detection
* Web interface using Streamlit or Flask
