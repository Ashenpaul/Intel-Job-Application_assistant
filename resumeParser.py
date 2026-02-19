import PyPDF2
import re

def parse_resume(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except:
        return "Error parsing PDF"

def extract_skills(text: str) -> list:
    skills = ['python', 'java', 'react', 'sql', 'docker', 'aws', 'ai']
    text_lower = text.lower()
    found_skills = []
    for skill in skills:
        if skill in text_lower:
            found_skills.append(skill.title())
    return found_skills
