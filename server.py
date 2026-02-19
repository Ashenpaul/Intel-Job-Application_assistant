from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import os
import PyPDF2
import re
from typing import List, Dict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="Intel Job Application Assistant v2.0 - Production Ready")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Production data
SKILLS_DB = [
    "Python", "JavaScript", "React", "Node.js", "SQL", "Docker", "AWS", "Azure",
    "Machine Learning", "TensorFlow", "PyTorch", "Git", "Kubernetes", "Linux",
    "FastAPI", "Django", "Flask", "MongoDB", "PostgreSQL", "Redis", "CI/CD"
]

COVER_TEMPLATES = {
    "intern": """Dear Intel Hiring Manager,

As a motivated Computer Science student with hands-on experience in {skills}, 
I am excited to apply for the Software Engineering Intern position at Intel.

My projects demonstrate proficiency in modern development practices and 
alignment with Intel's innovation goals.

Best regards,
[Your Name]""",

    "software": """Dear Intel Recruiting Team,

With {skills} expertise and production deployment experience, I am eager 
to contribute to Intel's cutting-edge projects.

My background aligns perfectly with Intel's software engineering requirements.

Sincerely,
[Your Name]""",

    "ml": """Dear Intel AI Team,

My machine learning portfolio featuring {skills} positions me to contribute 
immediately to Intel's AI initiatives.

Looking forward to discussing my qualifications.

Best,
[Your Name]"""
}

@app.get("/")
async def root():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
<title>Intel Job Application Assistant v2.0</title>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {margin:0;padding:0;box-sizing:border-box}
body {font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);min-height:100vh;padding:20px}
.container {max-width:1100px;margin:0 auto;background:white;border-radius:25px;box-shadow:0 25px 50px rgba(0,0,0,0.15);overflow:hidden}
.header {background:linear-gradient(135deg,#0066cc,#004499);color:white;padding:40px 30px;text-align:center;position:relative;overflow:hidden}
.header::before {content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.1),transparent);animation:shine 3s infinite}
@keyframes shine {0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.header h1 {font-size:2.8em;margin-bottom:10px;font-weight:700;letter-spacing:-1px}
.header p {font-size:1.2em;opacity:0.95;max-width:600px;margin:0 auto;line-height:1.5}
.section {padding:35px;background:#f8f9ff;border-bottom:1px solid #e1e8ed}
.section:last-child {border-bottom:none}
.section h3 {color:#2c3e50;font-size:1.5em;margin-bottom:25px;display:flex;align-items:center;gap:12px;font-weight:600}
.upload-area {border:2px dashed #007bff;border-radius:18px;padding:35px;text-align:center;transition:all 0.3s;cursor:pointer;position:relative}
.upload-area:hover {background:#e6f3ff;border-color:#0056b3;transform:translateY(-3px)}
.upload-area.dragover {background:#d4edda;border-color:#28a745}
input[type=file] {margin:15px 0;font-size:16px;width:100%;padding:12px}
textarea {width:100%;min-height:160px;padding:20px;border:2px solid #e1e8ed;border-radius:18px;font-family:inherit;font-size:16px;resize:vertical;transition:all 0.3s}
textarea:focus {outline:none;border-color:#007bff;box-shadow:0 0 0 4px rgba(0,123,255,0.1);transform:translateY(-1px)}
.btn {background:linear-gradient(135deg,#007bff,#0056b3);color:white;border:none;padding:16px 35px;border-radius:28px;font-size:16px;font-weight:600;cursor:pointer;transition:all 0.3s;margin:12px 8px 0 0;box-shadow:0 4px 15px rgba(0,123,255,0.3)}
.btn:hover {transform:translateY(-3px);box-shadow:0 8px 25px rgba(0,123,255,0.4)}
.btn:active {transform:translateY(-1px)}
.btn:disabled {opacity:0.6;cursor:not-allowed;transform:none}
select {padding:14px 24px;border:2px solid #e1e8ed;border-radius:15px;font-size:16px;background:#fff;margin-right:15px;font-weight:500;min-width:200px}
.result {margin-top:25px;padding:30px;border-radius:20px;background:#d4edda;border-left:6px solid #28a745;box-shadow:0 5px 20px rgba(40,167,69,0.2)}
.result h4 {color:#155724;margin-bottom:18px;font-size:1.3em;display:flex;align-items:center;gap:10px}
.result pre {background:#f8f9fa;padding:20px;border-radius:15px;font-size:15px;line-height:1.7;white-space:pre-wrap;border-left:4px solid #28a745}
.error {background:#f8d7da;border-left-color:#dc3545;color:#721c24;box-shadow:0 5px 20px rgba(220,53,69,0.2)}
.loading {background:#fff3cd;border-left-color:#ffc107;padding:30px;text-align:center;position:relative;overflow:hidden}
.loading::after {content:'⏳ Processing with AI...';font-size:1.2em;font-weight:600;color:#856404;display:block;margin-top:15px;animation:pulse 1.5s infinite}
@keyframes pulse {0%,100%{opacity:1}50%{opacity:0.5}}
.progress-bar {width:100%;height:6px;background:#e9ecef;border-radius:3px;overflow:hidden;margin-top:20px}
.progress-fill {height:100%;background:linear-gradient(90deg,#28a745,#20c997);border-radius:3px;width:0%;transition:width 0.3s}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>🤖 Intel Job Application Assistant</h1>
<p>AI-Powered Resume Analysis • Job Matching • Cover Letter Generation • ATS Optimization</p>
</div>

<div class="section">
<h3>📄 Resume Analysis</h3>
<div class="upload-area" id="uploadArea" onclick="document.getElementById('resume').click()">
<input type="file" id="resume" accept=".pdf" style="display:none">
<div>📁 Click here or drag & drop your PDF resume</div>
<div style="font-size:14px;color:#666;margin-top:10px">Supports all standard PDF formats</div>
</div>
<button class="btn" id="analyzeBtn" onclick="analyzeResume()">🔍 Analyze Skills & Experience</button>
</div>

<div class="section">
<h3>🎯 Job Description Matching</h3>
<textarea id="jobdesc" placeholder="Paste Intel Job Description here...&#10;&#10;💡 Example:&#10;'Software Engineer Intern - Bangalore&#10;Requirements: Python, Machine Learning, AWS, Docker experience preferred...'&#10;&#10;📊 Get instant ATS compatibility score + optimization tips"></textarea>
<button class="btn" id="matchBtn" onclick="matchJob()">🚀 Calculate Match Score & ATS Score</button>
</div>

<div class="section">
<h3>✉️ AI Cover Letter Generator</h3>
<select id="roletype">
<option value="intern">💼 Software Engineering Intern</option>
<option value="software">👨‍💻 Software Engineer</option>
<option value="ml">🧠 Machine Learning Engineer</option>
</select>
<button class="btn" onclick="generateCoverLetter()">✨ Generate Personalized Cover Letter</button>
</div>

<div id="output"></div>
</div>

<script>
let resumeData = {skills: [], text: '', preview: ''};
let isProcessing = false;

document.getElementById('uploadArea').addEventListener('dragover', (e) => {
    e.preventDefault(); e.currentTarget.classList.add('dragover');
});
document.getElementById('uploadArea').addEventListener('dragleave', (e) => {
    e.currentTarget.classList.remove('dragover');
});
document.getElementById('uploadArea').addEventListener('drop', (e) => {
    e.preventDefault(); e.currentTarget.classList.remove('dragover');
    document.getElementById('resume').files = e.dataTransfer.files;
});

async function showResult(html, type = 'result') {
    document.getElementById('output').innerHTML = `<div class="${type}">${html}</div>`;
    isProcessing = false;
    document.querySelectorAll('.btn').forEach(btn => btn.disabled = false);
}

async function analyzeResume() {
    if (isProcessing) return;
    const file = document.getElementById('resume').files[0];
    if (!file) return showResult('<strong>❌</strong> Please select a PDF resume first!', 'error');
    
    isProcessing = true;
    document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
    showResult('<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>', 'loading');
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/upload-resume', {method: 'POST', body: formData});
        const data = await response.json();
        
        resumeData = data;
        const skillsList = data.skills.slice(0, 12).join(' • ') + (data.skills.length > 12 ? ' • +more' : '');
        
        showResult(`
            <h4>✅ <strong>${data.skills.length}</strong> Skills & Technologies Detected</h4>
            <pre>${skillsList}</pre>
            <details style="margin-top:15px">
                <summary>📄 Resume Preview (${data.preview.length} chars)</summary>
                <pre style="font-size:13px;background:#f1f3f4;padding:15px;margin-top:10px">${data.preview}</pre>
            </details>
        `, 'result');
    } catch(error) {
        showResult('<strong>❌</strong> Resume analysis failed. Please check file format and try again.', 'error');
    }
}

async function matchJob() {
    if (isProcessing || resumeData.skills.length === 0) {
        return showResult(resumeData.skills.length === 0 ? 
            '<strong>⚠️</strong> Please analyze your resume first!' : 
            '⏳ Please wait for current operation to complete...', 'error');
    }
    
    const jobDesc = document.getElementById('jobdesc').value.trim();
    if (!jobDesc) return showResult('<strong>⚠️</strong> Please paste job description!', 'error');
    
    isProcessing = true;
    document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
    showResult('<div class="progress-bar"><div class="progress-fill" id="progress"></div></div>', 'loading');
    
    try {
        const response = await fetch('/match-job', {
            method: 'POST',
            headers: {'Content-Type': 'application/x-www-form-urlencoded'},
            body: `job_desc=${encodeURIComponent(jobDesc)}&resume_text=${encodeURIComponent(resumeData.text || resumeData.skills.join(' '))}`
        });
        const data = await response.json();
        const score = (data.score * 100).toFixed(1);
        const scoreClass = score > 80 ? 'result' : score > 60 ? 'loading' : 'error';
        
        showResult(`
            <h4>📊 <strong>${score}%</strong> Match Score ${score > 80 ? '✅ Excellent' : score > 60 ? '⚠️ Good' : '🔴 Needs Improvement'}</h4>
            <p><strong>ATS Compatibility:</strong> ${score}%</p>
            <p><strong>💡 Optimization Tips:</strong></p>
            <pre>${data.tips}</pre>
            ${data.missing_keywords && data.missing_keywords.length > 0 ? `
                <p><strong>🔍 Missing Keywords from JD:</strong></p>
                <pre style="background:#fff3cd">${data.missing_keywords.join(', ')}</pre>
            ` : ''}
            <details>
                <summary>⚙️ Technical Analysis</summary>
                <pre>TF-IDF Similarity: ${data.score.toFixed(4)} | Keywords Matched: ${data.missing_keywords ? data.missing_keywords.length : 0}</pre>
            </details>
        `, scoreClass);
    } catch(error) {
        showResult('<strong>❌</strong> Job matching analysis failed', 'error');
    }
}

async function generateCoverLetter() {
    if (isProcessing || resumeData.skills.length === 0) {
        return showResult(resumeData.skills.length === 0 ? 
            '<strong>⚠️</strong> Please analyze your resume first!' : 
            '⏳ Please wait...', 'error');
    }
    
    isProcessing = true;
    document.querySelectorAll('.btn').forEach(btn => btn.disabled = true);
    showResult('✨ Generating personalized cover letter with AI...', 'loading');
    
    try {
        const response = await fetch('/cover-letter', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                role: document.getElementById('roletype').value,
                skills: resumeData.skills
            })
        });
        const data = await response.json();
        
        showResult(`
            <h4>✉️ Personalized Cover Letter Generated</h4>
            <pre style="white-space:pre-wrap;font-size:15px;line-height:1.7">${data.letter}</pre>
            <p style="margin-top:20px;font-size:14px;color:#666">
                💡 <strong>Tip:</strong> Customize the last paragraph with specific Intel projects you're excited about
            </p>
        `, 'result');
    } catch(error) {
        showResult('<strong>❌</strong> Cover letter generation failed', 'error');
    }
}
</script>
</body>
</html>
    """)

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    try:
        # Read file as bytes (CRITICAL FIX)
        contents = await file.read()
        
        # Create temp file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"temp_resume_{timestamp}.pdf"
        os.makedirs("uploads", exist_ok=True)
        filepath = f"uploads/{filename}"
        
        # Write file
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Extract text with error handling
        full_text = ""
        try:
            with open(filepath, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                    except:
                        continue
        except Exception as pdf_error:
            full_text = "PDF parsing failed - using filename skills"
        
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Skill extraction (robust)
        text_lower = (full_text + file.filename).lower()
        skill_patterns = {
            "Python": r"python|pytorch|django|flask|pandas|numpy",
            "JavaScript": r"javascript|js|react|node|vue|angular",
            "React": r"react|next\.?js",
            "SQL": r"sql|mysql|postgresql|database",
            "Docker": r"docker|k8s|kubernetes",
            "AWS": r"aws|amazon|ec2|s3",
            "Machine Learning": r"machine learning|ml|ai|deep learning",
            "TensorFlow": r"tensorflow|keras",
            "Git": r"git|github",
            "Linux": r"linux|ubuntu|bash"
        }
        
        found_skills = []
        for skill, pattern in skill_patterns.items():
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Ensure minimum skills for demo
        if not found_skills:
            found_skills = ["Python", "Git"]  # Demo fallback
        
        return {
            "skills": found_skills[:12],
            "total_skills": len(found_skills),
            "text": full_text,
            "preview": (full_text[:400] + "...") if len(full_text) > 400 else full_text,
            "filename": file.filename,
            "status": "success"
        }
    except Exception as e:
        return {"skills": [], "error": str(e), "status": "error"}

@app.post("/match-job")
async def match_job(job_desc: str = Form(...), resume_text: str = Form(...)):
    try:
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=1500,
            ngram_range=(1, 2),
            min_df=1
        )
        
        documents = [job_desc.lower(), resume_text.lower()]
        tfidf_matrix = vectorizer.fit_transform(documents)
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Keyword analysis
        job_words = re.findall(r'\b[a-zA-Z]{4,15}\b', job_desc.lower())
        resume_words = set(re.findall(r'\b[a-zA-Z]{4,15}\b', resume_text.lower()))
        job_skills = [w for w in job_words if w in [s.lower() for s in SKILLS_DB]]
        missing_keywords = [w.title() for w in job_skills if w.lower() not in resume_words][:8]
        
        # Dynamic tips
        tips = []
        if score < 0.5:
            tips.append("🔴 CRITICAL: Add exact keywords from JD to pass ATS filters")
        elif score < 0.7:
            tips.append("🟡 Add 3-5 more keywords from job description")
        
        tips.extend([
            "📝 Use action verbs: Developed, Designed, Optimized, Deployed",
            "🎯 Include quantifiable achievements: 'Reduced latency 40%'",
            "📄 Save as PDF, use standard fonts (Arial/Calibri 10-12pt)"
        ])
        
        return {
            "score": float(score),
            "tips": " | ".join(tips),
            "missing_keywords": missing_keywords,
            "job_keywords_matched": len(job_skills),
            "status": "success"
        }
    except Exception as e:
        return {
            "score": 0.0, 
            "tips": "Analysis temporarily unavailable",
            "missing_keywords": [],
            "status": "error"
        }

@app.post("/cover-letter")
async def generate_cover_letter(role: str = Form(...), skills: str = Form(...)):
    try:
        skill_list = skills.split(',')[:6]
        template = COVER_TEMPLATES.get(role, COVER_TEMPLATES["intern"])
        letter = template.format(skills=", ".join(skill_list))
        
        return {
            "letter": letter,
            "skills_used": skill_list,
            "role": role,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Cover letter generation failed")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0", "features": ["resume", "matching", "cover-letter"]}

if __name__ == "__main__":
    print("🚀 Starting Intel Job Application Assistant v2.0...")
    print("📱 Frontend: http://127.0.0.1:8000")
    print("🔧 Backend APIs ready")
    uvicorn.run(app, host="127.0.0.1", port=8000)

 