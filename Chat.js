import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './Chat.css';

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [jobDesc, setJobDesc] = useState('');
  const [resumeText, setResumeText] = useState('');
  const [file, setFile] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const addMessage = (text, isBot = false) => {
    setMessages(prev => [...prev, { text, isBot, time: new Date().toLocaleTimeString() }]);
  };

  const uploadResume = async () => {
    if (!file) return addMessage("Please select a resume file", true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await axios.post('http://localhost:8000/upload-resume', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setResumeText(response.data.text);
      addMessage(`✅ Skills found: ${response.data.skills.join(', ')}`, true);
    } catch (error) {
      addMessage("❌ Error uploading resume", true);
    }
  };

  const analyzeJob = async () => {
    if (!jobDesc || !resumeText) {
      addMessage("Please upload resume and enter job description", true);
      return;
    }
    
    try {
      const response = await axios.post('http://localhost:8000/analyze-job', 
        `job_desc=${encodeURIComponent(jobDesc)}&resume_text=${encodeURIComponent(resumeText)}`,
        { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
      );
      
      const { match_score, tips } = response.data;
      addMessage(`📊 Match Score: ${(match_score * 100).toFixed(1)}%`, true);
      addMessage(`💡 Tips: ${tips}`, true);
    } catch (error) {
      addMessage("❌ Error analyzing job", true);
    }
  };

  const sendMessage = () => {
    if (!input.trim()) return;
    addMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.isBot ? 'bot' : 'user'}`}>
            <span className="time">{msg.time}</span>
            <span>{msg.text}</span>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type your message..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>

      <div className="job-tools">
        <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={uploadResume}>Analyze Resume</button>
        
        <textarea
          value={jobDesc}
          onChange={(e) => setJobDesc(e.target.value)}
          placeholder="Paste Job Description here..."
          rows="4"
        />
        <button onClick={analyzeJob}>Match Job</button>
      </div>
    </div>
  );
}

export default Chat;
