import React from 'react';
import Chat from './Chat';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>🤖 Intel Job Application Assistant</h1>
        <p>Upload resume & match with job descriptions</p>
      </header>
      <Chat />
    </div>
  );
}

export default App;
