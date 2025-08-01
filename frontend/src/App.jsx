import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import TopicPage from './pages/TopicPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/topic/:topicId" element={<TopicPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;