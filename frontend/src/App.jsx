import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ModernHomePage from './pages/ModernHomePage';
import ArticleFeedPage from './pages/ArticleFeedPage';
import ArticleDetailPage from './pages/ArticleDetailPage';
import TopicsHomePage from './pages/TopicsHomePage';
import TopicDetailPage from './pages/TopicDetailPage';
import HomePage from './pages/HomePage';
import TopicPage from './pages/TopicPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<ArticleFeedPage />} />
          <Route path="/articles" element={<ArticleFeedPage />} />
          <Route path="/article/:articleId" element={<ArticleDetailPage />} />
          <Route path="/topics" element={<TopicsHomePage />} />
          <Route path="/topic/:topicKey" element={<TopicDetailPage />} />
          <Route path="/modern" element={<ModernHomePage />} />
          <Route path="/classic" element={<HomePage />} />
          <Route path="/topic-old/:topicId" element={<TopicPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;