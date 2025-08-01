import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Filter, SortAsc, RefreshCw, Brain } from 'lucide-react';
import ArticleCard from '../components/ArticleCard';
import BiasSpectrum from '../components/BiasSpectrum';
import { topics, articles, sortOptions } from '../data/dummyData';

const TopicPage = () => {
  const { topicId } = useParams();
  const [sortBy, setSortBy] = useState('bias-desc');
  const [isLoading, setIsLoading] = useState(false);
  
  const topic = topics.find(t => t.id === topicId);
  const topicArticles = articles[topicId] || [];
  
  // Sort articles based on selected option
  const sortedArticles = [...topicArticles].sort((a, b) => {
    switch (sortBy) {
      case 'bias-desc':
        return b.biasScore - a.biasScore;
      case 'bias-asc':
        return a.biasScore - b.biasScore;
      case 'date-desc':
        return new Date(b.publishedDate) - new Date(a.publishedDate);
      case 'date-asc':
        return new Date(a.publishedDate) - new Date(b.publishedDate);
      case 'confidence-desc':
        return b.confidence - a.confidence;
      case 'confidence-asc':
        return a.confidence - b.confidence;
      case 'source-asc':
        return a.source.localeCompare(b.source);
      case 'source-desc':
        return b.source.localeCompare(a.source);
      default:
        return 0;
    }
  });

  const handleRefresh = () => {
    setIsLoading(true);
    // Simulate refresh
    setTimeout(() => setIsLoading(false), 1500);
  };

  if (!topic) {
    return (
      <div className="min-h-screen bg-secondary-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-secondary-900 mb-4">Topic Not Found</h1>
          <Link to="/" className="btn-primary">
            Back to Home
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-secondary-50">
      {/* Header */}
      <header className="bg-white border-b border-secondary-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-6">
            <div className="flex items-center space-x-4">
              <Link 
                to="/" 
                className="inline-flex items-center text-secondary-600 hover:text-secondary-900 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 mr-2" />
                Back to Topics
              </Link>
              
              <div className="h-6 w-px bg-secondary-300"></div>
              
              <div className="flex items-center space-x-3">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${topic.gradient} flex items-center justify-center text-lg`}>
                  {topic.icon}
                </div>
                <div>
                  <h1 className="text-xl font-bold text-secondary-900">{topic.title}</h1>
                  <p className="text-sm text-secondary-600">{topicArticles.length} articles analyzed</p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <button
                onClick={handleRefresh}
                disabled={isLoading}
                className="btn-secondary"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                {isLoading ? 'Updating...' : 'Refresh'}
              </button>
              
              <div className="flex items-center space-x-2 text-sm text-secondary-600 bg-secondary-100 px-3 py-2 rounded-lg">
                <Brain className="w-4 h-4" />
                <span>AI Analysis Active</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Bias Spectrum */}
            <BiasSpectrum articles={sortedArticles} currentTopic={topicId} />
            
            {/* Sort Controls */}
            <div className="bg-white rounded-xl p-6 shadow-card border border-secondary-200">
              <div className="flex items-center space-x-2 mb-4">
                <SortAsc className="w-5 h-5 text-secondary-600" />
                <h3 className="text-lg font-semibold text-secondary-900">Sort Articles</h3>
              </div>
              
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="w-full p-3 border border-secondary-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              >
                {sortOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-xs text-blue-800">
                  <strong>Current sort:</strong> {sortOptions.find(opt => opt.value === sortBy)?.label}
                </p>
              </div>
            </div>

            {/* Topic Info */}
            <div className="bg-white rounded-xl p-6 shadow-card border border-secondary-200">
              <h3 className="text-lg font-semibold text-secondary-900 mb-3">About This Analysis</h3>
              <p className="text-sm text-secondary-600 leading-relaxed mb-4">
                {topic.description}
              </p>
              
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-secondary-600">Articles:</span>
                  <span className="font-medium text-secondary-900">{topicArticles.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600">Avg Confidence:</span>
                  <span className="font-medium text-secondary-900">
                    {Math.round(topicArticles.reduce((sum, a) => sum + a.confidence, 0) / topicArticles.length * 100)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-secondary-600">Sources:</span>
                  <span className="font-medium text-secondary-900">
                    {new Set(topicArticles.map(a => a.source)).size}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            {/* Results Header */}
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="text-2xl font-bold text-secondary-900">Article Analysis</h2>
                <p className="text-secondary-600 mt-1">
                  Showing {sortedArticles.length} articles sorted by{' '}
                  <span className="font-medium">
                    {sortOptions.find(opt => opt.value === sortBy)?.label.toLowerCase()}
                  </span>
                </p>
              </div>
              
              <div className="flex items-center space-x-2 text-sm text-secondary-600">
                <Filter className="w-4 h-4" />
                <span>All articles</span>
              </div>
            </div>

            {/* Articles Grid */}
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <RefreshCw className="w-8 h-8 animate-spin text-primary-600 mx-auto mb-4" />
                  <p className="text-secondary-600">Updating analysis...</p>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {sortedArticles.map((article, index) => (
                  <div 
                    key={article.id}
                    className="animate-fade-in"
                    style={{ animationDelay: `${index * 0.1}s` }}
                  >
                    <ArticleCard article={article} />
                  </div>
                ))}
                
                {sortedArticles.length === 0 && (
                  <div className="text-center py-12">
                    <div className="text-secondary-400 mb-4">
                      <Filter className="w-12 h-12 mx-auto" />
                    </div>
                    <h3 className="text-lg font-medium text-secondary-900 mb-2">No articles found</h3>
                    <p className="text-secondary-600">
                      Try adjusting your filters or check back later for new articles.
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TopicPage;