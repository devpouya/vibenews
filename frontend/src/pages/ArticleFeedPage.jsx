import React, { useState, useEffect } from 'react';
import { 
  Search, 
  Filter, 
  Globe, 
  TrendingUp, 
  Clock, 
  RefreshCw, 
  Loader2,
  Tag,
  ChevronDown
} from 'lucide-react';
import ArticleCard from '../components/ArticleCard';

const ArticleFeedPage = () => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('all');
  const [selectedTopic, setSelectedTopic] = useState('all');
  const [selectedBias, setSelectedBias] = useState('all');
  const [feedType, setFeedType] = useState('recent'); // recent, trending, all
  const [availableTopics, setAvailableTopics] = useState([]);
  const [showFilters, setShowFilters] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  const languages = [
    { code: 'all', name: 'All Languages', flag: '🌐' },
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'it', name: 'Italiano', flag: '🇮🇹' }
  ];

  const biasFilters = [
    { key: 'all', label: 'All Bias', color: 'text-gray-600' },
    { key: 'left', label: 'Left', color: 'text-blue-600' },
    { key: 'center_left', label: 'Center Left', color: 'text-blue-500' },
    { key: 'center', label: 'Center', color: 'text-gray-600' },
    { key: 'center_right', label: 'Center Right', color: 'text-red-500' },
    { key: 'right', label: 'Right', color: 'text-red-600' }
  ];

  const feedTypes = [
    { key: 'recent', label: 'Recent', icon: Clock },
    { key: 'trending', label: 'Trending', icon: TrendingUp },
    { key: 'all', label: 'All Articles', icon: Globe }
  ];

  useEffect(() => {
    fetchArticles();
    fetchTopics();
  }, [feedType, selectedLanguage, selectedTopic, selectedBias]);

  useEffect(() => {
    if (searchQuery) {
      const debounceTimer = setTimeout(() => {
        performSearch();
      }, 500);
      
      return () => clearTimeout(debounceTimer);
    } else {
      fetchArticles();
    }
  }, [searchQuery]);

  const fetchArticles = async () => {
    try {
      setLoading(true);
      
      const params = new URLSearchParams();
      if (selectedLanguage !== 'all') params.append('language', selectedLanguage);
      if (selectedTopic !== 'all') params.append('topic', selectedTopic);
      if (selectedBias !== 'all') params.append('bias', selectedBias);
      
      // Use simplified API endpoint - all feed types use the same /articles endpoint
      let endpoint = `http://localhost:8000/articles`;
      if (params.toString()) {
        endpoint += `?${params.toString()}`;
      }
      
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error('Failed to fetch articles');
      }
      
      const data = await response.json();
      setArticles(data.articles || []);
      setLastUpdated(new Date());
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const performSearch = async () => {
    try {
      setLoading(true);
      
      const params = new URLSearchParams();
      params.append('q', searchQuery);
      if (selectedLanguage !== 'all') params.append('language', selectedLanguage);
      if (selectedTopic !== 'all') params.append('topics', selectedTopic);
      
      const response = await fetch(`http://localhost:8000/search?${params.toString()}`);
      if (!response.ok) {
        throw new Error('Search failed');
      }
      
      const data = await response.json();
      setArticles(data.articles || []);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchTopics = async () => {
    try {
      // For now, disable topics since our simplified API doesn't have topics endpoint
      // const response = await fetch('http://localhost:8000/topics');
      // if (response.ok) {
      //   const data = await response.json();
      //   setAvailableTopics(data.topics || []);
      // }
    } catch (err) {
      console.error('Failed to fetch topics:', err);
    }
  };

  const handleRefresh = () => {
    setSearchQuery('');
    fetchArticles();
  };

  const formatLastUpdated = (date) => {
    if (!date) return '';
    
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Top Navigation */}
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">VN</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">VibeNews</h1>
                <p className="text-sm text-gray-600">Swiss News Analysis</p>
              </div>
            </div>
            
            {/* Feed Type Selector */}
            <div className="flex items-center space-x-1 bg-gray-100 rounded-lg p-1">
              {feedTypes.map((type) => {
                const Icon = type.icon;
                return (
                  <button
                    key={type.key}
                    onClick={() => setFeedType(type.key)}
                    className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                      feedType === type.key
                        ? 'bg-white text-blue-600 shadow-sm'
                        : 'text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    <Icon className="w-4 h-4 mr-1" />
                    {type.label}
                  </button>
                );
              })}
            </div>
          </div>
          
          {/* Search and Filters */}
          <div className="pb-4">
            <div className="flex items-center space-x-4">
              {/* Search */}
              <div className="flex-1 max-w-2xl">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <input
                    type="text"
                    placeholder="Search articles..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
              
              {/* Filter Toggle */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`flex items-center px-4 py-3 border rounded-lg font-medium transition-colors ${
                  showFilters 
                    ? 'border-blue-300 bg-blue-50 text-blue-700' 
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Filter className="w-4 h-4 mr-2" />
                Filters
                <ChevronDown className={`w-4 h-4 ml-1 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
              </button>
              
              {/* Refresh */}
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="flex items-center px-4 py-3 text-gray-600 hover:text-gray-900 transition-colors"
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </button>
            </div>
            
            {/* Filters Panel */}
            {showFilters && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Language Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Language</label>
                    <select
                      value={selectedLanguage}
                      onChange={(e) => setSelectedLanguage(e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {languages.map((lang) => (
                        <option key={lang.code} value={lang.code}>
                          {lang.flag} {lang.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Topic Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Topic</label>
                    <select
                      value={selectedTopic}
                      onChange={(e) => setSelectedTopic(e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="all">All Topics</option>
                      {availableTopics.map((topic) => (
                        <option key={topic} value={topic}>
                          #{topic}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  {/* Bias Filter */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Political Bias</label>
                    <select
                      value={selectedBias}
                      onChange={(e) => setSelectedBias(e.target.value)}
                      className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      {biasFilters.map((bias) => (
                        <option key={bias.key} value={bias.key}>
                          {bias.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            {/* Active Filters Display */}
            {(selectedLanguage !== 'all' || selectedTopic !== 'all' || selectedBias !== 'all' || searchQuery) && (
              <div className="mt-3 flex items-center space-x-2">
                <span className="text-sm text-gray-500">Active filters:</span>
                
                {searchQuery && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    Search: "{searchQuery}"
                  </span>
                )}
                
                {selectedLanguage !== 'all' && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                    {languages.find(l => l.code === selectedLanguage)?.name}
                  </span>
                )}
                
                {selectedTopic !== 'all' && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                    #{selectedTopic}
                  </span>
                )}
                
                {selectedBias !== 'all' && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-orange-100 text-orange-800">
                    {biasFilters.find(b => b.key === selectedBias)?.label}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        
        {/* Status Bar */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <span>{articles.length} articles</span>
            {lastUpdated && (
              <span>Updated {formatLastUpdated(lastUpdated)}</span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {loading && (
              <div className="flex items-center text-sm text-gray-500">
                <Loader2 className="w-4 h-4 animate-spin mr-1" />
                Loading...
              </div>
            )}
          </div>
        </div>

        {/* Loading State */}
        {loading && articles.length === 0 && (
          <div className="text-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
            <p className="text-gray-600">Loading articles...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="text-center py-12">
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md mx-auto">
              <p className="text-red-600 mb-4">Error: {error}</p>
              <button
                onClick={handleRefresh}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {/* Articles Feed */}
        {!loading && !error && (
          <>
            {articles.length > 0 ? (
              <div className="space-y-6">
                {articles.map((article, index) => (
                  <ArticleCard 
                    key={article.id || index} 
                    article={article}
                    language={selectedLanguage !== 'all' ? selectedLanguage : 'en'}
                    showSimilarity={searchQuery ? true : false}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-gray-400 mb-4">
                  <Search className="w-16 h-16 mx-auto" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No articles found</h3>
                <p className="text-gray-500 mb-4">
                  {searchQuery 
                    ? `No articles match your search "${searchQuery}" with the current filters.`
                    : 'No articles available with the current filters.'
                  }
                </p>
                <button
                  onClick={() => {
                    setSearchQuery('');
                    setSelectedLanguage('all');
                    setSelectedTopic('all');
                    setSelectedBias('all');
                    setShowFilters(false);
                  }}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Clear all filters
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ArticleFeedPage;