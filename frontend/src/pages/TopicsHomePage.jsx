import React, { useState, useEffect } from 'react';
import { Globe, Search, Loader2, RefreshCw, TrendingUp, Clock } from 'lucide-react';
import TopicCard from '../components/TopicCard';

const TopicsHomePage = () => {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [searchQuery, setSearchQuery] = useState('');
  const [lastUpdated, setLastUpdated] = useState(null);

  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'it', name: 'Italiano', flag: '🇮🇹' }
  ];

  useEffect(() => {
    fetchTopics();
  }, [currentLanguage]);

  const fetchTopics = async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/topics/homepage?lang=${currentLanguage}&limit=12`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch topics');
      }
      
      const data = await response.json();
      setTopics(data.topics || []);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = (langCode) => {
    setCurrentLanguage(langCode);
  };

  const handleRefresh = () => {
    fetchTopics();
  };

  const filteredTopics = topics.filter(topic =>
    topic.topic_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (topic.latest_article?.title || '').toLowerCase().includes(searchQuery.toLowerCase())
  );

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
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Top Navigation */}
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">VN</span>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">VibeNews</h1>
                <p className="text-sm text-gray-600">Swiss News Bias Analysis</p>
              </div>
            </div>
            
            {/* Language Selector */}
            <div className="flex items-center space-x-2">
              <Globe className="w-4 h-4 text-gray-500" />
              <div className="flex space-x-1">
                {languages.map((lang) => (
                  <button
                    key={lang.code}
                    onClick={() => handleLanguageChange(lang.code)}
                    className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                      currentLanguage === lang.code
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                    }`}
                  >
                    {lang.flag} {lang.name}
                  </button>
                ))}
              </div>
            </div>
          </div>
          
          {/* Title and Description */}
          <div className="pb-6">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Swiss News Topics
            </h2>
            <p className="text-lg text-gray-600 mb-6">
              Explore how Swiss media covers major topics with AI-powered bias analysis across German, French, Italian, and English sources.
            </p>
            
            {/* Search and Controls */}
            <div className="flex items-center justify-between">
              <div className="flex-1 max-w-md">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                  <input
                    type="text"
                    placeholder="Search topics..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>
              </div>
              
              <div className="flex items-center space-x-4 ml-4">
                {lastUpdated && (
                  <div className="flex items-center text-sm text-gray-500">
                    <Clock className="w-4 h-4 mr-1" />
                    Updated {formatLastUpdated(lastUpdated)}
                  </div>
                )}
                
                <button
                  onClick={handleRefresh}
                  disabled={loading}
                  className="flex items-center px-3 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
                >
                  <RefreshCw className={`w-4 h-4 mr-1 ${loading ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
            <p className="text-gray-600">Loading topics...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="text-center py-12">
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md mx-auto">
              <p className="text-red-600 mb-4">Error loading topics: {error}</p>
              <button
                onClick={handleRefresh}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Try Again
              </button>
            </div>
          </div>
        )}

        {/* Topics Grid */}
        {!loading && !error && (
          <>
            {/* Stats Header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-6">
                <div className="flex items-center text-sm text-gray-600">
                  <TrendingUp className="w-4 h-4 mr-1" />
                  {filteredTopics.length} active topics
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <Globe className="w-4 h-4 mr-1" />
                  Multilingual analysis
                </div>
              </div>
              
              {searchQuery && (
                <div className="text-sm text-gray-500">
                  Showing {filteredTopics.length} results for "{searchQuery}"
                </div>
              )}
            </div>

            {/* Topics Grid */}
            {filteredTopics.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredTopics.map((topic) => (
                  <TopicCard 
                    key={`${topic.topic_id}_${currentLanguage}`} 
                    topic={topic} 
                    language={currentLanguage}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-gray-400 mb-4">
                  <Search className="w-16 h-16 mx-auto" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">No topics found</h3>
                <p className="text-gray-500">
                  {searchQuery 
                    ? `No topics match "${searchQuery}". Try a different search term.`
                    : 'No topics are currently available. Please check back later.'
                  }
                </p>
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="mt-4 text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Clear search
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3">About VibeNews</h3>
              <p className="text-sm text-gray-600 leading-relaxed">
                Advanced AI-powered bias detection for Swiss news sources. 
                Analyze political bias across German, French, Italian, and English articles.
              </p>
            </div>
            
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Analysis Features</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• 27 bias types detected</li>
                <li>• Real-time topic clustering</li>
                <li>• Multilingual translation</li>
                <li>• Swiss political spectrum mapping</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Data Sources</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>• SRF (German/Romansh)</li>
                <li>• Watson (German)</li>
                <li>• RTS (French)</li>
                <li>• RSI (Italian)</li>
                <li>• Blick (German)</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-gray-200 mt-8 pt-8 text-center text-sm text-gray-500">
            <p>&copy; 2025 VibeNews. Powered by advanced NLP and bias detection algorithms.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TopicsHomePage;