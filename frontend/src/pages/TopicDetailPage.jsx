import React, { useState, useEffect } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { 
  ArrowLeft, 
  Filter, 
  Globe, 
  Clock, 
  ExternalLink, 
  TrendingUp, 
  Users,
  BarChart3,
  Loader2
} from 'lucide-react';
import { Link } from 'react-router-dom';

const TopicDetailPage = () => {
  const { topicKey } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const [topicData, setTopicData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const currentLanguage = searchParams.get('lang') || 'en';
  const currentBiasFilter = searchParams.get('bias') || 'all';

  const languages = [
    { code: 'en', name: 'English', flag: '🇬🇧' },
    { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
    { code: 'fr', name: 'Français', flag: '🇫🇷' },
    { code: 'it', name: 'Italiano', flag: '🇮🇹' }
  ];

  const biasFilters = [
    { key: 'all', label: 'All Articles', color: 'bg-gray-100 text-gray-700' },
    { key: 'left', label: 'Left', color: 'bg-blue-100 text-blue-700' },
    { key: 'center_left', label: 'Center Left', color: 'bg-blue-50 text-blue-600' },
    { key: 'center', label: 'Center', color: 'bg-gray-100 text-gray-600' },
    { key: 'center_right', label: 'Center Right', color: 'bg-red-50 text-red-600' },
    { key: 'right', label: 'Right', color: 'bg-red-100 text-red-700' }
  ];

  useEffect(() => {
    fetchTopicData();
  }, [topicKey, currentLanguage, currentBiasFilter]);

  const fetchTopicData = async () => {
    try {
      setLoading(true);
      const biasParam = currentBiasFilter !== 'all' ? `&bias=${currentBiasFilter}` : '';
      const response = await fetch(`/api/topics/${topicKey}?lang=${currentLanguage}${biasParam}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch topic data');
      }
      
      const data = await response.json();
      setTopicData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLanguageChange = (langCode) => {
    const newSearchParams = new URLSearchParams(searchParams);
    newSearchParams.set('lang', langCode);
    setSearchParams(newSearchParams);
  };

  const handleBiasFilterChange = (biasKey) => {
    const newSearchParams = new URLSearchParams(searchParams);
    if (biasKey === 'all') {
      newSearchParams.delete('bias');
    } else {
      newSearchParams.set('bias', biasKey);
    }
    setSearchParams(newSearchParams);
  };

  const formatTimeAgo = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMinutes = Math.floor((now - date) / (1000 * 60));
    
    if (diffInMinutes < 60) {
      return `${diffInMinutes}m ago`;
    } else if (diffInMinutes < 1440) {
      return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
      return `${Math.floor(diffInMinutes / 1440)}d ago`;
    }
  };

  const getSourceLogo = (source) => {
    const sourceLogos = {
      'srf.ch': '/logos/srf.png',
      'watson.ch': '/logos/watson.png',
      'blick.ch': '/logos/blick.png',
      'rts.ch': '/logos/rts.png',
      'rsi.ch': '/logos/rsi.png',
      '20min.ch': '/logos/20min.png'
    };
    
    return sourceLogos[source] || '/logos/default-news.png';
  };

  const getBiasScoreColor = (score) => {
    if (score < -0.3) return 'text-blue-700 bg-blue-100';
    if (score < -0.1) return 'text-blue-600 bg-blue-50';
    if (score <= 0.1) return 'text-gray-600 bg-gray-100';
    if (score <= 0.3) return 'text-red-600 bg-red-50';
    return 'text-red-700 bg-red-100';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading topic data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">Error: {error}</p>
          <Link to="/" className="text-blue-600 hover:text-blue-700">
            ← Back to Home
          </Link>
        </div>
      </div>
    );
  }

  if (!topicData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Topic not found</p>
          <Link to="/" className="text-blue-600 hover:text-blue-700">
            ← Back to Home
          </Link>
        </div>
      </div>
    );
  }

  const { topic, articles_by_spectrum, spectrum_statistics, total_articles } = topicData;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link 
                to="/" 
                className="flex items-center text-gray-500 hover:text-gray-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 mr-2" />
                Back to Topics
              </Link>
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
          
          {/* Topic Header */}
          <div className="mt-6">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              {topic.name}
            </h1>
            {topic.description && (
              <p className="text-lg text-gray-600 mb-4">
                {topic.description}
              </p>
            )}
            
            {/* Topic Stats */}
            <div className="flex items-center space-x-6 text-sm text-gray-500">
              <div className="flex items-center">
                <Users className="w-4 h-4 mr-1" />
                {total_articles} articles
              </div>
              <div className="flex items-center">
                <TrendingUp className="w-4 h-4 mr-1" />
                Trending Score: {topic.trending_score.toFixed(1)}
              </div>
              <div className="flex items-center">
                <Clock className="w-4 h-4 mr-1" />
                Last updated: {formatTimeAgo(topic.last_article_date)}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* Sidebar - Bias Spectrum Filter */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg border border-gray-200 p-6 sticky top-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Filter className="w-5 h-5 mr-2" />
                Bias Spectrum
              </h3>
              
              <div className="space-y-2">
                {biasFilters.map((filter) => (
                  <button
                    key={filter.key}
                    onClick={() => handleBiasFilterChange(filter.key)}
                    className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      currentBiasFilter === filter.key
                        ? 'bg-blue-100 text-blue-700 border border-blue-200'
                        : 'hover:bg-gray-50 text-gray-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{filter.label}</span>
                      <span className="text-xs text-gray-500">
                        {spectrum_statistics[filter.key] || 0}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
              
              {/* Bias Distribution Chart */}
              <div className="mt-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Distribution</h4>
                <div className="space-y-2">
                  {biasFilters.slice(1).map((filter) => {
                    const count = spectrum_statistics[filter.key] || 0;
                    const percentage = total_articles > 0 ? (count / total_articles) * 100 : 0;
                    
                    return (
                      <div key={filter.key} className="flex items-center">
                        <div className="w-16 text-xs text-gray-500 text-right mr-2">
                          {percentage.toFixed(0)}%
                        </div>
                        <div className="flex-1 bg-gray-200 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${filter.color.replace('text-', 'bg-').replace('100', '200')}`}
                            style={{ width: `${Math.max(percentage, 2)}%` }}
                          />
                        </div>
                        <div className="w-8 text-xs text-gray-500 text-left ml-2">
                          {count}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Main Content - Articles */}
          <div className="lg:col-span-3">
            <div className="space-y-6">
              
              {/* Show articles for current bias filter */}
              {currentBiasFilter === 'all' ? (
                // Show all categories
                Object.entries(articles_by_spectrum)
                  .filter(([category, articles]) => articles.length > 0)
                  .map(([category, articles]) => {
                    const categoryInfo = biasFilters.find(f => f.key === category);
                    if (!categoryInfo) return null;
                    
                    return (
                      <div key={category} className="bg-white rounded-lg border border-gray-200">
                        <div className="px-6 py-4 border-b border-gray-200">
                          <h3 className={`text-lg font-semibold ${categoryInfo.color}`}>
                            {categoryInfo.label} ({articles.length})
                          </h3>
                        </div>
                        
                        <div className="divide-y divide-gray-200">
                          {articles.map((article) => (
                            <ArticleCard 
                              key={article.id} 
                              article={article} 
                              getSourceLogo={getSourceLogo}
                              formatTimeAgo={formatTimeAgo}
                              getBiasScoreColor={getBiasScoreColor}
                            />
                          ))}
                        </div>
                      </div>
                    );
                  })
              ) : (
                // Show single category
                <div className="bg-white rounded-lg border border-gray-200">
                  <div className="px-6 py-4 border-b border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900">
                      {biasFilters.find(f => f.key === currentBiasFilter)?.label} Articles ({articles_by_spectrum[currentBiasFilter]?.length || 0})
                    </h3>
                  </div>
                  
                  <div className="divide-y divide-gray-200">
                    {(articles_by_spectrum[currentBiasFilter] || []).map((article) => (
                      <ArticleCard 
                        key={article.id} 
                        article={article} 
                        getSourceLogo={getSourceLogo}
                        formatTimeAgo={formatTimeAgo}
                        getBiasScoreColor={getBiasScoreColor}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* No articles message */}
              {total_articles === 0 && (
                <div className="text-center py-12">
                  <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No articles found</h3>
                  <p className="text-gray-500">
                    There are no articles available for this topic and filter combination.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Article Card Component
const ArticleCard = ({ article, getSourceLogo, formatTimeAgo, getBiasScoreColor }) => {
  return (
    <div className="p-6 hover:bg-gray-50 transition-colors">
      <div className="flex gap-4">
        {/* Source Logo */}
        <div className="flex-shrink-0">
          <img 
            src={getSourceLogo(article.source)}
            alt={article.source}
            className="w-8 h-8 rounded object-contain"
            onError={(e) => {
              e.target.src = '/logos/default-news.png';
            }}
          />
        </div>
        
        {/* Article Content */}
        <div className="flex-1 min-w-0">
          <h4 className="text-lg font-medium text-gray-900 mb-2 leading-tight">
            {article.title}
          </h4>
          
          {article.summary && (
            <p className="text-gray-600 mb-3 leading-relaxed">
              {article.summary}
            </p>
          )}
          
          {/* Article Meta */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <span className="font-medium">{article.source}</span>
              <span>•</span>
              <div className="flex items-center">
                <Clock className="w-3 h-3 mr-1" />
                {formatTimeAgo(article.published_date)}
              </div>
              <span>•</span>
              <span>{article.word_count} words</span>
            </div>
            
            <div className="flex items-center space-x-2">
              {/* Bias Score */}
              {article.bias_info && (
                <div className={`px-2 py-1 rounded-full text-xs font-medium ${getBiasScoreColor(article.bias_info.overall_score)}`}>
                  {article.bias_info.political_leaning.replace('_', '-')} ({article.bias_info.overall_score.toFixed(2)})
                </div>
              )}
              
              {/* External Link */}
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-blue-600 hover:text-blue-700 text-sm font-medium"
              >
                Read more
                <ExternalLink className="w-3 h-3 ml-1" />
              </a>
            </div>
          </div>
          
          {/* Bias Explanation */}
          {article.bias_info && article.bias_info.explanation && (
            <div className="mt-3 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                <strong>Bias Analysis:</strong> {article.bias_info.explanation}
              </p>
              {article.bias_info.bias_types_detected > 0 && (
                <p className="text-xs text-blue-600 mt-1">
                  {article.bias_info.bias_types_detected} bias types detected
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TopicDetailPage;