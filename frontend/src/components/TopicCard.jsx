import React from 'react';
import { Clock, Users, TrendingUp, ExternalLink, Globe } from 'lucide-react';
import { Link } from 'react-router-dom';

const TopicCard = ({ topic, language = 'en' }) => {
  const { latest_article } = topic;
  
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

  const getBiasColor = (leaning) => {
    switch (leaning) {
      case 'left': return 'text-blue-600 bg-blue-50';
      case 'center_left': return 'text-blue-500 bg-blue-25';
      case 'center': return 'text-gray-600 bg-gray-50';
      case 'center_right': return 'text-red-500 bg-red-25';
      case 'right': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
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

  const getLanguageFlag = (lang) => {
    const flags = {
      'de': '🇩🇪',
      'fr': '🇫🇷', 
      'it': '🇮🇹',
      'en': '🇬🇧'
    };
    return flags[lang] || '🌐';
  };

  return (
    <Link 
      to={`/topic/${topic.topic_name_key}?lang=${language}`}
      className="block bg-white rounded-lg border border-gray-200 hover:border-gray-300 hover:shadow-md transition-all duration-200 group"
    >
      <div className="p-4">
        {/* Topic Header */}
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600 transition-colors line-clamp-2">
              {topic.topic_name}
            </h3>
            <div className="flex items-center gap-2 mt-1">
              <div className="flex items-center text-sm text-gray-500">
                <Users className="w-4 h-4 mr-1" />
                {topic.article_count} articles
              </div>
              {topic.trending_score > 0.5 && (
                <div className="flex items-center text-sm text-orange-600 bg-orange-50 px-2 py-1 rounded-full">
                  <TrendingUp className="w-3 h-3 mr-1" />
                  Trending
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Latest Article */}
        {latest_article && (
          <div className="border-l-4 border-blue-100 pl-4 mb-3">
            <div className="flex items-start gap-3">
              {/* Source Logo */}
              <div className="flex-shrink-0 mt-1">
                <img 
                  src={getSourceLogo(latest_article.source)}
                  alt={latest_article.source}
                  className="w-6 h-6 rounded object-contain"
                  onError={(e) => {
                    e.target.src = '/logos/default-news.png';
                  }}
                />
              </div>
              
              {/* Article Content */}
              <div className="flex-1 min-w-0">
                <h4 className="text-base font-medium text-gray-900 group-hover:text-blue-700 transition-colors line-clamp-2 mb-2">
                  {latest_article.title}
                </h4>
                
                {latest_article.summary && (
                  <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                    {latest_article.summary}
                  </p>
                )}
                
                {/* Article Meta */}
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <div className="flex items-center gap-3">
                    <div className="flex items-center">
                      <Clock className="w-3 h-3 mr-1" />
                      {formatTimeAgo(latest_article.published_date)}
                    </div>
                    
                    <div className="flex items-center">
                      <span className="mr-1">{getLanguageFlag(latest_article.language)}</span>
                      {latest_article.language.toUpperCase()}
                    </div>
                    
                    <span className="text-gray-400">•</span>
                    <span className="font-medium">{latest_article.source}</span>
                  </div>
                  
                  {/* Bias Indicator */}
                  {latest_article.bias_info && (
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getBiasColor(latest_article.bias_info.political_leaning)}`}>
                      {latest_article.bias_info.political_leaning.replace('_', '-')}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Topic Footer */}
        <div className="flex items-center justify-between pt-2 border-t border-gray-100">
          <div className="text-xs text-gray-500">
            Updated {formatTimeAgo(topic.last_updated)}
          </div>
          
          <div className="flex items-center text-xs text-blue-600 group-hover:text-blue-700 font-medium">
            View all articles
            <ExternalLink className="w-3 h-3 ml-1 group-hover:translate-x-0.5 transition-transform" />
          </div>
        </div>
      </div>
    </Link>
  );
};

export default TopicCard;