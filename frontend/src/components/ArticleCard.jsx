import React from 'react';
import { 
  Clock, 
  ExternalLink, 
  Tag, 
  TrendingUp, 
  Globe,
  BarChart3,
  Eye,
  Share2
} from 'lucide-react';
import { Link } from 'react-router-dom';

const ArticleCard = ({ article, language = 'en', showSimilarity = false }) => {
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
      case 'left': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'center_left': return 'bg-blue-50 text-blue-700 border-blue-100';
      case 'center': return 'bg-gray-100 text-gray-700 border-gray-200';
      case 'center_right': return 'bg-red-50 text-red-700 border-red-100';
      case 'right': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const getBiasScoreColor = (score) => {
    const absScore = Math.abs(score);
    if (absScore > 0.6) return 'text-red-600';
    if (absScore > 0.3) return 'text-orange-600';
    return 'text-green-600';
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

  const truncateText = (text, maxLength = 150) => {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substr(0, maxLength) + '...';
  };

  // Extract data from article (handle both vector store and API formats)
  const articleData = article.metadata || article;
  const articleContent = article.content || article.summary || '';
  
  // Use article.id as the primary ID, fallback to article_id from metadata
  const articleId = article.id || articleData.article_id;
  
  const {
    title = '',
    source = '',
    language: articleLang = 'en',
    published_date = '',
    bias_score = 0,
    political_leaning = 'center',
    bias_analyzed = false,
    topic_tags = '[]',
    url = '',
    word_count = 0,
    canton = '',
    similarity_score,
    relevance_score,
    trending_score
  } = articleData;

  // Parse topic tags
  let topicTags = [];
  try {
    topicTags = typeof topic_tags === 'string' ? JSON.parse(topic_tags) : (topic_tags || []);
  } catch {
    topicTags = [];
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 hover:border-gray-300 hover:shadow-md transition-all duration-200 group">
      <div className="p-6">
        {/* Header with source and metadata */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            {/* Source Logo */}
            <img 
              src={getSourceLogo(source)}
              alt={source}
              className="w-8 h-8 rounded object-contain"
              onError={(e) => {
                e.target.src = '/logos/default-news.png';
              }}
            />
            
            {/* Source and Meta Info */}
            <div className="text-sm text-gray-600">
              <div className="font-medium text-gray-900">{source}</div>
              <div className="flex items-center space-x-2">
                <span>{getLanguageFlag(articleLang)} {articleLang.toUpperCase()}</span>
                <span>•</span>
                <Clock className="w-3 h-3" />
                <span>{formatTimeAgo(published_date)}</span>
                {canton && (
                  <>
                    <span>•</span>
                    <span>{canton}</span>
                  </>
                )}
              </div>
            </div>
          </div>
          
          {/* Similarity/Relevance Score */}
          {(showSimilarity && (similarity_score || relevance_score)) && (
            <div className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {similarity_score ? 
                `${Math.round(similarity_score * 100)}% similar` :
                `${Math.round(relevance_score * 100)}% relevant`
              }
            </div>
          )}
          
          {/* Trending Indicator */}
          {trending_score > 0.7 && (
            <div className="flex items-center text-xs text-orange-600 bg-orange-50 px-2 py-1 rounded-full">
              <TrendingUp className="w-3 h-3 mr-1" />
              Trending
            </div>
          )}
        </div>

        {/* Article Title */}
        <Link 
          to={`/article/${articleId}?lang=${language}`}
          className="block mb-3"
        >
          <h2 className="text-xl font-semibold text-gray-900 group-hover:text-blue-600 transition-colors leading-tight">
            {title}
          </h2>
        </Link>

        {/* Article Summary */}
        {articleContent && (
          <p className="text-gray-600 leading-relaxed mb-4">
            {truncateText(articleContent, 200)}
          </p>
        )}

        {/* Topic Tags */}
        {topicTags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {topicTags.slice(0, 4).map((tag, index) => (
              <span
                key={index}
                className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-purple-100 text-purple-700 border border-purple-200"
              >
                <Tag className="w-3 h-3 mr-1" />
                {tag}
              </span>
            ))}
            {topicTags.length > 4 && (
              <span className="text-xs text-gray-500">
                +{topicTags.length - 4} more
              </span>
            )}
          </div>
        )}

        {/* Bias Analysis Section */}
        {bias_analyzed && (
          <div className="bg-blue-50 border border-blue-100 rounded-lg p-3 mb-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center text-sm font-medium text-blue-900">
                <BarChart3 className="w-4 h-4 mr-1" />
                Bias Analysis
              </div>
              <div className={`text-sm font-mono ${getBiasScoreColor(bias_score)}`}>
                {bias_score > 0 ? '+' : ''}{bias_score.toFixed(2)}
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${getBiasColor(political_leaning)}`}>
                {political_leaning.replace('_', ' ')}
              </span>
              
              {/* Bias visualization bar */}
              <div className="flex-1 mx-4">
                <div className="relative h-2 bg-gray-200 rounded-full">
                  <div 
                    className={`absolute h-2 rounded-full ${
                      bias_score < -0.3 ? 'bg-blue-500' :
                      bias_score < -0.1 ? 'bg-blue-300' :
                      bias_score <= 0.1 ? 'bg-gray-400' :
                      bias_score <= 0.3 ? 'bg-red-300' : 'bg-red-500'
                    }`}
                    style={{
                      left: `${Math.max(0, 50 + (bias_score * 50))}%`,
                      width: '4px'
                    }}
                  />
                  {/* Center line */}
                  <div className="absolute left-1/2 top-0 w-px h-2 bg-gray-400 transform -translate-x-1/2" />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer with actions */}
        <div className="flex items-center justify-between pt-4 border-t border-gray-100">
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <span>{word_count} words</span>
            
            {/* Similar Articles Link */}
            <Link 
              to={`/article/${articleId}?lang=${language}`}
              className="flex items-center text-blue-600 hover:text-blue-700 font-medium"
            >
              <Eye className="w-4 h-4 mr-1" />
              Similar Articles
            </Link>
          </div>
          
          <div className="flex items-center space-x-2">
            {/* External Link */}
            {url && (
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
                title="Read original article"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
            
            {/* Share Button */}
            <button
              onClick={() => {
                if (navigator.share) {
                  navigator.share({
                    title: title,
                    url: url || window.location.href
                  });
                } else {
                  // Fallback: copy to clipboard
                  navigator.clipboard.writeText(url || window.location.href);
                }
              }}
              className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
              title="Share article"
            >
              <Share2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ArticleCard;