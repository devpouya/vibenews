import React, { useState } from 'react';
import { Calendar, ExternalLink, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';
import BiasIndicator from './BiasIndicator';

const ArticleCard = ({ article }) => {
  const [showDetails, setShowDetails] = useState(false);

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-green-600 bg-green-50';
    if (confidence >= 0.8) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="card p-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-secondary-900 mb-2 leading-tight">
            {article.title}
          </h3>
          
          <div className="flex items-center space-x-4 text-sm text-secondary-600 mb-3">
            <span className="font-medium text-secondary-700">{article.source}</span>
            <div className="flex items-center space-x-1">
              <Calendar className="w-4 h-4" />
              <span>{formatDate(article.publishedDate)}</span>
            </div>
            <a 
              href={article.url} 
              className="flex items-center space-x-1 text-primary-600 hover:text-primary-700 transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              <span>Read</span>
            </a>
          </div>
        </div>
      </div>

      {/* Bias and Confidence Indicators */}
      <div className="flex items-center justify-between mb-4">
        <BiasIndicator 
          score={article.biasScore} 
          label={article.biasLabel}
        />
        
        <div className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${getConfidenceColor(article.confidence)}`}>
          <TrendingUp className="w-3 h-3 mr-1" />
          <span>{Math.round(article.confidence * 100)}% confidence</span>
        </div>
      </div>

      {/* AI Summary */}
      <div className="bg-secondary-50 rounded-lg p-4 mb-4">
        <h4 className="text-sm font-semibold text-secondary-800 mb-2">AI Analysis Summary</h4>
        <p className="text-sm text-secondary-700 leading-relaxed">
          {article.aiSummary}
        </p>
      </div>

      {/* Expandable Details */}
      <div className="border-t border-secondary-200 pt-4">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center justify-between w-full text-left text-sm font-medium text-secondary-700 hover:text-secondary-900 transition-colors"
        >
          <span>Detailed Bias Analysis</span>
          {showDetails ? (
            <ChevronUp className="w-4 h-4" />
          ) : (
            <ChevronDown className="w-4 h-4" />
          )}
        </button>
        
        {showDetails && (
          <div className="mt-3 p-4 bg-blue-50 rounded-lg animate-slide-up">
            <h5 className="text-sm font-semibold text-blue-900 mb-2">AI Reasoning</h5>
            <p className="text-sm text-blue-800 leading-relaxed">
              {article.aiReasoning}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ArticleCard;