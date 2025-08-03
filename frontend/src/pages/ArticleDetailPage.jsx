import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  ExternalLink, 
  Calendar, 
  User, 
  Globe, 
  TrendingUp,
  Loader2,
  AlertCircle,
  BookOpen
} from 'lucide-react';
import ArticleCard from '../components/ArticleCard';
import apiClient from '../api/client';

const ArticleDetailPage = () => {
  const { articleId } = useParams();
  const navigate = useNavigate();
  
  const [article, setArticle] = useState(null);
  const [similarArticles, setSimilarArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [similarLoading, setSimilarLoading] = useState(true);

  useEffect(() => {
    fetchArticleData();
  }, [articleId]);

  const fetchArticleData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Fetch article details and similar articles in parallel
      const [articleResponse, similarResponse] = await Promise.all([
        apiClient.getArticle(articleId),
        apiClient.getSimilarArticles(articleId, 4)
      ]);
      
      setArticle(articleResponse.article);
      setSimilarArticles(similarResponse.similar_articles);
      
    } catch (err) {
      console.error('Error fetching article data:', err);
      setError(err.message);
    } finally {
      setLoading(false);
      setSimilarLoading(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'Unknown date';
    }
  };

  const getBiasColor = (biasScore) => {
    const score = parseFloat(biasScore) || 0;
    if (score < -0.2) return 'text-blue-600 bg-blue-50';
    if (score > 0.2) return 'text-red-600 bg-red-50';
    return 'text-gray-600 bg-gray-50';
  };

  const getBiasLabel = (biasScore) => {
    const score = parseFloat(biasScore) || 0;
    if (score < -0.3) return 'Left';
    if (score < -0.1) return 'Center Left';
    if (score > 0.3) return 'Right';
    if (score > 0.1) return 'Center Right';
    return 'Center';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading article...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Error Loading Article</h2>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Back to Articles
          </button>
        </div>
      </div>
    );
  }

  if (!article) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <BookOpen className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Article Not Found</h2>
          <p className="text-gray-600 mb-4">The article you're looking for doesn't exist.</p>
          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Back to Articles
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <button
            onClick={() => navigate('/')}
            className="flex items-center text-gray-600 hover:text-gray-900 transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Articles
          </button>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">VN</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">VibeNews</h1>
                <p className="text-sm text-gray-600">Article Details</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Article Header */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-gray-900 mb-4 leading-tight">
                {article.title}
              </h1>
              
              <div className="flex flex-wrap items-center gap-4 text-sm text-gray-600 mb-4">
                <div className="flex items-center">
                  <User className="w-4 h-4 mr-1" />
                  {article.source}
                </div>
                
                <div className="flex items-center">
                  <Calendar className="w-4 h-4 mr-1" />
                  {formatDate(article.published_date)}
                </div>
                
                <div className="flex items-center">
                  <Globe className="w-4 h-4 mr-1" />
                  {article.language.toUpperCase()}
                </div>
                
                <div className="flex items-center">
                  <BookOpen className="w-4 h-4 mr-1" />
                  {article.word_count} words
                </div>
              </div>

              {/* Bias Score */}
              <div className="flex items-center gap-2 mb-4">
                <span className="text-sm text-gray-600">Political Leaning:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getBiasColor(article.bias_score)}`}>
                  {getBiasLabel(article.bias_score)} ({article.bias_score})
                </span>
              </div>
            </div>
            
            {article.url && (
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center px-3 py-2 text-sm text-blue-600 hover:text-blue-700 border border-blue-200 hover:border-blue-300 rounded-lg transition-colors"
              >
                <ExternalLink className="w-4 h-4 mr-1" />
                Original
              </a>
            )}
          </div>
        </div>

        {/* Article Content */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <div className="prose prose-lg max-w-none">
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
              {article.content}
            </p>
          </div>
        </div>

        {/* Similar Articles Section */}
        <div className="mb-8">
          <div className="flex items-center mb-6">
            <TrendingUp className="w-5 h-5 text-blue-600 mr-2" />
            <h2 className="text-xl font-semibold text-gray-900">Similar Articles</h2>
          </div>
          
          {similarLoading ? (
            <div className="text-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-blue-600 mx-auto mb-2" />
              <p className="text-gray-600">Loading similar articles...</p>
            </div>
          ) : similarArticles.length > 0 ? (
            <div className="grid gap-6">
              {similarArticles.map((similarArticle) => (
                <div key={similarArticle.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h3 
                        className="text-lg font-semibold text-gray-900 mb-2 cursor-pointer hover:text-blue-600 transition-colors"
                        onClick={() => navigate(`/article/${similarArticle.id}`)}
                      >
                        {similarArticle.title}
                      </h3>
                      <p className="text-gray-600 mb-3 leading-relaxed">
                        {similarArticle.content}
                      </p>
                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span>{similarArticle.source}</span>
                        <span>{formatDate(similarArticle.published_date)}</span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getBiasColor(similarArticle.bias_score)}`}>
                          {getBiasLabel(similarArticle.bias_score)}
                        </span>
                        {similarArticle.similarity_score && (
                          <span className="text-blue-600">
                            {Math.round(similarArticle.similarity_score * 100)}% similar
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <TrendingUp className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No similar articles found</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ArticleDetailPage;