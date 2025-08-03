// Simplified API client for VibeNews backend
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class APIClient {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      console.log(`Making request to: ${url}`);
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      console.log(`Response from ${endpoint}:`, data);
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Simplified article endpoints that match our new API
  async getArticles(params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const endpoint = `/articles${queryString ? `?${queryString}` : ''}`;
    return this.request(endpoint);
  }

  async searchArticles(query, params = {}) {
    const allParams = { q: query, ...params };
    const queryString = new URLSearchParams(allParams).toString();
    return this.request(`/search?${queryString}`);
  }

  async getStats() {
    return this.request('/stats');
  }

  async getHealth() {
    return this.request('/health');
  }

  // Legacy methods for backward compatibility - redirect to simplified endpoints
  async getRecentArticles(params = {}) {
    return this.getArticles(params);
  }

  async getAllArticles(params = {}) {
    return this.getArticles(params);
  }

  async getTrendingArticles(params = {}) {
    return this.getArticles(params);
  }

  async getArticleStats() {
    return this.getStats();
  }

  // New methods for article details and similar articles
  async getArticle(articleId) {
    return this.request(`/articles/${articleId}`);
  }

  async getSimilarArticles(articleId, limit = 5) {
    const params = new URLSearchParams({ limit: limit.toString() });
    return this.request(`/articles/${articleId}/similar?${params.toString()}`);
  }
}

// Create singleton instance
const apiClient = new APIClient();

export default apiClient;