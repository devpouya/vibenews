import React from 'react';
import { TrendingUp, Zap, Shield, Brain } from 'lucide-react';
import TopicCard from '../components/TopicCard';
import { topics } from '../data/dummyData';

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-50 via-white to-primary-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-secondary-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between py-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-secondary-900">VibeNews</h1>
                <p className="text-sm text-secondary-600">Swiss News Bias Analyzer</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-secondary-600">
              <Shield className="w-4 h-4" />
              <span>AI-Powered Analysis</span>
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-secondary-900 mb-4">
            Understand Media Bias with{' '}
            <span className="bg-gradient-to-r from-primary-600 to-purple-600 bg-clip-text text-transparent">
              AI Analysis
            </span>
          </h2>
          <p className="text-xl text-secondary-600 max-w-3xl mx-auto leading-relaxed">
            Explore how Swiss news sources cover major topics. Our AI analyzes bias patterns, 
            sentiment, and perspective to help you understand the full story.
          </p>
        </div>

        {/* Features */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          <div className="text-center p-6">
            <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <TrendingUp className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-secondary-900 mb-2">Bias Spectrum Analysis</h3>
            <p className="text-secondary-600 text-sm">
              Articles ranked from -1 to +1 on topic-specific bias scales with detailed reasoning
            </p>
          </div>
          
          <div className="text-center p-6">
            <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Zap className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-secondary-900 mb-2">Real-time Updates</h3>
            <p className="text-secondary-600 text-sm">
              Fresh analysis of the latest articles from major Swiss news sources
            </p>
          </div>
          
          <div className="text-center p-6">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h3 className="text-lg font-semibold text-secondary-900 mb-2">AI Explanations</h3>
            <p className="text-secondary-600 text-sm">
              Understand exactly why each article received its bias score with detailed analysis
            </p>
          </div>
        </div>

        {/* Topics Grid */}
        <div>
          <div className="flex items-center justify-between mb-8">
            <h3 className="text-2xl font-bold text-secondary-900">Current Topics</h3>
            <span className="text-sm text-secondary-600 bg-secondary-100 px-3 py-1 rounded-full">
              {topics.length} active analyses
            </span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {topics.map((topic) => (
              <TopicCard key={topic.id} topic={topic} />
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-secondary-900 text-secondary-300 py-12 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold text-white">VibeNews</span>
              </div>
              <p className="text-sm leading-relaxed">
                Analyzing Swiss news bias with advanced AI to promote media literacy and informed decision-making.
              </p>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-3">About the Analysis</h4>
              <ul className="space-y-2 text-sm">
                <li>• AI-powered bias detection</li>
                <li>• 27 different bias types analyzed</li>
                <li>• Swiss political spectrum mapping</li>
                <li>• Real-time article processing</li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-3">Data Sources</h4>
              <ul className="space-y-2 text-sm">
                <li>• Major Swiss news outlets</li>
                <li>• BABE research dataset</li>
                <li>• BiasScanner methodology</li>
                <li>• Updated continuously</li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-secondary-800 mt-8 pt-8 text-center text-sm text-secondary-400">
            <p>&copy; 2025 VibeNews. Built with advanced NLP and bias detection algorithms.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;