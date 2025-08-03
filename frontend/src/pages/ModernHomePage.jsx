import React, { useEffect, useState } from 'react';
import { ArrowRight, BarChart3, Globe, Shield, Zap, Brain, TrendingUp } from 'lucide-react';

const ModernHomePage = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [statsCount, setStatsCount] = useState({ types: 0, sources: 0, languages: 0 });

  useEffect(() => {
    setIsVisible(true);
    
    // Animate stats counter
    const animateStats = () => {
      const duration = 2000;
      const steps = 60;
      const stepDuration = duration / steps;
      
      let currentStep = 0;
      const timer = setInterval(() => {
        currentStep++;
        const progress = currentStep / steps;
        const easeOut = 1 - Math.pow(1 - progress, 3);
        
        setStatsCount({
          types: Math.floor(easeOut * 27),
          sources: Math.floor(easeOut * 5),
          languages: Math.floor(easeOut * 3)
        });
        
        if (currentStep >= steps) {
          clearInterval(timer);
          setStatsCount({ types: 27, sources: 5, languages: 3 });
        }
      }, stepDuration);
    };
    
    const statsTimer = setTimeout(animateStats, 1000);
    return () => {
      clearTimeout(statsTimer);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navigation */}
      <nav className={`relative z-50 px-6 py-6 transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'}`}>
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="text-2xl font-bold text-orange-400 hover:text-orange-300 transition-colors duration-300 cursor-pointer">
            VibeNews
          </div>
          
          <div className="hidden md:flex items-center space-x-8 text-gray-300">
            <a href="#about" className="hover:text-white transition-colors duration-200 flex items-center">
              About <span className="ml-1 text-xs">▼</span>
            </a>
            <a href="#analysis" className="hover:text-white transition-colors duration-200 flex items-center">
              Analysis <span className="ml-1 text-xs">▼</span>
            </a>
            <a href="#sources" className="hover:text-white transition-colors duration-200 flex items-center">
              Sources <span className="ml-1 text-xs">▼</span>
            </a>
            <a href="#methodology" className="hover:text-white transition-colors duration-200 flex items-center">
              Methodology <span className="ml-1 text-xs">▼</span>
            </a>
          </div>
          
          <div className="flex items-center space-x-4">
            <button className="bg-orange-500 hover:bg-orange-600 text-white px-6 py-2 rounded-full font-medium transition-all duration-300 transform hover:scale-105 hover:shadow-lg hover:shadow-orange-500/25">
              Analyze Now
            </button>
            <div className="w-8 h-8 bg-gray-700 rounded border border-gray-600 flex items-center justify-center hover:bg-gray-600 transition-colors duration-300 cursor-pointer">
              <div className="w-2 h-2 bg-gray-400 rounded-sm"></div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 pt-20 pb-32">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Left Column - Content */}
          <div className={`space-y-8 transition-all duration-1000 delay-300 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-8'}`}>
            <div className="inline-block">
              <span className="text-purple-400 text-sm font-medium uppercase tracking-wider animate-pulse">
                INTRODUCING VIBENEWS™
              </span>
            </div>
            
            <div className="space-y-6">
              <h1 className="text-6xl lg:text-7xl font-bold leading-tight">
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-orange-400 bg-clip-text text-transparent">
                  The Future of
                </span>
                <br />
                <span className="text-white">
                  Swiss News
                </span>
                <br />
                <span className="text-white">
                  Analysis
                </span>
              </h1>
              
              <p className="text-xl text-gray-300 leading-relaxed max-w-lg">
                Accelerating media literacy through AI-powered bias detection.
                Transform your news consumption with our cutting-edge 
                methodology.
              </p>
            </div>
            
            <div className="pt-4 flex space-x-4">
              <a 
                href="/articles"
                className="group bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-8 py-4 rounded-full font-medium text-lg transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/25 flex items-center space-x-3"
              >
                <span>Read Articles</span>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-2 transition-transform duration-300" />
              </a>
              
              <a 
                href="/topics"
                className="group bg-white border-2 border-purple-200 text-purple-700 hover:bg-purple-50 px-8 py-4 rounded-full font-medium text-lg transition-all duration-300 transform hover:scale-105 flex items-center space-x-3"
              >
                <span>Browse Topics</span>
              </a>
            </div>
          </div>

          {/* Right Column - Gradient Circle */}
          <div className={`relative flex items-center justify-center transition-all duration-1000 delay-500 ${isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8'}`}>
            <div className="relative w-96 h-96 lg:w-[500px] lg:h-[500px]">
              {/* Main gradient circle */}
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500 via-pink-500 to-orange-500 rounded-full opacity-90 blur-sm animate-pulse"></div>
              <div className="absolute inset-2 bg-gradient-to-br from-purple-600 via-pink-600 to-orange-600 rounded-full animate-spin" style={{animationDuration: '20s'}}></div>
              
              {/* Inner content */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-white text-center space-y-4">
                  <div className="text-4xl font-bold">
                    VibeNews™
                  </div>
                  <div className="text-lg opacity-90">
                    Swiss Bias Analysis
                  </div>
                </div>
              </div>
              
              {/* Floating elements around the circle */}
              <div className="absolute -top-8 -right-8 w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl flex items-center justify-center backdrop-blur-sm animate-bounce hover:scale-110 transition-transform duration-300 cursor-pointer" style={{animationDelay: '0s', animationDuration: '3s'}}>
                <BarChart3 className="w-8 h-8 text-white" />
              </div>
              
              <div className="absolute -bottom-8 -left-8 w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl flex items-center justify-center backdrop-blur-sm animate-bounce hover:scale-110 transition-transform duration-300 cursor-pointer" style={{animationDelay: '1s', animationDuration: '3s'}}>
                <Brain className="w-8 h-8 text-white" />
              </div>
              
              <div className="absolute top-20 -left-12 w-12 h-12 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-xl flex items-center justify-center backdrop-blur-sm animate-bounce hover:scale-110 transition-transform duration-300 cursor-pointer" style={{animationDelay: '2s', animationDuration: '3s'}}>
                <Zap className="w-6 h-6 text-white animate-pulse" />
              </div>
              
              <div className="absolute bottom-20 -right-12 w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center backdrop-blur-sm animate-bounce hover:scale-110 transition-transform duration-300 cursor-pointer" style={{animationDelay: '0.5s', animationDuration: '3s'}}>
                <Shield className="w-6 h-6 text-white animate-pulse" />
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Features Section */}
      <section className={`max-w-7xl mx-auto px-6 py-20 transition-all duration-1000 delay-700 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-8 hover:bg-slate-800/70 hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10 hover:-translate-y-2 transition-all duration-500 group">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <TrendingUp className="w-6 h-6 text-white group-hover:rotate-12 transition-transform duration-300" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-4 group-hover:text-purple-300 transition-colors duration-300">Bias Spectrum</h3>
            <p className="text-gray-300 leading-relaxed">
              Advanced AI analysis mapping Swiss news articles across political bias spectrums with unprecedented accuracy.
            </p>
          </div>
          
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-8 hover:bg-slate-800/70 hover:border-blue-500/50 hover:shadow-xl hover:shadow-blue-500/10 hover:-translate-y-2 transition-all duration-500 group" style={{transitionDelay: '100ms'}}>
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Globe className="w-6 h-6 text-white group-hover:rotate-180 transition-transform duration-500" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-4 group-hover:text-blue-300 transition-colors duration-300">Multi-Language</h3>
            <p className="text-gray-300 leading-relaxed">
              Comprehensive analysis across German, French, and Italian Swiss news sources for complete coverage.
            </p>
          </div>
          
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-2xl p-8 hover:bg-slate-800/70 hover:border-green-500/50 hover:shadow-xl hover:shadow-green-500/10 hover:-translate-y-2 transition-all duration-500 group" style={{transitionDelay: '200ms'}}>
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Zap className="w-6 h-6 text-white group-hover:animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-4 group-hover:text-green-300 transition-colors duration-300">Real-Time</h3>
            <p className="text-gray-300 leading-relaxed">
              Live analysis of breaking news with instant bias detection and sentiment scoring across major topics.
            </p>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className={`max-w-7xl mx-auto px-6 py-20 transition-all duration-1000 delay-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>
        <div className="bg-gradient-to-r from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600 rounded-3xl p-12 hover:border-slate-500 hover:shadow-2xl hover:shadow-slate-500/20 transition-all duration-500">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 text-center">
            <div className="group hover:scale-105 transition-transform duration-300">
              <div className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                {statsCount.types}
              </div>
              <div className="text-gray-300 group-hover:text-purple-300 transition-colors duration-300">Bias Types Detected</div>
            </div>
            <div className="group hover:scale-105 transition-transform duration-300">
              <div className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                {statsCount.sources}
              </div>
              <div className="text-gray-300 group-hover:text-blue-300 transition-colors duration-300">Swiss News Sources</div>
            </div>
            <div className="group hover:scale-105 transition-transform duration-300">
              <div className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent">
                {statsCount.languages}
              </div>
              <div className="text-gray-300 group-hover:text-green-300 transition-colors duration-300">Languages Analyzed</div>
            </div>
            <div className="group hover:scale-105 transition-transform duration-300">
              <div className="text-4xl font-bold text-white mb-2 bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent animate-pulse">
                24/7
              </div>
              <div className="text-gray-300 group-hover:text-orange-300 transition-colors duration-300">Real-Time Updates</div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer Wave */}
      <div className="relative">
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-purple-900/20 to-transparent"></div>
      </div>
    </div>
  );
};

export default ModernHomePage;