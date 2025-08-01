import React from 'react';
import { ArrowRight, Clock, FileText } from 'lucide-react';
import { Link } from 'react-router-dom';

const TopicCard = ({ topic }) => {
  const formatLastUpdated = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffHours = Math.floor((now - date) / (1000 * 60 * 60));
    
    if (diffHours < 1) return 'Updated just now';
    if (diffHours < 24) return `Updated ${diffHours}h ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <Link 
      to={`/topic/${topic.id}`}
      className="group block"
    >
      <div className="card p-6 h-full flex flex-col transition-all duration-300 group-hover:scale-[1.02] group-hover:-translate-y-1">
        {/* Header with icon and gradient */}
        <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${topic.gradient} flex items-center justify-center text-2xl mb-4 group-hover:scale-110 transition-transform duration-300`}>
          {topic.icon}
        </div>
        
        {/* Content */}
        <div className="flex-1">
          <h3 className="text-xl font-bold text-secondary-900 mb-2 group-hover:text-primary-700 transition-colors">
            {topic.title}
          </h3>
          
          <p className="text-secondary-600 text-sm leading-relaxed mb-4">
            {topic.description}
          </p>
        </div>
        
        {/* Stats */}
        <div className="space-y-3">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2 text-secondary-600">
              <FileText className="w-4 h-4" />
              <span>{topic.articleCount} articles</span>
            </div>
            <div className="flex items-center space-x-2 text-secondary-500">
              <Clock className="w-4 h-4" />
              <span>{formatLastUpdated(topic.lastUpdated)}</span>
            </div>
          </div>
          
          {/* Action */}
          <div className="flex items-center justify-between pt-2 border-t border-secondary-100">
            <span className="text-sm font-medium text-primary-600 group-hover:text-primary-700 transition-colors">
              Analyze Bias
            </span>
            <ArrowRight className="w-4 h-4 text-primary-600 group-hover:text-primary-700 group-hover:translate-x-1 transition-all duration-200" />
          </div>
        </div>
      </div>
    </Link>
  );
};

export default TopicCard;