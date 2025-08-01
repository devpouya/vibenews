import React from 'react';
import { getBiasColor } from '../data/dummyData';

const BiasIndicator = ({ score, label, className = '' }) => {
  const colorClass = getBiasColor(score);
  
  return (
    <div className={`bias-indicator ${colorClass} ${className}`}>
      <div className="flex items-center space-x-1">
        <div className="w-2 h-2 rounded-full bg-current opacity-75"></div>
        <span className="font-medium">{label}</span>
        <span className="text-xs opacity-90">({score > 0 ? '+' : ''}{score.toFixed(1)})</span>
      </div>
    </div>
  );
};

export default BiasIndicator;