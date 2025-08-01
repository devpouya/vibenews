import React from 'react';

const BiasSpectrum = ({ articles, currentTopic = 'russia-ukraine-conflict' }) => {
  // Create bias distribution for visualization
  const biasDistribution = articles.reduce((acc, article) => {
    const bucket = Math.floor((article.biasScore + 1) * 10 / 2); // Convert -1 to 1 range to 0-10 buckets
    acc[Math.min(9, Math.max(0, bucket))] = (acc[Math.min(9, Math.max(0, bucket))] || 0) + 1;
    return acc;
  }, {});

  const maxCount = Math.max(...Object.values(biasDistribution));
  
  // Define labels based on topic
  const getSpectrumLabels = (topic) => {
    switch (topic) {
      case 'russia-ukraine-conflict':
        return { left: 'Pro-Ukraine', right: 'Pro-Russia' };
      case 'climate-change-policy':
        return { left: 'Pro-Environment', right: 'Pro-Business' };
      case 'economic-inflation':
        return { left: 'Pro-Regulation', right: 'Anti-Regulation' };
      case 'immigration-policy':
        return { left: 'Pro-Immigration', right: 'Anti-Immigration' };
      case 'tech-regulation':
        return { left: 'Pro-Regulation', right: 'Pro-Tech Industry' };
      default:
        return { left: 'Left Bias', right: 'Right Bias' };
    }
  };

  const labels = getSpectrumLabels(currentTopic);

  return (
    <div className="bg-white rounded-xl p-6 shadow-card border border-secondary-200">
      <h3 className="text-lg font-semibold text-secondary-900 mb-4">Bias Distribution</h3>
      
      {/* Spectrum bar */}
      <div className="relative mb-6">
        <div className="flex h-8 rounded-lg overflow-hidden bg-gradient-to-r from-bias-strong-green via-bias-neutral to-bias-strong-red">
          {/* Distribution markers */}
          {Object.entries(biasDistribution).map(([bucket, count]) => {
            const position = (parseInt(bucket) / 9) * 100;
            const height = Math.max(2, (count / maxCount) * 20);
            return (
              <div
                key={bucket}
                className="absolute bg-secondary-900 rounded-full opacity-70"
                style={{
                  left: `${position}%`,
                  bottom: '100%',
                  width: '4px',
                  height: `${height}px`,
                  transform: 'translateX(-50%)',
                  marginBottom: '4px'
                }}
              />
            );
          })}
        </div>
        
        {/* Scale labels */}
        <div className="flex justify-between mt-2 text-sm text-secondary-600">
          <span className="font-medium text-bias-strong-green">{labels.left}</span>
          <span className="font-medium text-bias-neutral">Neutral</span>
          <span className="font-medium text-bias-strong-red">{labels.right}</span>
        </div>
        
        {/* Scale numbers */}
        <div className="flex justify-between mt-1 text-xs text-secondary-400">
          <span>-1.0</span>
          <span>0</span>
          <span>+1.0</span>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="bg-secondary-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-bias-strong-green">
            {articles.filter(a => a.biasScore < -0.3).length}
          </div>
          <div className="text-xs text-secondary-600">{labels.left}</div>
        </div>
        <div className="bg-secondary-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-bias-neutral">
            {articles.filter(a => Math.abs(a.biasScore) <= 0.3).length}
          </div>
          <div className="text-xs text-secondary-600">Neutral</div>
        </div>
        <div className="bg-secondary-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-bias-strong-red">
            {articles.filter(a => a.biasScore > 0.3).length}
          </div>
          <div className="text-xs text-secondary-600">{labels.right}</div>
        </div>
      </div>
    </div>
  );
};

export default BiasSpectrum;