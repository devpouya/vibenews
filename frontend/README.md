# VibeNews Frontend

A modern React frontend for the VibeNews Swiss news bias analyzer.

## Features

- **Modern Design**: Built with React 18 and Tailwind CSS
- **Bias Visualization**: Interactive bias spectrum with color-coded articles
- **Smart Sorting**: Multiple sorting options (bias, date, confidence, source)
- **AI Analysis**: Detailed AI-generated summaries and bias reasoning
- **Responsive**: Desktop-first design with mobile support
- **Real-time**: Simulated real-time updates and refresh functionality

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── ArticleCard.jsx  # Individual article display
│   ├── BiasIndicator.jsx # Bias score visualization
│   ├── BiasSpectrum.jsx # Bias distribution chart
│   └── TopicCard.jsx    # Topic overview cards
├── pages/               # Route components
│   ├── HomePage.jsx     # Main landing page
│   └── TopicPage.jsx    # Topic analysis page
├── data/                # Mock data and utilities
│   └── dummyData.js     # Sample articles and topics
└── App.jsx              # Main app component
```

## Key Components

### BiasSpectrum
Interactive visualization showing bias distribution across articles with:
- Color-coded spectrum from green (pro-topic) to red (anti-topic)
- Article count distribution
- Statistics breakdown

### ArticleCard
Comprehensive article display featuring:
- Bias indicator with descriptive labels
- Confidence scores
- AI-generated analysis summaries
- Expandable detailed reasoning
- Source and timestamp information

### Sorting System
Multiple sorting options:
- **Bias**: Most/least biased articles first
- **Date**: Newest/oldest articles first  
- **Confidence**: Highest/lowest confidence first
- **Source**: Alphabetical source ordering

## Styling

- **Tailwind CSS**: Utility-first CSS framework
- **Modern Colors**: Custom bias color palette (red-green spectrum)
- **Animations**: Smooth transitions and micro-interactions
- **Typography**: Inter font for clean, readable text

## Data Structure

Articles include:
- `title`, `source`, `publishedDate`
- `biasScore` (-1 to +1), `biasLabel` (descriptive)
- `confidence` (0-1), `aiSummary`, `aiReasoning`

Topics include:
- `title`, `description`, `articleCount`
- `icon`, `gradient` (for visual theming)
- `lastUpdated`

## Browser Support

- Chrome/Edge 88+
- Firefox 85+
- Safari 14+

## Development

Built with modern React patterns:
- Function components with hooks
- React Router for navigation
- Responsive design principles
- Accessibility considerations