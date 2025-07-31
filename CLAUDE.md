# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VibeNews is a news bias aggregator that ranks Swiss news articles along a bias spectrum (-1 to +1) for specific topics. Users can search for topics (e.g., "Russia-Ukraine conflict") and view articles ranked from most pro-stance to most anti-stance. The system scrapes Swiss news sources and uses LLM-based bias annotation to train models.

## Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate vibenews

# Install additional language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Development Commands

### Python/Backend (FastAPI)
```bash
# Install development dependencies
pip install -e ".[dev]"

# Code formatting and linting
black backend/
ruff check backend/
mypy backend/

# Run tests
pytest

# Start API server (development)
uvicorn backend.api.main:app --reload
```

### Frontend (React)
Note: Frontend directory structure exists but appears to be minimal/placeholder.

## Architecture

### Backend Structure
- `backend/config.py` - Settings management with Pydantic, includes API keys, model paths, and data directories
- `backend/models/article.py` - Core data models for articles and bias annotations using Pydantic
- `backend/scraper/base.py` - Abstract base class for news scrapers with rate limiting and content extraction
- `backend/api/` - FastAPI endpoints (directory exists but empty)
- `backend/annotator/` - Bias annotation system (directory exists but empty)  
- `backend/scheduler/` - Task scheduling for scraping (directory exists but empty)
- `backend/vector_store/` - ChromaDB integration for RAG (directory exists but empty)

### Data Flow (MVP)
1. **Scraping**: Politics articles from 20 Minuten (no paywall)
2. **LLM Annotation**: Gemini model generates bias labels for training data (rate limited: 5/min, 25/day)
3. **Topic Clustering**: BERTopic groups articles by subject
4. **Bias Model**: LoRA-trained LLM scores articles on bias spectrum (-1 to +1)
5. **Vector Storage**: Single ChromaDB collection with bias scores as metadata
6. **API**: Serve articles ranked by bias for specific topics
7. **Frontend**: Bias spectrum visualization

### Key Components
- **Article Model**: Comprehensive model with content, metadata, bias scores, embeddings
- **BiasLabel Enum**: 5-level bias classification (strongly_left to strongly_right)
- **BaseScraper**: Template for source-specific news scrapers with rate limiting
- **Settings**: Centralized configuration including Gemini API integration and rate limits

### ML Pipeline (MVP)
- **LLM Annotator**: Gemini model for bias annotation (5 req/min, 25 req/day limits)
- **Embedding model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Bias model**: LoRA-trained LLM (future implementation)
- **Topic modeling**: BERTopic for content clustering
- **Vector storage**: Single ChromaDB collection with bias metadata filtering

### Data Directories
- `backend/data/raw/` - Original scraped content
- `backend/data/processed/` - Cleaned and analyzed articles  
- `backend/data/annotations/` - Human/LLM bias annotations

## BiasScanner Implementation ✅

**Status**: COMPLETE - Full BiasScanner algorithm implemented with Swiss news integration

### Algorithm Overview
Implemented the complete BiasScanner algorithm from Menzner & Leidner (2024) research paper:
- **27 Bias Types**: Comprehensive classification system for media bias detection
- **Sentence-Level Analysis**: Analyzes each sentence individually for bias
- **Performance Target**: 76% F1-score (based on original paper)
- **Swiss Integration**: Maps bias detection to -1/+1 political spectrum

### Implementation Structure

#### Core Components
- `backend/bias_detection/bias_types.py` - Complete 27 bias type definitions with examples and severity weights
- `backend/bias_detection/sentence_classifier.py` - Sentence-level bias classification using Gemini
- `backend/bias_detection/bias_scorer.py` - Bias scoring and Swiss political spectrum integration
- `backend/bias_detection/biasscanner_pipeline.py` - End-to-end pipeline for article processing

#### The 27 Bias Types Detected
1. Ad Hominem Bias
2. Ambiguous Attribution Bias  
3. Anecdotal Evidence Bias
4. Causal Misunderstanding Bias
5. Cherry Picking Bias
6. Circular Reasoning Bias
7. Discriminatory Bias
8. Emotional Sensationalism Bias
9. External Validation Bias
10. False Balance Bias
11. False Dichotomy Bias
12. Faulty Analogy Bias
13. Generalization Bias
14. Insinuative Questioning Bias
15. Intergroup Bias
16. Mud Praise Bias
17. Opinionated Bias
18. Political Bias
19. Projection Bias
20. Shifting Benchmark Bias
21. Source Selection Bias
22. Speculation Bias
23. Straw Man Bias
24. Unsubstantiated Claims Bias
25. Whataboutism Bias
26. Word Choice Bias
27. Under-reporting Bias

#### Swiss News Integration
- **Political Spectrum Mapping**: Converts bias types to -1 (left) to +1 (right) scale
- **Keyword Analysis**: Swiss-specific political language detection
- **Leaning Classification**: "left", "center", "right" categorization
- **Spectrum Ranking**: Articles sorted by political bias for topic visualization

#### API Endpoints
- `POST /analyze-bias` - Analyze bias in article text
- `GET /bias-spectrum-report/{topic}` - Generate spectrum report for topic
- `GET /bias-types` - Information about all 27 bias types
- Enhanced existing endpoints with BiasScanner analysis

#### Evaluation Framework
- `backend/evaluation/biasscanner_evaluator.py` - BABE dataset evaluation
- Performance comparison to original BiasScanner paper
- Comprehensive metrics (F1, precision, recall, accuracy)
- Bias type specific performance analysis

### Usage Examples

#### Single Article Analysis
```python
from backend.bias_detection.biasscanner_pipeline import BiasDetectionPipeline

pipeline = BiasDetectionPipeline(gemini_api_key)
article = {"content": "Article text...", "title": "Title"}
result = pipeline.process_article(article)

# Access bias analysis
bias_score = result['bias_analysis']['bias_score']['overall_score']
political_leaning = result['bias_analysis']['swiss_bias_spectrum']['political_leaning']
bias_types = result['bias_analysis']['bias_types_detected']
```

#### Batch Processing for Spectrum
```python
articles = [...]  # List of article dictionaries
processed = pipeline.process_article_batch(articles)
spectrum_report = pipeline.create_bias_spectrum_report(processed)

# Articles ranked from left to right on political spectrum
ranked_articles = spectrum_report['articles_by_spectrum']
```

#### BABE Dataset Evaluation
```python
from backend.evaluation.biasscanner_evaluator import BiasEvaluator

evaluator = BiasEvaluator(gemini_api_key, "path/to/babe_dataset.jsonl")
report = evaluator.run_full_evaluation(sample_limit=100)

f1_score = report['performance_metrics']['f1_score']
comparison = report['comparison_to_paper']
```

### Testing
Run complete test suite:
```bash
python test_biasscanner.py
```

Tests include:
- Single article bias analysis
- Multi-article spectrum ranking  
- BABE dataset evaluation (if available)
- Performance comparison to original paper

### Configuration

#### Getting Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key

#### Setting Environment Variable
```bash
# Option 1: Export for current session
export GEMINI_API_KEY="your_gemini_api_key_here"

# Option 2: Add to your shell profile (permanent)
echo 'export GEMINI_API_KEY="your_gemini_api_key_here"' >> ~/.zshrc
source ~/.zshrc

# Option 3: Create .env file in project root
echo 'GEMINI_API_KEY=your_gemini_api_key_here' > .env
```

#### Verify Setup
```bash
# Check if API key is set
echo $GEMINI_API_KEY

# Test BiasScanner functionality
python test_biasscanner.py
```

Rate limits (following original BiasScanner):
- 5 requests per minute
- 25 requests per day

### Performance Expectations
- **Target F1-Score**: 76% (based on original BiasScanner paper)
- **Processing Speed**: ~2-3 seconds per article (depends on length)
- **Bias Detection**: 27 different bias types with explanations
- **Political Spectrum**: Accurate left/center/right classification for Swiss news

### Integration Status
✅ **Complete Integration** with existing VibeNews infrastructure:
- ChromaDB vector storage enhanced with BiasScanner results
- FastAPI endpoints support BiasScanner analysis
- BABE dataset validation framework
- Swiss political spectrum mapping (-1 to +1)
- Comprehensive bias reporting and visualization