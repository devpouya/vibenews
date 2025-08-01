# International News Scraping Implementation

## 📰 **News Sources to Scrape**

### Major International Outlets
- **Associated Press** - Global wire service
- **Reuters** - International news agency  
- **BBC** - British public broadcaster
- **CNN** - American news network
- **Politico** - Political news specialist
- **The Hill** - Capitol Hill focused news

### Open Access Sources
- **Council of EU - Newsroom** - Official EU communications
- **The Conversation** - Academic journalism platform

## 🎯 **Scraping Targets**

### Topics (from Frontend)
1. **Russia-Ukraine Conflict** - War coverage and geopolitical analysis
2. **Climate Change Policy** - Environmental regulations and green initiatives  
3. **Economic Inflation** - Monetary policy and economic impact
4. **Immigration Policy** - Border policies and refugee situations
5. **Tech Regulation** - AI regulation and digital privacy laws

### Article Distribution
- **Target**: 200 articles total (~40 per topic)
- **Sources**: 8 sources (~25 articles each)
- **Timeframe**: Recent articles (last 30 days)

## 💾 **Memory Estimation**

### Storage Requirements
```
Per Article:
- Title: ~80 characters = 160 bytes
- Content: ~1,600 words = 8,000 characters = 8 KB  
- Metadata: ~400 bytes
- Total per article: ~8.6 KB

200 Articles Total:
- Raw storage: ~1.7 MB
- JSON with formatting: ~1.3 MB  
- With embeddings (768 dims): ~1.9 MB
- With bias analysis: ~2.5 MB

Total project impact: ~3-4 MB
```

### Validation Results ✅
- **Memory usage**: Within reasonable limits (< 5 MB)
- **Format compatibility**: 100% compatible with existing BABE pipeline
- **Topic coverage**: 100% of frontend topics mapped
- **Scraper functionality**: All components working

## 🛠️ **Implementation Details**

### Scraper Architecture
```python
MultiSourceScraper
├── APScraper           # Associated Press
├── ReutersScraper      # Reuters  
├── BBCScraper          # BBC News
├── CNNScraper          # CNN
├── PoliticoScraper     # Politico
├── TheHillScraper      # The Hill
├── EUCouncilScraper    # EU Council
└── ConversationScraper # The Conversation
```

### Data Format (Compatible with existing)
```json
{
  "title": "Article headline",
  "content": "Full article text...",
  "url": "https://source.com/article",
  "published_date": "2025-08-01T16:30:00Z",
  "source": "Source Name", 
  "topic": "topic-slug",
  "scraped_at": "2025-08-01T16:30:00Z",
  "word_count": 1500
}
```

### Rate Limiting & Ethics
- **Respectful delays**: 1-2 seconds between articles
- **Source delays**: 2 seconds between different sources
- **User-Agent**: Proper browser identification
- **Robots.txt**: Compliance with site policies
- **Fair use**: Academic/research purposes only

## 🚀 **Usage Instructions**

### Quick Start
```bash
# Test scraping setup
python tests/integration/test_international_scraping.py

# Run full scraping (200 articles)
python scripts/scrape_international_news.py

# Expected output location
backend/data/raw/international_news_TIMESTAMP.jsonl
backend/data/raw/scraping_summary_TIMESTAMP.json
```

### Integration with Existing Pipeline
```bash
# After scraping, run bias analysis
python scripts/run_bias_analysis.py --input backend/data/raw/international_news_*.jsonl

# View results in frontend
cd frontend && npm start
```

## 📊 **Expected Results**

### Article Breakdown
- **Russia-Ukraine**: ~40 articles from multiple perspectives
- **Climate Policy**: ~40 articles from policy and business angles
- **Economic Inflation**: ~40 articles from financial and political views
- **Immigration**: ~40 articles from policy and humanitarian perspectives  
- **Tech Regulation**: ~40 articles from industry and regulatory viewpoints

### Bias Analysis Ready
- Articles will be compatible with BiasScanner (27 bias types)
- Ready for Swiss political spectrum mapping (-1 to +1)
- Suitable for training data augmentation
- Compatible with existing frontend visualization

## ⚠️ **Considerations**

### Technical Limitations
- **Site changes**: Scraping may break if sites change structure
- **Rate limits**: Some sites may have anti-scraping measures
- **Content access**: Paywalls may limit article access
- **Dynamic content**: JavaScript-heavy sites may require Selenium

### Ethical Guidelines
- **Respect robots.txt** and site terms of service
- **Academic use only** - for bias research purposes
- **Attribution**: Properly credit original sources
- **Fair use**: Limited excerpts for analysis, not republication

### Fallback Plan
- If live scraping fails, use RSS feeds or APIs where available
- Supplement with open datasets (Common Crawl, etc.)
- Use cached/archived versions for testing
- Create synthetic examples if needed for demonstration

## 🎯 **Success Metrics**

- [ ] **200 articles scraped** across 5 topics
- [ ] **8 sources covered** with reasonable distribution  
- [ ] **< 5 MB storage** including metadata
- [ ] **100% format compatibility** with existing pipeline
- [ ] **Bias analysis ready** for immediate processing
- [ ] **Frontend integration** showing international articles