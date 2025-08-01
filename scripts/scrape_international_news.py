#!/usr/bin/env python3
"""
Scraper for international news sources
Collects articles from major outlets for bias analysis
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.scraper.news_sources import MultiSourceScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main scraping workflow"""
    
    # Topics from frontend
    topics = [
        'russia-ukraine-conflict',
        'climate-change-policy', 
        'economic-inflation',
        'immigration-policy',
        'tech-regulation'
    ]
    
    print("🌍 International News Scraper")
    print("=" * 50)
    print(f"Topics: {', '.join(topics)}")
    print(f"Target: 200 articles total (~40 per topic)")
    print(f"Sources: AP, Reuters, BBC, CNN, Politico, The Hill, EU Council, The Conversation")
    
    # Initialize scraper
    scraper = MultiSourceScraper()
    
    # Scrape articles
    print("\n🔍 Starting scraping process...")
    articles = scraper.scrape_all_sources(topics, max_articles_per_source=25)
    
    if not articles:
        print("❌ No articles scraped. Check network connection and site accessibility.")
        return
    
    print(f"\n📊 Scraping Results:")
    print(f"   Total articles: {len(articles)}")
    
    # Analyze by source
    source_counts = {}
    topic_counts = {}
    
    for article in articles:
        source = article.get('source', 'unknown')
        topic = article.get('topic', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    print(f"\n📈 By Source:")
    for source, count in sorted(source_counts.items()):
        print(f"   {source}: {count} articles")
    
    print(f"\n📈 By Topic:")
    for topic, count in sorted(topic_counts.items()):
        print(f"   {topic}: {count} articles")
    
    # Calculate memory usage
    total_content = sum(len(article.get('content', '')) for article in articles)
    estimated_memory = len(json.dumps(articles, indent=2).encode('utf-8'))
    
    print(f"\n💾 Memory Estimation:")
    print(f"   Total content chars: {total_content:,}")
    print(f"   JSON file size: {estimated_memory / 1024 / 1024:.2f} MB")
    print(f"   Average per article: {estimated_memory // len(articles) if articles else 0:,} bytes")
    
    # Save data
    output_dir = project_root / 'backend' / 'data' / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'international_news_{timestamp}.jsonl'
    
    print(f"\n💾 Saving to: {output_file}")
    
    # Save as JSONL (same format as 20 Minuten data)
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(articles)} articles")
    
    # Create summary file
    summary = {
        'scraping_session': {
            'timestamp': datetime.now().isoformat(),
            'total_articles': len(articles),
            'topics_covered': list(topic_counts.keys()),
            'sources_used': list(source_counts.keys()),
            'estimated_memory_mb': estimated_memory / 1024 / 1024
        },
        'source_breakdown': source_counts,
        'topic_breakdown': topic_counts,
        'sample_articles': articles[:3]  # First 3 for inspection
    }
    
    summary_file = output_dir / f'scraping_summary_{timestamp}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Summary saved to: {summary_file}")
    
    # Integration check
    print(f"\n🔗 Integration Notes:")
    print(f"   Data format: Compatible with existing BABE pipeline")
    print(f"   Next steps: Run bias analysis on scraped articles")
    print(f"   BiasScanner: Ready to process {len(articles)} articles")
    
    return articles

def estimate_full_scraping_memory():
    """Estimate memory for full 200-article scraping"""
    
    print("\n💾 Full Scraping Memory Estimation")
    print("=" * 40)
    
    # Estimates based on typical article sizes
    estimates = {
        'avg_title_chars': 80,
        'avg_content_chars': 8000,  # ~1600 words
        'avg_metadata_chars': 400,
        'articles_target': 200
    }
    
    # Calculate storage requirements
    total_chars = estimates['articles_target'] * (
        estimates['avg_title_chars'] + 
        estimates['avg_content_chars'] + 
        estimates['avg_metadata_chars']
    )
    
    # JSON overhead (indentation, keys, etc.)
    json_overhead = 1.3
    raw_size_mb = (total_chars * json_overhead) / 1024 / 1024
    
    # With embeddings (768 dimensions * 4 bytes float32)
    embedding_size_mb = (estimates['articles_target'] * 768 * 4) / 1024 / 1024
    
    print(f"📊 Storage Estimates:")
    print(f"   Raw text (200 articles): {raw_size_mb:.2f} MB")
    print(f"   With embeddings: {raw_size_mb + embedding_size_mb:.2f} MB")
    print(f"   With bias analysis: {raw_size_mb * 1.5:.2f} MB")
    print(f"   Total project impact: ~{raw_size_mb * 2:.1f} MB")
    
    return raw_size_mb

if __name__ == "__main__":
    # Show memory estimates first
    estimate_full_scraping_memory()
    
    print(f"\n" + "=" * 50)
    
    # Run scraping
    try:
        articles = main()
        print(f"\n🎉 Scraping completed successfully!")
        print(f"   Use 'python scripts/run_bias_analysis.py' to analyze articles")
    except KeyboardInterrupt:
        print(f"\n⏹️  Scraping interrupted by user")
    except Exception as e:
        print(f"\n❌ Scraping failed: {e}")
        import traceback
        traceback.print_exc()