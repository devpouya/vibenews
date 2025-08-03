#!/usr/bin/env python3
"""
Scrape Swiss news articles from the past 2 months and add to vector database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from datetime import datetime, timedelta
from typing import List, Dict
import json

from backend.scraper.swiss_news_scraper import SwissNewsScraper
from backend.vector_store.article_vector_store import ArticleVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def scrape_swiss_articles_2_months():
    """Scrape Swiss articles from the past 2 months"""
    
    # Calculate 2 months back (roughly 60 days)
    hours_back = 60 * 24  # 60 days * 24 hours
    
    logger.info(f"Starting Swiss news scraping for past {hours_back} hours (2 months)")
    
    try:
        # Initialize scraper
        swiss_scraper = SwissNewsScraper()
        
        # Scrape articles
        logger.info("Scraping articles from all Swiss sources...")
        articles = swiss_scraper.scrape_all_swiss_sources(hours_back=hours_back)
        
        logger.info(f"Successfully scraped {len(articles)} articles from Swiss sources")
        
        # Filter and process articles
        processed_articles = []
        for article in articles:
            try:
                # Convert SwissArticle to dict format for vector store
                article_dict = {
                    'id': str(hash(article.url)),  # Generate unique ID from URL
                    'title': article.title,
                    'content': article.content,
                    'url': article.url,
                    'source': article.source,
                    'language': article.language,
                    'published_date': article.published_date,
                    'bias_score': 0.0,  # Default bias score
                    'word_count': article.word_count,
                    'canton': article.canton,
                    'paywall_status': article.paywall_status.value,
                    'scraped_at': article.scraped_at,
                    'content_hash': article.content_hash
                }
                
                # Only include articles with substantial content
                if len(article.content.strip()) >= 100:
                    processed_articles.append(article_dict)
                else:
                    logger.warning(f"Skipping short article: {article.title}")
                    
            except Exception as e:
                logger.error(f"Error processing article {article.title}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_articles)} articles with substantial content")
        return processed_articles
        
    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        return []

def add_articles_to_vector_db(articles: List[Dict]):
    """Add scraped articles to the vector database"""
    
    logger.info(f"Adding {len(articles)} articles to vector database...")
    
    try:
        # Initialize vector store
        vector_store = ArticleVectorStore()
        
        # Add articles to vector store
        successful_adds = 0
        for article in articles:
            try:
                success = vector_store.add_simple_article(
                    article_id=str(abs(hash(article['url'])) % 10**8),  # Ensure positive integer ID
                    title=article['title'],
                    content=article['content'],
                    metadata={
                        'url': article['url'],
                        'source': article['source'],
                        'language': article['language'],
                        'published_date': article['published_date'],
                        'bias_score': article['bias_score'],
                        'word_count': article['word_count'],
                        'canton': article.get('canton', ''),
                        'paywall_status': article.get('paywall_status', 'unknown'),
                        'scraped_at': article['scraped_at'],
                        'content_hash': article['content_hash'],
                        'topic_tags': json.dumps(['swiss', 'news'])  # Basic tags
                    }
                )
                
                if success:
                    successful_adds += 1
                    if successful_adds % 10 == 0:
                        logger.info(f"Added {successful_adds} articles so far...")
                        
            except Exception as e:
                logger.error(f"Error adding article to vector store: {e}")
                continue
        
        logger.info(f"Successfully added {successful_adds}/{len(articles)} articles to vector database")
        
        # Get final statistics
        stats = vector_store.get_statistics()
        logger.info(f"Vector store now contains {stats.get('total_articles', 0)} articles")
        
        return successful_adds
        
    except Exception as e:
        logger.error(f"Error adding articles to vector database: {e}")
        return 0

def main():
    """Main execution function"""
    logger.info("=== Swiss News Scraping Started ===")
    
    # Step 1: Scrape articles
    articles = scrape_swiss_articles_2_months()
    
    if not articles:
        logger.error("No articles were scraped. Exiting.")
        return False
    
    # Step 2: Add to vector database
    successful_adds = add_articles_to_vector_db(articles)
    
    # Step 3: Summary
    logger.info("=== Swiss News Scraping Complete ===")
    logger.info(f"Total articles scraped: {len(articles)}")
    logger.info(f"Articles added to vector DB: {successful_adds}")
    
    # Save summary to file
    summary = {
        'timestamp': datetime.now().isoformat(),
        'articles_scraped': len(articles),
        'articles_added_to_db': successful_adds,
        'sources': list(set([article['source'] for article in articles])),
        'languages': list(set([article['language'] for article in articles])),
        'date_range': '2 months back'
    }
    
    with open('swiss_scraping_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Summary saved to swiss_scraping_summary.json")
    
    return successful_adds > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Swiss news scraping completed successfully!")
    else:
        print("❌ Swiss news scraping failed.")
        sys.exit(1)