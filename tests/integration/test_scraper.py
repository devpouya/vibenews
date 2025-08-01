"""
Test script to scrape articles from 20 Minuten without annotation
"""
import asyncio
import logging
from backend.scraper.twentymin_scraper import TwentyMinScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_scraping():
    """Test scraping articles from 20 Minuten"""
    scraper = TwentyMinScraper()
    
    logger.info("Starting to scrape 20 Minuten politics articles...")
    articles = scraper.scrape_politics_articles(limit=20)
    
    logger.info(f"Successfully scraped {len(articles)} articles")
    
    # Display results
    for i, article in enumerate(articles, 1):
        print(f"\n--- Article {i} ---")
        print(f"ID: {article.id}")
        print(f"Title: {article.title}")
        print(f"URL: {article.url}")
        print(f"Author: {article.author or 'Unknown'}")
        print(f"Published: {article.published_at}")
        print(f"Content preview: {article.content[:200]}...")
        print(f"Content length: {len(article.content)} characters")
    
    return articles


if __name__ == "__main__":
    articles = asyncio.run(test_scraping())
    print(f"\n🎉 Total articles scraped: {len(articles)}")