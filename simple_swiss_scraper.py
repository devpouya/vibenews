#!/usr/bin/env python3
"""
Simple Swiss News Scraper - Focus on free/accessible content
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import hashlib

from backend.vector_store.article_vector_store import ArticleVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSwissScraper:
    """Simple Swiss news scraper focusing on accessible content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept-Language': 'de-CH,de;q=0.9,fr-CH,fr;q=0.8,en;q=0.6'
        })
        
    def scrape_srf_rss(self, max_articles: int = 50) -> List[Dict]:
        """Scrape SRF using RSS feeds to avoid paywall issues"""
        articles = []
        
        # SRF RSS feeds
        rss_feeds = [
            'https://www.srf.ch/news/bnf/rss/1890',  # Swiss news
            'https://www.srf.ch/news/bnf/rss/1646',  # International
            'https://www.srf.ch/news/bnf/rss/1570',  # Politics
        ]
        
        for feed_url in rss_feeds:
            try:
                logger.info(f"Scraping SRF RSS: {feed_url}")
                response = self.session.get(feed_url, timeout=10)
                response.raise_for_status()
                
                # Parse RSS XML
                from xml.etree import ElementTree as ET
                root = ET.fromstring(response.content)
                
                items = root.findall('.//item')[:max_articles // len(rss_feeds)]
                
                for item in items:
                    try:
                        title_elem = item.find('title')
                        link_elem = item.find('link')
                        pub_date_elem = item.find('pubDate')
                        description_elem = item.find('description')
                        
                        if title_elem is not None and link_elem is not None:
                            title = title_elem.text or "No title"
                            url = link_elem.text or ""
                            pub_date = pub_date_elem.text if pub_date_elem is not None else datetime.now().isoformat()
                            description = description_elem.text if description_elem is not None else ""
                            
                            # Create article with RSS description as content
                            article = {
                                'title': title,
                                'content': description or title,  # Use description or fallback to title
                                'url': url,
                                'source': 'SRF',
                                'language': 'de',
                                'published_date': pub_date,
                                'scraped_at': datetime.now().isoformat(),
                                'word_count': len((description or title).split())
                            }
                            
                            if len(article['content']) >= 50:  # Minimum content length
                                articles.append(article)
                                
                    except Exception as e:
                        logger.error(f"Error processing SRF RSS item: {e}")
                        continue
                        
                time.sleep(1)  # Be respectful
                
            except Exception as e:
                logger.error(f"Error scraping SRF RSS {feed_url}: {e}")
                continue
                
        logger.info(f"Scraped {len(articles)} articles from SRF RSS")
        return articles
    
    def scrape_watson_simple(self, max_articles: int = 30) -> List[Dict]:
        """Simple Watson scraping focusing on accessible content"""
        articles = []
        
        try:
            # Use Watson's simple front page
            response = self.session.get('https://www.watson.ch', timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and 'watson.ch' in href and len(href.split('/')) >= 4:
                    if href not in urls and 'privacy' not in href:
                        urls.append(href)
                        
            # Scrape a few articles with simple content extraction
            for url in urls[:max_articles]:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        
                        # Extract title
                        title_elem = soup.find('h1') or soup.find('title')
                        title = title_elem.get_text().strip() if title_elem else "No title"
                        
                        # Extract simple content (meta description or first paragraph)
                        content = ""
                        meta_desc = soup.find('meta', attrs={'name': 'description'})
                        if meta_desc:
                            content = meta_desc.get('content', '')
                        
                        if not content:
                            # Fallback to first paragraph
                            first_p = soup.find('p')
                            if first_p:
                                content = first_p.get_text().strip()
                        
                        if len(content) >= 50 and len(title) > 5:
                            articles.append({
                                'title': title,
                                'content': content,
                                'url': url,
                                'source': 'Watson',
                                'language': 'de',
                                'published_date': datetime.now().isoformat(),
                                'scraped_at': datetime.now().isoformat(),
                                'word_count': len(content.split())
                            })
                            
                    time.sleep(1)  # Be respectful
                    
                except Exception as e:
                    logger.error(f"Error scraping Watson article {url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Watson: {e}")
            
        logger.info(f"Scraped {len(articles)} articles from Watson")
        return articles
    
    def scrape_twentymin_simple(self, max_articles: int = 30) -> List[Dict]:
        """Simple 20 Minuten scraping"""
        articles = []
        
        try:
            # Use 20min RSS feed
            rss_url = 'https://www.20min.ch/rss/rss.tmpl'
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            
            from xml.etree import ElementTree as ET
            root = ET.fromstring(response.content)
            
            items = root.findall('.//item')[:max_articles]
            
            for item in items:
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    description_elem = item.find('description')
                    
                    if title_elem is not None and link_elem is not None:
                        title = title_elem.text or "No title"
                        url = link_elem.text or ""
                        pub_date = pub_date_elem.text if pub_date_elem is not None else datetime.now().isoformat()
                        description = description_elem.text if description_elem is not None else ""
                        
                        if len(description) >= 50:
                            articles.append({
                                'title': title,
                                'content': description,
                                'url': url,
                                'source': '20 Minuten',
                                'language': 'de',
                                'published_date': pub_date,
                                'scraped_at': datetime.now().isoformat(),
                                'word_count': len(description.split())
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing 20min RSS item: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping 20min RSS: {e}")
            
        logger.info(f"Scraped {len(articles)} articles from 20 Minuten")
        return articles

def main():
    """Main scraping function"""
    logger.info("=== Simple Swiss News Scraping Started ===")
    
    scraper = SimpleSwissScraper()
    all_articles = []
    
    # Scrape from different sources
    sources = [
        ('SRF RSS', scraper.scrape_srf_rss, 80),
        ('Watson', scraper.scrape_watson_simple, 40),
        ('20 Minuten', scraper.scrape_twentymin_simple, 40)
    ]
    
    for source_name, scrape_func, max_articles in sources:
        logger.info(f"Scraping {source_name}...")
        try:
            articles = scrape_func(max_articles)
            all_articles.extend(articles)
            logger.info(f"✅ {source_name}: {len(articles)} articles")
        except Exception as e:
            logger.error(f"❌ {source_name} failed: {e}")
        
        time.sleep(2)  # Delay between sources
    
    logger.info(f"Total articles scraped: {len(all_articles)}")
    
    if not all_articles:
        logger.error("No articles were scraped!")
        return False
    
    # Add to vector database
    logger.info("Adding articles to vector database...")
    try:
        vector_store = ArticleVectorStore()
        successful_adds = 0
        
        for article in all_articles:
            try:
                # Generate unique ID
                article_id = str(abs(hash(article['url'])) % 10**8)
                
                success = vector_store.add_simple_article(
                    article_id=article_id,
                    title=article['title'],
                    content=article['content'],
                    metadata={
                        'url': article['url'],
                        'source': article['source'],
                        'language': article['language'],
                        'published_date': article['published_date'],
                        'bias_score': 0.0,
                        'word_count': article['word_count'],
                        'scraped_at': article['scraped_at'],
                        'topic_tags': json.dumps(['swiss', 'news', 'current'])
                    }
                )
                
                if success:
                    successful_adds += 1
                    
            except Exception as e:
                logger.error(f"Error adding article: {e}")
                continue
        
        logger.info(f"✅ Added {successful_adds}/{len(all_articles)} articles to vector database")
        
        # Get final stats
        stats = vector_store.get_statistics()
        logger.info(f"Vector store total: {stats.get('total_articles', 0)} articles")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'articles_scraped': len(all_articles),
            'articles_added': successful_adds,
            'sources': list(set([a['source'] for a in all_articles])),
            'languages': list(set([a['language'] for a in all_articles]))
        }
        
        with open('simple_swiss_scraping_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("=== Simple Swiss News Scraping Complete ===")
        return successful_adds > 0
        
    except Exception as e:
        logger.error(f"Error adding to vector database: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Swiss news scraping completed successfully!")
    else:
        print("❌ Swiss news scraping failed.")
        sys.exit(1)