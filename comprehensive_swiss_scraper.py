#!/usr/bin/env python3
"""
Comprehensive Swiss News Website Scraper
Scrapes 200+ articles directly from Swiss news websites
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Set
import logging
import hashlib
import re
from urllib.parse import urljoin, urlparse
import random

from backend.vector_store.article_vector_store import ArticleVectorStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SwissWebsiteScraper:
    """Scrapes articles directly from Swiss news websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'de-CH,de;q=0.9,fr-CH,fr;q=0.8,en;q=0.6',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.scraped_urls: Set[str] = set()
        
    def scrape_srf_website(self, max_articles: int = 60) -> List[Dict]:
        """Scrape SRF news website directly"""
        articles = []
        
        # SRF section URLs
        srf_sections = [
            'https://www.srf.ch/news/schweiz',
            'https://www.srf.ch/news/international', 
            'https://www.srf.ch/news/wirtschaft',
            'https://www.srf.ch/news/politik',
            'https://www.srf.ch/news/gesellschaft'
        ]
        
        for section_url in srf_sections:
            try:
                logger.info(f"Scraping SRF section: {section_url}")
                response = self.session.get(section_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links on SRF
                article_links = []
                
                # Look for various SRF article link patterns
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href:
                        # SRF article URLs typically contain /news/
                        if '/news/' in href and href not in self.scraped_urls:
                            if href.startswith('/'):
                                full_url = 'https://www.srf.ch' + href
                            else:
                                full_url = href
                            
                            if 'srf.ch' in full_url and full_url not in article_links:
                                article_links.append(full_url)
                
                # Scrape individual articles
                section_articles = 0
                for url in article_links[:max_articles // len(srf_sections)]:
                    try:
                        article = self._scrape_srf_article(url)
                        if article:
                            articles.append(article)
                            section_articles += 1
                            self.scraped_urls.add(url)
                            
                        time.sleep(random.uniform(1, 3))  # Random delay
                        
                    except Exception as e:
                        logger.error(f"Error scraping SRF article {url}: {e}")
                        continue
                        
                logger.info(f"SRF {section_url}: {section_articles} articles")
                time.sleep(2)  # Delay between sections
                
            except Exception as e:
                logger.error(f"Error scraping SRF section {section_url}: {e}")
                continue
                
        logger.info(f"Total SRF articles: {len(articles)}")
        return articles
    
    def _scrape_srf_article(self, url: str) -> Dict:
        """Scrape individual SRF article"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_selectors = ['h1', '.article-title', '[data-urn] h1', '.title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            if not title:
                return None
                
            # Extract content
            content = ""
            content_selectors = [
                '.article-content',
                '.text-container', 
                '[data-urn] .text',
                '.article-body',
                '.content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Get all paragraphs within the content
                    paragraphs = content_elem.find_all('p')
                    if paragraphs:
                        content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    else:
                        content = content_elem.get_text().strip()
                    break
            
            # If no structured content, try to get paragraphs directly
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs[:5] if len(p.get_text().strip()) > 50])
            
            # Extract date - try multiple methods
            published_date = datetime.now().isoformat()
            date_selectors = ['time', '[datetime]', '.date', '.publish-date']
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    if date_elem.get('datetime'):
                        published_date = date_elem.get('datetime')
                    else:
                        date_text = date_elem.get_text().strip()
                        if date_text:
                            published_date = date_text
                    break
            
            if len(content) >= 100 and len(title) > 10:
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'source': 'SRF',
                    'language': 'de',
                    'published_date': published_date,
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content.split())
                }
                
        except Exception as e:
            logger.error(f"Error parsing SRF article {url}: {e}")
            
        return None
    
    def scrape_watson_website(self, max_articles: int = 50) -> List[Dict]:
        """Scrape Watson news website directly"""
        articles = []
        
        # Watson section URLs
        watson_sections = [
            'https://www.watson.ch/schweiz',
            'https://www.watson.ch/international', 
            'https://www.watson.ch/wirtschaft',
            'https://www.watson.ch/digital',
            'https://www.watson.ch/leben'
        ]
        
        for section_url in watson_sections:
            try:
                logger.info(f"Scraping Watson section: {section_url}")
                response = self.session.get(section_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and 'watson.ch' in href and len(href.split('/')) >= 4:
                        if href not in self.scraped_urls and 'privacy' not in href:
                            article_links.append(href)
                
                # Scrape individual articles
                section_articles = 0
                for url in article_links[:max_articles // len(watson_sections)]:
                    try:
                        article = self._scrape_watson_article(url)
                        if article:
                            articles.append(article)
                            section_articles += 1
                            self.scraped_urls.add(url)
                            
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        logger.error(f"Error scraping Watson article {url}: {e}")
                        continue
                        
                logger.info(f"Watson {section_url}: {section_articles} articles")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping Watson section {section_url}: {e}")
                continue
                
        logger.info(f"Total Watson articles: {len(articles)}")
        return articles
    
    def _scrape_watson_article(self, url: str) -> Dict:
        """Scrape individual Watson article"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_selectors = ['h1', '.headline', '.article-title', '.title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            # Extract content
            content = ""
            content_selectors = [
                '.article-content',
                '.story-content',
                '.content',
                '.text'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    paragraphs = content_elem.find_all('p')
                    if paragraphs:
                        content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    break
            
            # Fallback content extraction
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs[:5] if len(p.get_text().strip()) > 30])
            
            if len(content) >= 100 and len(title) > 10:
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'source': 'Watson',
                    'language': 'de',
                    'published_date': datetime.now().isoformat(),
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content.split())
                }
                
        except Exception as e:
            logger.error(f"Error parsing Watson article {url}: {e}")
            
        return None
    
    def scrape_blick_website(self, max_articles: int = 40) -> List[Dict]:
        """Scrape Blick news website"""
        articles = []
        
        # Blick URLs
        blick_sections = [
            'https://www.blick.ch/schweiz/',
            'https://www.blick.ch/ausland/',
            'https://www.blick.ch/wirtschaft/',
            'https://www.blick.ch/politik/'
        ]
        
        for section_url in blick_sections:
            try:
                logger.info(f"Scraping Blick section: {section_url}")
                response = self.session.get(section_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and 'blick.ch' in href and len(href.split('/')) >= 4:
                        if href not in self.scraped_urls:
                            article_links.append(href)
                
                # Scrape individual articles  
                section_articles = 0
                for url in article_links[:max_articles // len(blick_sections)]:
                    try:
                        article = self._scrape_blick_article(url)
                        if article:
                            articles.append(article)
                            section_articles += 1
                            self.scraped_urls.add(url)
                            
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        logger.error(f"Error scraping Blick article {url}: {e}")
                        continue
                        
                logger.info(f"Blick {section_url}: {section_articles} articles")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping Blick section {section_url}: {e}")
                continue
                
        logger.info(f"Total Blick articles: {len(articles)}")
        return articles
    
    def _scrape_blick_article(self, url: str) -> Dict:
        """Scrape individual Blick article"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_elem = soup.select_one('h1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract content
            content = ""
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs[:8] if len(p.get_text().strip()) > 20])
            
            if len(content) >= 100 and len(title) > 10:
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'source': 'Blick',
                    'language': 'de',
                    'published_date': datetime.now().isoformat(),
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content.split())
                }
                
        except Exception as e:
            logger.error(f"Error parsing Blick article {url}: {e}")
            
        return None
    
    def scrape_twentymin_website(self, max_articles: int = 50) -> List[Dict]:
        """Scrape 20 Minuten website"""
        articles = []
        
        # 20min sections
        twentymin_sections = [
            'https://www.20min.ch/schweiz',
            'https://www.20min.ch/ausland', 
            'https://www.20min.ch/wirtschaft',
            'https://www.20min.ch/politik'
        ]
        
        for section_url in twentymin_sections:
            try:
                logger.info(f"Scraping 20min section: {section_url}")
                response = self.session.get(section_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links
                article_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and '20min.ch' in href and '/story/' in href:
                        if href not in self.scraped_urls:
                            article_links.append(href)
                
                # Scrape individual articles
                section_articles = 0
                for url in article_links[:max_articles // len(twentymin_sections)]:
                    try:
                        article = self._scrape_twentymin_article(url)
                        if article:
                            articles.append(article)
                            section_articles += 1
                            self.scraped_urls.add(url)
                            
                        time.sleep(random.uniform(1, 3))
                        
                    except Exception as e:
                        logger.error(f"Error scraping 20min article {url}: {e}")
                        continue
                        
                logger.info(f"20min {section_url}: {section_articles} articles")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping 20min section {section_url}: {e}")
                continue
                
        logger.info(f"Total 20min articles: {len(articles)}")
        return articles
    
    def _scrape_twentymin_article(self, url: str) -> Dict:
        """Scrape individual 20min article"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ""
            title_elem = soup.select_one('h1')
            if title_elem:
                title = title_elem.get_text().strip()
            
            # Extract content
            content = ""
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs[:6] if len(p.get_text().strip()) > 25])
            
            if len(content) >= 100 and len(title) > 10:
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'source': '20 Minuten',
                    'language': 'de',
                    'published_date': datetime.now().isoformat(),
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content.split())
                }
                
        except Exception as e:
            logger.error(f"Error parsing 20min article {url}: {e}")
            
        return None

def main():
    """Main scraping function"""
    logger.info("=== Comprehensive Swiss Website Scraping Started ===")
    logger.info("Target: 200+ articles from Swiss news websites")
    
    scraper = SwissWebsiteScraper()
    all_articles = []
    
    # Scrape from different sources
    sources = [
        ('SRF Website', scraper.scrape_srf_website, 80),
        ('Watson Website', scraper.scrape_watson_website, 60),
        ('Blick Website', scraper.scrape_blick_website, 40),
        ('20 Minuten Website', scraper.scrape_twentymin_website, 40)
    ]
    
    for source_name, scrape_func, max_articles in sources:
        logger.info(f"\n🔍 Scraping {source_name} (target: {max_articles} articles)...")
        try:
            articles = scrape_func(max_articles)
            all_articles.extend(articles)
            logger.info(f"✅ {source_name}: {len(articles)} articles scraped")
        except Exception as e:
            logger.error(f"❌ {source_name} failed: {e}")
        
        # Longer delay between sources
        time.sleep(5)
    
    logger.info(f"\n📊 TOTAL ARTICLES SCRAPED: {len(all_articles)}")
    
    if len(all_articles) < 50:
        logger.error("❌ Less than 50 articles scraped - something went wrong!")
        return False
    
    # Add to vector database
    logger.info(f"\n💾 Adding {len(all_articles)} articles to vector database...")
    
    try:
        # Clear existing database first
        vector_store = ArticleVectorStore()
        
        # Clear existing articles
        try:
            all_results = vector_store.articles_collection.get()
            if all_results['ids']:
                vector_store.articles_collection.delete(ids=all_results['ids'])
                logger.info(f"Cleared {len(all_results['ids'])} existing articles")
        except Exception as e:
            logger.warning(f"Error clearing database: {e}")
        
        # Add new articles
        successful_adds = 0
        for i, article in enumerate(all_articles):
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
                        'topic_tags': json.dumps(['schweiz', 'news', 'aktuell'])
                    }
                )
                
                if success:
                    successful_adds += 1
                    if successful_adds % 25 == 0:
                        logger.info(f"   💾 Added {successful_adds} articles...")
                        
            except Exception as e:
                logger.error(f"Error adding article {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully added {successful_adds}/{len(all_articles)} articles to vector database")
        
        # Get final stats
        stats = vector_store.get_statistics()
        logger.info(f"📈 Vector store total: {stats.get('total_articles', 0)} articles")
        
        # Save detailed summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_scraped': len(all_articles),
            'total_added_to_db': successful_adds,
            'sources': {
                source: len([a for a in all_articles if a['source'] == source])
                for source in set([a['source'] for a in all_articles])
            },
            'languages': list(set([a['language'] for a in all_articles])),
            'average_word_count': sum([a['word_count'] for a in all_articles]) // len(all_articles) if all_articles else 0
        }
        
        with open('comprehensive_swiss_scraping_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("📄 Summary saved to comprehensive_swiss_scraping_summary.json")
        logger.info("=== Comprehensive Swiss Website Scraping Complete ===")
        
        return successful_adds >= 50  # Success if we got at least 50 articles
        
    except Exception as e:
        logger.error(f"Error adding to vector database: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Swiss website scraping completed successfully!")
    else:
        print("❌ Swiss website scraping failed.")
        sys.exit(1)