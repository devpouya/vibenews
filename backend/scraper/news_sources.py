"""
Multi-source news scraper for international outlets
Scrapes articles from major news sources for bias analysis
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin, urlparse
import logging

from .base import BaseScraper

logger = logging.getLogger(__name__)

class MultiSourceScraper:
    """Scraper for multiple international news sources"""
    
    def __init__(self):
        self.sources = {
            'ap': APScraper(),
            'reuters': ReutersScraper(), 
            'bbc': BBCScraper(),
            'cnn': CNNScraper(),
            'politico': PoliticoScraper(),
            'thehill': TheHillScraper(),
            'eu_council': EUCouncilScraper(),
            'conversation': ConversationScraper()
        }
        
    def scrape_all_sources(self, topics: List[str], max_articles_per_source: int = 25) -> List[Dict]:
        """Scrape articles from all sources for given topics"""
        all_articles = []
        
        for source_name, scraper in self.sources.items():
            logger.info(f"Scraping {source_name}...")
            try:
                articles = scraper.scrape_topics(topics, max_articles_per_source)
                for article in articles:
                    article['source'] = source_name
                all_articles.extend(articles)
                time.sleep(2)  # Be respectful between sources
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
                continue
                
        logger.info(f"Scraped {len(all_articles)} total articles")
        return all_articles


class APScraper:
    """Associated Press scraper"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://apnews.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        # AP topic search URLs
        topic_urls = {
            'russia-ukraine-conflict': '/hub/russia-ukraine',
            'climate-change-policy': '/hub/climate-and-environment', 
            'economic-inflation': '/hub/inflation',
            'immigration-policy': '/hub/immigration',
            'tech-regulation': '/hub/technology'
        }
        
        for topic in topics:
            if topic in topic_urls:
                url = self.base_url + topic_urls[topic]
                topic_articles = self._scrape_topic_page(url, topic, max_articles // len(topics))
                articles.extend(topic_articles)
                
        return articles
    
    def _scrape_topic_page(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/article/' in href:
                    full_url = urljoin(self.base_url, href)
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping AP topic {topic}: {e}")
            
        return articles
    
    def _scrape_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Find article content
            content_elem = soup.find('div', class_='RichTextStoryBody') or soup.find('div', class_='ArticleBody')
            content = ""
            if content_elem:
                paragraphs = content_elem.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Extract publication date
            date_elem = soup.find('time') or soup.find('[datetime]')
            pub_date = datetime.now().isoformat()
            if date_elem and date_elem.get('datetime'):
                pub_date = date_elem.get('datetime')
            
            if len(content) < 100:  # Skip very short articles
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'Associated Press',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping AP article {url}: {e}")
            return None


class ReutersScraper:
    """Reuters scraper using RSS feeds to avoid 401 errors"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.reuters.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        # Use Reuters RSS feeds instead of direct pages
        rss_feeds = {
            'russia-ukraine-conflict': 'https://feeds.reuters.com/reuters/worldNews',
            'climate-change-policy': 'https://feeds.reuters.com/reuters/environment',
            'economic-inflation': 'https://feeds.reuters.com/reuters/businessNews',
            'immigration-policy': 'https://feeds.reuters.com/reuters/worldNews',
            'tech-regulation': 'https://feeds.reuters.com/reuters/technologyNews'
        }
        
        for topic in topics:
            if topic in rss_feeds:
                try:
                    topic_articles = self._scrape_reuters_rss(rss_feeds[topic], topic, max_articles // len(topics))
                    articles.extend(topic_articles)
                except Exception as e:
                    logger.error(f"Error scraping Reuters RSS for {topic}: {e}")
                    
        return articles
    
    def _scrape_reuters_rss(self, rss_url: str, topic: str, max_articles: int) -> List[Dict]:
        """Scrape articles from Reuters RSS feed"""
        articles = []
        
        try:
            import xml.etree.ElementTree as ET
            
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            
            for item in items[:max_articles]:
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    
                    if title_elem is not None and link_elem is not None:
                        title = title_elem.text
                        url = link_elem.text
                        pub_date = pub_date_elem.text if pub_date_elem is not None else datetime.now().isoformat()
                        
                        # Get full article content
                        article = self._scrape_reuters_article(url, topic)
                        if article:
                            article['title'] = title
                            article['published_date'] = pub_date
                            articles.append(article)
                            
                    time.sleep(0.5)  # Be respectful
                    
                except Exception as e:
                    logger.error(f"Error processing Reuters RSS item: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Reuters RSS {rss_url}: {e}")
            
        return articles
    
    def _scrape_reuters_section(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and re.match(r'/\w+/\d{4}/\d{2}/\d{2}/', href):
                    full_url = urljoin(self.base_url, href)
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_reuters_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping Reuters section {topic}: {e}")
            
        return articles
    
    def _scrape_reuters_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content
            content_elem = soup.find('div', {'data-testid': 'Body'})
            content = ""
            if content_elem:
                paragraphs = content_elem.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Extract date
            date_elem = soup.find('time')
            pub_date = datetime.now().isoformat()
            if date_elem and date_elem.get('datetime'):
                pub_date = date_elem.get('datetime')
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'Reuters',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping Reuters article {url}: {e}")
            return None


class BBCScraper:
    """BBC News scraper"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://www.bbc.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        # BBC RSS feeds - more reliable than scraping pages
        rss_feeds = {
            'russia-ukraine-conflict': 'http://feeds.bbci.co.uk/news/world/europe/rss.xml',
            'climate-change-policy': 'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml',
            'economic-inflation': 'http://feeds.bbci.co.uk/news/business/rss.xml',
            'immigration-policy': 'http://feeds.bbci.co.uk/news/world/rss.xml',
            'tech-regulation': 'http://feeds.bbci.co.uk/news/technology/rss.xml'
        }
        
        for topic in topics:
            if topic in rss_feeds:
                try:
                    topic_articles = self._scrape_bbc_rss(rss_feeds[topic], topic, max_articles // len(topics))
                    articles.extend(topic_articles)
                except Exception as e:
                    logger.error(f"Error scraping BBC RSS for {topic}: {e}")
                
        return articles
    
    def _scrape_bbc_rss(self, rss_url: str, topic: str, max_articles: int) -> List[Dict]:
        """Scrape articles from BBC RSS feed"""
        articles = []
        
        try:
            import xml.etree.ElementTree as ET
            
            response = self.session.get(rss_url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            
            for item in items[:max_articles]:
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    description_elem = item.find('description')
                    
                    if title_elem is not None and link_elem is not None:
                        title = title_elem.text
                        url = link_elem.text
                        pub_date = pub_date_elem.text if pub_date_elem is not None else datetime.now().isoformat()
                        
                        # Get full article content
                        article = self._scrape_bbc_article(url, topic)
                        if article:
                            article['title'] = title
                            article['published_date'] = pub_date
                            articles.append(article)
                        elif description_elem is not None:
                            # Fallback to description if full article fails
                            articles.append({
                                'title': title,
                                'content': description_elem.text or '',
                                'url': url,
                                'published_date': pub_date,
                                'source': 'BBC',
                                'topic': topic,
                                'scraped_at': datetime.now().isoformat(),
                                'word_count': len((description_elem.text or '').split())
                            })
                            
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error processing BBC RSS item: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping BBC RSS {rss_url}: {e}")
            
        return articles
    
    def _scrape_bbc_section(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/news/' in href and 'live' not in href:
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_bbc_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping BBC section {topic}: {e}")
            
        return articles
    
    def _scrape_bbc_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('[data-testid="headline"]')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content
            content_divs = soup.find_all('div', {'data-component': 'text-block'})
            content = ""
            if content_divs:
                content = ' '.join([div.get_text().strip() for div in content_divs])
            
            # Extract date
            time_elem = soup.find('time')
            pub_date = datetime.now().isoformat()
            if time_elem and time_elem.get('datetime'):
                pub_date = time_elem.get('datetime')
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'BBC',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping BBC article {url}: {e}")
            return None


class CNNScraper:
    """CNN scraper"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://www.cnn.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        topic_sections = {
            'russia-ukraine-conflict': '/world',
            'climate-change-policy': '/climate',
            'economic-inflation': '/economy',
            'immigration-policy': '/politics',
            'tech-regulation': '/business/tech'
        }
        
        for topic in topics:
            if topic in topic_sections:
                url = self.base_url + topic_sections[topic]
                topic_articles = self._scrape_cnn_section(url, topic, max_articles // len(topics))
                articles.extend(topic_articles)
                
        return articles
    
    def _scrape_cnn_section(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links - CNN uses specific selectors
            article_links = soup.find_all(['a'], href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/2025/' in href and 'index.html' in href:
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_cnn_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping CNN section {topic}: {e}")
            
        return articles
    
    def _scrape_cnn_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('[data-component="headline"]')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content - CNN uses specific structure
            content_divs = soup.find_all('div', class_='zn-body__paragraph') or soup.find_all('p')
            content = ""
            if content_divs:
                content = ' '.join([div.get_text().strip() for div in content_divs if div.get_text().strip()])
            
            # Extract date
            date_elem = soup.find('div', class_='timestamp') or soup.find('time')
            pub_date = datetime.now().isoformat()
            if date_elem:
                date_text = date_elem.get_text().strip()
                if date_text:
                    pub_date = date_text
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'CNN',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping CNN article {url}: {e}")
            return None


# Simplified implementations for other sources
class PoliticoScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://www.politico.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        topic_sections = {
            'russia-ukraine-conflict': '/news/national-security',  
            'climate-change-policy': '/news/energy-environment',
            'economic-inflation': '/news/economy',
            'immigration-policy': '/news/congress',
            'tech-regulation': '/news/technology'
        }
        
        for topic in topics:
            if topic in topic_sections:
                url = self.base_url + topic_sections[topic]
                topic_articles = self._scrape_politico_section(url, topic, max_articles // len(topics))
                articles.extend(topic_articles)
                
        return articles
    
    def _scrape_politico_section(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/news/' in href and '/2025/' in href:
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_politico_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping Politico section {topic}: {e}")
            
        return articles
    
    def _scrape_politico_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content
            content_divs = soup.find_all('p')
            content = ""
            if content_divs:
                content = ' '.join([p.get_text().strip() for p in content_divs if p.get_text().strip()])
            
            # Extract date
            date_elem = soup.find('time')
            pub_date = datetime.now().isoformat()
            if date_elem and date_elem.get('datetime'):
                pub_date = date_elem.get('datetime')
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'Politico',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping Politico article {url}: {e}")
            return None

class TheHillScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://thehill.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        topic_sections = {
            'russia-ukraine-conflict': '/policy/international',
            'climate-change-policy': '/policy/energy-environment', 
            'economic-inflation': '/business-a-lobbying',
            'immigration-policy': '/latino',
            'tech-regulation': '/policy/technology'
        }
        
        for topic in topics:
            if topic in topic_sections:
                url = self.base_url + topic_sections[topic]
                topic_articles = self._scrape_thehill_section(url, topic, max_articles // len(topics))
                articles.extend(topic_articles)
                
        return articles
    
    def _scrape_thehill_section(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/homenews/' in href:
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_thehill_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping The Hill section {topic}: {e}")
            
        return articles
    
    def _scrape_thehill_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('h2', class_='title')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content
            content_div = soup.find('div', class_='submitted-by') or soup.find('div', class_='field-item')
            content = ""
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            if not content:  # Fallback
                content_divs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in content_divs if p.get_text().strip()])
            
            # Extract date
            date_elem = soup.find('time') or soup.find('span', class_='submitted-date')
            pub_date = datetime.now().isoformat()
            if date_elem:
                if date_elem.get('datetime'):
                    pub_date = date_elem.get('datetime')
                else:
                    pub_date = date_elem.get_text().strip()
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'The Hill',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping The Hill article {url}: {e}")
            return None

class EUCouncilScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://newsroom.consilium.europa.eu"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        # EU Council uses general news with keyword filtering
        news_url = f"{self.base_url}/en/news"
        
        try:
            all_articles = self._scrape_eu_council_news(news_url, max_articles)
            # Filter articles by topic keywords
            for topic in topics:
                topic_articles = self._filter_articles_by_topic(all_articles, topic)
                for article in topic_articles[:max_articles // len(topics)]:
                    article['topic'] = topic
                    articles.append(article)
        except Exception as e:
            logger.error(f"Error scraping EU Council: {e}")
                
        return articles
    
    def _scrape_eu_council_news(self, url: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news item links
            news_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in news_links:
                href = link.get('href')
                if href and '/en/press/' in href:
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_eu_council_article(url)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping EU Council news: {e}")
            
        return articles
    
    def _scrape_eu_council_article(self, url: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('h2')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content
            content_div = soup.find('div', class_='field-item') or soup.find('div', class_='content')
            content = ""
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            if not content:  # Fallback
                content_divs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in content_divs if p.get_text().strip()])
            
            # Extract date
            date_elem = soup.find('time') or soup.find('span', class_='date')
            pub_date = datetime.now().isoformat()
            if date_elem:
                if date_elem.get('datetime'):
                    pub_date = date_elem.get('datetime')
                else:
                    pub_date = date_elem.get_text().strip()
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'EU Council',
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping EU Council article {url}: {e}")
            return None
    
    def _filter_articles_by_topic(self, articles: List[Dict], topic: str) -> List[Dict]:
        """Filter articles by topic keywords"""
        topic_keywords = {
            'russia-ukraine-conflict': ['ukraine', 'russia', 'war', 'sanctions', 'security'],
            'climate-change-policy': ['climate', 'environment', 'green', 'carbon', 'energy'],
            'economic-inflation': ['economy', 'inflation', 'monetary', 'finance', 'trade'],
            'immigration-policy': ['migration', 'asylum', 'border', 'refugee', 'visa'],
            'tech-regulation': ['digital', 'technology', 'ai', 'data', 'cyber']
        }
        
        keywords = topic_keywords.get(topic, [])
        filtered = []
        
        for article in articles:
            text = (article['title'] + ' ' + article['content']).lower()
            if any(keyword in text for keyword in keywords):
                filtered.append(article)
                
        return filtered

class ConversationScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.base_url = "https://theconversation.com"
        
    def scrape_topics(self, topics: List[str], max_articles: int = 25) -> List[Dict]:
        articles = []
        
        # The Conversation Europe sections
        topic_sections = {
            'russia-ukraine-conflict': '/europe/topics/ukraine-24543',
            'climate-change-policy': '/europe/topics/climate-change-3801',
            'economic-inflation': '/europe/topics/inflation-15804',
            'immigration-policy': '/europe/topics/migration-11704', 
            'tech-regulation': '/europe/topics/artificial-intelligence-ai-13079'
        }
        
        for topic in topics:
            if topic in topic_sections:
                url = self.base_url + topic_sections[topic]
                topic_articles = self._scrape_conversation_section(url, topic, max_articles // len(topics))
                articles.extend(topic_articles)
            else:
                # Fallback: search by keywords
                topic_articles = self._scrape_conversation_search(topic, max_articles // len(topics))
                articles.extend(topic_articles)
                
        return articles
    
    def _scrape_conversation_section(self, url: str, topic: str, max_articles: int) -> List[Dict]:
        articles = []
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/europe/' in href and len(href.split('-')) > 2:  # Typical article URL pattern
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls and 'topics' not in href:
                        article_urls.append(full_url)
                        
            # Scrape individual articles
            for url in article_urls[:max_articles]:
                article = self._scrape_conversation_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error scraping The Conversation section {topic}: {e}")
            
        return articles
    
    def _scrape_conversation_search(self, topic: str, max_articles: int) -> List[Dict]:
        """Fallback search method"""
        search_terms = {
            'russia-ukraine-conflict': 'ukraine',
            'climate-change-policy': 'climate',
            'economic-inflation': 'inflation',
            'immigration-policy': 'migration',
            'tech-regulation': 'artificial intelligence'
        }
        
        search_term = search_terms.get(topic, topic.replace('-', ' '))
        search_url = f"{self.base_url}/europe/search?q={search_term.replace(' ', '+')}"
        
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article URLs from search results
            article_links = soup.find_all('a', href=True)
            article_urls = []
            
            for link in article_links:
                href = link.get('href')
                if href and '/europe/' in href and len(href.split('-')) > 2:
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    if full_url not in article_urls:
                        article_urls.append(full_url)
            
            articles = []
            for url in article_urls[:max_articles]:
                article = self._scrape_conversation_article(url, topic)
                if article:
                    articles.append(article)
                time.sleep(1)
                    
            return articles
            
        except Exception as e:
            logger.error(f"Error in Conversation search for {topic}: {e}")
            return []
    
    def _scrape_conversation_article(self, url: str, topic: str) -> Optional[Dict]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('h1') or soup.find('h2', class_='entry-title')
            title = title_elem.get_text().strip() if title_elem else "No title"
            
            # Extract content - The Conversation has specific structure
            content_div = soup.find('div', class_='entry-content') or soup.find('div', class_='content-body')
            content = ""
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            if not content:  # Fallback
                content_divs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in content_divs if p.get_text().strip()])
            
            # Extract date
            date_elem = soup.find('time') or soup.find('span', class_='date')
            pub_date = datetime.now().isoformat()
            if date_elem:
                if date_elem.get('datetime'):
                    pub_date = date_elem.get('datetime')
                else:
                    pub_date = date_elem.get_text().strip()
            
            if len(content) < 100:
                return None
                
            return {
                'title': title,
                'content': content,
                'url': url,
                'published_date': pub_date,
                'source': 'The Conversation',
                'topic': topic,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split())
            }
            
        except Exception as e:
            logger.error(f"Error scraping The Conversation article {url}: {e}")
            return None