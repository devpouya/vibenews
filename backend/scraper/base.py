from abc import ABC, abstractmethod
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import hashlib
import logging

from backend.models.article import Article


logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """Base class for news scrapers"""
    
    def __init__(self, source_name: str, base_url: str, delay: float = 1.0):
        self.source_name = source_name
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def _get_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a webpage"""
        try:
            time.sleep(self.delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _generate_article_id(self, url: str, title: str) -> str:
        """Generate unique article ID"""
        content = f"{url}:{title}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @abstractmethod
    def get_politics_section_url(self) -> str:
        """Return URL for politics section"""
        pass
    
    @abstractmethod
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract article URLs from section page"""
        pass
    
    @abstractmethod
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Article]:
        """Extract article content from article page"""
        pass
    
    def scrape_politics_articles(self, limit: int = 50) -> List[Article]:
        """Scrape politics articles from the news source"""
        articles = []
        
        # Get politics section page
        politics_url = self.get_politics_section_url()
        soup = self._get_page(politics_url)
        if not soup:
            return articles
        
        # Extract article links
        article_links = self.extract_article_links(soup)
        logger.info(f"Found {len(article_links)} article links from {self.source_name}")
        
        # Scrape individual articles
        for link in article_links[:limit]:
            article_soup = self._get_page(link)
            if article_soup:
                article = self.extract_article_content(article_soup, link)
                if article:
                    articles.append(article)
                    logger.info(f"Scraped: {article.title[:50]}...")
        
        return articles