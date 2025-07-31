from typing import List, Optional
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from backend.scraper.base import BaseScraper
from backend.models.article import Article


class TwentyMinScraper(BaseScraper):
    """Scraper for 20 Minuten politics section"""
    
    def __init__(self):
        super().__init__(
            source_name="20min",
            base_url="https://www.20min.ch",
            delay=1.0
        )
    
    def get_politics_section_url(self) -> str:
        """Return URL for 20min politics section"""
        return "https://www.20min.ch/themen/politik"
    
    def extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract article URLs from 20min politics page"""
        links = []
        
        # 20min uses various selectors for article links
        # Look for article links in the politics section
        article_selectors = [
            'a[href*="/story/"]',  # Story URLs
            '.teaser a[href]',      # Teaser links
            '.story-teaser a[href]' # Story teaser links
        ]
        
        for selector in article_selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    # Convert relative URLs to absolute
                    if href.startswith('/'):
                        href = urljoin(self.base_url, href)
                    
                    # Filter for politics stories
                    if '/story/' in href and href not in links:
                        links.append(href)
        
        return links[:50]  # Limit to 50 articles
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Article]:
        """Extract article content from 20min article page"""
        try:
            # Extract title
            title_selectors = ['h1', '.story-title', '.article-title']
            title = None
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            if not title:
                return None
            
            # Extract content - 20min uses direct paragraphs
            content_parts = []
            paragraphs = soup.select('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Filter out very short paragraphs
                    content_parts.append(text)
            
            if not content_parts:
                return None
            
            content = ' '.join(content_parts)
            
            # Extract author
            author = None
            author_selectors = ['.author', '.byline', '[data-testid="author"]']
            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)
                    break
            
            # Extract publication date
            published_at = datetime.now()  # Default to now
            date_selectors = ['time', '.date', '[data-testid="date"]']
            for selector in date_selectors:
                date_elem = soup.select_one(selector)
                if date_elem:
                    date_str = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    if date_str:
                        try:
                            # Try to parse various date formats
                            from dateutil import parser
                            published_at = parser.parse(date_str)
                        except:
                            pass  # Keep default date
                    break
            
            # Create article
            article = Article(
                id=self._generate_article_id(url, title),
                url=url,
                title=title,
                content=content,
                author=author,
                published_at=published_at,
                source=self.source_name,
                language="de"  # 20min is primarily German
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {e}")
            return None