"""
Swiss News Real-Time Scraper
Handles paywall detection and focuses on free Swiss news sources
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import re
from urllib.parse import urljoin, urlparse
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

# from .base import BaseScraper  # Not using base scraper for Swiss system

logger = logging.getLogger(__name__)

class PaywallStatus(Enum):
    FREE = "free"
    BLOCKED = "paywall_blocked"
    SOFT_PAYWALL = "soft_paywall"
    UNKNOWN = "unknown"

@dataclass
class SwissArticle:
    """Swiss article data structure"""
    title: str
    content: str
    url: str
    published_date: str
    source: str
    language: str
    canton: Optional[str]
    content_hash: str
    scraped_at: str
    word_count: int
    paywall_status: PaywallStatus
    
class PaywallDetector:
    """Detects paywalls in Swiss news content"""
    
    # Multi-language paywall indicators
    PAYWALL_INDICATORS = {
        'german': [
            'premium', 'abo', 'abonnement', 'kostenpflichtig', 'bezahlartikel',
            'registrierung erforderlich', 'anmelden', 'weiterlesen',
            'vollständigen artikel', 'premium-inhalt', 'plus-artikel'
        ],
        'french': [
            'premium', 'abonnement', 'payant', 'réservé aux abonnés',
            's\'abonner', 'contenu premium', 'article complet',
            'inscription requise', 'lire la suite'
        ],
        'italian': [
            'premium', 'abbonamento', 'riservato agli abbonati',
            'contenuto premium', 'articolo completo', 'iscriviti',
            'continua a leggere'
        ],
        'common': [
            'paywall', 'subscription', 'subscribe', 'register',
            'login required', 'premium content', 'locked content',
            'blocked content', 'full article'
        ]
    }
    
    # HTML/CSS indicators
    HTML_INDICATORS = [
        'paywall-container', 'subscription-required', 'premium-article',
        'locked-content', 'paywall-overlay', 'subscription-wall',
        'premium-content', 'blocked-article', 'register-wall'
    ]
    
    def detect_paywall(self, content: str, html: str, url: str) -> PaywallStatus:
        """
        Detect paywall status from content and HTML
        
        Args:
            content: Extracted text content
            html: Raw HTML content  
            url: Article URL
            
        Returns:
            PaywallStatus indicating access level
        """
        content_lower = content.lower()
        html_lower = html.lower()
        
        # Check for hard paywall indicators
        if self._check_hard_paywall(content_lower, html_lower):
            logger.info(f"Hard paywall detected: {url}")
            return PaywallStatus.BLOCKED
            
        # Check for soft paywall (partial content)
        if self._check_soft_paywall(content_lower, html_lower):
            logger.info(f"Soft paywall detected: {url}")
            return PaywallStatus.SOFT_PAYWALL
            
        # Check content length (very short might indicate blocking)
        if len(content.strip()) < 50:
            logger.warning(f"Suspiciously short content: {url}")
            return PaywallStatus.UNKNOWN
            
        return PaywallStatus.FREE
    
    def _check_hard_paywall(self, content: str, html: str) -> bool:
        """Check for definitive paywall indicators"""
        
        # Check all language indicators
        all_indicators = []
        for lang_indicators in self.PAYWALL_INDICATORS.values():
            all_indicators.extend(lang_indicators)
            
        # Text-based detection
        for indicator in all_indicators:
            if indicator in content:
                return True
                
        # HTML-based detection
        for indicator in self.HTML_INDICATORS:
            if indicator in html:
                return True
                
        return False
    
    def _check_soft_paywall(self, content: str, html: str) -> bool:
        """Check for soft paywall (partial content) indicators"""
        
        soft_indicators = [
            'weiterlesen', 'lire la suite', 'continua a leggere',
            'read more', 'artikel fortsetzung', 'suite de l\'article'
        ]
        
        for indicator in soft_indicators:
            if indicator in content:
                return True
                
        return False

class SwissNewsScraper:
    """Main Swiss news scraper with paywall handling"""
    
    def __init__(self):
        self.session = requests.Session()
        self.paywall_detector = PaywallDetector()
        self.scraped_urls: Set[str] = set()
        
        # Swiss-specific headers
        self.session.headers.update({
            'Accept-Language': 'de-CH,de;q=0.9,fr-CH,fr;q=0.8,it-CH,it;q=0.7,en;q=0.6',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        })
        
    def scrape_all_swiss_sources(self, hours_back: int = 4) -> List[SwissArticle]:
        """
        Scrape all configured Swiss news sources
        
        Args:
            hours_back: How many hours back to look for new articles
            
        Returns:
            List of SwissArticle objects
        """
        scrapers = {
            'srf': SRFScraper(),
            'watson': WatsonScraper(),
            'blick': BlickScraper(),
            'rts': RTSScraper(),
            'rsi': RSIScraper()
        }
        
        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for source_name, scraper in scrapers.items():
            logger.info(f"Scraping Swiss source: {source_name}")
            try:
                articles = scraper.scrape_recent_articles(cutoff_time)
                
                # Filter out already scraped articles
                new_articles = [
                    article for article in articles 
                    if article.url not in self.scraped_urls
                ]
                
                # Update scraped URLs set
                self.scraped_urls.update(article.url for article in new_articles)
                
                all_articles.extend(new_articles)
                logger.info(f"Retrieved {len(new_articles)} new articles from {source_name}")
                
                # Respectful delay between sources
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
                continue
                
        logger.info(f"Total new Swiss articles scraped: {len(all_articles)}")
        return all_articles
    
    def _create_content_hash(self, title: str, content: str) -> str:
        """Create hash for duplicate detection"""
        combined = f"{title.strip()}{content.strip()}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

class SRFScraper:
    """SRF (Swiss Radio and Television) scraper"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept-Language': 'de-CH,de;q=0.9,fr-CH,fr;q=0.8,it-CH,it;q=0.7,en;q=0.6',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        })
        self.base_url = "https://www.srf.ch"
        self.paywall_detector = PaywallDetector()
        
    def scrape_recent_articles(self, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape recent articles from SRF"""
        articles = []
        
        # SRF news sections (excluding sports)
        sections = [
            '/news',
            '/news/schweiz', 
            '/news/international',
            '/news/wirtschaft'
        ]
        
        for section in sections:
            try:
                section_articles = self._scrape_section(section, cutoff_time)
                articles.extend(section_articles)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error scraping SRF section {section}: {e}")
                
        return articles
    
    def _scrape_section(self, section: str, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape articles from a specific SRF section"""
        articles = []
        url = self.base_url + section
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (SRF uses specific patterns)
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href')
                if href and self._is_article_url(href):
                    full_url = urljoin(self.base_url, href)
                    
                    # Scrape individual article
                    article = self._scrape_article(full_url)
                    if article and self._is_recent(article.published_date, cutoff_time):
                        articles.append(article)
                        
                    time.sleep(0.5)  # Respectful delay
                    
        except Exception as e:
            logger.error(f"Error scraping SRF section {section}: {e}")
            
        return articles
    
    def _is_article_url(self, href: str) -> bool:
        """Check if URL looks like an SRF article"""
        return (
            '/news/' in href and 
            len(href.split('/')) > 4 and
            not href.endswith(('.jpg', '.png', '.pdf', '.mp4'))
        )
    
    def _scrape_article(self, url: str) -> Optional[SwissArticle]:
        """Scrape individual SRF article"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            html_content = response.text
            
            # Extract article data
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            pub_date = self._extract_date(soup)
            
            if not title or not content:
                return None
                
            # Check for paywall
            paywall_status = self.paywall_detector.detect_paywall(
                content, html_content, url
            )
            
            if paywall_status == PaywallStatus.BLOCKED:
                logger.warning(f"SRF article blocked by paywall: {url}")
                return None
                
            # Create content hash
            content_hash = hashlib.md5(f"{title}{content}".encode()).hexdigest()
            
            return SwissArticle(
                title=title,
                content=content,
                url=url,
                published_date=pub_date,
                source='SRF',
                language='de',
                canton='national',
                content_hash=content_hash,
                scraped_at=datetime.now().isoformat(),
                word_count=len(content.split()),
                paywall_status=paywall_status
            )
            
        except Exception as e:
            logger.error(f"Error scraping SRF article {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from SRF article"""
        title_elem = (
            soup.find('h1') or 
            soup.find('title') or
            soup.find('h1', class_='article-title')
        )
        return title_elem.get_text().strip() if title_elem else ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract content from SRF article"""
        # SRF uses specific content containers
        content_selectors = [
            'div.article-content',
            'div.text-container', 
            'div.article-body',
            'article'
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                paragraphs = content_elem.find_all('p')
                if paragraphs:
                    return ' '.join([p.get_text().strip() for p in paragraphs])
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text().strip() for p in paragraphs[:10]])  # Limit fallback
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        """Extract publication date from SRF article"""
        date_elem = (
            soup.find('time') or
            soup.find('span', class_='date') or
            soup.find('div', class_='publish-date')
        )
        
        if date_elem:
            if date_elem.get('datetime'):
                return date_elem.get('datetime')
            else:
                return date_elem.get_text().strip()
                
        return datetime.now().isoformat()
    
    def _is_recent(self, date_str: str, cutoff_time: datetime) -> bool:
        """Check if article is recent enough"""
        try:
            # Try to parse various date formats
            if 'T' in date_str:
                article_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                # Handle Swiss date formats
                article_time = datetime.now()  # Fallback
                
            return article_time > cutoff_time
        except:
            return True  # Include if we can't parse date

class WatsonScraper:
    """Watson.ch scraper with correct URL structure"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept-Language': 'de-CH,de;q=0.9,fr-CH,fr;q=0.8,it-CH,it;q=0.7,en;q=0.6',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        })
        self.base_url = "https://www.watson.ch"
        self.paywall_detector = PaywallDetector()
        
    def scrape_recent_articles(self, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape recent articles from Watson.ch"""
        articles = []
        
        # Watson.ch sections (excluding sports, focusing on news/politics)
        sections = [
            '/schweiz',
            '/international', 
            '/wirtschaft'
        ]
        
        for section in sections:
            try:
                section_articles = self._scrape_section(section, cutoff_time)
                articles.extend(section_articles)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error scraping Watson section {section}: {e}")
                
        return articles
    
    def _scrape_section(self, section: str, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape articles from Watson section"""
        articles = []
        url = self.base_url + section
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article links (generic approach)
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href')
                if href and self._is_article_url(href):
                    full_url = urljoin(self.base_url, href)
                    
                    article = self._scrape_article(full_url)
                    if article and self._is_recent(article.published_date, cutoff_time):
                        articles.append(article)
                        
                    time.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error scraping Watson section {section}: {e}")
            
        return articles[:10]  # Limit per section
    
    def _is_article_url(self, href: str) -> bool:
        """Check if URL looks like a Watson article"""
        return (
            len(href.split('/')) >= 3 and
            not href.endswith(('.jpg', '.png', '.pdf', '.mp4')) and
            not href.startswith('#') and
            not 'javascript:' in href
        )
    
    def _scrape_article(self, url: str) -> Optional[SwissArticle]:
        """Scrape individual Watson article"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            html_content = response.text
            
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            pub_date = self._extract_date(soup)
            
            if not title or not content:
                return None
                
            paywall_status = self.paywall_detector.detect_paywall(content, html_content, url)
            if paywall_status == PaywallStatus.BLOCKED:
                return None
                
            content_hash = hashlib.md5(f"{title}{content}".encode()).hexdigest()
            
            return SwissArticle(
                title=title,
                content=content,
                url=url,
                published_date=pub_date,
                source='Watson',
                language='de',
                canton='national',
                content_hash=content_hash,
                scraped_at=datetime.now().isoformat(),
                word_count=len(content.split()),
                paywall_status=paywall_status
            )
            
        except Exception as e:
            logger.error(f"Error scraping Watson article {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_elem = soup.find('h1') or soup.find('title')
        return title_elem.get_text().strip() if title_elem else ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text().strip() for p in paragraphs[:15]])
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        date_elem = soup.find('time')
        if date_elem and date_elem.get('datetime'):
            return date_elem.get('datetime')
        return datetime.now().isoformat()
    
    def _is_recent(self, date_str: str, cutoff_time: datetime) -> bool:
        try:
            if 'T' in date_str:
                article_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return article_time > cutoff_time
        except:
            pass
        return True

class BlickScraper:
    """Blick.ch scraper with anti-blocking measures"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept-Language': 'de-CH,de;q=0.9,fr-CH,fr;q=0.8,it-CH,it;q=0.7,en;q=0.6',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.base_url = "https://www.blick.ch"
        self.paywall_detector = PaywallDetector()
        
    def scrape_recent_articles(self, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape recent articles from Blick.ch - try RSS first"""
        articles = []
        
        # Try RSS feeds first (less likely to be blocked, no sports)
        rss_feeds = [
            '/rss',
            '/rss/schweiz',
            '/rss/ausland',
            '/rss/wirtschaft'
        ]
        
        for feed_path in rss_feeds:
            try:
                feed_articles = self._scrape_rss_feed(feed_path, cutoff_time)
                articles.extend(feed_articles)
                time.sleep(2)  # Longer delay for Blick
            except Exception as e:
                logger.error(f"Error scraping Blick RSS {feed_path}: {e}")
                
        return articles[:20]  # Limit total
    
    def _scrape_rss_feed(self, feed_path: str, cutoff_time: datetime) -> List[SwissArticle]:
        """Try to scrape Blick RSS feed"""
        import xml.etree.ElementTree as ET
        articles = []
        
        try:
            url = self.base_url + feed_path
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            items = root.findall('.//item')
            
            for item in items[:5]:  # Limit per feed
                try:
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    pub_date_elem = item.find('pubDate')
                    desc_elem = item.find('description')
                    
                    if title_elem is not None and link_elem is not None:
                        # Use RSS content directly to avoid being blocked
                        content = desc_elem.text if desc_elem is not None else ""
                        
                        if len(content) > 50:  # Minimum content check
                            content_hash = hashlib.md5(f"{title_elem.text}{content}".encode()).hexdigest()
                            
                            article = SwissArticle(
                                title=title_elem.text,
                                content=content,
                                url=link_elem.text,
                                published_date=pub_date_elem.text if pub_date_elem is not None else datetime.now().isoformat(),
                                source='Blick',
                                language='de',
                                canton='national',
                                content_hash=content_hash,
                                scraped_at=datetime.now().isoformat(),
                                word_count=len(content.split()),
                                paywall_status=PaywallStatus.FREE
                            )
                            articles.append(article)
                            
                except Exception as e:
                    logger.error(f"Error processing Blick RSS item: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error with Blick RSS feed {feed_path}: {e}")
            
        return articles

class RTSScraper:
    """RTS.ch scraper for French Swiss content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept-Language': 'fr-CH,fr;q=0.9,de-CH,de;q=0.8,en;q=0.7',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        })
        self.base_url = "https://www.rts.ch"
        self.paywall_detector = PaywallDetector()
        
    def scrape_recent_articles(self, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape recent articles from RTS"""
        articles = []
        
        # RTS sections (French structure, no sports)
        sections = [
            '/info',
            '/info/suisse',
            '/info/monde',
            '/info/economie'
        ]
        
        for section in sections:
            try:
                section_articles = self._scrape_section(section, cutoff_time)
                articles.extend(section_articles)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error scraping RTS section {section}: {e}")
                
        return articles[:15]
    
    def _scrape_section(self, section: str, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape RTS section"""
        articles = []
        url = self.base_url + section
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href')
                if href and self._is_article_url(href):
                    full_url = urljoin(self.base_url, href)
                    
                    article = self._scrape_article(full_url)
                    if article and self._is_recent(article.published_date, cutoff_time):
                        articles.append(article)
                        
                    time.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error scraping RTS section {section}: {e}")
            
        return articles[:5]  # Limit per section
    
    def _is_article_url(self, href: str) -> bool:
        return (
            '/info/' in href and
            len(href.split('/')) >= 4 and
            not href.endswith(('.jpg', '.png', '.pdf', '.mp4'))
        )
    
    def _scrape_article(self, url: str) -> Optional[SwissArticle]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            html_content = response.text
            
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            pub_date = self._extract_date(soup)
            
            if not title or not content or len(content) < 100:
                return None
                
            paywall_status = self.paywall_detector.detect_paywall(content, html_content, url)
            if paywall_status == PaywallStatus.BLOCKED:
                return None
                
            content_hash = hashlib.md5(f"{title}{content}".encode()).hexdigest()
            
            return SwissArticle(
                title=title,
                content=content,
                url=url,
                published_date=pub_date,
                source='RTS',
                language='fr',
                canton='national',
                content_hash=content_hash,
                scraped_at=datetime.now().isoformat(),
                word_count=len(content.split()),
                paywall_status=paywall_status
            )
            
        except Exception as e:
            logger.error(f"Error scraping RTS article {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_elem = soup.find('h1') or soup.find('title')
        return title_elem.get_text().strip() if title_elem else ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text().strip() for p in paragraphs[:15]])
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        date_elem = soup.find('time')
        if date_elem and date_elem.get('datetime'):
            return date_elem.get('datetime')
        return datetime.now().isoformat()
    
    def _is_recent(self, date_str: str, cutoff_time: datetime) -> bool:
        try:
            if 'T' in date_str:
                article_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return article_time > cutoff_time
        except:
            pass
        return True

class RSIScraper:
    """RSI.ch scraper for Italian Swiss content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept-Language': 'it-CH,it;q=0.9,de-CH,de;q=0.8,fr-CH,fr;q=0.7,en;q=0.6',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'
        })
        self.base_url = "https://www.rsi.ch"
        self.paywall_detector = PaywallDetector()
        
    def scrape_recent_articles(self, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape recent articles from RSI"""
        articles = []
        
        # RSI sections (Italian structure, no sports)
        sections = [
            '/news',
            '/news/svizzera',
            '/news/mondo',
            '/news/economia'
        ]
        
        for section in sections:
            try:
                section_articles = self._scrape_section(section, cutoff_time)
                articles.extend(section_articles)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error scraping RSI section {section}: {e}")
                
        return articles[:10]
    
    def _scrape_section(self, section: str, cutoff_time: datetime) -> List[SwissArticle]:
        """Scrape RSI section"""
        articles = []
        url = self.base_url + section
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href')
                if href and self._is_article_url(href):
                    full_url = urljoin(self.base_url, href)
                    
                    article = self._scrape_article(full_url)
                    if article and self._is_recent(article.published_date, cutoff_time):
                        articles.append(article)
                        
                    time.sleep(0.5)
                    
        except Exception as e:
            logger.error(f"Error scraping RSI section {section}: {e}")
            
        return articles[:3]  # Smaller limit for RSI
    
    def _is_article_url(self, href: str) -> bool:
        return (
            '/news/' in href and
            len(href.split('/')) >= 4 and
            not href.endswith(('.jpg', '.png', '.pdf', '.mp4'))
        )
    
    def _scrape_article(self, url: str) -> Optional[SwissArticle]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            html_content = response.text
            
            title = self._extract_title(soup)
            content = self._extract_content(soup)
            pub_date = self._extract_date(soup)
            
            if not title or not content or len(content) < 100:
                return None
                
            paywall_status = self.paywall_detector.detect_paywall(content, html_content, url)
            if paywall_status == PaywallStatus.BLOCKED:
                return None
                
            content_hash = hashlib.md5(f"{title}{content}".encode()).hexdigest()
            
            return SwissArticle(
                title=title,
                content=content,
                url=url,
                published_date=pub_date,
                source='RSI',
                language='it',
                canton='ticino',
                content_hash=content_hash,
                scraped_at=datetime.now().isoformat(),
                word_count=len(content.split()),
                paywall_status=paywall_status
            )
            
        except Exception as e:
            logger.error(f"Error scraping RSI article {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        title_elem = soup.find('h1') or soup.find('title')
        return title_elem.get_text().strip() if title_elem else ""
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text().strip() for p in paragraphs[:10]])
    
    def _extract_date(self, soup: BeautifulSoup) -> str:
        date_elem = soup.find('time')
        if date_elem and date_elem.get('datetime'):
            return date_elem.get('datetime')
        return datetime.now().isoformat()
    
    def _is_recent(self, date_str: str, cutoff_time: datetime) -> bool:
        try:
            if 'T' in date_str:
                article_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                return article_time > cutoff_time
        except:
            pass
        return True