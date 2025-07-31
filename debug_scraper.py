"""
Debug script to understand 20 Minuten page structure
"""
import requests
from bs4 import BeautifulSoup
from backend.scraper.twentymin_scraper import TwentyMinScraper

def debug_scraper():
    scraper = TwentyMinScraper()
    
    # Get the politics page
    soup = scraper._get_page("https://www.20min.ch/themen/politik")
    if not soup:
        print("Failed to get politics page")
        return
    
    # Extract links
    links = scraper.extract_article_links(soup)
    print(f"Found {len(links)} article links:")
    for i, link in enumerate(links[:5]):  # Show first 5
        print(f"{i+1}. {link}")
    
    if links:
        # Try to scrape the first article
        first_link = links[0]
        print(f"\nTrying to scrape: {first_link}")
        
        article_soup = scraper._get_page(first_link)
        if article_soup:
            print("Successfully got article page")
            
            # Debug title extraction
            title_selectors = ['h1', '.story-title', '.article-title']
            for selector in title_selectors:
                title_elem = article_soup.select_one(selector)
                if title_elem:
                    print(f"Found title with '{selector}': {title_elem.get_text(strip=True)}")
                    break
            else:
                print("No title found with any selector")
                # Show all h1 tags
                h1_tags = article_soup.find_all('h1')
                print(f"Found {len(h1_tags)} h1 tags:")
                for h1 in h1_tags[:3]:
                    print(f"  - {h1.get_text(strip=True)}")
            
            # Debug content extraction
            content_selectors = [
                '.story-body',
                '.article-body', 
                '.content-text',
                '[data-testid="article-body"]'
            ]
            
            for selector in content_selectors:
                content_elem = article_soup.select_one(selector)
                if content_elem:
                    print(f"Found content with '{selector}': {len(content_elem.get_text())} chars")
                    break
            else:
                print("No content found with any selector")
                # Show some divs or paragraphs
                paragraphs = article_soup.find_all('p')[:5]
                print(f"Found {len(paragraphs)} paragraphs in total")
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 20:
                        print(f"  - {text[:100]}...")
        else:
            print("Failed to get article page")

if __name__ == "__main__":
    debug_scraper()