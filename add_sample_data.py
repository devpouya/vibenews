#!/usr/bin/env python3
"""Add sample articles to the vector database for testing"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.vector_store.article_vector_store import ArticleVectorStore
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_sample_articles():
    """Add sample articles to test the system"""
    try:
        # Initialize vector store
        vector_store = ArticleVectorStore()
        
        # Sample articles
        sample_articles = [
            {
                "id": "1",
                "title": "Swiss Economy Shows Resilience Despite Global Uncertainty",
                "content": "Switzerland's economy continues to demonstrate remarkable resilience in the face of global economic uncertainty. Recent indicators suggest steady growth in key sectors including technology, finance, and manufacturing. The Swiss National Bank maintains its cautious approach to monetary policy while supporting economic stability.",
                "url": "https://example.com/swiss-economy-resilience",
                "source": "Swiss Economic Times",
                "language": "en",
                "published_date": datetime.now() - timedelta(hours=2),
                "bias_score": 0.1,
                "word_count": 150,
                "topic_tags": ["economy", "finance", "switzerland"]
            },
            {
                "id": "2", 
                "title": "Climate Action: Switzerland Advances Green Energy Initiatives",
                "content": "Switzerland has announced new initiatives to accelerate its transition to renewable energy sources. The government's comprehensive plan includes increased investment in solar and wind power, as well as improvements to energy storage infrastructure. Environmental groups have welcomed the announcement while calling for even more ambitious targets.",
                "url": "https://example.com/swiss-green-energy",
                "source": "Environmental News Swiss",
                "language": "en",
                "published_date": datetime.now() - timedelta(hours=5),
                "bias_score": -0.2,
                "word_count": 120,
                "topic_tags": ["environment", "energy", "climate"]
            },
            {
                "id": "3",
                "title": "Technology Sector Drives Innovation in Swiss Cities",
                "content": "Swiss cities are becoming increasingly attractive to technology companies, with Zurich and Geneva leading the way in startup formation and venture capital investment. The combination of skilled workforce, stable political environment, and strategic location continues to draw international tech firms to establish European headquarters in Switzerland.",
                "url": "https://example.com/swiss-tech-innovation",
                "source": "Tech Weekly Switzerland",
                "language": "en", 
                "published_date": datetime.now() - timedelta(hours=8),
                "bias_score": 0.0,
                "word_count": 95,
                "topic_tags": ["technology", "innovation", "cities"]
            },
            {
                "id": "4",
                "title": "Swiss Healthcare System Adapts to Population Aging",
                "content": "Switzerland's healthcare system is implementing new strategies to address the challenges of an aging population. Healthcare providers are investing in digital health technologies and preventive care programs. The government is reviewing healthcare financing mechanisms to ensure long-term sustainability while maintaining quality of care.",
                "url": "https://example.com/swiss-healthcare-aging",
                "source": "Health Journal Switzerland",
                "language": "en",
                "published_date": datetime.now() - timedelta(hours=12),
                "bias_score": 0.05,
                "word_count": 110,
                "topic_tags": ["healthcare", "aging", "digital"]
            },
            {
                "id": "5",
                "title": "Swiss Tourism Industry Rebounds After Challenging Period",
                "content": "The Swiss tourism industry is showing strong signs of recovery, with visitor numbers approaching pre-pandemic levels. Mountain resorts and urban destinations are reporting increased bookings from both domestic and international travelers. The industry is focusing on sustainable tourism practices while rebuilding capacity.",
                "url": "https://example.com/swiss-tourism-rebound",
                "source": "Tourism Today Switzerland",
                "language": "en",
                "published_date": datetime.now() - timedelta(hours=18),
                "bias_score": -0.1,
                "word_count": 85,
                "topic_tags": ["tourism", "recovery", "sustainability"]
            }
        ]
        
        # Add articles to vector store
        for article in sample_articles:
            try:
                vector_store.add_simple_article(
                    article_id=article["id"],
                    title=article["title"],
                    content=article["content"],
                    metadata={
                        "url": article["url"],
                        "source": article["source"],
                        "language": article["language"],
                        "published_date": article["published_date"].isoformat(),
                        "bias_score": article["bias_score"],
                        "word_count": article["word_count"],
                        "topic_tags": article["topic_tags"]
                    }
                )
                logger.info(f"Added article: {article['title']}")
            except Exception as e:
                logger.error(f"Failed to add article {article['id']}: {e}")
        
        # Get statistics
        stats = vector_store.get_statistics()
        logger.info(f"Vector store now contains {stats.get('total_articles', 0)} articles")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to add sample articles: {e}")
        return False

if __name__ == "__main__":
    success = add_sample_articles()
    if success:
        print("Sample articles added successfully!")
    else:
        print("Failed to add sample articles.")
        sys.exit(1)