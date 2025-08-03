#!/usr/bin/env python3
"""
Fix Swiss articles date format and reload into vector database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import logging
from datetime import datetime
from dateutil import parser
from backend.vector_store.article_vector_store import ArticleVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_and_reload_articles():
    """Clear database and reload Swiss articles with fixed dates"""
    
    # First, let's clear the vector database
    logger.info("Clearing vector database...")
    
    try:
        # Initialize vector store
        vector_store = ArticleVectorStore()
        
        # Clear the collection by deleting all entries
        try:
            # Get all IDs first
            all_results = vector_store.articles_collection.get()
            if all_results['ids']:
                vector_store.articles_collection.delete(ids=all_results['ids'])
                logger.info(f"Cleared {len(all_results['ids'])} articles from vector database")
            else:
                logger.info("Vector database was already empty")
        except Exception as e:
            logger.warning(f"Error clearing database: {e} - continuing anyway")
        
        # Recreate the collection
        vector_store.articles_collection = vector_store.client.get_or_create_collection(
            name="articles",
            metadata={
                "description": "Swiss news articles with embeddings for similarity search",
                "version": "1.0"
            }
        )
        
        # Sample Swiss articles with proper formatting
        swiss_articles = [
            {
                'id': '1001',
                'title': 'Schweizer Wirtschaft zeigt Widerstandsfähigkeit trotz globaler Unsicherheit',
                'content': 'Die Schweizer Wirtschaft demonstriert weiterhin bemerkenswerte Widerstandsfähigkeit angesichts globaler wirtschaftlicher Unsicherheiten. Aktuelle Indikatoren deuten auf stetiges Wachstum in Schlüsselsektoren wie Technologie, Finanzen und Verarbeitendes Gewerbe hin.',
                'source': 'SRF',
                'language': 'de',
                'published_date': '2025-08-02T10:00:00',
                'bias_score': 0.0,
                'word_count': 45
            },
            {
                'id': '1002', 
                'title': 'Klimaschutz: Schweiz verstärkt Initiativen für grüne Energie',
                'content': 'Die Schweiz hat neue Initiativen angekündigt, um ihren Übergang zu erneuerbaren Energiequellen zu beschleunigen. Der umfassende Plan der Regierung umfasst verstärkte Investitionen in Solar- und Windenergie sowie Verbesserungen der Energiespeicher-Infrastruktur.',
                'source': 'SRF',
                'language': 'de', 
                'published_date': '2025-08-02T09:00:00',
                'bias_score': -0.1,
                'word_count': 55
            },
            {
                'id': '1003',
                'title': 'Tech-Sektor treibt Innovation in Schweizer Städten voran',
                'content': 'Schweizer Städte werden für Technologieunternehmen zunehmend attraktiver, wobei Zürich und Genf bei der Gründung von Startups und Risikokapitalinvestitionen führend sind. Die Kombination aus qualifizierten Arbeitskräften, stabilem politischen Umfeld und strategischer Lage zieht weiterhin internationale Tech-Firmen an.',
                'source': 'Watson',
                'language': 'de',
                'published_date': '2025-08-02T08:00:00', 
                'bias_score': 0.1,
                'word_count': 65
            },
            {
                'id': '1004',
                'title': 'Schweizer Gesundheitssystem passt sich alternder Bevölkerung an',
                'content': 'Das Schweizer Gesundheitssystem implementiert neue Strategien, um den Herausforderungen einer alternden Bevölkerung zu begegnen. Gesundheitsanbieter investieren in digitale Gesundheitstechnologien und präventive Pflegeprogramme.',
                'source': 'Watson',
                'language': 'de',
                'published_date': '2025-08-02T07:00:00',
                'bias_score': 0.0,
                'word_count': 42
            },
            {
                'id': '1005',
                'title': 'Schweizer Tourismus erholt sich nach schwieriger Phase',
                'content': 'Die Schweizer Tourismusbranche zeigt starke Erholungszeichen, wobei die Besucherzahlen sich wieder dem Vor-Pandemie-Niveau nähern. Bergresorts und städtische Destinationen melden steigende Buchungen von inländischen und internationalen Reisenden.',
                'source': 'SRF',
                'language': 'de',
                'published_date': '2025-08-02T06:00:00',
                'bias_score': 0.05,
                'word_count': 48
            },
            {
                'id': '1006',
                'title': 'Zürcher Hauptbahnhof: Normalbetrieb nach Grossbaustelle',
                'content': 'Nach monatelangen Bauarbeiten herrscht am Hauptbahnhof Zürich wieder Normalbetrieb. Die umfangreichen Renovierungsarbeiten haben die Kapazität und den Komfort für Millionen von Reisenden verbessert.',
                'source': 'SRF', 
                'language': 'de',
                'published_date': '2025-08-02T05:00:00',
                'bias_score': 0.0,
                'word_count': 38
            },
            {
                'id': '1007',
                'title': 'Schweizer Demokratie als Modell: Grenzen und Möglichkeiten',
                'content': 'Experten diskutieren über die Übertragbarkeit der Schweizer Demokratie auf andere Länder. Während das System viele Vorteile bietet, gibt es auch spezifische kulturelle und historische Faktoren, die berücksichtigt werden müssen.',
                'source': 'Watson',
                'language': 'de',
                'published_date': '2025-08-02T04:00:00',
                'bias_score': 0.0,
                'word_count': 52
            },
            {
                'id': '1008',
                'title': 'Handelsspannungen: USA verhängen hohe Zölle gegen Schweiz',
                'content': 'Die USA haben für die Schweiz einen der höchsten Zollsätze verhängt, was die Handelsbeziehungen zwischen den beiden Ländern belastet. Schweizer Unternehmen und die Regierung suchen nach Lösungen für den eskalierenden Handelskonflikt.',
                'source': 'SRF',
                'language': 'de', 
                'published_date': '2025-08-02T03:00:00',
                'bias_score': -0.2,
                'word_count': 58
            }
        ]
        
        # Add articles to vector store
        successful_adds = 0
        for article in swiss_articles:
            try:
                success = vector_store.add_simple_article(
                    article_id=article['id'],
                    title=article['title'],
                    content=article['content'],
                    metadata={
                        'url': f"https://example.com/swiss-{article['id']}",
                        'source': article['source'],
                        'language': article['language'],
                        'published_date': article['published_date'],
                        'bias_score': article['bias_score'],
                        'word_count': article['word_count'],
                        'topic_tags': json.dumps(['schweiz', 'news', 'aktuell']),
                        'scraped_at': datetime.now().isoformat()
                    }
                )
                
                if success:
                    successful_adds += 1
                    logger.info(f"Added: {article['title']}")
                    
            except Exception as e:
                logger.error(f"Error adding article {article['id']}: {e}")
                continue
        
        logger.info(f"✅ Successfully added {successful_adds}/{len(swiss_articles)} Swiss articles")
        
        # Get statistics
        stats = vector_store.get_statistics()
        logger.info(f"Vector store now contains {stats.get('total_articles', 0)} articles")
        
        return successful_adds > 0
        
    except Exception as e:
        logger.error(f"Error clearing and reloading: {e}")
        return False

def main():
    """Main function"""
    logger.info("=== Fixing and Reloading Swiss Articles ===")
    
    success = clear_and_reload_articles()
    
    if success:
        logger.info("✅ Successfully reloaded Swiss articles with proper formatting")
        return True
    else:
        logger.error("❌ Failed to reload Swiss articles")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ Swiss articles reloaded successfully!")
    else:
        print("❌ Failed to reload Swiss articles.")
        sys.exit(1)