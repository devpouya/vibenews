"""
Main pipeline for scraping, annotating, and storing articles
"""
import asyncio
import logging
from typing import List
from datetime import datetime

from sentence_transformers import SentenceTransformer

from backend.scraper.twentymin_scraper import TwentyMinScraper
from backend.annotator.gemini_annotator import GeminiAnnotator
from backend.vector_store.chroma_store import ChromaVectorStore
from backend.models.article import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VibeNewsPipeline:
    """Main pipeline orchestrating the entire process"""
    
    def __init__(self):
        self.scraper = TwentyMinScraper()
        self.annotator = GeminiAnnotator()
        self.vector_store = ChromaVectorStore()
        
        # Load embedding model
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        logger.info("Embedding model loaded")
    
    async def scrape_and_process(self, topics: List[str], max_articles: int = 20) -> int:
        """
        Scrape articles and process them through the full pipeline
        
        Args:
            topics: List of topics to annotate for (e.g., ['ukraine', 'climate'])
            max_articles: Maximum number of articles to process
        
        Returns:
            Number of articles successfully processed
        """
        processed_count = 0
        
        try:
            # Step 1: Scrape articles
            logger.info("Starting article scraping...")
            articles = self.scraper.scrape_politics_articles(limit=max_articles)
            logger.info(f"Scraped {len(articles)} articles")
            
            if not articles:
                logger.warning("No articles scraped")
                return 0
            
            # Step 2: Process each article for each topic
            for article in articles:
                logger.info(f"Processing article: {article.title[:50]}...")
                
                # Generate embedding
                embedding = self.embedding_model.encode(article.content).tolist()
                
                # Annotate for each topic
                for topic in topics:
                    logger.info(f"Annotating for topic: {topic}")
                    
                    annotation = await self.annotator.annotate_article(article, topic)
                    if annotation:
                        # Store in vector database
                        success = self.vector_store.add_article(
                            article=article,
                            embedding=embedding,
                            bias_score=annotation['bias_score'],
                            topic=topic
                        )
                        
                        if success:
                            processed_count += 1
                            logger.info(
                                f"Successfully processed article for topic '{topic}' "
                                f"with bias score {annotation['bias_score']:.2f}"
                            )
                        else:
                            logger.error(f"Failed to store article for topic '{topic}'")
                    else:
                        logger.error(f"Failed to annotate article for topic '{topic}'")
                
                # Brief pause between articles to be respectful
                await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
        
        logger.info(f"Pipeline completed. Processed {processed_count} article-topic pairs")
        return processed_count
    
    async def process_single_article(self, article: Article, topics: List[str]) -> bool:
        """Process a single article for multiple topics"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(article.content).tolist()
            
            # Annotate for each topic
            for topic in topics:
                annotation = await self.annotator.annotate_article(article, topic)
                if annotation:
                    success = self.vector_store.add_article(
                        article=article,
                        embedding=embedding,
                        bias_score=annotation['bias_score'],
                        topic=topic
                    )
                    if not success:
                        return False
                else:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing single article: {e}")
            return False


async def main():
    """Example usage of the pipeline"""
    pipeline = VibeNewsPipeline()
    
    # Define topics to analyze
    topics = [
        "ukraine",
        "climate change", 
        "immigration",
        "EU relations"
    ]
    
    # Run the pipeline
    processed = await pipeline.scrape_and_process(topics, max_articles=5)
    print(f"Successfully processed {processed} article-topic pairs")


if __name__ == "__main__":
    asyncio.run(main())