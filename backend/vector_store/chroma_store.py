import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from backend.models.article import Article
from backend.config import settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store for articles with bias metadata"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        if persist_directory is None:
            persist_directory = str(settings.data_dir / "chroma_db")
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="articles",
            metadata={"description": "Swiss news articles with bias scores"}
        )
        
        logger.info(f"ChromaDB initialized with {self.collection.count()} articles")
    
    def add_article(self, article: Article, embedding: List[float], bias_score: float, topic: str) -> bool:
        """Add article to vector store with bias metadata"""
        try:
            metadata = {
                "source": article.source,
                "published_at": article.published_at.isoformat(),
                "author": article.author or "",
                "language": article.language,
                "bias_score": bias_score,
                "topic": topic,
                "title": article.title,
                "url": article.url
            }
            
            self.collection.add(
                documents=[article.content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[article.id]
            )
            
            logger.info(f"Added article {article.id[:8]}... to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding article {article.id} to vector store: {e}")
            return False
    
    def search_by_topic(
        self, 
        query: str, 
        topic: Optional[str] = None,
        bias_range: Optional[tuple] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search articles by topic with optional bias filtering"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if topic:
                where_clause["topic"] = topic
            
            if bias_range:
                min_bias, max_bias = bias_range
                where_clause["bias_score"] = {"$gte": min_bias, "$lte": max_bias}
            
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            articles = []
            if results['documents'][0]:  # Check if results exist
                for i in range(len(results['documents'][0])):
                    article_data = {
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i]
                    }
                    articles.append(article_data)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []
    
    def get_articles_by_bias_range(
        self, 
        topic: str, 
        min_bias: float = -1.0, 
        max_bias: float = 1.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get articles for a topic within a bias range, sorted by bias score"""
        try:
            where_clause = {
                "topic": topic,
                "bias_score": {"$gte": min_bias, "$lte": max_bias}
            }
            
            # Get all matching articles
            results = self.collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            # Format and sort by bias score
            articles = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    article_data = {
                        "id": results['ids'][i],
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    articles.append(article_data)
                
                # Sort by bias score
                articles.sort(key=lambda x: x['metadata']['bias_score'])
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting articles by bias range: {e}")
            return []
    
    def get_topics(self) -> List[str]:
        """Get all unique topics in the collection"""
        try:
            # This is a simplified approach - ChromaDB doesn't have direct aggregation
            # In practice, you might want to maintain a separate topics collection
            results = self.collection.get(
                include=["metadatas"]
            )
            
            topics = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    if 'topic' in metadata:
                        topics.add(metadata['topic'])
            
            return sorted(list(topics))
            
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return []
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article from vector store"""
        try:
            self.collection.delete(ids=[article_id])
            logger.info(f"Deleted article {article_id[:8]}... from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting article {article_id}: {e}")
            return False