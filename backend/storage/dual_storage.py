from typing import List, Dict, Any, Optional
import logging

from backend.models.article import Article
from backend.storage.json_storage import JSONLStorage
from backend.vector_store.chroma_store import ChromaVectorStore

# Optional dependency
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class DualStorage:
    """
    Dual storage system: JSON Lines for ML training + ChromaDB for API serving
    """
    
    def __init__(self, load_embedding_model: bool = True):
        self.json_storage = JSONLStorage()
        self.vector_store = ChromaVectorStore()
        
        # Load embedding model if needed
        self.embedding_model = None
        if load_embedding_model:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("sentence-transformers not available, skipping embedding model")
            else:
                logger.info("Loading embedding model...")
                self.embedding_model = SentenceTransformer(
                    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                )
                logger.info("Embedding model loaded")
    
    def store_articles(
        self, 
        articles: List[Article], 
        jsonl_filename: Optional[str] = None,
        store_in_vector_db: bool = False,
        default_topic: str = "general",
        default_bias_score: float = 0.0
    ) -> Dict[str, str]:
        """
        Store articles in both JSON Lines and optionally ChromaDB
        
        Args:
            articles: List of Article objects
            jsonl_filename: Optional filename for JSON Lines storage
            store_in_vector_db: Whether to also store in ChromaDB
            default_topic: Default topic for vector storage
            default_bias_score: Default bias score for vector storage
        
        Returns:
            Dictionary with file paths and storage info
        """
        results = {}
        
        # Always store in JSON Lines for ML training
        jsonl_path = self.json_storage.save_articles(articles, jsonl_filename)
        results['jsonl_path'] = jsonl_path
        results['articles_count'] = len(articles)
        
        # Optionally store in ChromaDB for API serving
        if store_in_vector_db:
            if not self.embedding_model:
                logger.warning("Embedding model not loaded, skipping vector storage")
                results['vector_storage'] = 'skipped - no embedding model'
            else:
                stored_count = 0
                for article in articles:
                    try:
                        # Generate embedding
                        embedding = self.embedding_model.encode(article.content).tolist()
                        
                        # Store in vector database
                        success = self.vector_store.add_article(
                            article=article,
                            embedding=embedding,
                            bias_score=default_bias_score,
                            topic=default_topic
                        )
                        
                        if success:
                            stored_count += 1
                        
                    except Exception as e:
                        logger.error(f"Error storing article {article.id} in vector DB: {e}")
                
                results['vector_storage'] = f'{stored_count}/{len(articles)} articles stored'
        else:
            results['vector_storage'] = 'skipped'
        
        logger.info(f"Dual storage completed: {results}")
        return results
    
    def store_annotated_articles(
        self, 
        articles: List[Article], 
        annotations: List[Dict[str, Any]],
        jsonl_filename: Optional[str] = None,
        annotations_filename: Optional[str] = None,
        store_in_vector_db: bool = True
    ) -> Dict[str, str]:
        """
        Store articles with their bias annotations
        
        Args:
            articles: List of Article objects
            annotations: List of annotation dictionaries with bias scores
            jsonl_filename: Optional filename for articles
            annotations_filename: Optional filename for annotations
            store_in_vector_db: Whether to store in ChromaDB
        
        Returns:
            Dictionary with storage results
        """
        results = {}
        
        # Store articles in JSON Lines
        jsonl_path = self.json_storage.save_articles(articles, jsonl_filename)
        results['articles_path'] = jsonl_path
        
        # Store annotations in JSON Lines
        annotations_path = self.json_storage.save_annotations(annotations, annotations_filename)
        results['annotations_path'] = annotations_path
        
        # Store in ChromaDB if requested
        if store_in_vector_db and self.embedding_model:
            stored_count = 0
            
            # Create lookup for annotations by article ID
            annotations_by_id = {ann['article_id']: ann for ann in annotations}
            
            for article in articles:
                annotation = annotations_by_id.get(article.id)
                if annotation:
                    try:
                        # Generate embedding
                        embedding = self.embedding_model.encode(article.content).tolist()
                        
                        # Store with bias annotation
                        success = self.vector_store.add_article(
                            article=article,
                            embedding=embedding,
                            bias_score=annotation['bias_score'],
                            topic=annotation['topic']
                        )
                        
                        if success:
                            stored_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error storing annotated article {article.id}: {e}")
            
            results['vector_storage'] = f'{stored_count}/{len(articles)} annotated articles stored'
        else:
            results['vector_storage'] = 'skipped'
        
        results['total_articles'] = len(articles)
        results['total_annotations'] = len(annotations)
        
        logger.info(f"Annotated articles storage completed: {results}")
        return results
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored data
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {}
        
        # JSON Lines files
        stats['jsonl_files'] = {
            'raw': self.json_storage.list_files('raw'),
            'processed': self.json_storage.list_files('processed'),
            'annotations': self.json_storage.list_files('annotations')
        }
        
        # ChromaDB stats
        try:
            vector_count = self.vector_store.collection.count()
            topics = self.vector_store.get_topics()
            stats['vector_store'] = {
                'total_articles': vector_count,
                'topics': topics,
                'topic_count': len(topics)
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            stats['vector_store'] = {'error': str(e)}
        
        return stats
    
    def load_for_ml_training(self, articles_file: str, annotations_file: str):
        """
        Load data for ML model training
        
        Args:
            articles_file: Articles JSON Lines file
            annotations_file: Annotations JSON Lines file
        
        Returns:
            Merged DataFrame ready for training
        """
        return self.json_storage.create_training_dataset(articles_file, annotations_file)