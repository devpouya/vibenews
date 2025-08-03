"""
Article-Centric Vector Store for VibeNews
Stores articles with embeddings for similarity search and topic tagging
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

from backend.models.swiss_article import SwissArticle
from backend.config import settings

logger = logging.getLogger(__name__)

class ArticleVectorStore:
    """Article-centric vector store with similarity search and topic tagging"""
    
    def __init__(self, persist_directory: Optional[str] = None, db_session=None):
        if persist_directory is None:
            persist_directory = str(settings.data_dir / "article_vector_db")
        
        self.db_session = db_session
        
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
        
        # Initialize collection for articles
        self.articles_collection = self.client.get_or_create_collection(
            name="articles",
            metadata={
                "description": "Swiss news articles with embeddings for similarity search",
                "version": "1.0"
            }
        )
        
        logger.info(f"ArticleVectorStore initialized with {self.articles_collection.count()} articles")
    
    def get_simple_similar_articles(self, article_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar articles using simple keyword matching (for testing with zero embeddings)"""
        try:
            # Get the target article
            target_result = self.articles_collection.get(
                ids=[f"article_{article_id}"],
                include=["documents", "metadatas"]
            )
            
            if not target_result['documents']:
                logger.warning(f"Article {article_id} not found in vector store")
                return []
            
            target_doc = target_result['documents'][0]
            target_metadata = target_result['metadatas'][0]
            
            # Get all other articles
            all_results = self.articles_collection.get(
                include=["documents", "metadatas"]
            )
            
            # Simple similarity based on topic tags or content keywords
            similar_articles = []
            target_topics = json.loads(target_metadata.get('topic_tags', '[]'))
            
            for i, doc in enumerate(all_results['documents']):
                article_id_str = all_results['ids'][i].replace("article_", "")
                
                # Skip the target article itself
                if article_id_str == str(article_id):
                    continue
                
                metadata = all_results['metadatas'][i]
                article_topics = json.loads(metadata.get('topic_tags', '[]'))
                
                # Calculate simple similarity score based on shared topics
                shared_topics = set(target_topics) & set(article_topics)
                similarity_score = len(shared_topics) / max(len(target_topics), len(article_topics), 1)
                
                # Add some randomness for variety when topics don't match
                if similarity_score == 0:
                    similarity_score = 0.1
                
                article_data = {
                    "id": article_id_str,
                    "title": metadata.get('title', 'Untitled'),
                    "content": doc[:300] + "..." if len(doc) > 300 else doc,
                    "url": metadata.get('url', ''),
                    "source": metadata.get('source', 'Unknown'),
                    "language": metadata.get('language', 'en'),
                    "published_date": metadata.get('published_date', ''),
                    "bias_score": metadata.get('bias_score', 0.0),
                    "word_count": metadata.get('word_count', 0),
                    "similarity_score": similarity_score,
                    "shared_topics": list(shared_topics),
                    "metadata": metadata
                }
                similar_articles.append(article_data)
            
            # Sort by similarity score and return top results
            similar_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Found {len(similar_articles[:limit])} similar articles for article {article_id}")
            return similar_articles[:limit]
            
        except Exception as e:
            logger.error(f"Error finding simple similar articles for {article_id}: {e}")
            return []
    
    def get_all_articles(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all articles without date filtering for testing"""
        try:
            # Get all articles
            results = self.articles_collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            # Format results
            articles = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    article_data = {
                        "id": results['ids'][i].replace("article_", ""),
                        "title": results['metadatas'][i].get('title', 'Untitled'),
                        "content": results['documents'][i][:300] + "..." if len(results['documents'][i]) > 300 else results['documents'][i],
                        "url": results['metadatas'][i].get('url', ''),
                        "source": results['metadatas'][i].get('source', 'Unknown'),
                        "language": results['metadatas'][i].get('language', 'en'),
                        "published_date": results['metadatas'][i].get('published_date', ''),
                        "bias_score": results['metadatas'][i].get('bias_score', 0.0),
                        "word_count": results['metadatas'][i].get('word_count', 0),
                        "metadata": results['metadatas'][i]
                    }
                    articles.append(article_data)
            
            logger.info(f"Retrieved {len(articles)} articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error getting all articles: {e}")
            return []
    
    def add_simple_article(self, article_id: str, title: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Add article directly with simple parameters for testing"""
        try:
            # Generate a simple embedding (just zeros for testing)
            embedding = [0.0] * 384  # Standard embedding size
            
            # Clean metadata - ensure no None values
            clean_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, list):
                        clean_metadata[key] = json.dumps(value)
                    else:
                        clean_metadata[key] = value
                else:
                    clean_metadata[key] = ""
            
            # Add required fields if missing
            clean_metadata.setdefault("article_id", article_id)
            clean_metadata.setdefault("title", title)
            clean_metadata.setdefault("bias_score", 0.0)
            clean_metadata.setdefault("bias_analyzed", False)
            clean_metadata.setdefault("political_leaning", "center")
            
            # Add to collection
            self.articles_collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                ids=[f"article_{article_id}"]
            )
            
            logger.info(f"Added simple article {article_id}: {title}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding simple article {article_id}: {e}")
            return False
    
    def add_article(self, article: SwissArticle, embedding: List[float], 
                   topic_tags: Optional[List[str]] = None,
                   bias_data: Optional[Dict] = None) -> bool:
        """Add article to vector store with topic tags and bias data"""
        try:
            # Prepare article metadata
            metadata = {
                "article_id": article.id,
                "title": article.title[:500],  # Truncate for metadata limits
                "source": article.source,
                "language": article.language,
                "published_date": article.published_date.isoformat(),
                "scraped_at": article.scraped_at.isoformat(),
                "word_count": article.word_count,
                "paywall_status": article.paywall_status,
                "canton": article.canton or "",
                "url": article.url[:500],  # Truncate URL
                "content_hash": article.content_hash
            }
            
            # Add topic tags
            if topic_tags:
                metadata["topic_tags"] = json.dumps(topic_tags[:10])  # Limit to 10 tags
                metadata["primary_topic"] = topic_tags[0] if topic_tags else ""
            else:
                metadata["topic_tags"] = "[]"
                metadata["primary_topic"] = ""
            
            # Add bias data if available
            if bias_data:
                metadata.update({
                    "bias_score": bias_data.get("bias_score", {}).get("overall_score", 0.0),
                    "political_leaning": bias_data.get("swiss_bias_spectrum", {}).get("political_leaning", "center"),
                    "bias_types_count": len(bias_data.get("bias_types_detected", [])),
                    "bias_analyzed": True
                })
            else:
                metadata.update({
                    "bias_score": 0.0,
                    "political_leaning": "center",
                    "bias_types_count": 0,
                    "bias_analyzed": False
                })
            
            # Clean metadata - remove None values as ChromaDB doesn't support them
            clean_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    clean_metadata[key] = value
                else:
                    # Provide default values for None
                    if key in ["canton"]:
                        clean_metadata[key] = ""
                    elif key in ["bias_score", "bias_types_count"]:
                        clean_metadata[key] = 0
                    elif key in ["bias_analyzed"]:
                        clean_metadata[key] = False
                    else:
                        clean_metadata[key] = ""
            
            # Add to collection
            self.articles_collection.add(
                documents=[article.content],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                ids=[f"article_{article.id}"]
            )
            
            logger.info(f"Added article {article.id} to vector store with {len(topic_tags or [])} topic tags")
            return True
            
        except Exception as e:
            logger.error(f"Error adding article {article.id} to vector store: {e}")
            return False
    
    def get_similar_articles(self, article_id: int, limit: int = 10, 
                           exclude_same_source: bool = True) -> List[Dict[str, Any]]:
        """Get articles similar to the given article"""
        try:
            # Get the target article's embedding
            target_result = self.articles_collection.get(
                ids=[f"article_{article_id}"],
                include=["embeddings", "metadatas"]
            )
            
            if not target_result['embeddings']:
                logger.warning(f"Article {article_id} not found in vector store")
                return []
            
            target_embedding = target_result['embeddings'][0]
            target_metadata = target_result['metadatas'][0]
            
            # Build where clause to exclude target article
            where_clause = {"article_id": {"$ne": article_id}}
            
            # Optionally exclude articles from same source
            if exclude_same_source and target_metadata.get("source"):
                where_clause["source"] = {"$ne": target_metadata["source"]}
            
            # Find similar articles
            results = self.articles_collection.query(
                query_embeddings=[target_embedding],
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            similar_articles = []
            if results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    article_data = {
                        "id": results['ids'][0][i].replace("article_", ""),
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                        "distance": results['distances'][0][i]
                    }
                    similar_articles.append(article_data)
            
            logger.info(f"Found {len(similar_articles)} similar articles for article {article_id}")
            return similar_articles
            
        except Exception as e:
            logger.error(f"Error finding similar articles for {article_id}: {e}")
            return []
    
    def search_articles(self, query: str, filters: Optional[Dict] = None, 
                       limit: int = 20) -> List[Dict[str, Any]]:
        """Search articles using semantic search"""
        try:
            # Build where clause from filters
            where_clause = {}
            
            if filters:
                if filters.get("language"):
                    where_clause["language"] = filters["language"]
                
                if filters.get("source"):
                    where_clause["source"] = filters["source"]
                
                if filters.get("topic"):
                    # Search within topic tags (note: this is basic string matching)
                    # For better topic search, we'd need to store each tag separately
                    pass  # Will implement topic filtering separately
                
                if filters.get("bias_range"):
                    min_bias, max_bias = filters["bias_range"]
                    where_clause["bias_score"] = {"$gte": min_bias, "$lte": max_bias}
                
                if filters.get("date_range"):
                    start_date, end_date = filters["date_range"]
                    where_clause["published_date"] = {
                        "$gte": start_date.isoformat(),
                        "$lte": end_date.isoformat()
                    }
            
            # Perform semantic search
            results = self.articles_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            articles = []
            if results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    article_data = {
                        "id": results['ids'][0][i].replace("article_", ""),
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "relevance_score": 1 - results['distances'][0][i],
                        "distance": results['distances'][0][i]
                    }
                    articles.append(article_data)
            
            logger.info(f"Search for '{query}' returned {len(articles)} results")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching articles: {e}")
            return []
    
    def get_articles_by_topic(self, topic: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get articles that have a specific topic tag"""
        try:
            # Note: This is a simplified approach. For better performance,
            # we should store topic tags as separate fields or use a separate collection
            
            # Get all articles and filter by topic in memory
            # In production, this should be optimized
            results = self.articles_collection.get(
                limit=limit * 3,  # Get more to filter
                include=["documents", "metadatas"]
            )
            
            filtered_articles = []
            if results['documents']:
                for i, metadata in enumerate(results['metadatas']):
                    topic_tags_str = metadata.get("topic_tags", "[]")
                    try:
                        topic_tags = json.loads(topic_tags_str)
                        if topic.lower() in [tag.lower() for tag in topic_tags]:
                            article_data = {
                                "id": results['ids'][i].replace("article_", ""),
                                "content": results['documents'][i],
                                "metadata": metadata
                            }
                            filtered_articles.append(article_data)
                            
                            if len(filtered_articles) >= limit:
                                break
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(filtered_articles)} articles for topic '{topic}'")
            return filtered_articles
            
        except Exception as e:
            logger.error(f"Error getting articles by topic {topic}: {e}")
            return []
    
    def get_recent_articles(self, limit: int = 50, language: Optional[str] = None,
                          hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get recent articles sorted by publication date"""
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(hours=hours_back)
            
            # Build where clause
            where_clause = {
                "published_date": {"$gte": cutoff_date.isoformat()}
            }
            
            if language:
                where_clause["language"] = language
            
            # Get articles
            results = self.articles_collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            # Format and sort by date
            articles = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    article_data = {
                        "id": results['ids'][i].replace("article_", ""),
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    articles.append(article_data)
                
                # Sort by published date (most recent first)
                articles.sort(
                    key=lambda x: x['metadata']['published_date'], 
                    reverse=True
                )
            
            logger.info(f"Retrieved {len(articles)} recent articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []
    
    def get_trending_articles(self, limit: int = 20, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get trending articles based on recency and engagement"""
        try:
            # For now, use recent articles with high bias diversity as "trending"
            # In a real system, this would consider clicks, shares, etc.
            
            recent_articles = self.get_recent_articles(limit * 2, hours_back=hours_back)
            
            # Score articles based on recency and bias diversity
            scored_articles = []
            for article in recent_articles:
                metadata = article['metadata']
                
                # Calculate recency score (0-1, newer = higher)
                pub_date = datetime.fromisoformat(metadata['published_date'])
                hours_old = (datetime.now() - pub_date).total_seconds() / 3600
                recency_score = max(0, 1 - (hours_old / hours_back))
                
                # Bias diversity score (articles with non-center bias get higher score)
                bias_score = abs(metadata.get('bias_score', 0))
                bias_diversity_score = min(bias_score * 2, 1)  # Normalize to 0-1
                
                # Topic diversity (articles with more topic tags get higher score)
                topic_tags_str = metadata.get("topic_tags", "[]")
                try:
                    topic_tags = json.loads(topic_tags_str)
                    topic_diversity_score = min(len(topic_tags) / 5, 1)  # Normalize to 0-1
                except:
                    topic_diversity_score = 0
                
                # Combine scores
                trending_score = (
                    recency_score * 0.5 + 
                    bias_diversity_score * 0.3 + 
                    topic_diversity_score * 0.2
                )
                
                article['trending_score'] = trending_score
                scored_articles.append(article)
            
            # Sort by trending score and take top articles
            scored_articles.sort(key=lambda x: x['trending_score'], reverse=True)
            trending_articles = scored_articles[:limit]
            
            logger.info(f"Retrieved {len(trending_articles)} trending articles")
            return trending_articles
            
        except Exception as e:
            logger.error(f"Error getting trending articles: {e}")
            return []
    
    def get_all_topic_tags(self) -> List[str]:
        """Get all unique topic tags across articles"""
        try:
            # Get all articles with topic tags
            results = self.articles_collection.get(
                include=["metadatas"]
            )
            
            all_topics = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    topic_tags_str = metadata.get("topic_tags", "[]")
                    try:
                        topic_tags = json.loads(topic_tags_str)
                        all_topics.update(topic_tags)
                    except json.JSONDecodeError:
                        continue
            
            sorted_topics = sorted(list(all_topics))
            logger.info(f"Found {len(sorted_topics)} unique topic tags")
            return sorted_topics
            
        except Exception as e:
            logger.error(f"Error getting topic tags: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the article collection"""
        try:
            total_count = self.articles_collection.count()
            
            if total_count == 0:
                return {"total_articles": 0}
            
            # Get sample of articles for statistics
            sample_size = min(1000, total_count)
            results = self.articles_collection.get(
                limit=sample_size,
                include=["metadatas"]
            )
            
            # Calculate statistics
            stats = {
                "total_articles": total_count,
                "sample_size": sample_size,
                "languages": {},
                "sources": {},
                "bias_distribution": {
                    "left": 0, "center_left": 0, "center": 0, 
                    "center_right": 0, "right": 0
                },
                "recent_articles": {
                    "last_24h": 0,
                    "last_week": 0
                },
                "topic_stats": {
                    "articles_with_topics": 0,
                    "avg_topics_per_article": 0
                }
            }
            
            now = datetime.now()
            total_topic_count = 0
            articles_with_topics = 0
            
            for metadata in results['metadatas']:
                # Language distribution
                lang = metadata.get('language', 'unknown')
                stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
                
                # Source distribution
                source = metadata.get('source', 'unknown')
                stats['sources'][source] = stats['sources'].get(source, 0) + 1
                
                # Bias distribution
                political_leaning = metadata.get('political_leaning', 'center')
                if political_leaning in stats['bias_distribution']:
                    stats['bias_distribution'][political_leaning] += 1
                
                # Recent articles
                pub_date = datetime.fromisoformat(metadata['published_date'])
                hours_diff = (now - pub_date).total_seconds() / 3600
                
                if hours_diff <= 24:
                    stats['recent_articles']['last_24h'] += 1
                if hours_diff <= 168:  # 7 days
                    stats['recent_articles']['last_week'] += 1
                
                # Topic statistics
                topic_tags_str = metadata.get("topic_tags", "[]")
                try:
                    topic_tags = json.loads(topic_tags_str)
                    if topic_tags:
                        articles_with_topics += 1
                        total_topic_count += len(topic_tags)
                except:
                    pass
            
            # Calculate topic averages
            stats['topic_stats']['articles_with_topics'] = articles_with_topics
            if articles_with_topics > 0:
                stats['topic_stats']['avg_topics_per_article'] = total_topic_count / articles_with_topics
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_old_articles(self, days_old: int = 90) -> int:
        """Remove articles older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            # Get old articles
            results = self.articles_collection.get(
                where={"published_date": {"$lt": cutoff_date.isoformat()}},
                include=["metadatas"]
            )
            
            if results['ids']:
                # Delete old articles
                self.articles_collection.delete(ids=results['ids'])
                logger.info(f"Cleaned up {len(results['ids'])} old articles")
                return len(results['ids'])
            
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up old articles: {e}")
            return 0