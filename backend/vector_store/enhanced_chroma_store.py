"""
Enhanced ChromaDB Vector Store for VibeNews
Integrates topic clustering, multilingual support, and real-time updates
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
from backend.models.topic import Topic, ArticleTopic, LanguageEnum
from backend.config import settings

logger = logging.getLogger(__name__)

class EnhancedChromaVectorStore:
    """Enhanced ChromaDB vector store with topic clustering and multilingual support"""
    
    def __init__(self, persist_directory: Optional[str] = None, db_session=None):
        if persist_directory is None:
            persist_directory = str(settings.data_dir / "enhanced_chroma_db")
        
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
        
        # Initialize collections
        self._init_collections()
        
        logger.info(f"Enhanced ChromaDB initialized with collections:")
        logger.info(f"  - Articles: {self.articles_collection.count()} items")
        logger.info(f"  - Topics: {self.topics_collection.count()} items")
    
    def _init_collections(self):
        """Initialize ChromaDB collections"""
        # Articles collection - stores individual articles with topic assignments
        self.articles_collection = self.client.get_or_create_collection(
            name="articles_v2",
            metadata={
                "description": "Swiss news articles with topic assignments and bias scores",
                "version": "2.0"
            }
        )
        
        # Topics collection - stores topic embeddings and metadata
        self.topics_collection = self.client.get_or_create_collection(
            name="topics_v2", 
            metadata={
                "description": "News topics with embeddings and statistics",
                "version": "2.0"
            }
        )
        
        # Translations collection - for multilingual search
        self.translations_collection = self.client.get_or_create_collection(
            name="translations_v2",
            metadata={
                "description": "Article translations for multilingual search",
                "version": "2.0"
            }
        )
    
    def add_article_with_topic(self, article: SwissArticle, topic: Topic, 
                              embedding: List[float], confidence_score: float,
                              bias_score: Optional[Dict] = None) -> bool:
        """Add article to vector store with topic assignment"""
        try:
            # Prepare article metadata
            metadata = {
                "article_id": article.id,
                "topic_id": topic.id,
                "topic_name_key": topic.name_key,
                "title": article.title[:500],  # Truncate to avoid metadata size limits
                "source": article.source,
                "language": article.language,
                "published_date": article.published_date.isoformat(),
                "scraped_at": article.scraped_at.isoformat(),
                "word_count": article.word_count,
                "paywall_status": article.paywall_status,
                "confidence_score": confidence_score,
                "bias_analyzed": article.bias_analyzed,
                "canton": article.canton or "",
                "url": article.url[:500]  # Truncate URL
            }
            
            # Add bias score if available
            if bias_score:
                metadata.update({
                    "bias_score": bias_score.get("overall_score", 0.0),
                    "political_leaning": bias_score.get("political_leaning", "center"),
                    "bias_types_count": len(bias_score.get("bias_types_detected", []))
                })
            
            # Add to articles collection
            self.articles_collection.add(
                documents=[article.content],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"article_{article.id}"]
            )
            
            logger.info(f"Added article {article.id} to vector store with topic {topic.name_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding article {article.id} to vector store: {e}")
            return False
    
    def add_topic_embedding(self, topic: Topic, embedding: List[float]) -> bool:
        """Add topic to topics collection with embedding"""
        try:
            # Prepare topic metadata
            metadata = {
                "topic_id": topic.id,
                "name_key": topic.name_key,
                "status": topic.status,
                "source": topic.source,
                "article_count": topic.article_count,
                "trending_score": topic.trending_score,
                "cluster_id": topic.cluster_id or "",
                "created_at": topic.created_at.isoformat(),
                "updated_at": topic.updated_at.isoformat()
            }
            
            # Add language-specific names
            if topic.names:
                for lang, name in topic.names.items():
                    metadata[f"name_{lang}"] = name[:200]  # Truncate
            
            # Create document from topic names and keywords
            topic_text_parts = []
            if topic.names:
                topic_text_parts.extend(topic.names.values())
            if topic.descriptions:
                topic_text_parts.extend(topic.descriptions.values())
            if topic.keywords:
                for lang_keywords in topic.keywords.values():
                    topic_text_parts.extend(lang_keywords[:10])  # Limit keywords
            
            topic_document = " ".join(topic_text_parts)
            
            # Add to topics collection
            self.topics_collection.add(
                documents=[topic_document],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"topic_{topic.id}"]
            )
            
            logger.info(f"Added topic {topic.name_key} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding topic {topic.id} to vector store: {e}")
            return False
    
    def search_articles_by_topic(self, topic_name_key: str, language: Optional[str] = None,
                               bias_range: Optional[Tuple[float, float]] = None,
                               limit: int = 20, 
                               sort_by: str = "published_date") -> List[Dict[str, Any]]:
        """Search articles by topic with optional filtering"""
        try:
            # Build where clause
            where_clause = {"topic_name_key": topic_name_key}
            
            if language:
                where_clause["language"] = language
            
            if bias_range:
                min_bias, max_bias = bias_range
                where_clause["bias_score"] = {"$gte": min_bias, "$lte": max_bias}
            
            # Get articles
            results = self.articles_collection.get(
                where=where_clause,
                limit=limit,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            articles = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    article_data = {
                        "id": results['ids'][i],
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    articles.append(article_data)
                
                # Sort results
                if sort_by == "published_date":
                    articles.sort(key=lambda x: x['metadata']['published_date'], reverse=True)
                elif sort_by == "bias_score":
                    articles.sort(key=lambda x: x['metadata'].get('bias_score', 0))
                elif sort_by == "confidence":
                    articles.sort(key=lambda x: x['metadata']['confidence_score'], reverse=True)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error searching articles by topic {topic_name_key}: {e}")
            return []
    
    def search_articles_semantic(self, query: str, language: Optional[str] = None,
                               topics: Optional[List[str]] = None,
                               limit: int = 20) -> List[Dict[str, Any]]:
        """Semantic search across articles"""
        try:
            # Build where clause
            where_clause = {}
            
            if language:
                where_clause["language"] = language
            
            if topics:
                where_clause["topic_name_key"] = {"$in": topics}
            
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
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity_score": 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    articles.append(article_data)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def get_topic_articles_by_bias_spectrum(self, topic_name_key: str, 
                                          language: Optional[str] = None,
                                          limit: int = 50) -> Dict[str, List[Dict]]:
        """Get articles for a topic organized by bias spectrum"""
        try:
            articles = self.search_articles_by_topic(
                topic_name_key, 
                language=language, 
                limit=limit,
                sort_by="bias_score"
            )
            
            # Organize by bias spectrum
            spectrum = {
                "left": [],      # bias_score < -0.3
                "center_left": [], # -0.3 <= bias_score < -0.1
                "center": [],    # -0.1 <= bias_score <= 0.1
                "center_right": [], # 0.1 < bias_score <= 0.3
                "right": []      # bias_score > 0.3
            }
            
            for article in articles:
                bias_score = article['metadata'].get('bias_score', 0.0)
                
                if bias_score < -0.3:
                    spectrum["left"].append(article)
                elif bias_score < -0.1:
                    spectrum["center_left"].append(article)
                elif bias_score <= 0.1:
                    spectrum["center"].append(article)
                elif bias_score <= 0.3:
                    spectrum["center_right"].append(article)
                else:
                    spectrum["right"].append(article)
            
            return spectrum
            
        except Exception as e:
            logger.error(f"Error getting bias spectrum for topic {topic_name_key}: {e}")
            return {"left": [], "center_left": [], "center": [], "center_right": [], "right": []}
    
    def get_trending_topics(self, language: Optional[str] = None, 
                          limit: int = 10, 
                          min_articles: int = 3) -> List[Dict[str, Any]]:
        """Get trending topics based on recent activity and article count"""
        try:
            # Build where clause
            where_clause = {
                "status": "active",
                "article_count": {"$gte": min_articles}
            }
            
            # Get topics
            results = self.topics_collection.get(
                where=where_clause,
                limit=limit * 2,  # Get more to filter and sort
                include=["documents", "metadatas"]
            )
            
            # Format and sort by trending score
            topics = []
            if results['documents']:
                for i in range(len(results['documents'])):
                    topic_data = {
                        "id": results['ids'][i],
                        "document": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    topics.append(topic_data)
                
                # Sort by trending score
                topics.sort(key=lambda x: x['metadata']['trending_score'], reverse=True)
                topics = topics[:limit]
            
            return topics
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []
    
    def get_topic_statistics(self, topic_name_key: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a topic"""
        try:
            # Get all articles for this topic
            articles = self.search_articles_by_topic(topic_name_key, limit=1000)
            
            if not articles:
                return {"error": "No articles found for topic"}
            
            # Calculate statistics
            stats = {
                "total_articles": len(articles),
                "language_breakdown": {},
                "source_breakdown": {},
                "bias_distribution": {
                    "left": 0, "center_left": 0, "center": 0, 
                    "center_right": 0, "right": 0
                },
                "recent_activity": {
                    "last_24h": 0,
                    "last_week": 0,
                    "last_month": 0
                },
                "avg_confidence": 0.0,
                "date_range": {"earliest": None, "latest": None}
            }
            
            total_confidence = 0
            dates = []
            
            # Process each article
            for article in articles:
                metadata = article['metadata']
                
                # Language breakdown
                lang = metadata.get('language', 'unknown')
                stats["language_breakdown"][lang] = stats["language_breakdown"].get(lang, 0) + 1
                
                # Source breakdown
                source = metadata.get('source', 'unknown')
                stats["source_breakdown"][source] = stats["source_breakdown"].get(source, 0) + 1
                
                # Bias distribution
                bias_score = metadata.get('bias_score', 0.0)
                if bias_score < -0.3:
                    stats["bias_distribution"]["left"] += 1
                elif bias_score < -0.1:
                    stats["bias_distribution"]["center_left"] += 1
                elif bias_score <= 0.1:
                    stats["bias_distribution"]["center"] += 1
                elif bias_score <= 0.3:
                    stats["bias_distribution"]["center_right"] += 1
                else:
                    stats["bias_distribution"]["right"] += 1
                
                # Confidence
                total_confidence += metadata.get('confidence_score', 0.0)
                
                # Date analysis
                pub_date = datetime.fromisoformat(metadata['published_date'].replace('Z', '+00:00'))
                dates.append(pub_date)
                
                # Recent activity
                now = datetime.now()
                if (now - pub_date).days <= 1:
                    stats["recent_activity"]["last_24h"] += 1
                if (now - pub_date).days <= 7:
                    stats["recent_activity"]["last_week"] += 1
                if (now - pub_date).days <= 30:
                    stats["recent_activity"]["last_month"] += 1
            
            # Final calculations
            stats["avg_confidence"] = total_confidence / len(articles) if articles else 0
            
            if dates:
                stats["date_range"]["earliest"] = min(dates).isoformat()
                stats["date_range"]["latest"] = max(dates).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating topic statistics for {topic_name_key}: {e}")
            return {"error": str(e)}
    
    def update_topic_embeddings(self, topic: Topic, new_embedding: List[float]) -> bool:
        """Update topic embedding in vector store"""
        try:
            # Update existing topic
            self.topics_collection.update(
                ids=[f"topic_{topic.id}"],
                embeddings=[new_embedding],
                metadatas=[{
                    "article_count": topic.article_count,
                    "trending_score": topic.trending_score,
                    "updated_at": datetime.now().isoformat()
                }]
            )
            
            logger.info(f"Updated embeddings for topic {topic.name_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating topic embeddings for {topic.id}: {e}")
            return False
    
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
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all collections"""
        try:
            return {
                "articles": {
                    "count": self.articles_collection.count(),
                    "sample_metadata": self.articles_collection.peek()
                },
                "topics": {
                    "count": self.topics_collection.count(),
                    "sample_metadata": self.topics_collection.peek()
                },
                "translations": {
                    "count": self.translations_collection.count()
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}