import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from backend.models.article import Article
from backend.config import settings

logger = logging.getLogger(__name__)


class JSONLStorage:
    """JSON Lines storage for articles - optimized for ML training"""
    
    def __init__(self):
        self.raw_dir = settings.raw_data_dir
        self.processed_dir = settings.processed_data_dir
        self.annotations_dir = settings.annotations_dir
        
        # Ensure directories exist
        for directory in [self.raw_dir, self.processed_dir, self.annotations_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_articles(self, articles: List[Article], filename: Optional[str] = None) -> str:
        """
        Save articles to JSON Lines format
        
        Args:
            articles: List of Article objects
            filename: Optional filename, defaults to articles_YYYYMMDD_HHMMSS.jsonl
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"articles_{timestamp}.jsonl"
        
        filepath = self.raw_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for article in articles:
                    # Convert Article to dictionary
                    article_dict = {
                        "id": article.id,
                        "url": article.url,
                        "title": article.title,
                        "content": article.content,
                        "summary": article.summary,
                        "author": article.author,
                        "published_at": article.published_at.isoformat(),
                        "source": article.source,
                        "language": article.language,
                        "topics": article.topics,
                        "keywords": article.keywords,
                        "scraped_at": article.scraped_at.isoformat(),
                        "metadata": article.metadata
                    }
                    f.write(json.dumps(article_dict, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(articles)} articles to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving articles: {e}")
            raise
    
    def load_articles(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load articles from JSON Lines file
        
        Args:
            filename: Name of file in raw_data_dir
        
        Returns:
            List of article dictionaries
        """
        filepath = self.raw_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
        
        try:
            articles = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    article_dict = json.loads(line.strip())
                    articles.append(article_dict)
            
            logger.info(f"Loaded {len(articles)} articles from {filepath}")
            return articles
            
        except Exception as e:
            logger.error(f"Error loading articles: {e}")
            raise
    
    def load_articles_as_dataframe(self, filename: str) -> pd.DataFrame:
        """
        Load articles as pandas DataFrame for ML training
        
        Args:
            filename: Name of file in raw_data_dir
        
        Returns:
            DataFrame with articles
        """
        filepath = self.raw_dir / filename
        
        try:
            df = pd.read_json(filepath, lines=True)
            logger.info(f"Loaded {len(df)} articles as DataFrame from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading articles as DataFrame: {e}")
            raise
    
    def save_annotations(self, annotations: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Save bias annotations to JSON Lines format
        
        Args:
            annotations: List of annotation dictionaries
            filename: Optional filename, defaults to annotations_YYYYMMDD_HHMMSS.jsonl
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotations_{timestamp}.jsonl"
        
        filepath = self.annotations_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for annotation in annotations:
                    f.write(json.dumps(annotation, ensure_ascii=False) + '\n')
            
            logger.info(f"Saved {len(annotations)} annotations to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving annotations: {e}")
            raise
    
    def load_annotations_as_dataframe(self, filename: str) -> pd.DataFrame:
        """
        Load annotations as pandas DataFrame
        
        Args:
            filename: Name of file in annotations_dir
        
        Returns:
            DataFrame with annotations
        """
        filepath = self.annotations_dir / filename
        
        try:
            df = pd.read_json(filepath, lines=True)
            logger.info(f"Loaded {len(df)} annotations as DataFrame from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading annotations as DataFrame: {e}")
            raise
    
    def create_training_dataset(
        self, 
        articles_file: str, 
        annotations_file: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create ML training dataset by merging articles and annotations
        
        Args:
            articles_file: Articles JSON Lines file
            annotations_file: Annotations JSON Lines file  
            output_file: Optional output file for processed dataset
        
        Returns:
            Merged DataFrame ready for training
        """
        try:
            # Load articles and annotations
            articles_df = self.load_articles_as_dataframe(articles_file)
            annotations_df = self.load_annotations_as_dataframe(annotations_file)
            
            # Merge on article ID
            training_df = articles_df.merge(annotations_df, on='id', how='inner')
            
            # Save processed dataset if requested
            if output_file:
                output_path = self.processed_dir / output_file
                training_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
                logger.info(f"Saved training dataset to {output_path}")
            
            logger.info(f"Created training dataset with {len(training_df)} samples")
            return training_df
            
        except Exception as e:
            logger.error(f"Error creating training dataset: {e}")
            raise
    
    def list_files(self, directory: str = "raw") -> List[str]:
        """
        List available files in a directory
        
        Args:
            directory: "raw", "processed", or "annotations"
        
        Returns:
            List of filenames
        """
        if directory == "raw":
            dir_path = self.raw_dir
        elif directory == "processed":
            dir_path = self.processed_dir
        elif directory == "annotations":
            dir_path = self.annotations_dir
        else:
            raise ValueError("Directory must be 'raw', 'processed', or 'annotations'")
        
        files = [f.name for f in dir_path.glob("*.jsonl")]
        return sorted(files)
    
    def get_file_stats(self, filename: str, directory: str = "raw") -> Dict[str, Any]:
        """
        Get statistics about a JSON Lines file
        
        Args:
            filename: Name of the file
            directory: "raw", "processed", or "annotations"
        
        Returns:
            Dictionary with file statistics
        """
        if directory == "raw":
            dir_path = self.raw_dir
        elif directory == "processed":
            dir_path = self.processed_dir
        elif directory == "annotations":
            dir_path = self.annotations_dir
        else:
            raise ValueError("Directory must be 'raw', 'processed', or 'annotations'")
        
        filepath = dir_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
        
        try:
            line_count = 0
            total_size = filepath.stat().st_size
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
            
            return {
                "filename": filename,
                "line_count": line_count,
                "file_size_bytes": total_size,
                "file_size_mb": round(total_size / (1024 * 1024), 2),
                "created_at": datetime.fromtimestamp(filepath.stat().st_ctime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting file stats: {e}")
            raise