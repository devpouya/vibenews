import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
import logging

from backend.storage.json_storage import JSONLStorage

logger = logging.getLogger(__name__)


class MLDataLoader:
    """Data loading utilities for ML model training"""
    
    def __init__(self):
        self.storage = JSONLStorage()
    
    def load_training_data(
        self, 
        articles_file: str, 
        annotations_file: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and split data for training
        
        Args:
            articles_file: Articles JSON Lines file
            annotations_file: Annotations JSON Lines file
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (train_df, test_df)
        """
        # Create merged dataset
        full_df = self.storage.create_training_dataset(articles_file, annotations_file)
        
        # Split data
        train_df, test_df = train_test_split(
            full_df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=full_df['bias_label'] if 'bias_label' in full_df.columns else None
        )
        
        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test samples")
        return train_df, test_df
    
    def prepare_text_features(self, df: pd.DataFrame) -> List[str]:
        """
        Prepare text features for model training
        
        Args:
            df: DataFrame with articles
        
        Returns:
            List of text strings for training
        """
        # Combine title and content
        texts = []
        for _, row in df.iterrows():
            text = f"{row['title']} {row['content']}"
            texts.append(text)
        
        return texts
    
    def prepare_labels(self, df: pd.DataFrame, label_column: str = 'bias_score') -> List[float]:
        """
        Prepare labels for model training
        
        Args:
            df: DataFrame with annotations
            label_column: Column name for labels
        
        Returns:
            List of labels
        """
        return df[label_column].tolist()
    
    def get_topic_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get distribution of topics in the dataset
        
        Args:
            df: DataFrame with topic annotations
        
        Returns:
            Dictionary with topic counts
        """
        if 'topic' not in df.columns:
            return {}
        
        return df['topic'].value_counts().to_dict()
    
    def get_bias_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get distribution of bias scores in the dataset
        
        Args:
            df: DataFrame with bias scores
        
        Returns:
            Dictionary with bias statistics
        """
        if 'bias_score' not in df.columns:
            return {}
        
        bias_stats = {
            'mean': df['bias_score'].mean(),
            'std': df['bias_score'].std(),
            'min': df['bias_score'].min(),
            'max': df['bias_score'].max(),
            'median': df['bias_score'].median()
        }
        
        # Distribution by ranges
        bias_ranges = {
            'strongly_negative': (df['bias_score'] <= -0.6).sum(),
            'moderately_negative': ((df['bias_score'] > -0.6) & (df['bias_score'] <= -0.2)).sum(),
            'neutral': ((df['bias_score'] > -0.2) & (df['bias_score'] <= 0.2)).sum(),
            'moderately_positive': ((df['bias_score'] > 0.2) & (df['bias_score'] <= 0.6)).sum(),
            'strongly_positive': (df['bias_score'] > 0.6).sum()
        }
        
        bias_stats['distribution'] = bias_ranges
        return bias_stats
    
    def filter_by_topic(self, df: pd.DataFrame, topics: List[str]) -> pd.DataFrame:
        """
        Filter dataset by specific topics
        
        Args:
            df: DataFrame with articles
            topics: List of topics to include
        
        Returns:
            Filtered DataFrame
        """
        if 'topic' not in df.columns:
            logger.warning("No topic column found, returning original DataFrame")
            return df
        
        filtered_df = df[df['topic'].isin(topics)]
        logger.info(f"Filtered to {len(filtered_df)} articles for topics: {topics}")
        return filtered_df
    
    def filter_by_bias_range(
        self, 
        df: pd.DataFrame, 
        min_bias: float = -1.0, 
        max_bias: float = 1.0
    ) -> pd.DataFrame:
        """
        Filter dataset by bias score range
        
        Args:
            df: DataFrame with bias scores
            min_bias: Minimum bias score
            max_bias: Maximum bias score
        
        Returns:
            Filtered DataFrame
        """
        if 'bias_score' not in df.columns:
            logger.warning("No bias_score column found, returning original DataFrame")
            return df
        
        filtered_df = df[
            (df['bias_score'] >= min_bias) & 
            (df['bias_score'] <= max_bias)
        ]
        logger.info(f"Filtered to {len(filtered_df)} articles with bias in range [{min_bias}, {max_bias}]")
        return filtered_df
    
    def sample_balanced_dataset(
        self, 
        df: pd.DataFrame, 
        samples_per_topic: int = 100,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create a balanced dataset with equal samples per topic
        
        Args:
            df: DataFrame with articles
            samples_per_topic: Number of samples per topic
            random_state: Random seed
        
        Returns:
            Balanced DataFrame
        """
        if 'topic' not in df.columns:
            logger.warning("No topic column found, returning sample of original DataFrame")
            return df.sample(n=min(len(df), samples_per_topic * 5), random_state=random_state)
        
        balanced_dfs = []
        for topic in df['topic'].unique():
            topic_df = df[df['topic'] == topic]
            sample_size = min(len(topic_df), samples_per_topic)
            topic_sample = topic_df.sample(n=sample_size, random_state=random_state)
            balanced_dfs.append(topic_sample)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"Created balanced dataset with {len(balanced_df)} samples")
        return balanced_df