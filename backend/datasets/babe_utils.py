"""
BABE Dataset Utilities
Tools for validation, analysis, and pretraining with BABE dataset
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from sklearn.model_selection import train_test_split
from collections import Counter

from backend.storage.json_storage import JSONLStorage
from backend.ml.data_loader import MLDataLoader

logger = logging.getLogger(__name__)


class BABEValidator:
    """Validation and analysis utilities for BABE dataset"""
    
    def __init__(self):
        self.json_storage = JSONLStorage()
        self.ml_loader = MLDataLoader()
    
    def load_babe_data(self, filename: str) -> pd.DataFrame:
        """Load BABE dataset from JSON Lines file"""
        return self.json_storage.load_articles_as_dataframe(filename)
    
    def analyze_bias_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze bias label distribution in BABE dataset
        
        Args:
            df: BABE DataFrame with bias_labels column
        
        Returns:
            Dictionary with bias analysis
        """
        analysis = {
            'total_samples': len(df),
            'bias_categories': {},
            'label_distribution': {},
            'text_statistics': {}
        }
        
        # Extract bias labels from original_data
        if 'original_data' in df.columns:
            bias_labels = []
            for _, row in df.iterrows():
                original = row['original_data']
                if isinstance(original, dict):
                    # Look for bias-related keys
                    for key, value in original.items():
                        if 'bias' in key.lower():
                            bias_labels.append({
                                'sample_id': row.get('id', ''),
                                'label_type': key,
                                'label_value': value
                            })
            
            if bias_labels:
                bias_df = pd.DataFrame(bias_labels)
                
                # Analyze each bias label type
                for label_type in bias_df['label_type'].unique():
                    type_data = bias_df[bias_df['label_type'] == label_type]
                    distribution = type_data['label_value'].value_counts().to_dict()
                    analysis['label_distribution'][label_type] = distribution
        
        # Text statistics
        if 'text' in df.columns:
            text_stats = {
                'avg_length': df['text'].str.len().mean(),
                'min_length': df['text'].str.len().min(),
                'max_length': df['text'].str.len().max(),
                'total_characters': df['text'].str.len().sum()
            }
            analysis['text_statistics'] = text_stats
        
        return analysis
    
    def create_validation_split(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation split for BABE dataset
        
        Args:
            df: BABE DataFrame
            test_size: Fraction for validation set
            random_state: Random seed
        
        Returns:
            Tuple of (train_df, val_df)
        """
        # Try to stratify by bias labels if available
        stratify_column = None
        
        if 'bias_labels' in df.columns:
            # Use first bias label for stratification
            first_bias = df['bias_labels'].apply(
                lambda x: list(x.values())[0] if isinstance(x, dict) and x else 'unknown'
            )
            if len(first_bias.unique()) > 1:
                stratify_column = first_bias
        
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_column
        )
        
        logger.info(f"Created validation split: {len(train_df)} train, {len(val_df)} validation")
        return train_df, val_df
    
    def compare_with_swiss_articles(
        self, 
        babe_df: pd.DataFrame, 
        swiss_filename: str
    ) -> Dict[str, Any]:
        """
        Compare BABE dataset with Swiss articles
        
        Args:
            babe_df: BABE DataFrame
            swiss_filename: Filename of Swiss articles JSON Lines
        
        Returns:
            Comparison analysis
        """
        # Load Swiss articles
        swiss_df = self.json_storage.load_articles_as_dataframe(swiss_filename)
        
        comparison = {
            'dataset_sizes': {
                'babe': len(babe_df),
                'swiss': len(swiss_df)
            },
            'text_lengths': {},
            'language_analysis': {},
            'content_overlap': {}
        }
        
        # Text length comparison
        babe_lengths = babe_df['text'].str.len() if 'text' in babe_df.columns else []
        swiss_lengths = swiss_df['content'].str.len() if 'content' in swiss_df.columns else []
        
        if len(babe_lengths) > 0 and len(swiss_lengths) > 0:
            comparison['text_lengths'] = {
                'babe_avg': float(babe_lengths.mean()),
                'swiss_avg': float(swiss_lengths.mean()),
                'babe_median': float(babe_lengths.median()),
                'swiss_median': float(swiss_lengths.median())
            }
        
        # Language analysis (basic)
        if 'text' in babe_df.columns and 'content' in swiss_df.columns:
            # Simple language detection based on common words
            english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            german_words = {'der', 'die', 'das', 'und', 'oder', 'aber', 'in', 'an', 'zu', 'für', 'von', 'mit'}
            
            babe_sample = ' '.join(babe_df['text'].head(100).str.lower())
            swiss_sample = ' '.join(swiss_df['content'].head(100).str.lower())
            
            babe_english_score = sum(1 for word in english_words if word in babe_sample)
            babe_german_score = sum(1 for word in german_words if word in babe_sample)
            
            swiss_english_score = sum(1 for word in english_words if word in swiss_sample)
            swiss_german_score = sum(1 for word in german_words if word in swiss_sample)
            
            comparison['language_analysis'] = {
                'babe_likely_english': babe_english_score > babe_german_score,
                'swiss_likely_german': swiss_german_score > swiss_english_score,
                'language_mismatch': (babe_english_score > babe_german_score) and (swiss_german_score > swiss_english_score)
            }
        
        return comparison
    
    def prepare_for_pretraining(
        self, 
        babe_df: pd.DataFrame,
        output_format: str = 'gemini'
    ) -> List[Dict[str, Any]]:
        """
        Prepare BABE data for Gemini LoRA pretraining
        
        Args:
            babe_df: BABE DataFrame
            output_format: Format for pretraining ('gemini', 'generic')
        
        Returns:
            List of training examples
        """
        training_examples = []
        
        for _, row in babe_df.iterrows():
            if 'text' not in row or 'bias_labels' not in row:
                continue
            
            text = row['text']
            bias_labels = row.get('bias_labels', {})
            
            if output_format == 'gemini':
                # Format for Gemini fine-tuning
                example = {
                    'input_text': f"Analyze the bias in this text: {text}",
                    'output_text': f"Bias analysis: {bias_labels}",
                    'metadata': {
                        'source': 'babe',
                        'original_id': row.get('id', ''),
                        'bias_labels': bias_labels
                    }
                }
            else:
                # Generic format
                example = {
                    'text': text,
                    'labels': bias_labels,
                    'source': 'babe',
                    'id': row.get('id', '')
                }
            
            training_examples.append(example)
        
        logger.info(f"Prepared {len(training_examples)} examples for {output_format} pretraining")
        return training_examples
    
    def export_for_validation(
        self, 
        babe_df: pd.DataFrame, 
        output_filename: str
    ) -> str:
        """
        Export BABE dataset in format suitable for model validation
        
        Args:
            babe_df: BABE DataFrame
            output_filename: Output filename
        
        Returns:
            Path to exported file
        """
        validation_data = []
        
        for _, row in babe_df.iterrows():
            if 'text' not in row:
                continue
            
            val_example = {
                'id': row.get('id', ''),
                'text': row['text'],
                'ground_truth_labels': row.get('bias_labels', {}),
                'source': 'babe_validation',
                'created_at': row.get('created_at', ''),
                'text_length': len(row['text']),
                'original_data': row.get('original_data', {})
            }
            validation_data.append(val_example)
        
        # Save validation data
        filepath = self.json_storage.processed_dir / output_filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            import json
            for example in validation_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(validation_data)} validation examples to {filepath}")
        return str(filepath)