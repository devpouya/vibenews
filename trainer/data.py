"""
Cloud-native data loading for Vertex AI
Handles GCS data sources and distributed training
"""

import os
import logging
from typing import List, Tuple, Dict, Any
import pandas as pd
from pathlib import Path
import json
import io

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from google.cloud import storage
from sklearn.model_selection import train_test_split

from trainer.config import DataConfig

logger = logging.getLogger(__name__)


class CloudBiasDataset(Dataset):
    """Cloud-optimized dataset for bias classification"""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class CloudDataLoader:
    """Cloud-native data loader for GCS and other sources"""
    
    LABEL_MAP = {
        "Non-biased": 0,
        "Biased": 1,
        "No agreement": 2
    }
    
    def __init__(self, data_path: str, config: DataConfig):
        self.data_path = data_path
        self.config = config
        self.storage_client = storage.Client()
        
        logger.info(f"Initialized CloudDataLoader")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Data source: {config.data_source}")
    
    def load_from_gcs(self, gcs_path: str) -> pd.DataFrame:
        """Load data from Google Cloud Storage"""
        try:
            # Parse GCS path
            bucket_name = gcs_path.split('/')[2]
            blob_path = '/'.join(gcs_path.split('/')[3:])
            
            # Download data
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Determine file type and load accordingly
            if blob_path.endswith('.jsonl'):
                content = blob.download_as_text()
                data = []
                for line in content.strip().split('\n'):
                    if line.strip():
                        data.append(json.loads(line))
                df = pd.DataFrame(data)
            
            elif blob_path.endswith('.json'):
                content = blob.download_as_text()
                data = json.loads(content)
                df = pd.DataFrame(data)
            
            elif blob_path.endswith('.csv'):
                content = blob.download_as_text()
                df = pd.read_csv(io.StringIO(content))
            
            else:
                raise ValueError(f"Unsupported file format: {blob_path}")
            
            logger.info(f"Loaded {len(df)} samples from GCS: {gcs_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from GCS: {e}")
            raise
    
    def load_from_local(self, local_path: str) -> pd.DataFrame:
        """Load data from local file (for testing)"""
        try:
            if local_path.endswith('.jsonl'):
                data = []
                with open(local_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                df = pd.DataFrame(data)
            
            elif local_path.endswith('.json'):
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            
            elif local_path.endswith('.csv'):
                df = pd.read_csv(local_path)
            
            else:
                raise ValueError(f"Unsupported file format: {local_path}")
            
            logger.info(f"Loaded {len(df)} samples from local: {local_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from local: {e}")
            raise
    
    def load_babe_data(self) -> pd.DataFrame:
        """Load BABE dataset from configured source"""
        if self.config.data_source == "gcs":
            return self.load_from_gcs(self.data_path)
        elif self.config.data_source == "local":
            return self.load_from_local(self.data_path)
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")
    
    def extract_texts_and_labels(self, df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """Extract texts and labels from BABE dataframe"""
        texts = []
        labels = []
        
        logger.info(f"Processing {len(df)} samples...")
        
        for _, row in df.iterrows():
            # Handle different data formats
            if 'text' in row and 'bias_labels' in row:
                # BABE format
                text = row['text']
                bias_label = row['bias_labels'].get('label_bias', '') if isinstance(row['bias_labels'], dict) else ''
            elif 'content' in row and 'label' in row:
                # Alternative format
                text = row['content']
                bias_label = row['label']
            else:
                continue
            
            # Skip if no valid label
            if bias_label not in self.LABEL_MAP:
                continue
            
            # Apply filters
            if self.config.filter_no_agreement and bias_label == "No agreement":
                continue
            
            texts.append(text)
            labels.append(self.LABEL_MAP[bias_label])
        
        logger.info(f"Extracted {len(texts)} valid samples")
        
        # Log label distribution
        label_counts = {}
        for label in labels:
            label_name = [k for k, v in self.LABEL_MAP.items() if v == label][0]
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
        
        logger.info(f"Label distribution: {label_counts}")
        
        return texts, labels
    
    def apply_data_augmentation(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Apply data augmentation if configured"""
        if not self.config.augmentation:
            return texts, labels
        
        # Simple augmentation strategies
        augmented_texts = texts.copy()
        augmented_labels = labels.copy()
        
        # Example: Add slight variations for minority classes
        # This would be implemented based on specific needs
        logger.info("Data augmentation not implemented yet")
        
        return augmented_texts, augmented_labels
    
    def balance_classes(self, texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Balance classes if configured"""
        if not self.config.balance_classes:
            return texts, labels
        
        # Group by labels
        label_groups = {}
        for i, label in enumerate(labels):
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(i)
        
        # Find minimum class size
        min_size = min(len(indices) for indices in label_groups.values())
        
        # Sample equally from each class
        balanced_indices = []
        for indices in label_groups.values():
            balanced_indices.extend(torch.randperm(len(indices))[:min_size].tolist())
            balanced_indices = [indices[i] for i in balanced_indices[-min_size:]]
        
        # Extract balanced data
        balanced_texts = [texts[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]
        
        logger.info(f"Balanced dataset: {len(balanced_texts)} samples per class")
        
        return balanced_texts, balanced_labels
    
    def create_datasets(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer
    ) -> Tuple[CloudBiasDataset, CloudBiasDataset]:
        """Create train and validation datasets"""
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=1 - self.config.train_split,
            random_state=42,
            stratify=labels
        )
        
        logger.info(f"Data split: {len(train_texts)} train, {len(val_texts)} validation")
        
        # Create datasets
        train_dataset = CloudBiasDataset(
            train_texts, train_labels, tokenizer, self.config.max_length
        )
        val_dataset = CloudBiasDataset(
            val_texts, val_labels, tokenizer, self.config.max_length
        )
        
        return train_dataset, val_dataset
    
    def prepare_datasets(self, tokenizer=None) -> Tuple[CloudBiasDataset, CloudBiasDataset, Dict[str, Any]]:
        """Complete data preparation pipeline"""
        try:
            # Load data
            df = self.load_babe_data()
            
            # Extract texts and labels
            texts, labels = self.extract_texts_and_labels(df)
            
            # Apply preprocessing
            if self.config.augmentation:
                texts, labels = self.apply_data_augmentation(texts, labels)
            
            if self.config.balance_classes:
                texts, labels = self.balance_classes(texts, labels)
            
            # Use provided tokenizer or create default
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            # Create datasets
            train_dataset, val_dataset = self.create_datasets(texts, labels, tokenizer)
            
            # Prepare data info
            data_info = {
                'total_samples': len(texts),
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'num_classes': len(self.LABEL_MAP),
                'class_names': list(self.LABEL_MAP.keys()),
                'data_source': self.config.data_source,
                'data_path': self.data_path,
                'preprocessing': {
                    'max_length': self.config.max_length,
                    'augmentation': self.config.augmentation,
                    'balance_classes': self.config.balance_classes,
                    'filter_no_agreement': self.config.filter_no_agreement
                }
            }
            
            logger.info("Data preparation completed successfully")
            return train_dataset, val_dataset, data_info
            
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise
    
    def save_processed_data_to_gcs(self, texts: List[str], labels: List[int], gcs_path: str):
        """Save processed data back to GCS for caching"""
        try:
            # Prepare data for saving
            processed_data = []
            for text, label in zip(texts, labels):
                label_name = [k for k, v in self.LABEL_MAP.items() if v == label][0]
                processed_data.append({
                    'text': text,
                    'label': label_name,
                    'label_id': label
                })
            
            # Convert to JSON Lines
            jsonl_content = '\n'.join(json.dumps(item, ensure_ascii=False) for item in processed_data)
            
            # Upload to GCS
            bucket_name = gcs_path.split('/')[2]
            blob_path = '/'.join(gcs_path.split('/')[3:])
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(jsonl_content, content_type='application/jsonl')
            
            logger.info(f"Saved processed data to GCS: {gcs_path}")
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")


class DataValidator:
    """Validate data quality and consistency"""
    
    @staticmethod
    def validate_babe_format(df: pd.DataFrame) -> List[str]:
        """Validate BABE dataset format"""
        errors = []
        
        required_columns = ['text', 'bias_labels']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        if 'text' in df.columns:
            empty_texts = df['text'].isna().sum()
            if empty_texts > 0:
                errors.append(f"Found {empty_texts} empty text entries")
        
        if 'bias_labels' in df.columns:
            invalid_labels = 0
            for _, row in df.iterrows():
                if isinstance(row['bias_labels'], dict):
                    if 'label_bias' not in row['bias_labels']:
                        invalid_labels += 1
                else:
                    invalid_labels += 1
            
            if invalid_labels > 0:
                errors.append(f"Found {invalid_labels} invalid bias label entries")
        
        return errors
    
    @staticmethod
    def get_data_statistics(texts: List[str], labels: List[int]) -> Dict[str, Any]:
        """Get comprehensive data statistics"""
        stats = {
            'total_samples': len(texts),
            'text_lengths': {
                'min': min(len(text) for text in texts),
                'max': max(len(text) for text in texts),
                'mean': sum(len(text) for text in texts) / len(texts),
                'median': sorted([len(text) for text in texts])[len(texts) // 2]
            },
            'label_distribution': {},
            'class_balance_ratio': 0.0
        }
        
        # Label distribution
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        stats['label_distribution'] = label_counts
        
        # Class balance (ratio of smallest to largest class)
        if label_counts:
            min_count = min(label_counts.values())
            max_count = max(label_counts.values())
            stats['class_balance_ratio'] = min_count / max_count if max_count > 0 else 0.0
        
        return stats