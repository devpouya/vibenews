"""
BERT-based bias classification pipeline for BABE dataset
No text normalization - preserves original bias signals
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import logging

logger = logging.getLogger(__name__)


class BABEBiasDataset(Dataset):
    """Dataset class for BABE bias classification"""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])  # Ensure string
        label = self.labels[idx]
        
        # Tokenize with no preprocessing - raw text preservation
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


class BiasClassifier:
    """BERT-based bias classifier with confidence scoring"""
    
    # Label mapping for BABE dataset
    LABEL_MAP = {
        "Non-biased": 0,
        "Biased": 1,
        "No agreement": 2
    }
    
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = None
        # Use CPU for compatibility (avoid MPS issues)
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_babe_data(self, babe_df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """
        Load and prepare BABE data for classification
        No text preprocessing - preserves original bias signals
        """
        texts = []
        labels = []
        
        for _, row in babe_df.iterrows():
            if 'text' not in row or 'bias_labels' not in row:
                continue
            
            text = row['text']
            bias_label = row['bias_labels'].get('label_bias', '')
            
            # Skip if no valid label
            if bias_label not in self.LABEL_MAP:
                continue
            
            # Raw text - no preprocessing
            texts.append(text)
            labels.append(self.LABEL_MAP[bias_label])
        
        logger.info(f"Loaded {len(texts)} samples")
        logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
        
        return texts, labels
    
    def create_datasets(
        self, 
        texts: List[str], 
        labels: List[int],
        train_split: float = 0.8
    ) -> Tuple[BABEBiasDataset, BABEBiasDataset]:
        """Create train/validation datasets"""
        
        # Split data
        split_idx = int(len(texts) * train_split)
        
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        val_texts = texts[split_idx:]
        val_labels = labels[split_idx:]
        
        # Create datasets
        train_dataset = BABEBiasDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = BABEBiasDataset(
            val_texts, val_labels, self.tokenizer
        )
        
        logger.info(f"Created datasets: {len(train_dataset)} train, {len(val_dataset)} val")
        return train_dataset, val_dataset
    
    def initialize_model(self, num_labels: int = 3):
        """Initialize BERT model with classification head"""
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        logger.info(f"Initialized BERT model with {num_labels} classes")
    
    def predict_with_confidence(self, texts: List[str]) -> List[Dict]:
        """
        Predict bias labels with confidence scores
        
        Returns:
            List of predictions with confidence metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call initialize_model() first.")
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize raw text (no preprocessing)
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Calculate probabilities and confidence
                probs = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1).item()
                max_prob = torch.max(probs).item()
                
                # Entropy-based uncertainty
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                
                prediction = {
                    'text': text,
                    'predicted_class': predicted_class,
                    'predicted_label': self.REVERSE_LABEL_MAP[predicted_class],
                    'confidence': max_prob,
                    'entropy': entropy,
                    'all_probabilities': {
                        self.REVERSE_LABEL_MAP[i]: prob.item() 
                        for i, prob in enumerate(probs[0])
                    }
                }
                predictions.append(prediction)
        
        return predictions
    
    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute evaluation metrics for training"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
        }
        
        # Add per-class metrics
        for i, label_name in self.REVERSE_LABEL_MAP.items():
            if i < len(precision):
                metrics[f'precision_{label_name}'] = precision[i]
                metrics[f'recall_{label_name}'] = recall[i]
                metrics[f'f1_{label_name}'] = f1[i]
        
        return metrics
    
    def setup_training(
        self,
        train_dataset: BABEBiasDataset,
        val_dataset: BABEBiasDataset,
        output_dir: str = "./bias_classifier_model",
        **training_kwargs
    ) -> Trainer:
        """Setup training pipeline (no actual training)"""
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            **training_kwargs
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Training pipeline setup complete (ready for training)")
        return trainer
    
    def analyze_sample(self, texts: List[str], labels: Optional[List[str]] = None):
        """Analyze a small sample for pipeline validation"""
        logger.info(f"Analyzing {len(texts)} samples:")
        
        for i, text in enumerate(texts[:3]):  # Show first 3
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Text: {text[:100]}...")
            logger.info(f"  Length: {len(text)} chars")
            if labels:
                logger.info(f"  Label: {labels[i]}")
            
            # Tokenize to check token count
            tokens = self.tokenizer.tokenize(text)
            logger.info(f"  Tokens: {len(tokens)}")