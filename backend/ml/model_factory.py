"""
Model factory for creating different bias classification architectures
Supports BERT, DistilBERT, RoBERTa, and other transformer models
"""

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    ElectraForSequenceClassification, ElectraTokenizer,
    AutoModel, AutoTokenizer
)
from typing import Dict, Any, Tuple, Optional
import logging

from .experiment_config import ModelConfig

logger = logging.getLogger(__name__)


class BaseBiasClassifier(nn.Module):
    """Base class for bias classification models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.num_labels = config.num_labels
        
    def get_tokenizer(self):
        """Get appropriate tokenizer for the model"""
        raise NotImplementedError
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        raise NotImplementedError
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        return {
            'architecture': self.config.architecture,
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class BertBiasClassifier(BaseBiasClassifier):
    """BERT-based bias classifier"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = BertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
        
        # Freeze layers if specified
        self._freeze_layers(config.freeze_layers)
    
    def get_tokenizer(self):
        return BertTokenizer.from_pretrained(self.model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers"""
        if num_layers > 0:
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            
            for layer in self.model.bert.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logger.info(f"Frozen first {num_layers} BERT layers")


class DistilBertBiasClassifier(BaseBiasClassifier):
    """DistilBERT-based bias classifier (faster, smaller)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            dropout=config.dropout
        )
        
        self._freeze_layers(config.freeze_layers)
    
    def get_tokenizer(self):
        return DistilBertTokenizer.from_pretrained(self.model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers"""
        if num_layers > 0:
            for param in self.model.distilbert.embeddings.parameters():
                param.requires_grad = False
            
            for layer in self.model.distilbert.transformer.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logger.info(f"Frozen first {num_layers} DistilBERT layers")


class RobertaBiasClassifier(BaseBiasClassifier):
    """RoBERTa-based bias classifier"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = RobertaForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
        
        self._freeze_layers(config.freeze_layers)
    
    def get_tokenizer(self):
        return RobertaTokenizer.from_pretrained(self.model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers"""
        if num_layers > 0:
            for param in self.model.roberta.embeddings.parameters():
                param.requires_grad = False
            
            for layer in self.model.roberta.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logger.info(f"Frozen first {num_layers} RoBERTa layers")


class ElectraBiasClassifier(BaseBiasClassifier):
    """ELECTRA-based bias classifier"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = ElectraForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            hidden_dropout_prob=config.dropout,
            attention_probs_dropout_prob=config.dropout
        )
        
        self._freeze_layers(config.freeze_layers)
    
    def get_tokenizer(self):
        return ElectraTokenizer.from_pretrained(self.model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers"""
        if num_layers > 0:
            for param in self.model.electra.embeddings.parameters():
                param.requires_grad = False
            
            for layer in self.model.electra.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logger.info(f"Frozen first {num_layers} ELECTRA layers")


class CustomBiasClassifier(BaseBiasClassifier):
    """Custom architecture with configurable components"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        # Load base transformer
        self.transformer = AutoModel.from_pretrained(config.model_name)
        
        # Custom classification head
        hidden_size = config.hidden_size or self.transformer.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_size // 2, config.num_labels)
        )
        
        self._freeze_layers(config.freeze_layers)
    
    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return type('ModelOutput', (), {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        })()
    
    def _freeze_layers(self, num_layers: int):
        """Freeze first N transformer layers"""
        if num_layers > 0:
            # This is generic - specific implementation depends on model architecture
            layers = list(self.transformer.children())
            for layer in layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logger.info(f"Frozen first {num_layers} transformer layers")


class ModelFactory:
    """Factory for creating bias classification models"""
    
    # Registry of available models
    MODEL_REGISTRY = {
        'bert': BertBiasClassifier,
        'distilbert': DistilBertBiasClassifier,
        'roberta': RobertaBiasClassifier,
        'electra': ElectraBiasClassifier,
        'custom': CustomBiasClassifier
    }
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseBiasClassifier:
        """Create model instance based on configuration"""
        architecture = config.architecture.lower()
        
        if architecture not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unsupported architecture: {architecture}. "
                           f"Available: {list(cls.MODEL_REGISTRY.keys())}")
        
        model_class = cls.MODEL_REGISTRY[architecture]
        model = model_class(config)
        
        logger.info(f"Created {architecture} model: {config.model_name}")
        logger.info(f"Model info: {model.get_model_info()}")
        
        return model
    
    @classmethod
    def get_available_architectures(cls) -> list:
        """Get list of available model architectures"""
        return list(cls.MODEL_REGISTRY.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: type):
        """Register new model architecture"""
        cls.MODEL_REGISTRY[name] = model_class
        logger.info(f"Registered new model architecture: {name}")


# Predefined model configurations
class ModelConfigs:
    """Predefined model configurations for common use cases"""
    
    @staticmethod
    def bert_base() -> ModelConfig:
        return ModelConfig(
            architecture="bert",
            model_name="bert-base-uncased",
            num_labels=3,
            dropout=0.1
        )
    
    @staticmethod
    def bert_large() -> ModelConfig:
        return ModelConfig(
            architecture="bert",
            model_name="bert-large-uncased",
            num_labels=3,
            dropout=0.1
        )
    
    @staticmethod
    def distilbert_base() -> ModelConfig:
        return ModelConfig(
            architecture="distilbert",
            model_name="distilbert-base-uncased",
            num_labels=3,
            dropout=0.1
        )
    
    @staticmethod
    def roberta_base() -> ModelConfig:
        return ModelConfig(
            architecture="roberta",
            model_name="roberta-base",
            num_labels=3,
            dropout=0.1
        )
    
    @staticmethod
    def multilingual_bert() -> ModelConfig:
        return ModelConfig(
            architecture="bert",
            model_name="bert-base-multilingual-cased",
            num_labels=3,
            dropout=0.1
        )
    
    @staticmethod
    def domain_adapted_bert() -> ModelConfig:
        """BERT fine-tuned on news/political text"""
        return ModelConfig(
            architecture="bert",
            model_name="nlptown/bert-base-multilingual-uncased-sentiment",
            num_labels=3,
            dropout=0.2
        )