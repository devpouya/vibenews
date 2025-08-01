"""
Cloud-optimized model factory for Vertex AI
Lightweight versions of bias classification models
"""

import logging
from typing import Tuple, Dict, Any
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from trainer.config import ModelConfig, VertexExperimentConfig
from trainer.data import CloudBiasDataset

logger = logging.getLogger(__name__)


class BiasModelFactory:
    """Factory for creating bias classification models optimized for Vertex AI"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def create_model_and_tokenizer(self) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Create model and tokenizer based on config"""
        
        logger.info(f"Creating {self.config.architecture} model: {self.config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True
        )
        
        # Load model with architecture-specific dropout parameters
        model_kwargs = {
            'num_labels': self.config.num_labels,
        }
        
        # Different architectures use different dropout parameter names
        if 'distilbert' in self.config.model_name.lower():
            model_kwargs['dropout'] = self.config.dropout
            model_kwargs['attention_dropout'] = self.config.dropout
        elif 'bert' in self.config.model_name.lower():
            model_kwargs['hidden_dropout_prob'] = self.config.dropout
            model_kwargs['attention_probs_dropout_prob'] = self.config.dropout
        elif 'roberta' in self.config.model_name.lower():
            model_kwargs['hidden_dropout_prob'] = self.config.dropout
            model_kwargs['attention_probs_dropout_prob'] = self.config.dropout
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Apply layer freezing if specified
        if self.config.freeze_layers > 0:
            self._freeze_layers(model, self.config.freeze_layers)
        
        # Move to device
        model = model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model, tokenizer
    
    def _freeze_layers(self, model, num_layers: int):
        """Freeze specified number of layers"""
        try:
            # Handle different model architectures
            if hasattr(model, 'bert'):
                encoder_layers = model.bert.encoder.layer
                embeddings = model.bert.embeddings
            elif hasattr(model, 'distilbert'):
                encoder_layers = model.distilbert.transformer.layer
                embeddings = model.distilbert.embeddings
            elif hasattr(model, 'roberta'):
                encoder_layers = model.roberta.encoder.layer
                embeddings = model.roberta.embeddings
            else:
                logger.warning("Unknown model architecture for layer freezing")
                return
            
            # Freeze embeddings
            for param in embeddings.parameters():
                param.requires_grad = False
            
            # Freeze specified layers
            for layer in encoder_layers[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
            
            logger.info(f"Frozen first {num_layers} layers + embeddings")
            
        except Exception as e:
            logger.warning(f"Could not freeze layers: {e}")
    
    def create_trainer(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        train_dataset: CloudBiasDataset,
        eval_dataset: CloudBiasDataset,
        config: VertexExperimentConfig,
        model_dir: str,
        tracker=None
    ) -> Trainer:
        """Create Hugging Face trainer for Vertex AI"""
        
        # Training arguments optimized for Vertex AI
        training_args = TrainingArguments(
            output_dir=model_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=config.training.epochs,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.eval_batch_size,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            warmup_steps=config.training.warmup_steps,
            max_grad_norm=config.training.max_grad_norm,
            
            # Evaluation and logging
            eval_strategy="steps",
            eval_steps=config.logging.eval_steps,
            logging_steps=config.logging.log_steps,
            save_steps=config.logging.save_steps,
            
            # Model saving
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            
            # Vertex AI optimizations
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            
            # Mixed precision for speed (if supported)
            fp16=config.training.fp16,
            
            # Reporting (tensorboard if enabled, otherwise none)
            report_to=["tensorboard"] if config.logging.tensorboard else [],
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self._compute_metrics,
        )
        
        # Add custom callback for cloud logging
        if tracker:
            trainer.add_callback(CloudLoggingCallback(tracker))
        
        logger.info("Trainer created successfully")
        return trainer
    
    def _compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
        }
        
        # Add per-class metrics
        class_names = ["non_biased", "biased", "no_agreement"]
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                metrics[f'precision_{class_name}'] = precision[i]
                metrics[f'recall_{class_name}'] = recall[i]
                metrics[f'f1_{class_name}'] = f1[i]
        
        return metrics


class CloudLoggingCallback:
    """Custom callback for cloud experiment tracking"""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs and self.tracker:
            step = state.global_step
            
            # Separate train and eval metrics
            train_metrics = {k.replace('train_', ''): v for k, v in logs.items() if k.startswith('train_')}
            eval_metrics = {k.replace('eval_', ''): v for k, v in logs.items() if k.startswith('eval_')}
            
            if train_metrics:
                self.tracker.log_metrics(train_metrics, step, prefix="train")
            if eval_metrics:
                self.tracker.log_metrics(eval_metrics, step, prefix="eval")
    
    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after evaluation"""
        if logs and self.tracker:
            step = state.global_step
            eval_metrics = {k.replace('eval_', ''): v for k, v in logs.items() if k.startswith('eval_')}
            
            if eval_metrics:
                self.tracker.log_metrics(eval_metrics, step, prefix="eval")


def get_model_recommendations(budget_per_hour: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """Get model recommendations based on budget"""
    
    recommendations = {
        "ultra_cheap": {
            "model_name": "distilbert-base-uncased",
            "architecture": "distilbert", 
            "batch_size": 32,
            "machine_type": "e2-standard-2",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "estimated_cost_per_hour": 0.25,
            "expected_performance": "75-78% F1",
            "training_time": "1-1.5 hours"
        },
        "cheap": {
            "model_name": "bert-base-uncased",
            "architecture": "bert",
            "batch_size": 16,
            "machine_type": "n1-standard-4", 
            "accelerator_type": "NVIDIA_TESLA_T4",
            "estimated_cost_per_hour": 0.54,
            "expected_performance": "78-82% F1",
            "training_time": "2-2.5 hours"
        },
        "balanced": {
            "model_name": "roberta-base",
            "architecture": "roberta",
            "batch_size": 16,
            "machine_type": "n1-standard-4",
            "accelerator_type": "NVIDIA_TESLA_V100", 
            "estimated_cost_per_hour": 2.74,
            "expected_performance": "80-84% F1",
            "training_time": "1.5-2 hours"
        }
    }
    
    # Filter by budget
    affordable = {k: v for k, v in recommendations.items() 
                 if v["estimated_cost_per_hour"] <= budget_per_hour}
    
    return affordable if affordable else {"ultra_cheap": recommendations["ultra_cheap"]}