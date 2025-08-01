#!/usr/bin/env python3
"""
Main experiment runner script
Supports local and cloud training with comprehensive tracking
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from ml.experiment_config import ExperimentManager, ExperimentConfig
from ml.model_factory import ModelFactory
from ml.experiment_tracker import ExperimentTracker
from ml.bias_classifier import BABEBiasDataset  # Updated import
from datasets.babe_utils import BABEValidator
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner with tracking and cloud support"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.experiment_manager = ExperimentManager()
        self.tracker = ExperimentTracker(config)
        self.babe_validator = BABEValidator()
        
        # Create run directory
        self.run_dir = self.experiment_manager.create_run_dir(config)
        
        logger.info(f"Starting experiment: {config.get_run_name()}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Run directory: {self.run_dir}")
    
    def load_data(self):
        """Load and prepare dataset"""
        logger.info("Loading BABE dataset...")
        
        # Load BABE data
        babe_df = self.babe_validator.load_babe_data('babe_with_annotations_20250731.jsonl')
        logger.info(f"Loaded {len(babe_df)} samples")
        
        # Extract texts and labels
        texts = []
        labels = []
        label_map = {"Non-biased": 0, "Biased": 1, "No agreement": 2}
        
        for _, row in babe_df.iterrows():
            if 'text' not in row or 'bias_labels' not in row:
                continue
            
            text = row['text']
            bias_label = row['bias_labels'].get('label_bias', '')
            
            if bias_label not in label_map:
                continue
            
            # Apply data config filters
            if self.config.data.filter_no_agreement and bias_label == "No agreement":
                continue
            
            texts.append(text)
            labels.append(label_map[bias_label])
        
        logger.info(f"Prepared {len(texts)} samples after filtering")
        
        # Split data
        split_idx = int(len(texts) * self.config.data.train_split)
        train_texts = texts[:split_idx]
        train_labels = labels[:split_idx]
        eval_texts = texts[split_idx:]
        eval_labels = labels[split_idx:]
        
        logger.info(f"Train: {len(train_texts)}, Eval: {len(eval_texts)}")
        
        return train_texts, train_labels, eval_texts, eval_labels
    
    def create_model_and_tokenizer(self):
        """Create model and tokenizer"""
        logger.info("Creating model...")
        
        model = ModelFactory.create_model(self.config.model)
        tokenizer = model.get_tokenizer()
        
        # Move to device
        model = model.to(self.device)
        
        # Log model info
        model_info = model.get_model_info()
        self.tracker.log_metrics(
            {f"model/{k}": v for k, v in model_info.items() if isinstance(v, (int, float))},
            step=0
        )
        
        return model, tokenizer
    
    def create_datasets(self, train_texts, train_labels, eval_texts, eval_labels, tokenizer):
        """Create PyTorch datasets"""
        train_dataset = BABEBiasDataset(
            train_texts, train_labels, tokenizer, self.config.data.max_length
        )
        eval_dataset = BABEBiasDataset(
            eval_texts, eval_labels, tokenizer, self.config.data.max_length
        )
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
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
        class_names = ["Non-biased", "Biased", "No agreement"]
        for i, class_name in enumerate(class_names):
            if i < len(precision):
                metrics[f'precision_{class_name}'] = precision[i]
                metrics[f'recall_{class_name}'] = recall[i]
                metrics[f'f1_{class_name}'] = f1[i]
        
        return metrics
    
    def setup_trainer(self, model, train_dataset, eval_dataset):
        """Setup Hugging Face trainer"""
        
        training_args = TrainingArguments(
            output_dir=str(self.run_dir / "checkpoints"),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.eval_batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_dir=str(self.run_dir / "logs"),
            logging_steps=self.config.logging.log_steps,
            eval_strategy="steps",
            eval_steps=self.config.logging.eval_steps,
            save_steps=self.config.logging.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to=["tensorboard"] if self.config.logging.tensorboard else [],
            run_name=self.config.get_run_name(),
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        return trainer
    
    def run_training(self):
        """Execute full training pipeline"""
        try:
            # Load data
            train_texts, train_labels, eval_texts, eval_labels = self.load_data()
            
            # Create model and tokenizer
            model, tokenizer = self.create_model_and_tokenizer()
            
            # Log model graph
            if self.config.logging.log_model_graph:
                sample_input = tokenizer(
                    "Sample text for graph",
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.data.max_length
                ).to(self.device)
                self.tracker.log_model_graph(model, sample_input['input_ids'])
            
            # Create datasets
            train_dataset, eval_dataset = self.create_datasets(
                train_texts, train_labels, eval_texts, eval_labels, tokenizer
            )
            
            # Setup trainer
            trainer = self.setup_trainer(model, train_dataset, eval_dataset)
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Final evaluation
            logger.info("Running final evaluation...")
            eval_result = trainer.evaluate()
            
            # Detailed evaluation with confusion matrix
            predictions = trainer.predict(eval_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = eval_labels
            probabilities = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
            
            # Log confusion matrix
            self.tracker.log_confusion_matrix(
                y_true, y_pred, step=trainer.state.global_step, title="Final Evaluation"
            )
            
            # Log prediction samples
            self.tracker.log_predictions_sample(
                eval_texts[:20], y_true[:20], y_pred[:20], probabilities[:20].tolist(),
                step=trainer.state.global_step
            )
            
            # Log learning curves
            self.tracker.log_learning_curves(trainer.state.global_step)
            
            # Save model
            model_path = self.run_dir / "final_model"
            trainer.save_model(str(model_path))
            tokenizer.save_pretrained(str(model_path))
            
            # Prepare final results
            final_results = {
                'train_result': train_result.metrics,
                'eval_result': eval_result,
                'model_path': str(model_path),
                'config': self.config.to_dict()
            }
            
            # Log hyperparameters
            hparams = {
                'learning_rate': self.config.training.learning_rate,
                'batch_size': self.config.training.batch_size,
                'epochs': self.config.training.epochs,
                'model_name': self.config.model.model_name,
                'architecture': self.config.model.architecture
            }
            self.tracker.log_hyperparameters(hparams, eval_result)
            
            # Save final results
            self.tracker.save_final_results(final_results)
            
            logger.info("Training completed successfully!")
            logger.info(f"Final metrics: {eval_result}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.tracker.close()
    
    def run_cloud_training(self):
        """Submit training job to cloud platform"""
        # This would implement cloud submission logic
        # For now, just run locally
        logger.info("Cloud training not implemented yet, running locally...")
        return self.run_training()


def main():
    parser = argparse.ArgumentParser(description="Run bias classification experiments")
    parser.add_argument("config", help="Path to experiment config file")
    parser.add_argument("--cloud", action="store_true", help="Run on cloud platform")
    parser.add_argument("--local", action="store_true", help="Force local training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Create and run experiment
    runner = ExperimentRunner(config)
    
    if args.cloud and not args.local:
        results = runner.run_cloud_training()
    else:
        results = runner.run_training()
    
    print(f"\n✅ Experiment completed: {config.get_run_name()}")
    print(f"📊 Results: {results['eval_result']}")
    print(f"💾 Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()