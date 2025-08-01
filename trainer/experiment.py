"""
Cloud experiment tracking for Vertex AI
Minimal tracking optimized for cost and performance
"""

import logging
import json
import os
from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from google.cloud import storage
import torch

from trainer.config import VertexExperimentConfig

logger = logging.getLogger(__name__)


class CloudExperimentTracker:
    """Lightweight experiment tracker for Vertex AI"""
    
    def __init__(
        self, 
        config: VertexExperimentConfig,
        tensorboard_log_dir: str = None,
        model_dir: str = None
    ):
        self.config = config
        self.tensorboard_log_dir = tensorboard_log_dir
        self.model_dir = model_dir
        
        # Initialize storage client
        self.storage_client = storage.Client()
        
        # Metrics storage
        self.metrics_history = []
        self.best_metrics = {}
        
        # Class names
        self.class_names = ["Non-biased", "Biased", "No agreement"]
        
        logger.info(f"Initialized CloudExperimentTracker")
        logger.info(f"Experiment: {config.experiment_name}")
        logger.info(f"Tensorboard: {tensorboard_log_dir}")
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log dataset information"""
        logger.info("Dataset Information:")
        logger.info(f"  Total samples: {data_info['total_samples']}")
        logger.info(f"  Train samples: {data_info['train_samples']}")
        logger.info(f"  Val samples: {data_info['val_samples']}")
        logger.info(f"  Classes: {data_info['num_classes']}")
        
        # Save data info
        self.data_info = data_info
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information"""
        logger.info("Model Information:")
        logger.info(f"  Architecture: {model_info['architecture']}")
        logger.info(f"  Model name: {model_info['model_name']}")
        logger.info(f"  Parameters: {model_info['num_parameters']:,}")
        logger.info(f"  Trainable: {model_info['trainable_parameters']:,}")
        
        # Save model info
        self.model_info = model_info
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics with optional prefix"""
        # Add prefix
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics
        
        # Store in history
        metrics_entry = {"step": step, **prefixed_metrics}
        self.metrics_history.append(metrics_entry)
        
        # Update best metrics
        for key, value in prefixed_metrics.items():
            if 'loss' in key.lower():
                # Lower is better for loss
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
            else:
                # Higher is better for most metrics
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
        
        # Log to console
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in prefixed_metrics.items()])
        logger.info(f"Step {step}: {metric_str}")
    
    def generate_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int],
        save_path: str = "confusion_matrix.png"
    ):
        """Generate and save confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.class_names,
                yticklabels=self.class_names
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Confusion matrix saved to: {save_path}")
            
            # Upload to GCS if model_dir is GCS path
            if self.model_dir and self.model_dir.startswith('gs://'):
                self._upload_file_to_gcs(save_path, f"analysis/confusion_matrix.png")
            
        except Exception as e:
            logger.error(f"Failed to generate confusion matrix: {e}")
    
    def generate_classification_report(
        self, 
        y_true: List[int], 
        y_pred: List[int],
        save_path: str = "classification_report.json"
    ):
        """Generate detailed classification report"""
        try:
            report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names, 
                output_dict=True
            )
            
            # Save report
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Classification report saved to: {save_path}")
            
            # Upload to GCS
            if self.model_dir and self.model_dir.startswith('gs://'):
                self._upload_file_to_gcs(save_path, f"analysis/classification_report.json")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate classification report: {e}")
            return {}
    
    def generate_training_curves(self, save_path: str = "training_curves.png"):
        """Generate training progress curves"""
        try:
            if len(self.metrics_history) < 2:
                logger.info("Not enough data for training curves")
                return
            
            # Extract data
            steps = [m['step'] for m in self.metrics_history]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Training Progress')
            
            # Loss curves
            train_losses = [m.get('train/loss', np.nan) for m in self.metrics_history]
            eval_losses = [m.get('eval/loss', np.nan) for m in self.metrics_history]
            
            if not all(np.isnan(train_losses)):
                axes[0, 0].plot(steps, train_losses, label='Train', color='blue')
                axes[0, 0].plot(steps, eval_losses, label='Validation', color='orange')
                axes[0, 0].set_title('Loss')
                axes[0, 0].set_xlabel('Step')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Accuracy curves
            train_acc = [m.get('train/accuracy', np.nan) for m in self.metrics_history]
            eval_acc = [m.get('eval/accuracy', np.nan) for m in self.metrics_history]
            
            if not all(np.isnan(eval_acc)):
                axes[0, 1].plot(steps, eval_acc, label='Validation', color='orange')
                if not all(np.isnan(train_acc)):
                    axes[0, 1].plot(steps, train_acc, label='Train', color='blue')
                axes[0, 1].set_title('Accuracy')
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # F1 curves
            eval_f1 = [m.get('eval/f1_macro', np.nan) for m in self.metrics_history]
            
            if not all(np.isnan(eval_f1)):
                axes[1, 0].plot(steps, eval_f1, label='F1 Macro', color='green')
                axes[1, 0].set_title('F1 Score')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Learning rate
            learning_rates = [m.get('train/learning_rate', np.nan) for m in self.metrics_history]
            
            if not all(np.isnan(learning_rates)):
                axes[1, 1].plot(steps, learning_rates, color='red')
                axes[1, 1].set_title('Learning Rate')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training curves saved to: {save_path}")
            
            # Upload to GCS
            if self.model_dir and self.model_dir.startswith('gs://'):
                self._upload_file_to_gcs(save_path, f"analysis/training_curves.png")
                
        except Exception as e:
            logger.error(f"Failed to generate training curves: {e}")
    
    def generate_prediction_samples(
        self,
        texts: List[str],
        y_true: List[int],
        y_pred: List[int],
        probabilities: List[List[float]],
        num_samples: int = 20,
        save_path: str = "prediction_samples.json"
    ):
        """Generate sample predictions for analysis"""
        try:
            samples = []
            
            for i in range(min(num_samples, len(texts))):
                sample = {
                    'text': texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                    'true_label': self.class_names[y_true[i]],
                    'predicted_label': self.class_names[y_pred[i]], 
                    'confidence': float(max(probabilities[i])),
                    'probabilities': {
                        self.class_names[j]: float(prob) 
                        for j, prob in enumerate(probabilities[i])
                    },
                    'correct': bool(y_true[i] == y_pred[i])
                }
                samples.append(sample)
            
            # Save samples
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Prediction samples saved to: {save_path}")
            
            # Upload to GCS
            if self.model_dir and self.model_dir.startswith('gs://'):
                self._upload_file_to_gcs(save_path, f"analysis/prediction_samples.json")
            
        except Exception as e:
            logger.error(f"Failed to generate prediction samples: {e}")
    
    def generate_final_analysis(self, trainer, eval_dataset) -> Dict[str, Any]:
        """Generate comprehensive final analysis"""
        try:
            logger.info("Generating final analysis...")
            
            # Get predictions
            predictions = trainer.predict(eval_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            probabilities = torch.softmax(torch.tensor(predictions.predictions), dim=1).numpy()
            
            # Generate visualizations
            self.generate_confusion_matrix(y_true, y_pred)
            report = self.generate_classification_report(y_true, y_pred)
            self.generate_training_curves()
            
            # Get sample texts for prediction analysis
            sample_texts = []
            sample_labels = []
            for i, item in enumerate(eval_dataset):
                if i >= 20:  # Limit to 20 samples
                    break
                # Decode text from tokenized input
                text = trainer.tokenizer.decode(item['input_ids'], skip_special_tokens=True)
                sample_texts.append(text)
                sample_labels.append(item['labels'].item())
            
            if sample_texts:
                self.generate_prediction_samples(
                    sample_texts[:20], 
                    sample_labels[:20],
                    y_pred[:20].tolist(), 
                    probabilities[:20].tolist()
                )
            
            # Compile analysis results
            analysis = {
                'confusion_matrix_generated': True,
                'classification_report': report,
                'training_curves_generated': True,
                'prediction_samples_generated': len(sample_texts) > 0,
                'final_metrics': {
                    'accuracy': float(report.get('accuracy', 0)),
                    'macro_f1': float(report.get('macro avg', {}).get('f1-score', 0)),
                    'weighted_f1': float(report.get('weighted avg', {}).get('f1-score', 0))
                }
            }
            
            logger.info("Final analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate final analysis: {e}")
            return {'error': str(e)}
    
    def _upload_file_to_gcs(self, local_path: str, gcs_blob_path: str):
        """Upload file to GCS"""
        try:
            if not self.model_dir.startswith('gs://'):
                return
            
            bucket_name = self.model_dir.split('/')[2]
            full_blob_path = '/'.join(self.model_dir.split('/')[3:]) + '/' + gcs_blob_path
            
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(full_blob_path)
            blob.upload_from_filename(local_path)
            
            logger.info(f"Uploaded {local_path} to gs://{bucket_name}/{full_blob_path}")
            
        except Exception as e:
            logger.error(f"Failed to upload file to GCS: {e}")
    
    def save_experiment_summary(self, final_results: Dict[str, Any]):
        """Save complete experiment summary"""
        try:
            summary = {
                'experiment_config': self.config.to_dict(),
                'data_info': getattr(self, 'data_info', {}),
                'model_info': getattr(self, 'model_info', {}),
                'best_metrics': self.best_metrics,
                'final_results': final_results,
                'metrics_history': self.metrics_history
            }
            
            # Save locally
            with open('experiment_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Upload to GCS
            if self.model_dir and self.model_dir.startswith('gs://'):
                self._upload_file_to_gcs('experiment_summary.json', 'experiment_summary.json')
            
            logger.info("Experiment summary saved")
            
        except Exception as e:
            logger.error(f"Failed to save experiment summary: {e}")