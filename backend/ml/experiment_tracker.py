"""
Experiment tracking with Tensorboard, Weights & Biases integration
Comprehensive logging for model comparison and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Optional, Union
import io
from PIL import Image

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("W&B not available. Install with: pip install wandb")

from .experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Comprehensive experiment tracking and logging"""
    
    def __init__(self, config: ExperimentConfig, log_dir: str = "runs"):
        self.config = config
        self.run_name = config.get_run_name()
        self.log_dir = Path(log_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.tensorboard_writer = None
        self.wandb_run = None
        
        if config.logging.tensorboard:
            self.tensorboard_writer = SummaryWriter(str(self.log_dir))
            logger.info(f"Tensorboard logging to: {self.log_dir}")
        
        if config.logging.wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project="vibenews-bias-detection",
                name=self.run_name,
                config=config.to_dict(),
                tags=config.tags
            )
            logger.info("W&B logging initialized")
        
        # Metrics storage
        self.metrics_history = []
        self.best_metrics = {}
        
        # Class names for visualization
        self.class_names = ["Non-biased", "Biased", "No agreement"]
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log scalar metrics to all trackers"""
        # Add prefix if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Store in history
        metrics_with_step = {"step": step, **metrics}
        self.metrics_history.append(metrics_with_step)
        
        # Tensorboard logging
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
        
        # W&B logging
        if self.wandb_run:
            self.wandb_run.log(metrics, step=step)
        
        # Update best metrics
        for key, value in metrics.items():
            if 'loss' in key.lower():
                # Lower is better for loss
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
            else:
                # Higher is better for most metrics
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
        
        logger.info(f"Step {step}: {metrics}")
    
    def log_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        step: int,
        title: str = "Confusion Matrix"
    ):
        """Log confusion matrix as image"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        # Log to tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(f"confusion_matrix/{title}", fig, step)
        
        # Log to W&B
        if self.wandb_run:
            self.wandb_run.log({f"confusion_matrix/{title}": wandb.Image(fig)}, step=step)
        
        # Save locally
        fig.savefig(self.log_dir / f"confusion_matrix_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Log classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        self._save_classification_report(report, step)
    
    def log_model_graph(self, model, sample_input):
        """Log model architecture graph"""
        if self.tensorboard_writer and self.config.logging.log_model_graph:
            try:
                self.tensorboard_writer.add_graph(model, sample_input)
                logger.info("Model graph logged to Tensorboard")
            except Exception as e:
                logger.warning(f"Failed to log model graph: {e}")
    
    def log_learning_curves(self, step: int):
        """Plot and log learning curves"""
        if len(self.metrics_history) < 2:
            return
        
        # Extract metrics over time
        steps = [m['step'] for m in self.metrics_history]
        
        # Plot training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress')
        
        # Loss curves
        if any('train/loss' in m for m in self.metrics_history):
            train_losses = [m.get('train/loss', np.nan) for m in self.metrics_history]
            val_losses = [m.get('eval/loss', np.nan) for m in self.metrics_history]
            
            axes[0, 0].plot(steps, train_losses, label='Train Loss', color='blue')
            axes[0, 0].plot(steps, val_losses, label='Val Loss', color='orange')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy curves
        if any('train/accuracy' in m for m in self.metrics_history):
            train_acc = [m.get('train/accuracy', np.nan) for m in self.metrics_history]
            val_acc = [m.get('eval/accuracy', np.nan) for m in self.metrics_history]
            
            axes[0, 1].plot(steps, train_acc, label='Train Acc', color='blue')
            axes[0, 1].plot(steps, val_acc, label='Val Acc', color='orange')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # F1 scores
        if any('train/f1_macro' in m for m in self.metrics_history):
            train_f1 = [m.get('train/f1_macro', np.nan) for m in self.metrics_history]
            val_f1 = [m.get('eval/f1_macro', np.nan) for m in self.metrics_history]
            
            axes[1, 0].plot(steps, train_f1, label='Train F1', color='blue')
            axes[1, 0].plot(steps, val_f1, label='Val F1', color='orange')
            axes[1, 0].set_title('F1 Score (Macro)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate
        if any('learning_rate' in m for m in self.metrics_history):
            lrs = [m.get('learning_rate', np.nan) for m in self.metrics_history]
            axes[1, 1].plot(steps, lrs, color='green')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Log to tensorboard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure("learning_curves", fig, step)
        
        # Log to W&B
        if self.wandb_run:
            self.wandb_run.log({"learning_curves": wandb.Image(fig)}, step=step)
        
        # Save locally
        fig.savefig(self.log_dir / f"learning_curves_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def log_predictions_sample(
        self, 
        texts: List[str], 
        y_true: List[int], 
        y_pred: List[int], 
        probabilities: List[List[float]],
        step: int,
        num_samples: int = 10
    ):
        """Log sample predictions for qualitative analysis"""
        samples = []
        
        for i in range(min(num_samples, len(texts))):
            sample = {
                'text': texts[i][:200] + "..." if len(texts[i]) > 200 else texts[i],
                'true_label': self.class_names[y_true[i]],
                'predicted_label': self.class_names[y_pred[i]],
                'confidence': max(probabilities[i]),
                'probabilities': {
                    self.class_names[j]: prob for j, prob in enumerate(probabilities[i])
                },
                'correct': y_true[i] == y_pred[i]
            }
            samples.append(sample)
        
        # Save samples
        samples_file = self.log_dir / f"predictions_sample_step_{step}.json"
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        # Log to W&B as table
        if self.wandb_run:
            table_data = []
            for sample in samples:
                table_data.append([
                    sample['text'],
                    sample['true_label'],
                    sample['predicted_label'],
                    sample['confidence'],
                    sample['correct']
                ])
            
            table = wandb.Table(
                columns=["Text", "True Label", "Predicted", "Confidence", "Correct"],
                data=table_data
            )
            self.wandb_run.log({f"predictions_sample_step_{step}": table}, step=step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with final metrics"""
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)
        
        # Save hyperparameters
        hparams_file = self.log_dir / "hparams.json"
        with open(hparams_file, 'w') as f:
            json.dump(hparams, f, indent=2, default=str)
    
    def save_final_results(self, results: Dict[str, Any]):
        """Save final experiment results"""
        # Add best metrics to results
        results['best_metrics'] = self.best_metrics
        results['config'] = self.config.to_dict()
        
        # Save results
        results_file = self.log_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to: {results_file}")
        
        # Log final metrics to W&B
        if self.wandb_run:
            self.wandb_run.summary.update(results)
    
    def close(self):
        """Close all loggers"""
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        logger.info(f"Experiment tracking closed for: {self.run_name}")
    
    def _save_classification_report(self, report: Dict[str, Any], step: int):
        """Save detailed classification report"""
        report_file = self.log_dir / f"classification_report_step_{step}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Extract per-class metrics for logging
        per_class_metrics = {}
        for class_name in self.class_names:
            if class_name in report:
                class_metrics = report[class_name]
                per_class_metrics[f"precision/{class_name}"] = class_metrics['precision']
                per_class_metrics[f"recall/{class_name}"] = class_metrics['recall']
                per_class_metrics[f"f1/{class_name}"] = class_metrics['f1-score']
        
        # Log per-class metrics
        if per_class_metrics:
            self.log_metrics(per_class_metrics, step, prefix="per_class")


class ExperimentComparison:
    """Compare multiple experiment runs"""
    
    def __init__(self, run_dirs: List[str]):
        self.run_dirs = [Path(d) for d in run_dirs]
        self.runs_data = {}
        self._load_runs_data()
    
    def _load_runs_data(self):
        """Load data from all runs"""
        for run_dir in self.run_dirs:
            run_name = run_dir.name
            
            # Load results
            results_file = run_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    self.runs_data[run_name] = json.load(f)
            else:
                logger.warning(f"No results found for run: {run_name}")
    
    def generate_comparison_report(self, save_path: str = "experiment_comparison.html"):
        """Generate HTML comparison report"""
        # This would generate a comprehensive HTML report
        # comparing all experiments with charts and tables
        pass
    
    def get_best_run(self, metric: str = "eval/f1_macro") -> str:
        """Get name of best performing run based on metric"""
        best_run = None
        best_value = float('-inf') if 'loss' not in metric else float('inf')
        
        for run_name, data in self.runs_data.items():
            if 'best_metrics' in data and metric in data['best_metrics']:
                value = data['best_metrics'][metric]
                
                if 'loss' in metric:
                    if value < best_value:
                        best_value = value
                        best_run = run_name
                else:
                    if value > best_value:
                        best_value = value
                        best_run = run_name
        
        return best_run