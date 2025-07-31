"""
BiasScanner Evaluation Framework using BABE Dataset
Validates BiasScanner performance against expert-annotated bias data
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging

from backend.bias_detection.biasscanner_pipeline import BiasDetectionPipeline
from backend.bias_detection.bias_types import BiasType

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for bias detection performance"""
    
    # Basic classification metrics
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    # Calculated metrics
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    
    # Additional metrics
    total_samples: int
    biased_samples: int
    non_biased_samples: int
    
    # Bias type specific metrics
    bias_type_performance: Dict[str, Dict[str, float]]


class BiasEvaluator:
    """
    Evaluation framework for BiasScanner using BABE dataset
    Provides comprehensive performance analysis and validation
    """
    
    def __init__(self, gemini_api_key: str, babe_dataset_path: str):
        """Initialize evaluator with BiasScanner pipeline and BABE dataset"""
        
        self.pipeline = BiasDetectionPipeline(gemini_api_key)
        self.babe_dataset_path = Path(babe_dataset_path)
        
        if not self.babe_dataset_path.exists():
            raise FileNotFoundError(f"BABE dataset not found: {babe_dataset_path}")
        
        logger.info(f"Initialized BiasEvaluator with BABE dataset: {babe_dataset_path}")
    
    def load_babe_dataset(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load BABE dataset for evaluation"""
        
        logger.info("Loading BABE dataset...")
        
        # Load from JSON Lines format
        data = []
        with open(self.babe_dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {i}: {e}")
                    continue
        
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples from BABE dataset")
        
        return df
    
    def prepare_evaluation_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare BABE data for BiasScanner evaluation
        Maps BABE labels to binary bias classification
        """
        
        evaluation_samples = []
        
        for _, row in df.iterrows():
            # Extract BABE bias labels
            bias_labels = row.get('bias_labels', {})
            label_bias = bias_labels.get('label_bias', '')
            
            # Map BABE labels to binary classification
            is_biased_ground_truth = self._map_babe_to_binary(label_bias)
            
            sample = {
                'id': row.get('id', ''),
                'text': row.get('text', ''),
                'ground_truth_biased': is_biased_ground_truth,
                'ground_truth_label': label_bias,
                'babe_metadata': {
                    'opinion_label': bias_labels.get('label_opinion', ''),
                    'outlet_type': bias_labels.get('outlet_type', ''),
                    'topic': row.get('topic', ''),
                    'outlet': row.get('outlet', '')
                }
            }
            
            evaluation_samples.append(sample)
        
        logger.info(f"Prepared {len(evaluation_samples)} samples for evaluation")
        return evaluation_samples
    
    def _map_babe_to_binary(self, babe_label: str) -> bool:
        """Map BABE bias labels to binary classification"""
        
        # BABE uses: "Biased", "Non-biased", "No agreement"
        biased_labels = ["Biased", "biased"]
        return babe_label in biased_labels
    
    def evaluate_biasscanner(self, evaluation_samples: List[Dict], 
                           batch_size: int = 10) -> EvaluationMetrics:
        """
        Evaluate BiasScanner performance against BABE ground truth
        
        Args:
            evaluation_samples: Prepared BABE samples
            batch_size: Number of samples to process in each batch
            
        Returns:
            Comprehensive evaluation metrics
        """
        
        logger.info(f"Starting BiasScanner evaluation on {len(evaluation_samples)} samples")
        
        predictions = []
        ground_truth = []
        bias_type_predictions = []
        
        # Process samples in batches (due to API rate limits)
        for i in range(0, len(evaluation_samples), batch_size):
            batch = evaluation_samples[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(evaluation_samples) + batch_size - 1)//batch_size}")
            
            for sample in batch:
                try:
                    # Run BiasScanner on sample
                    article_data = {
                        'content': sample['text'],
                        'title': f"Sample {sample['id']}",
                        'url': '',
                        'source': 'babe_evaluation'
                    }
                    
                    result = self.pipeline.process_article(article_data)
                    
                    # Extract prediction
                    bias_analysis = result.get('bias_analysis', {})
                    
                    if bias_analysis.get('success', False):
                        bias_score = bias_analysis.get('bias_score', {})
                        predicted_biased = bias_score.get('overall_score', 0.0) > 0.3  # Threshold
                        
                        predictions.append(predicted_biased)
                        ground_truth.append(sample['ground_truth_biased'])
                        
                        # Track bias types
                        detected_types = bias_analysis.get('bias_types_detected', [])
                        bias_type_predictions.append(detected_types)
                        
                    else:
                        # Failed prediction - count as non-biased
                        predictions.append(False)
                        ground_truth.append(sample['ground_truth_biased'])
                        bias_type_predictions.append([])
                        
                        logger.warning(f"Failed to analyze sample {sample['id']}")
                
                except Exception as e:
                    logger.error(f"Error processing sample {sample['id']}: {e}")
                    # Failed prediction - count as non-biased
                    predictions.append(False)
                    ground_truth.append(sample['ground_truth_biased'])
                    bias_type_predictions.append([])
        
        # Calculate evaluation metrics
        metrics = self._calculate_metrics(predictions, ground_truth, bias_type_predictions)
        
        logger.info(f"Evaluation completed - F1 Score: {metrics.f1_score:.3f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions: List[bool], ground_truth: List[bool],
                          bias_type_predictions: List[List[str]]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""
        
        # Convert to numpy arrays for easier calculation
        pred_array = np.array(predictions)
        truth_array = np.array(ground_truth)
        
        # Calculate confusion matrix components
        tp = np.sum((pred_array == True) & (truth_array == True))
        fp = np.sum((pred_array == True) & (truth_array == False))
        fn = np.sum((pred_array == False) & (truth_array == True))
        tn = np.sum((pred_array == False) & (truth_array == False))
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        
        # Bias type performance (simplified)
        bias_type_performance = self._analyze_bias_type_performance(bias_type_predictions)
        
        return EvaluationMetrics(
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn),
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            total_samples=len(predictions),
            biased_samples=int(np.sum(truth_array)),
            non_biased_samples=int(np.sum(~truth_array)),
            bias_type_performance=bias_type_performance
        )
    
    def _analyze_bias_type_performance(self, bias_type_predictions: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by bias type"""
        
        # Count bias type occurrences
        type_counts = {}
        total_predictions = len(bias_type_predictions)
        
        for prediction in bias_type_predictions:
            for bias_type in prediction:
                type_counts[bias_type] = type_counts.get(bias_type, 0) + 1
        
        # Calculate relative frequencies
        type_performance = {}
        for bias_type, count in type_counts.items():
            type_performance[bias_type] = {
                'frequency': count / total_predictions if total_predictions > 0 else 0.0,
                'absolute_count': count
            }
        
        return type_performance
    
    def run_full_evaluation(self, sample_limit: Optional[int] = 100) -> Dict:
        """
        Run complete evaluation pipeline
        
        Args:
            sample_limit: Limit number of samples for evaluation (None for all)
            
        Returns:
            Complete evaluation report
        """
        
        start_time = datetime.now()
        logger.info("Starting full BiasScanner evaluation")
        
        try:
            # Load and prepare data
            df = self.load_babe_dataset(limit=sample_limit)
            evaluation_samples = self.prepare_evaluation_data(df)
            
            # Run evaluation
            metrics = self.evaluate_biasscanner(evaluation_samples)
            
            # Generate report
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            report = {
                "evaluation_summary": {
                    "algorithm": "BiasScanner v1.0.0",
                    "dataset": "BABE (Media Bias Annotations by Experts)",
                    "evaluation_date": start_time.isoformat(),
                    "duration_seconds": duration,
                    "sample_size": metrics.total_samples
                },
                
                "performance_metrics": {
                    "f1_score": metrics.f1_score,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "accuracy": metrics.accuracy
                },
                
                "confusion_matrix": {
                    "true_positives": metrics.true_positives,
                    "false_positives": metrics.false_positives,
                    "false_negatives": metrics.false_negatives,
                    "true_negatives": metrics.true_negatives
                },
                
                "dataset_statistics": {
                    "total_samples": metrics.total_samples,
                    "biased_samples": metrics.biased_samples,
                    "non_biased_samples": metrics.non_biased_samples,
                    "bias_ratio": metrics.biased_samples / metrics.total_samples if metrics.total_samples > 0 else 0.0
                },
                
                "bias_type_analysis": metrics.bias_type_performance,
                
                "comparison_to_paper": {
                    "paper_f1_score": 0.758,  # From BiasScanner paper
                    "our_f1_score": metrics.f1_score,
                    "performance_ratio": metrics.f1_score / 0.758 if 0.758 > 0 else 0.0,
                    "note": "Comparison to BiasScanner paper results (Table 1)"
                }
            }
            
            logger.info(f"Evaluation completed successfully in {duration:.1f}s")
            logger.info(f"Performance: F1={metrics.f1_score:.3f}, Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}")
            
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "error": str(e),
                "evaluation_date": start_time.isoformat(),
                "status": "failed"
            }
    
    def save_evaluation_report(self, report: Dict, output_path: str) -> str:
        """Save evaluation report to JSON file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation report to {output_file}")
        return str(output_file)