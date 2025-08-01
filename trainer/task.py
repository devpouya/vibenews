"""
Main Vertex AI training task
Cloud-native entry point for bias classification training
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import json

import torch
from google.cloud import storage, aiplatform
from google.cloud.aiplatform import tensorboard

from trainer.model import BiasModelFactory
from trainer.data import CloudDataLoader
from trainer.experiment import CloudExperimentTracker
from trainer.config import VertexExperimentConfig

# Setup logging for cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VertexTrainingTask:
    """Main Vertex AI training task"""
    
    def __init__(self, args):
        self.args = args
        
        # Initialize Vertex AI
        aiplatform.init(
            project=args.project_id,
            location=args.region,
            staging_bucket=args.staging_bucket
        )
        
        # Setup paths
        self.model_dir = args.model_dir
        self.data_path = args.data_path
        self.tensorboard_log_dir = args.tensorboard_log_dir
        
        # Initialize components
        self.config = None
        self.data_loader = None
        self.model_factory = None
        self.tracker = None
        
        logger.info(f"Initialized Vertex AI training task")
        logger.info(f"Project: {args.project_id}")
        logger.info(f"Region: {args.region}")
        logger.info(f"Model dir: {self.model_dir}")
    
    def load_config(self):
        """Load experiment configuration from cloud storage"""
        try:
            if self.args.config_path.startswith('gs://'):
                # Load from GCS
                client = storage.Client()
                bucket_name = self.args.config_path.split('/')[2]
                blob_path = '/'.join(self.args.config_path.split('/')[3:])
                
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                config_content = blob.download_as_text()
                
                # Parse config
                import yaml
                config_dict = yaml.safe_load(config_content)
                self.config = VertexExperimentConfig.from_dict(config_dict)
            else:
                # Load from local file (for testing)
                self.config = VertexExperimentConfig.from_yaml(self.args.config_path)
            
            logger.info(f"Loaded config: {self.config.experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def setup_components(self):
        """Initialize training components"""
        try:
            # Data loader
            self.data_loader = CloudDataLoader(
                data_path=self.data_path,
                config=self.config.data
            )
            
            # Model factory
            self.model_factory = BiasModelFactory(self.config.model)
            
            # Experiment tracker
            self.tracker = CloudExperimentTracker(
                config=self.config,
                tensorboard_log_dir=self.tensorboard_log_dir,
                model_dir=self.model_dir
            )
            
            logger.info("Components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup components: {e}")
            return False
    
    def run_training(self):
        """Execute full training pipeline"""
        try:
            logger.info("Starting Vertex AI training pipeline...")
            
            # Create model first to get tokenizer
            logger.info(f"Creating {self.config.model.architecture} model...")
            model, tokenizer = self.model_factory.create_model_and_tokenizer()
            
            # Load and prepare data with correct tokenizer
            logger.info("Loading data...")
            train_dataset, eval_dataset, data_info = self.data_loader.prepare_datasets(tokenizer=tokenizer)
            
            # Log data info
            self.tracker.log_data_info(data_info)
            
            # Model already created above
            
            # Log model info
            model_info = {
                'architecture': self.config.model.architecture,
                'model_name': self.config.model.model_name,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            self.tracker.log_model_info(model_info)
            
            # Setup training
            logger.info("Setting up trainer...")
            trainer = self.model_factory.create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=self.config,
                model_dir=self.model_dir,
                tracker=self.tracker
            )
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train()
            
            # Final evaluation
            logger.info("Running final evaluation...")
            eval_result = trainer.evaluate()
            
            # Generate detailed analysis
            logger.info("Generating analysis...")
            analysis_results = self.tracker.generate_final_analysis(
                trainer, eval_dataset
            )
            
            # Save results
            final_results = {
                'experiment_name': self.config.experiment_name,
                'experiment_id': self.config.experiment_id,
                'train_result': train_result.metrics,
                'eval_result': eval_result,
                'analysis': analysis_results,
                'model_info': model_info,
                'data_info': data_info,
                'config': self.config.to_dict()
            }
            
            # Save to cloud storage
            self.save_results(final_results)
            
            # Save model artifacts
            self.save_model_artifacts(trainer, tokenizer)
            
            logger.info("Training completed successfully!")
            logger.info(f"Final F1 Score: {eval_result.get('eval_f1_macro', 'N/A'):.4f}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_results(self, results):
        """Save results to cloud storage"""
        try:
            # Save to model directory (GCS)
            results_path = os.path.join(self.model_dir, 'results.json')
            
            with open('results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Upload to GCS if model_dir is GCS path
            if self.model_dir.startswith('gs://'):
                client = storage.Client()
                bucket_name = self.model_dir.split('/')[2]
                blob_path = '/'.join(self.model_dir.split('/')[3:]) + '/results.json'
                
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.upload_from_filename('results.json')
                
                logger.info(f"Results saved to: {self.model_dir}/results.json")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def save_model_artifacts(self, trainer, tokenizer):
        """Save model and tokenizer artifacts"""
        try:
            # Save locally first
            local_model_path = "final_model"
            trainer.save_model(local_model_path)
            tokenizer.save_pretrained(local_model_path)
            
            # Upload to GCS if needed
            if self.model_dir.startswith('gs://'):
                import shutil
                
                # Create tarball
                shutil.make_archive('model_artifacts', 'tar.gz', local_model_path)
                
                # Upload
                client = storage.Client()
                bucket_name = self.model_dir.split('/')[2]
                blob_path = '/'.join(self.model_dir.split('/')[3:]) + '/model_artifacts.tar.gz'
                
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.upload_from_filename('model_artifacts.tar.gz')
                
                logger.info(f"Model artifacts saved to: {self.model_dir}/model_artifacts.tar.gz")
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {e}")


def parse_args():
    """Parse command line arguments for Vertex AI"""
    parser = argparse.ArgumentParser(description='Vertex AI Bias Classification Training')
    
    # Required Vertex AI arguments
    parser.add_argument('--model-dir', type=str, required=True,
                       help='GCS path for model output')
    parser.add_argument('--data-path', type=str, required=True,
                       help='GCS path to training data')
    parser.add_argument('--config-path', type=str, required=True,
                       help='GCS path to experiment config')
    
    # GCP settings
    parser.add_argument('--project-id', type=str, required=True,
                       help='GCP project ID')
    parser.add_argument('--region', type=str, default='us-central1',
                       help='GCP region')
    parser.add_argument('--staging-bucket', type=str, required=True,
                       help='GCS staging bucket')
    
    # Optional arguments
    parser.add_argument('--tensorboard-log-dir', type=str,
                       help='Tensorboard log directory (GCS path)')
    parser.add_argument('--job-name', type=str,
                       help='Vertex AI job name')
    parser.add_argument('--experiment-name', type=str,
                       help='Experiment name for tracking')
    
    # Development/testing
    parser.add_argument('--local-test', action='store_true',
                       help='Run in local test mode')
    
    return parser.parse_args()


def main():
    """Main entry point for Vertex AI training"""
    try:
        # Parse arguments
        args = parse_args()
        
        logger.info("Starting Vertex AI Bias Classification Training")
        logger.info(f"Arguments: {vars(args)}")
        
        # Initialize and run training task
        task = VertexTrainingTask(args)
        
        # Load configuration
        if not task.load_config():
            sys.exit(1)
        
        # Setup components
        if not task.setup_components():
            sys.exit(1)
        
        # Run training
        results = task.run_training()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()