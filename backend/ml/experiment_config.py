"""
Experiment configuration management system
Handles YAML configs, parameter validation, and experiment metadata
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    architecture: str = "bert"  # bert, distilbert, roberta, etc.
    model_name: str = "bert-base-uncased"
    num_labels: int = 3
    dropout: float = 0.1
    freeze_layers: int = 0  # Number of layers to freeze
    hidden_size: Optional[int] = None  # Override model hidden size


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    scheduler: str = "linear"  # linear, cosine, polynomial
    optimizer: str = "adamw"


@dataclass
class DataConfig:
    """Data processing configuration"""
    dataset: str = "babe"
    train_split: float = 0.8
    preprocessing: str = "none"  # none, clean, normalize
    max_length: int = 512
    augmentation: bool = False
    balance_classes: bool = False
    filter_no_agreement: bool = False


@dataclass
class LoggingConfig:
    """Experiment tracking configuration"""
    tensorboard: bool = True
    wandb: bool = False
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    log_model_graph: bool = True
    log_confusion_matrix: bool = True


@dataclass
class CloudConfig:
    """Cloud training configuration"""
    platform: str = "gcp"  # gcp, aws, azure
    machine_type: str = "n1-standard-4"
    gpu_type: str = "nvidia-tesla-v100"
    gpu_count: int = 1
    preemptible: bool = True
    region: str = "us-central1"
    zone: str = "us-central1-a"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    
    # Auto-generated metadata
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'experiment_name': self.experiment_name,
            'experiment_id': self.experiment_id,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__,
            'cloud': self.cloud.__dict__
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        cloud_config = CloudConfig(**config_dict.get('cloud', {}))
        
        return cls(
            experiment_name=config_dict['experiment_name'],
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', []),
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config,
            cloud=cloud_config
        )
    
    def save_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def get_run_name(self) -> str:
        """Generate unique run name for tracking"""
        return f"{self.experiment_name}_{self.experiment_id}"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Model validation
        if self.model.architecture not in ['bert', 'distilbert', 'roberta', 'electra']:
            errors.append(f"Unsupported architecture: {self.model.architecture}")
        
        if self.model.num_labels < 2:
            errors.append("num_labels must be >= 2")
        
        # Training validation
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        
        if self.training.batch_size <= 0:
            errors.append("batch_size must be > 0")
        
        if self.training.epochs <= 0:
            errors.append("epochs must be > 0")
        
        # Data validation
        if not 0 < self.data.train_split < 1:
            errors.append("train_split must be between 0 and 1")
        
        if self.data.max_length <= 0:
            errors.append("max_length must be > 0")
        
        return errors


class ExperimentManager:
    """Manage experiment configurations and runs"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.configs_dir = self.experiments_dir / "configs"
        self.runs_dir = self.experiments_dir / "runs"
        
        # Create directories
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_name: str) -> ExperimentConfig:
        """Load experiment configuration"""
        config_path = self.configs_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        return ExperimentConfig.from_yaml(str(config_path))
    
    def save_config(self, config: ExperimentConfig, config_name: str):
        """Save experiment configuration"""
        config_path = self.configs_dir / f"{config_name}.yaml"
        config.save_yaml(str(config_path))
    
    def list_configs(self) -> List[str]:
        """List available configuration files"""
        return [f.stem for f in self.configs_dir.glob("*.yaml")]
    
    def create_run_dir(self, config: ExperimentConfig) -> Path:
        """Create directory for experiment run"""
        run_dir = self.runs_dir / config.get_run_name()
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config to run directory
        config.save_yaml(str(run_dir / "config.yaml"))
        
        return run_dir
    
    def get_run_results(self, run_name: str) -> Dict[str, Any]:
        """Load results from completed run"""
        run_dir = self.runs_dir / run_name
        results_file = run_dir / "results.yaml"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                return yaml.safe_load(f)
        
        return {}
    
    def compare_runs(self, run_names: List[str]) -> Dict[str, Any]:
        """Compare results across multiple runs"""
        comparison = {
            'configs': {},
            'results': {},
            'summary': {}
        }
        
        for run_name in run_names:
            # Load config
            run_dir = self.runs_dir / run_name
            config_file = run_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    comparison['configs'][run_name] = yaml.safe_load(f)
            
            # Load results
            results = self.get_run_results(run_name)
            if results:
                comparison['results'][run_name] = results
        
        # Generate comparison summary
        if comparison['results']:
            metrics = ['accuracy', 'f1_macro', 'f1_weighted']
            for metric in metrics:
                comparison['summary'][metric] = {}
                for run_name in run_names:
                    if run_name in comparison['results']:
                        value = comparison['results'][run_name].get(metric)
                        if value is not None:
                            comparison['summary'][metric][run_name] = value
        
        return comparison