"""
Vertex AI compatible experiment configuration
Cloud-native version of experiment config with GCS support
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

from google.cloud import storage


@dataclass
class ModelConfig:
    """Model configuration for Vertex AI"""
    architecture: str = "bert"
    model_name: str = "bert-base-uncased"
    num_labels: int = 3
    dropout: float = 0.1
    freeze_layers: int = 0
    hidden_size: Optional[int] = None
    custom_head: bool = False


@dataclass
class TrainingConfig:
    """Training configuration for Vertex AI"""
    batch_size: int = 16
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    scheduler: str = "linear"
    optimizer: str = "adamw"
    gradient_accumulation_steps: int = 1
    fp16: bool = False


@dataclass
class DataConfig:
    """Data configuration for cloud storage"""
    dataset: str = "babe"
    train_split: float = 0.8
    preprocessing: str = "none"
    max_length: int = 512
    augmentation: bool = False
    balance_classes: bool = False
    filter_no_agreement: bool = False
    data_source: str = "gcs"  # gcs, local, url
    cache_dir: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration for Vertex AI"""
    tensorboard: bool = True
    log_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    log_model_graph: bool = False  # Disabled for cloud
    log_confusion_matrix: bool = True
    log_predictions: bool = True
    save_checkpoints: bool = True


@dataclass
class VertexConfig:
    """Vertex AI specific configuration"""
    machine_type: str = "n1-standard-4"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    disk_type: str = "pd-ssd"
    disk_size_gb: int = 100
    max_replica_count: int = 1
    preemptible: bool = True
    service_account: Optional[str] = None
    network: Optional[str] = None
    enable_web_access: bool = False


@dataclass
class VertexExperimentConfig:
    """Complete Vertex AI experiment configuration"""
    experiment_name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)
    
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
            'vertex': self.vertex.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VertexExperimentConfig':
        """Create from dictionary"""
        # Create nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        vertex_config = VertexConfig(**config_dict.get('vertex', {}))
        
        return cls(
            experiment_name=config_dict['experiment_name'],
            description=config_dict.get('description', ''),
            tags=config_dict.get('tags', []),
            experiment_id=config_dict.get('experiment_id', str(uuid.uuid4())[:8]),
            created_at=config_dict.get('created_at', datetime.now().isoformat()),
            model=model_config,
            training=training_config,
            data=data_config,
            logging=logging_config,
            vertex=vertex_config
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'VertexExperimentConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_gcs(cls, gcs_path: str) -> 'VertexExperimentConfig':
        """Load configuration from GCS"""
        client = storage.Client()
        bucket_name = gcs_path.split('/')[2]
        blob_path = '/'.join(gcs_path.split('/')[3:])
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        config_content = blob.download_as_text()
        
        config_dict = yaml.safe_load(config_content)
        return cls.from_dict(config_dict)
    
    def save_to_gcs(self, gcs_path: str):
        """Save configuration to GCS"""
        client = storage.Client()
        bucket_name = gcs_path.split('/')[2]
        blob_path = '/'.join(gcs_path.split('/')[3:])
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Convert to YAML and upload
        config_yaml = yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
        blob.upload_from_string(config_yaml, content_type='text/yaml')
    
    def get_job_name(self) -> str:
        """Generate Vertex AI job name"""
        # Vertex AI job names must match: ^[a-z]([a-z0-9-]{0,126}[a-z0-9])?$
        base_name = self.experiment_name.lower().replace('_', '-')
        return f"{base_name}-{self.experiment_id}"
    
    def get_model_dir(self, base_bucket: str) -> str:
        """Generate GCS model directory path"""
        return f"gs://{base_bucket}/models/{self.get_job_name()}"
    
    def get_tensorboard_dir(self, base_bucket: str) -> str:
        """Generate GCS Tensorboard directory path"""
        return f"gs://{base_bucket}/tensorboard/{self.get_job_name()}"
    
    def validate_vertex(self) -> List[str]:
        """Validate Vertex AI specific configuration"""
        errors = []
        
        # Machine type validation
        valid_machine_types = [
            'n1-standard-4', 'n1-standard-8', 'n1-standard-16',
            'n1-highmem-4', 'n1-highmem-8',
            'e2-standard-4', 'e2-standard-8'
        ]
        if self.vertex.machine_type not in valid_machine_types:
            errors.append(f"Invalid machine_type: {self.vertex.machine_type}")
        
        # Accelerator validation
        valid_accelerators = [
            'NVIDIA_TESLA_T4', 'NVIDIA_TESLA_V100', 'NVIDIA_TESLA_L4'
        ]
        if self.vertex.accelerator_type not in valid_accelerators:
            errors.append(f"Invalid accelerator_type: {self.vertex.accelerator_type}")
        
        # Count validation
        if not 1 <= self.vertex.accelerator_count <= 8:
            errors.append("accelerator_count must be between 1 and 8")
        
        # Disk size validation
        if self.vertex.disk_size_gb < 50:
            errors.append("disk_size_gb must be at least 50GB")
        
        return errors
    
    def validate(self) -> List[str]:
        """Complete validation including Vertex AI"""
        errors = []
        
        # Basic validation
        if not self.experiment_name:
            errors.append("experiment_name is required")
        
        # Model validation
        if self.model.num_labels < 2:
            errors.append("num_labels must be >= 2")
        
        # Training validation
        if self.training.learning_rate <= 0:
            errors.append("learning_rate must be > 0")
        if self.training.batch_size <= 0:
            errors.append("batch_size must be > 0")
        
        # Data validation
        if not 0 < self.data.train_split < 1:
            errors.append("train_split must be between 0 and 1")
        
        # Vertex AI specific validation
        errors.extend(self.validate_vertex())
        
        return errors
    
    def estimate_cost(self, training_hours: float = 2.0) -> Dict[str, float]:
        """Estimate Vertex AI training cost"""
        # Base pricing (approximate, varies by region)
        pricing = {
            'n1-standard-4': 0.1899,
            'n1-standard-8': 0.3798,
            'n1-standard-16': 0.7596,
            'NVIDIA_TESLA_T4': 0.35,
            'NVIDIA_TESLA_V100': 2.55,
            'NVIDIA_TESLA_L4': 0.60,
            'pd-ssd': 0.17 / (24 * 30)  # per GB per hour
        }
        
        # Calculate costs
        machine_cost = pricing.get(self.vertex.machine_type, 0.19) * training_hours
        gpu_cost = pricing.get(self.vertex.accelerator_type, 0.35) * self.vertex.accelerator_count * training_hours
        storage_cost = self.vertex.disk_size_gb * pricing['pd-ssd'] * training_hours
        
        # Vertex AI management fee (approximately 10%)
        base_cost = machine_cost + gpu_cost + storage_cost
        management_fee = base_cost * 0.1
        
        # Preemptible discount
        if self.vertex.preemptible:
            base_cost *= 0.3  # 70% discount
            management_fee *= 0.3
        
        total_cost = base_cost + management_fee
        
        return {
            'machine_cost': round(machine_cost, 2),
            'gpu_cost': round(gpu_cost, 2),
            'storage_cost': round(storage_cost, 4),
            'management_fee': round(management_fee, 2),
            'total_cost': round(total_cost, 2),
            'hourly_rate': round(total_cost / training_hours, 2),
            'preemptible_discount': self.vertex.preemptible
        }


class VertexConfigManager:
    """Manager for Vertex AI experiment configurations"""
    
    def __init__(self, bucket_name: str, configs_prefix: str = "configs"):
        self.bucket_name = bucket_name
        self.configs_prefix = configs_prefix
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
    
    def save_config(self, config: VertexExperimentConfig, config_name: str):
        """Save config to GCS"""
        blob_path = f"{self.configs_prefix}/{config_name}.yaml"
        gcs_path = f"gs://{self.bucket_name}/{blob_path}"
        config.save_to_gcs(gcs_path)
    
    def load_config(self, config_name: str) -> VertexExperimentConfig:
        """Load config from GCS"""
        blob_path = f"{self.configs_prefix}/{config_name}.yaml"
        gcs_path = f"gs://{self.bucket_name}/{blob_path}"
        return VertexExperimentConfig.from_gcs(gcs_path)
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        blobs = self.bucket.list_blobs(prefix=f"{self.configs_prefix}/")
        config_names = []
        
        for blob in blobs:
            if blob.name.endswith('.yaml'):
                name = blob.name.split('/')[-1].replace('.yaml', '')
                config_names.append(name)
        
        return config_names
    
    def compare_configs(self, config_names: List[str]) -> Dict[str, Any]:
        """Compare multiple configurations"""
        comparison = {'configs': {}, 'differences': {}}
        
        for name in config_names:
            try:
                config = self.load_config(name)
                comparison['configs'][name] = config.to_dict()
            except Exception as e:
                comparison['configs'][name] = f"Error loading: {e}"
        
        # Identify key differences
        if len(comparison['configs']) > 1:
            keys_to_compare = ['model.architecture', 'model.model_name', 
                             'training.learning_rate', 'training.batch_size']
            
            for key in keys_to_compare:
                values = {}
                for name, config_dict in comparison['configs'].items():
                    if isinstance(config_dict, dict):
                        # Navigate nested dict
                        value = config_dict
                        for k in key.split('.'):
                            value = value.get(k, 'N/A')
                        values[name] = value
                
                if len(set(values.values())) > 1:  # Different values
                    comparison['differences'][key] = values
        
        return comparison