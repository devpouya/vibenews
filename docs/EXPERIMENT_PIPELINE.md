# Model Development Pipeline 🧪

A comprehensive experiment pipeline for bias classification with multiple model architectures, tracking, and cloud training support.

## 🏗️ Architecture Overview

```
Experiment Config (YAML) → Model Factory → Training → Tracking → Results
                                ↓
                        BERT / DistilBERT / RoBERTa
                                ↓
                        Tensorboard + W&B Logging
                                ↓
                        Local / GCP Cloud Training
```

## 🔧 Components

### 1. **Experiment Configuration System**
- **YAML-based configs** for reproducible experiments
- **Validation** ensures config correctness
- **Modular structure**: model, training, data, logging, cloud settings
- **Auto-generated metadata**: experiment IDs, timestamps

### 2. **Model Factory**
- **Multiple architectures**: BERT, DistilBERT, RoBERTa, ELECTRA, Custom
- **Configurable components**: dropout, layer freezing, custom heads
- **Tokenizer management** for each model type
- **Model info logging** (parameters, architecture details)

### 3. **Experiment Tracking**
- **Tensorboard integration** with comprehensive logging
- **W&B support** for advanced experiment management
- **Metrics tracking**: accuracy, F1, per-class metrics
- **Visualizations**: confusion matrices, learning curves, prediction samples
- **Model graph logging** for architecture visualization

### 4. **Cloud Training Support**
- **GCP integration** with Compute Engine + GPUs
- **Cost estimation** before training
- **Automated setup** and teardown
- **Results synchronization** between local and cloud

## 🚀 Quick Start

### Local Training

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Test pipeline:**
```bash
python test_experiment_pipeline.py
```

3. **Run experiments:**
```bash
# BERT baseline
python run_experiment.py experiments/configs/bert_baseline.yaml

# DistilBERT (faster)
python run_experiment.py experiments/configs/distilbert_fast.yaml

# RoBERTa (potentially better performance)
python run_experiment.py experiments/configs/roberta_strong.yaml
```

4. **View results:**
```bash
# Tensorboard
tensorboard --logdir experiments/runs

# Results directory
ls experiments/runs/
```

### Cloud Training (GCP)

1. **Setup GCP:**
```bash
# Setup APIs and storage
python scripts/gcp_setup.py --project YOUR_PROJECT_ID --setup-only

# Create training instance
python scripts/gcp_setup.py --project YOUR_PROJECT_ID
```

2. **Run cloud training:**
```bash
python run_experiment.py experiments/configs/bert_baseline.yaml --cloud
```

## 📋 Experiment Configurations

### Available Configs

| Config | Model | Description | Training Time | Cost (GCP) |
|--------|-------|-------------|---------------|------------|
| `bert_baseline.yaml` | BERT-base | Standard baseline | ~2-3 hours | ~$5-8 |
| `distilbert_fast.yaml` | DistilBERT | Faster, smaller | ~1-2 hours | ~$3-5 |
| `roberta_strong.yaml` | RoBERTa-base | Potentially better | ~2-3 hours | ~$5-8 |

### Configuration Structure

```yaml
experiment_name: "bert_baseline_v1"
description: "BERT-base baseline for bias classification"
tags: ["baseline", "bert", "babe"]

model:
  architecture: "bert"
  model_name: "bert-base-uncased"
  num_labels: 3
  dropout: 0.1
  freeze_layers: 0

training:
  batch_size: 16
  learning_rate: 2.0e-5
  epochs: 3
  warmup_steps: 500
  weight_decay: 0.01

data:
  dataset: "babe"
  train_split: 0.8
  preprocessing: "none"
  max_length: 512
  filter_no_agreement: false

logging:
  tensorboard: true
  wandb: false
  log_steps: 100
  save_steps: 1000

cloud:
  platform: "gcp"
  machine_type: "n1-standard-4"
  gpu_type: "nvidia-tesla-v100"
  preemptible: true
```

## 📊 Experiment Tracking Features

### Metrics Logged
- **Training/Validation**: Loss, accuracy, F1 (macro/weighted)
- **Per-class metrics**: Precision, recall, F1 for each bias class
- **Learning curves**: Real-time training progress
- **Hyperparameters**: Complete config tracking

### Visualizations
- **Confusion matrices**: Detailed classification analysis
- **Learning curves**: Loss and metrics over time
- **Prediction samples**: Qualitative result analysis
- **Model graphs**: Architecture visualization

### Storage
```
experiments/
├── configs/           # YAML configuration files
├── runs/             # Experiment results
│   └── experiment_name_id/
│       ├── config.yaml
│       ├── results.json
│       ├── checkpoints/
│       ├── logs/     # Tensorboard logs
│       └── final_model/
└── comparison.html   # Multi-experiment comparison
```

## 🛠️ Custom Experiments

### Create New Configuration
```python
from backend.ml.experiment_config import ExperimentConfig, ModelConfig

# Create custom config
config = ExperimentConfig(
    experiment_name="custom_experiment",
    description="My custom bias classification experiment",
    tags=["custom", "experimental"]
)

# Customize model
config.model = ModelConfig(
    architecture="roberta",
    model_name="roberta-large",
    dropout=0.2,
    freeze_layers=6  # Freeze first 6 layers
)

# Save config
config.save_yaml("experiments/configs/custom_experiment.yaml")
```

### Add New Model Architecture
```python
from backend.ml.model_factory import ModelFactory, BaseBiasClassifier

class MyCustomClassifier(BaseBiasClassifier):
    def __init__(self, config):
        super().__init__(config)
        # Custom implementation
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Custom forward pass
        pass

# Register new architecture
ModelFactory.register_model("my_custom", MyCustomClassifier)
```

## 📈 Result Analysis

### Compare Experiments
```python
from backend.ml.experiment_config import ExperimentManager

manager = ExperimentManager()
comparison = manager.compare_runs([
    "bert_baseline_v1_abc123",
    "distilbert_fast_v1_def456",
    "roberta_strong_v1_ghi789"
])

print("Best F1 scores:")
for run, f1 in comparison['summary']['f1_macro'].items():
    print(f"{run}: {f1:.3f}")
```

### Load Trained Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "experiments/runs/bert_baseline_v1_abc123/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Use for prediction
inputs = tokenizer("Sample text", return_tensors="pt")
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=-1)
```

## 💰 Cost Analysis

### GCP Training Costs (Estimated)

| Instance Type | GPU | Cost/Hour | 2h Training | 4h Training |
|---------------|-----|-----------|-------------|-------------|
| n1-standard-4 + V100 | 1x V100 | $2.74 | $5.48 | $10.96 |
| n1-standard-4 + T4 | 1x T4 | $1.14 | $2.28 | $4.56 |
| n1-standard-8 + V100 | 1x V100 | $3.12 | $6.24 | $12.48 |

*Prices with preemptible instances (80% discount)*

### Cost Optimization Tips
1. **Use preemptible instances** for 80% cost savings
2. **DistilBERT** trains faster than BERT (lower costs)
3. **Monitor training** and stop early if converged
4. **Batch experiments** to minimize setup overhead

## 🔍 Troubleshooting

### Common Issues

**Config validation errors:**
```bash
# Check config before running
from backend.ml.experiment_config import ExperimentConfig
config = ExperimentConfig.from_yaml("path/to/config.yaml")
errors = config.validate()
```

**Out of memory:**
- Reduce `batch_size` in config
- Use gradient accumulation
- Try DistilBERT instead of BERT

**GCP connection issues:**
```bash
# Check gcloud setup
gcloud auth list
gcloud config get-value project
```

## 🎯 Best Practices

### Experiment Design
1. **Start with baselines** (BERT, DistilBERT)
2. **Systematic comparison** (change one thing at a time)
3. **Proper validation** (hold-out test set)
4. **Document everything** (tags, descriptions)

### Performance Optimization
1. **Use appropriate batch sizes** (16-32 typical)
2. **Learning rate scheduling** (warmup + decay)
3. **Early stopping** based on validation metrics
4. **Layer freezing** for transfer learning

### Resource Management
1. **Monitor costs** with GCP billing alerts
2. **Clean up instances** after training
3. **Use storage efficiently** (compress results)
4. **Batch similar experiments** together

## 🔄 Integration with VibeNews

### Deploy Trained Model
```python
# Load best model
best_run = manager.get_best_run("eval/f1_macro")
model_path = f"experiments/runs/{best_run}/final_model"

# Integrate with existing bias_classifier.py
classifier = BiasClassifier()
classifier.load_model(model_path)

# Use in production pipeline
results = classifier.predict_with_confidence(articles)
```

### Continuous Learning
1. **Regular retraining** with new Swiss articles
2. **A/B testing** between model versions
3. **Performance monitoring** in production
4. **Feedback incorporation** from user interactions

---

**Ready to start experimenting!** 🚀

Choose your approach:
- **Quick local test**: `python run_experiment.py experiments/configs/distilbert_fast.yaml`
- **Full comparison**: Run all three configs and compare results
- **Cloud scale**: Set up GCP and run with GPUs for faster training