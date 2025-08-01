# VibeNews Project Structure

Clean, organized codebase for the Swiss news bias analyzer.

## 📁 Project Overview

```
vibenews/
├── backend/                    # Core bias detection system
│   ├── api/                   # FastAPI endpoints (empty - future)
│   ├── bias_detection/        # BiasScanner implementation
│   ├── config.py             # Settings management
│   ├── models/               # Data models (Pydantic)
│   └── scraper/              # News scraping utilities
├── trainer/                   # Vertex AI training pipeline
│   ├── config.py             # Training configuration
│   ├── data.py               # Cloud data loading
│   ├── experiment.py         # Experiment tracking
│   ├── model.py              # Model factory and training
│   └── task.py               # Main training entry point
├── frontend/                  # React web application
│   ├── public/               # Static assets
│   ├── src/                  # React components and pages
│   │   ├── components/       # Reusable UI components
│   │   ├── data/            # Mock data and utilities
│   │   └── pages/           # Route components
│   └── package.json         # Frontend dependencies
├── tests/                     # Organized test suite
│   ├── unit/                # Fast component tests
│   ├── integration/         # Workflow tests
│   ├── validation/          # Deployment checks
│   └── run_all_tests.py     # Master test runner
├── scripts/                   # Automation and utilities
│   ├── gcp_setup.py         # Google Cloud setup
│   ├── monitor_and_submit.py # Job monitoring
│   ├── run_experiment.py    # Experiment runner
│   └── submit_vertex_job.py # Vertex AI job submission
├── debug/                     # Debugging utilities
│   ├── check_job_status.py  # Job status monitoring
│   ├── debug_scraper.py     # Scraper debugging
│   └── debug_submission.py  # Submission debugging
├── vertex_configs/           # Training configurations
│   ├── distilbert_cpu_only.yaml  # CPU training config
│   ├── bert_baseline_job.yaml     # BERT baseline
│   └── distilbert_ultra_cheap.yaml # Ultra-cheap training
├── experiments/              # Experiment configurations
│   └── configs/             # Different model configs
└── docs/                     # Documentation (future)
```

## 🧪 Test Organization

### **Unit Tests** (`tests/unit/`)
Fast, lightweight tests that don't require external dependencies:
- `test_syntax_only.py` - Python syntax validation
- `test_minimal_runtime.py` - Runtime pattern validation  
- `test_model_kwargs.py` - Model parameter validation

### **Integration Tests** (`tests/integration/`)
Complete workflow tests that may require dependencies:
- `test_babe_integration.py` - BABE dataset integration
- `test_bert_pipeline.py` - Training pipeline tests
- `test_biasscanner.py` - BiasScanner functionality
- `test_trainer_local.py` - Local trainer validation

### **Validation Tests** (`tests/validation/`)
Deployment readiness and configuration validation:
- `check_known_issues.py` - Known issue detection
- `test_runtime_issues.py` - Runtime issue validation
- `test_training_args_validation.py` - Training argument validation
- `test_vertex_setup.py` - Vertex AI setup validation

### **Debug Utilities** (`debug/`)
Troubleshooting and monitoring tools:
- `check_job_status.py` - Real-time job monitoring
- `debug_scraper.py` - Scraper issue diagnosis
- `debug_submission.py` - Job submission debugging

## 🚀 Quick Start

### Run Tests
```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test category
python tests/unit/test_syntax_only.py
python tests/validation/check_known_issues.py
```

### Train Models
```bash
# Submit Vertex AI job
python scripts/submit_vertex_job.py --config vertex_configs/distilbert_cpu_only.yaml

# Monitor job status
python debug/check_job_status.py
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

## 📋 Development Workflow

1. **Make changes** to code
2. **Run unit tests** for quick feedback
   ```bash
   python tests/unit/test_syntax_only.py
   ```
3. **Run validation tests** before deployment
   ```bash
   python tests/validation/check_known_issues.py
   ```
4. **Run full test suite** before major releases
   ```bash
   python tests/run_all_tests.py
   ```
5. **Use debug utilities** for troubleshooting
   ```bash
   python debug/check_job_status.py
   ```

## 🛠️ Key Commands

| Task | Command |
|------|---------|
| **Full test suite** | `python tests/run_all_tests.py` |
| **Quick validation** | `python tests/validation/check_known_issues.py` |
| **Submit training** | `python scripts/submit_vertex_job.py --config vertex_configs/distilbert_cpu_only.yaml` |
| **Check job status** | `python debug/check_job_status.py` |
| **Start frontend** | `cd frontend && npm start` |
| **Debug scraper** | `python debug/debug_scraper.py` |

## 📦 Dependencies

- **Core**: Python 3.8+, Google Cloud SDK
- **Training**: PyTorch, Transformers, scikit-learn
- **Frontend**: Node.js 16+, React 18, Tailwind CSS
- **Cloud**: Google Cloud Storage, Vertex AI

## 🗂️ Configuration Files

- **Training**: `vertex_configs/*.yaml` - Vertex AI job configurations
- **Experiments**: `experiments/configs/*.yaml` - Model experiment configs
- **Frontend**: `frontend/package.json` - React dependencies and scripts
- **Docker**: `Dockerfile` - Container definition for training
- **Python**: `requirements-vertex.txt` - Cloud training dependencies