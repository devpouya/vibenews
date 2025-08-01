# VibeNews Test Suite

Comprehensive testing suite for the VibeNews bias detection pipeline.

## Test Structure

```
tests/
├── unit/              # Unit tests for individual components
│   ├── test_syntax_only.py       # Python syntax validation
│   ├── test_minimal_runtime.py   # Runtime pattern validation
│   └── test_model_kwargs.py      # Model parameter validation
├── integration/       # Integration tests for complete workflows
│   ├── test_babe_integration.py  # BABE dataset integration
│   ├── test_bert_pipeline.py     # BERT training pipeline
│   ├── test_experiment_pipeline.py # Experiment management
│   ├── test_biasscanner.py       # BiasScanner implementation
│   ├── test_trainer_local.py     # Local trainer validation
│   ├── test_babe_simple.py       # Simple BABE data tests
│   ├── test_scraper.py           # News scraper tests
│   └── test_storage.py           # Storage integration tests
├── validation/        # Validation tests for deployment readiness
│   ├── check_known_issues.py     # Known issue detection
│   ├── test_runtime_issues.py    # Runtime issue validation
│   ├── test_training_args_validation.py # Training arguments validation
│   └── test_vertex_setup.py      # Vertex AI setup validation
└── debug/            # Debugging and monitoring utilities
    ├── debug_scraper.py          # Scraper debugging
    ├── debug_submission.py       # Job submission debugging
    └── check_job_status.py       # Job status monitoring
```

## Running Tests

### Quick Validation (Pre-deployment)
```bash
# Run all validation tests
python -m pytest tests/validation/ -v

# Check for known issues
python tests/validation/check_known_issues.py

# Validate training arguments
python tests/validation/test_training_args_validation.py
```

### Unit Tests
```bash
# Test syntax and basic patterns
python tests/unit/test_syntax_only.py

# Test model parameter generation
python tests/unit/test_model_kwargs.py

# Test runtime patterns
python tests/unit/test_minimal_runtime.py
```

### Integration Tests
```bash
# Test BABE dataset integration
python tests/integration/test_babe_integration.py

# Test BiasScanner functionality
python tests/integration/test_biasscanner.py

# Test complete training pipeline
python tests/integration/test_bert_pipeline.py
```

### Debugging Utilities
```bash
# Check Vertex AI job status
python debug/check_job_status.py

# Debug scraper issues
python debug/debug_scraper.py

# Debug job submission
python debug/debug_submission.py
```

## Test Categories

### 🔧 Unit Tests
- **Syntax validation**: Ensures all Python files have correct syntax
- **Parameter validation**: Tests model parameter generation for different architectures
- **Runtime patterns**: Validates key patterns work without heavy dependencies

### 🔗 Integration Tests
- **Data pipeline**: Tests BABE dataset loading and processing
- **Model training**: Tests complete training workflows
- **BiasScanner**: Tests bias detection implementation
- **Storage**: Tests Google Cloud Storage integration

### ✅ Validation Tests
- **Deployment readiness**: Checks for common deployment issues
- **Configuration validation**: Ensures configs are correctly formatted
- **Vertex AI setup**: Validates cloud infrastructure setup

### 🐛 Debug Utilities
- **Job monitoring**: Real-time job status and log access
- **Issue diagnosis**: Automated issue detection and reporting
- **Performance analysis**: Resource usage and optimization insights

## Best Practices

1. **Run validation tests** before any deployment
2. **Use unit tests** for rapid feedback during development  
3. **Run integration tests** before major releases
4. **Use debug utilities** for troubleshooting production issues

## Requirements

- Python 3.8+
- Dependencies from `requirements.txt` (for integration tests)
- Google Cloud SDK (for cloud tests)
- Valid Gemini API key (for BiasScanner tests)