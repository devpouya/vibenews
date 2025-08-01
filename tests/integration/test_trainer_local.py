#!/usr/bin/env python3
"""
Local test script for Vertex AI trainer components
Tests all components without requiring cloud deployment
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
        print("✅ Transformers")
        
        from trainer.model import BiasModelFactory
        from trainer.data import CloudDataLoader, CloudBiasDataset
        from trainer.experiment import CloudExperimentTracker
        from trainer.config import VertexExperimentConfig
        print("✅ All trainer components")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test config loading from YAML"""
    print("\n🔍 Testing config loading...")
    
    try:
        from trainer.config import VertexExperimentConfig
        import yaml
        
        # Test loading CPU config
        config_path = "vertex_configs/distilbert_cpu_only.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = VertexExperimentConfig.from_dict(config_dict)
        print(f"✅ Config loaded: {config.experiment_name}")
        print(f"   Model: {config.model.architecture}")
        print(f"   Batch size: {config.training.batch_size}")
        print(f"   Epochs: {config.training.epochs}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_factory():
    """Test model and tokenizer creation"""
    print("\n🔍 Testing model factory...")
    
    try:
        from trainer.model import BiasModelFactory
        from trainer.config import VertexExperimentConfig
        import yaml
        
        # Load config
        with open("vertex_configs/distilbert_cpu_only.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        config = VertexExperimentConfig.from_dict(config_dict)
        
        # Create factory
        factory = BiasModelFactory(config.model)
        
        # Test model creation
        print("   Creating model and tokenizer...")
        model, tokenizer = factory.create_model_and_tokenizer()
        
        print(f"✅ Model created: {type(model).__name__}")
        print(f"✅ Tokenizer created: {type(tokenizer).__name__}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model factory error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_arguments():
    """Test TrainingArguments creation - this is where the duplicate report_to error occurred"""
    print("\n🔍 Testing TrainingArguments creation...")
    
    try:
        from trainer.model import BiasModelFactory
        from trainer.config import VertexExperimentConfig
        from transformers import TrainingArguments
        import yaml
        import tempfile
        
        # Load config
        with open("vertex_configs/distilbert_cpu_only.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        config = VertexExperimentConfig.from_dict(config_dict)
        
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test TrainingArguments creation directly
            training_args = TrainingArguments(
                output_dir=temp_dir,
                overwrite_output_dir=True,
                
                # Training parameters
                num_train_epochs=config.training.epochs,
                per_device_train_batch_size=config.training.batch_size,
                per_device_eval_batch_size=config.training.eval_batch_size,
                gradient_accumulation_steps=config.training.gradient_accumulation_steps,
                
                # Optimization
                learning_rate=config.training.learning_rate,
                weight_decay=config.training.weight_decay,
                warmup_steps=config.training.warmup_steps,
                max_grad_norm=config.training.max_grad_norm,
                
                # Evaluation and logging
                eval_strategy="steps",
                eval_steps=config.logging.eval_steps,
                logging_steps=config.logging.log_steps,
                save_steps=config.logging.save_steps,
                
                # Model saving
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="f1_macro",
                greater_is_better=True,
                
                # Vertex AI optimizations
                dataloader_num_workers=0,
                remove_unused_columns=False,
                
                # Mixed precision for speed (if supported)
                fp16=config.training.fp16,
                
                # Reporting (should not be duplicated)
                report_to=["tensorboard"] if config.logging.tensorboard else [],
            )
            
            print("✅ TrainingArguments created successfully")
            print(f"   Output dir: {training_args.output_dir}")
            print(f"   Epochs: {training_args.num_train_epochs}")
            print(f"   Batch size: {training_args.per_device_train_batch_size}")
            print(f"   Report to: {training_args.report_to}")
            
            return True
    
    except Exception as e:
        print(f"❌ TrainingArguments creation error: {e}")
        if "keyword argument repeated" in str(e):
            print("   🚨 This is the duplicate parameter error!")
        import traceback
        traceback.print_exc()
        return False

def test_dummy_dataset():
    """Test dataset creation with dummy data"""
    print("\n🔍 Testing dataset creation...")
    
    try:
        from trainer.data import CloudBiasDataset
        from transformers import AutoTokenizer
        import pandas as pd
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'text': [
                "This is a neutral statement about politics.",
                "This politician is absolutely terrible and corrupt.",
                "The economic policy shows promising results."
            ],
            'bias_label': ["Non-biased", "Biased", "Non-biased"],
            'label': [0, 1, 0]
        })
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        # Create dataset
        dataset = CloudBiasDataset(dummy_data, tokenizer, max_length=128)
        
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Input shape: {sample['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_trainer_creation():
    """Test full trainer creation pipeline"""
    print("\n🔍 Testing full trainer creation...")
    
    try:
        from trainer.model import BiasModelFactory
        from trainer.data import CloudBiasDataset
        from trainer.config import VertexExperimentConfig
        from transformers import AutoTokenizer
        import yaml
        import pandas as pd
        import tempfile
        
        # Load config
        with open("vertex_configs/distilbert_cpu_only.yaml", 'r') as f:
            config_dict = yaml.safe_load(f)
        config = VertexExperimentConfig.from_dict(config_dict)
        
        # Create model factory
        factory = BiasModelFactory(config.model)
        model, tokenizer = factory.create_model_and_tokenizer()
        
        # Create dummy datasets
        dummy_data = pd.DataFrame({
            'text': ["Sample text " + str(i) for i in range(10)],
            'bias_label': ["Non-biased" if i % 2 == 0 else "Biased" for i in range(10)],
            'label': [0 if i % 2 == 0 else 1 for i in range(10)]
        })
        
        train_dataset = CloudBiasDataset(dummy_data, tokenizer, max_length=128)
        eval_dataset = CloudBiasDataset(dummy_data, tokenizer, max_length=128)
        
        # Create trainer with temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = factory.create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                config=config,
                model_dir=temp_dir,
                tracker=None
            )
            
            print("✅ Full trainer created successfully")
            print(f"   Model: {type(trainer.model).__name__}")
            print(f"   Train dataset size: {len(trainer.train_dataset)}")
            print(f"   Eval dataset size: {len(trainer.eval_dataset)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Trainer creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_argument_parsing():
    """Test command line argument parsing"""
    print("\n🔍 Testing argument parsing...")
    
    try:
        from trainer.task import parse_args
        import sys
        
        # Mock command line arguments
        test_args = [
            "--model-dir", "gs://test-bucket/models/test",
            "--data-path", "gs://test-bucket/data/test.jsonl",
            "--config-path", "gs://test-bucket/configs/test.yaml",
            "--project-id", "test-project",
            "--region", "us-central1",
            "--staging-bucket", "gs://test-staging",
            "--tensorboard-log-dir", "gs://test-bucket/logs",
            "--job-name", "test-job"
        ]
        
        # Temporarily replace sys.argv
        original_argv = sys.argv[:]
        sys.argv = ["trainer/task.py"] + test_args
        
        try:
            args = parse_args()
            print("✅ Arguments parsed successfully")
            print(f"   Model dir: {args.model_dir}")
            print(f"   Data path: {args.data_path}")
            print(f"   Project ID: {args.project_id}")
            
            return True
            
        finally:
            sys.argv = original_argv
            
    except Exception as e:
        print(f"❌ Argument parsing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running Local Trainer Tests")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
        ("Model Factory", test_model_factory),
        ("TrainingArguments", test_training_arguments),
        ("Dataset Creation", test_dummy_dataset),
        ("Full Trainer", test_full_trainer_creation),
        ("Argument Parsing", test_argument_parsing),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)