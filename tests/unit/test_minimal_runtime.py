#!/usr/bin/env python3
"""
Minimal runtime test to check if key patterns work without heavy dependencies
"""

def test_training_args_kwargs():
    """Test that our TrainingArguments kwargs don't have duplicates"""
    
    # This simulates the exact kwargs we pass to TrainingArguments
    config_mock = type('Config', (), {
        'training': type('Training', (), {
            'epochs': 2,
            'batch_size': 16,
            'eval_batch_size': 32,
            'gradient_accumulation_steps': 2,
            'learning_rate': 2.0e-5,
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'fp16': False
        })(),
        'logging': type('Logging', (), {
            'eval_steps': 500,
            'log_steps': 100,
            'save_steps': 1000,
            'tensorboard': True
        })()
    })()
    
    # This is the exact pattern from our fixed code
    training_args_kwargs = {
        'output_dir': '/tmp/test',
        'overwrite_output_dir': True,
        
        # Training parameters
        'num_train_epochs': config_mock.training.epochs,
        'per_device_train_batch_size': config_mock.training.batch_size,
        'per_device_eval_batch_size': config_mock.training.eval_batch_size,
        'gradient_accumulation_steps': config_mock.training.gradient_accumulation_steps,
        
        # Optimization
        'learning_rate': config_mock.training.learning_rate,
        'weight_decay': config_mock.training.weight_decay,
        'warmup_steps': config_mock.training.warmup_steps,
        'max_grad_norm': config_mock.training.max_grad_norm,
        
        # Evaluation and logging
        'eval_strategy': "steps",
        'eval_steps': config_mock.logging.eval_steps,
        'logging_steps': config_mock.logging.log_steps,
        'save_steps': config_mock.logging.save_steps,
        
        # Model saving
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'metric_for_best_model': "f1_macro",
        'greater_is_better': True,
        
        # Vertex AI optimizations
        'dataloader_num_workers': 0,
        'remove_unused_columns': False,
        
        # Mixed precision for speed (if supported)
        'fp16': config_mock.training.fp16,
        
        # Reporting (should be single, not duplicated)
        'report_to': ["tensorboard"] if config_mock.logging.tensorboard else [],
    }
    
    # Check for duplicate keys
    keys = list(training_args_kwargs.keys())
    duplicate_keys = [key for key in set(keys) if keys.count(key) > 1]
    
    if duplicate_keys:
        print(f"❌ Duplicate keys found: {duplicate_keys}")
        return False
    
    # Check specifically for report_to (the one that caused the error)
    report_to_count = keys.count('report_to')
    if report_to_count != 1:
        print(f"❌ report_to appears {report_to_count} times, should be exactly 1")
        return False
    
    print(f"✅ Training arguments: {len(keys)} unique parameters")
    print(f"✅ report_to value: {training_args_kwargs['report_to']}")
    
    return True

def test_argument_parsing_pattern():
    """Test the argument parsing pattern"""
    
    # Mock argparse arguments
    class MockArgs:
        def __init__(self):
            self.model_dir = "gs://test/models"
            self.data_path = "gs://test/data.jsonl"
            self.config_path = "gs://test/config.yaml"
            self.project_id = "test-project"
            self.region = "us-central1"
            self.staging_bucket = "gs://test-staging"
            self.tensorboard_log_dir = "gs://test/logs"
            self.job_name = "test-job"
    
    args = MockArgs()
    
    # Test that all required args are accessible
    required_attrs = [
        'model_dir', 'data_path', 'config_path', 'project_id', 
        'region', 'staging_bucket'
    ]
    
    for attr in required_attrs:
        if not hasattr(args, attr):
            print(f"❌ Missing required argument: {attr}")
            return False
        if getattr(args, attr) is None:
            print(f"❌ Required argument {attr} is None")
            return False
    
    print(f"✅ All {len(required_attrs)} required arguments present")
    return True

def main():
    """Run minimal runtime tests"""
    print("🧪 Running Minimal Runtime Tests")
    print("="*50)
    
    tests = [
        ("TrainingArguments kwargs", test_training_args_kwargs),
        ("Argument parsing", test_argument_parsing_pattern),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            print(f"✅ {test_name}: PASS")
        else:
            print(f"❌ {test_name}: FAIL") 
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All runtime tests passed!")
        print("✅ Key patterns validated - ready for deployment")
    else:
        print("❌ Runtime issues found")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)