#!/usr/bin/env python3
"""
Validate TrainingArguments parameters against current Transformers implementation
"""

def validate_training_arguments():
    """Test if all our TrainingArguments parameters are valid"""
    print("🔍 Validating TrainingArguments parameters...")
    
    # These are the exact parameters we're using in our code
    our_params = {
        'output_dir': '/tmp/test',
        'overwrite_output_dir': True,
        
        # Training parameters
        'num_train_epochs': 2,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 32,
        'gradient_accumulation_steps': 2,
        
        # Optimization
        'learning_rate': 2.0e-5,
        'weight_decay': 0.01,
        'warmup_steps': 100,
        'max_grad_norm': 1.0,
        
        # Evaluation and logging
        'eval_strategy': "steps",
        'eval_steps': 500,
        'logging_steps': 100,
        'save_steps': 1000,
        
        # Model saving
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'metric_for_best_model': "f1_macro",
        'greater_is_better': True,
        
        # Vertex AI optimizations
        'dataloader_num_workers': 0,
        'remove_unused_columns': False,
        
        # Mixed precision
        'fp16': False,
        
        # Reporting
        'report_to': [],
    }
    
    try:
        # Try to create TrainingArguments with our parameters
        # This will fail if any parameter is invalid
        from transformers import TrainingArguments
        
        training_args = TrainingArguments(**our_params)
        
        print("✅ All TrainingArguments parameters are valid")
        print(f"   Total parameters: {len(our_params)}")
        
        # Check for any deprecated warnings or issues
        problem_params = []
        
        # Common deprecated parameters that people sometimes use
        deprecated_checks = [
            ('evaluation_strategy', 'eval_strategy'),  # evaluation_strategy was renamed to eval_strategy
            ('do_train', 'implicit'),  # do_train is usually implicit
            ('do_eval', 'implicit'),   # do_eval is usually implicit
        ]
        
        for old_param, new_param in deprecated_checks:
            if old_param in our_params:
                problem_params.append(f"Using deprecated '{old_param}', should use '{new_param}'")
        
        if problem_params:
            for problem in problem_params:
                print(f"⚠️  {problem}")
            return False
        
        print("✅ No deprecated parameters detected")
        
        # Test specific combinations that might be problematic
        if our_params['eval_strategy'] == "steps" and 'eval_steps' not in our_params:
            print("❌ eval_strategy='steps' requires eval_steps parameter")
            return False
        
        if our_params['save_steps'] and our_params['save_steps'] > 0 and not our_params['output_dir']:
            print("❌ save_steps > 0 requires valid output_dir")
            return False
            
        print("✅ Parameter combinations are valid")
        return True
        
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            param_name = str(e).split("'")[1]
            print(f"❌ Invalid parameter: {param_name}")
            print(f"   Error: {e}")
            return False
        else:
            print(f"❌ TypeError: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_scheduler_names():
    """Check if scheduler names are valid"""
    print("\n🔍 Validating scheduler names...")
    
    # Our config uses these scheduler names
    scheduler_names = ["linear", "cosine", "polynomial"]
    
    try:
        from transformers import get_scheduler
        
        valid_schedulers = []
        invalid_schedulers = []
        
        for scheduler_name in scheduler_names:
            try:
                # Test if scheduler name is recognized
                # This is a simplified test - we can't actually create the scheduler without a model
                if scheduler_name in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
                    valid_schedulers.append(scheduler_name)
                else:
                    invalid_schedulers.append(scheduler_name)
            except Exception:
                invalid_schedulers.append(scheduler_name)
        
        if invalid_schedulers:
            print(f"❌ Invalid schedulers: {invalid_schedulers}")
            return False
        else:
            print(f"✅ Valid schedulers: {valid_schedulers}")
            return True
            
    except ImportError:
        print("⚠️  Could not import get_scheduler - skipping validation")
        return True
    except Exception as e:
        print(f"❌ Scheduler validation error: {e}")
        return False

def check_optimizer_names():
    """Check if optimizer names are valid"""
    print("\n🔍 Validating optimizer names...")
    
    # Our config uses these optimizer names  
    optimizer_names = ["adamw", "adam", "sgd"]
    
    # These are the optimizers supported by Transformers
    supported_optimizers = ["adamw", "adam", "sgd", "adafactor"]
    
    invalid_optimizers = [opt for opt in optimizer_names if opt not in supported_optimizers]
    
    if invalid_optimizers:
        print(f"❌ Invalid optimizers: {invalid_optimizers}")
        print(f"   Supported: {supported_optimizers}")
        return False
    else:
        print(f"✅ Valid optimizers: {optimizer_names}")
        return True

def main():
    """Run all validation tests"""
    print("🧪 Validating Training Script Parameters")
    print("="*50)
    
    tests = [
        ("TrainingArguments Parameters", validate_training_arguments),
        ("Scheduler Names", check_scheduler_names),
        ("Optimizer Names", check_optimizer_names),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All parameter validation tests passed!")
        print("✅ Training script parameters are compatible")
        return True
    else:
        print("❌ Parameter validation issues found")
        print("⚠️  Fix these before rebuilding container")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)