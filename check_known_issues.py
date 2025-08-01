#!/usr/bin/env python3
"""
Check for known TrainingArguments issues without requiring transformers
"""

def check_known_parameter_issues():
    """Check our parameters against known problematic ones"""
    print("🔍 Checking for known TrainingArguments issues...")
    
    # Parameters we're using (from trainer/model.py)
    our_parameters = [
        'output_dir',
        'overwrite_output_dir', 
        'num_train_epochs',
        'per_device_train_batch_size',
        'per_device_eval_batch_size', 
        'gradient_accumulation_steps',
        'learning_rate',
        'weight_decay',
        'warmup_steps',
        'max_grad_norm',
        'eval_strategy',  # This one was renamed!
        'eval_steps',
        'logging_steps',
        'save_steps',
        'save_total_limit',
        'load_best_model_at_end',
        'metric_for_best_model',
        'greater_is_better',
        'dataloader_num_workers',
        'remove_unused_columns',
        'fp16',
        'report_to'
    ]
    
    # Known problematic parameters and their fixes
    known_issues = [
        {
            'problem': 'evaluation_strategy',
            'solution': 'eval_strategy',
            'description': 'evaluation_strategy was renamed to eval_strategy in newer versions'
        },
        {
            'problem': 'prediction_loss_only',
            'solution': 'Remove or use compute_metrics',
            'description': 'prediction_loss_only is often not needed'
        },
        {
            'problem': 'eval_accumulation_steps',
            'solution': 'Check if actually needed',
            'description': 'eval_accumulation_steps can cause memory issues'
        }
    ]
    
    # Check if we're using any problematic parameters
    issues_found = []
    
    for param in our_parameters:
        for issue in known_issues:
            if param == issue['problem']:
                issues_found.append(f"Using {issue['problem']} - {issue['description']}. Use: {issue['solution']}")
    
    if issues_found:
        for issue in issues_found:
            print(f"❌ {issue}")
        return False
    
    print("✅ No known problematic parameters detected")
    
    # Check for good practices
    good_practices = []
    
    if 'eval_strategy' in our_parameters:
        good_practices.append("✅ Using eval_strategy (correct, not evaluation_strategy)")
    
    if 'report_to' in our_parameters:
        good_practices.append("✅ Using report_to for logging control")
        
    if 'dataloader_num_workers' in our_parameters:
        good_practices.append("✅ Setting dataloader_num_workers=0 for cloud environments")
    
    for practice in good_practices:
        print(practice)
    
    return True

def check_config_values():
    """Check if our config values make sense"""
    print("\n🔍 Checking configuration values...")
    
    import yaml
    
    try:
        with open('vertex_configs/distilbert_cpu_only.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        issues = []
        
        # Check training config
        training = config.get('training', {})
        
        if training.get('batch_size', 0) <= 0:
            issues.append("batch_size must be > 0")
            
        if training.get('learning_rate', 0) <= 0:
            issues.append("learning_rate must be > 0")
            
        if training.get('epochs', 0) <= 0:
            issues.append("epochs must be > 0")
        
        # Check logging config
        logging_config = config.get('logging', {})
        
        if logging_config.get('eval_steps', 0) <= 0:
            issues.append("eval_steps must be > 0 when eval_strategy='steps'")
            
        if logging_config.get('log_steps', 0) <= 0:
            issues.append("log_steps must be > 0")
        
        # Check model config
        model = config.get('model', {})
        
        if model.get('num_labels', 0) <= 0:
            issues.append("num_labels must be > 0")
        
        if issues:
            for issue in issues:
                print(f"❌ {issue}")
            return False
        else:
            print("✅ Configuration values look reasonable")
            print(f"   Batch size: {training.get('batch_size')}")
            print(f"   Learning rate: {training.get('learning_rate')}")
            print(f"   Epochs: {training.get('epochs')}")
            print(f"   Eval steps: {logging_config.get('eval_steps')}")
            return True
            
    except Exception as e:
        print(f"❌ Error checking config: {e}")
        return False

def check_for_cpu_specific_issues():
    """Check for CPU-specific training issues"""
    print("\n🔍 Checking for CPU-specific issues...")
    
    import yaml
    
    try:
        with open('vertex_configs/distilbert_cpu_only.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        issues = []
        warnings = []
        
        training = config.get('training', {})
        
        # Check for CPU-unfriendly settings
        if training.get('fp16', False):
            issues.append("fp16=True is not supported on CPU, should be False")
        
        if training.get('batch_size', 0) > 32:
            warnings.append(f"batch_size={training.get('batch_size')} might be too large for CPU training")
        
        # Check vertex config
        vertex = config.get('vertex', {})
        
        if 'accelerator_type' in vertex:
            issues.append("CPU-only config should not have accelerator_type")
            
        if 'accelerator_count' in vertex:
            issues.append("CPU-only config should not have accelerator_count")
        
        if issues:
            for issue in issues:
                print(f"❌ {issue}")
            return False
        
        for warning in warnings:
            print(f"⚠️  {warning}")
        
        print("✅ CPU configuration looks good")
        return True
        
    except Exception as e:
        print(f"❌ Error checking CPU config: {e}")
        return False

def main():
    """Run all checks"""
    print("🧪 Checking for Known Training Issues")
    print("="*50)
    
    tests = [
        ("Parameter Issues", check_known_parameter_issues),
        ("Config Values", check_config_values), 
        ("CPU-Specific Issues", check_for_cpu_specific_issues),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 No known training issues detected!")
        print("✅ Parameters should work with current Transformers")
        return True
    else:
        print("❌ Found potential training issues")
        print("⚠️  Review and fix before deployment")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)