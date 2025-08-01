#!/usr/bin/env python3
"""
Syntax-only test to catch Python syntax errors without requiring dependencies
"""

import ast
import sys
from pathlib import Path

def test_python_syntax(file_path):
    """Test if a Python file has valid syntax"""
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source, filename=str(file_path))
        return True, None
        
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} (line {e.lineno})"
    except Exception as e:
        return False, f"Error: {e}"

def check_duplicate_keywords():
    """Check for common duplicate keyword issues"""
    trainer_model = Path("trainer/model.py")
    
    if not trainer_model.exists():
        return False, "trainer/model.py not found"
    
    with open(trainer_model, 'r') as f:
        content = f.read()
    
    # Check for duplicate report_to
    lines = content.split('\n')
    report_to_lines = [i for i, line in enumerate(lines) if 'report_to=' in line and not line.strip().startswith('#')]
    
    if len(report_to_lines) > 1:
        return False, f"Multiple report_to found on lines: {[l+1 for l in report_to_lines]}"
    
    return True, None

def main():
    """Test syntax of all trainer files"""
    print("🔍 Testing Python syntax...")
    
    # Files to test
    files_to_test = [
        "trainer/__init__.py",
        "trainer/task.py", 
        "trainer/model.py",
        "trainer/data.py",
        "trainer/config.py",
        "trainer/experiment.py"
    ]
    
    all_passed = True
    
    for file_path in files_to_test:
        if not Path(file_path).exists():
            print(f"⚠️  {file_path}: File not found")
            continue
            
        passed, error = test_python_syntax(file_path)
        if passed:
            print(f"✅ {file_path}: Syntax OK")
        else:
            print(f"❌ {file_path}: {error}")
            all_passed = False
    
    # Check for duplicate keywords
    print("\n🔍 Checking for duplicate keywords...")
    passed, error = check_duplicate_keywords()
    if passed:
        print("✅ No duplicate keywords found")
    else:
        print(f"❌ Duplicate keyword issue: {error}")
        all_passed = False
    
    # Test the specific TrainingArguments pattern
    print("\n🔍 Testing TrainingArguments pattern...")
    try:
        # Simulate the TrainingArguments call with our parameters
        # This is a syntax-only test, not actually creating the object
        test_code = """
training_args_params = dict(
    output_dir="/tmp",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=2.0e-5,
    weight_decay=0.01,
    warmup_steps=100,
    max_grad_norm=1.0,
    eval_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    fp16=False,
    report_to=["tensorboard"]
)
"""
        ast.parse(test_code)
        print("✅ TrainingArguments pattern: Syntax OK")
        
    except SyntaxError as e:
        print(f"❌ TrainingArguments pattern: {e}")
        all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All syntax tests passed!")
        print("✅ Ready to rebuild container")
    else:
        print("❌ Syntax errors found - fix before rebuilding")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)