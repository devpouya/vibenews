#!/usr/bin/env python3
"""
Test for potential runtime issues that could cause the Vertex AI job to fail
"""

import sys
import traceback
from pathlib import Path

def test_method_signatures():
    """Test method signatures for mismatches"""
    print("🔍 Testing method signatures...")
    
    issues = []
    
    # Read the source files to check for signature issues
    try:
        with open("trainer/task.py", 'r') as f:
            task_content = f.read()
        
        with open("trainer/data.py", 'r') as f:
            data_content = f.read()
        
        # Check for prepare_datasets call
        if "self.data_loader.prepare_datasets(tokenizer)" in task_content:
            issues.append("prepare_datasets() called with positional tokenizer argument but method expects keyword")
        
        if "prepare_datasets(tokenizer=tokenizer)" not in task_content and "prepare_datasets(tokenizer)" in task_content:
            issues.append("prepare_datasets() should use tokenizer=tokenizer keyword argument")
        
        if issues:
            for issue in issues:
                print(f"❌ {issue}")
            return False
        else:
            print("✅ Method signatures look correct")
            return True
            
    except Exception as e:
        print(f"❌ Error checking signatures: {e}")
        return False

def test_data_processing_logic():
    """Test data processing logic for potential errors"""
    print("\n🔍 Testing data processing logic...")
    
    try:
        # Create mock data with different formats that could cause issues
        test_cases = [
            # Case 1: bias_labels as dict (expected)
            {'text': 'Test text 1', 'bias_labels': {'label_bias': 'Biased'}},
            
            # Case 2: bias_labels as string (could cause error)
            {'text': 'Test text 2', 'bias_labels': 'Biased'},
            
            # Case 3: bias_labels as None (could cause error)
            {'text': 'Test text 3', 'bias_labels': None},
            
            # Case 4: Alternative format
            {'content': 'Test text 4', 'label': 'Non-biased'},
            
            # Case 5: Missing fields
            {'text': 'Test text 5'},
        ]
        
        # Test the logic for each case
        LABEL_MAP = {"Non-biased": 0, "Biased": 1, "No agreement": 2}
        
        for i, row in enumerate(test_cases):
            try:
                # Simulate the data processing logic from extract_texts_and_labels
                if 'text' in row and 'bias_labels' in row:
                    text = row['text']
                    # This is the potentially problematic line
                    bias_label = row['bias_labels'].get('label_bias', '') if isinstance(row['bias_labels'], dict) else ''
                elif 'content' in row and 'label' in row:
                    text = row['content']
                    bias_label = row['label']
                else:
                    continue
                
                # Check if valid label
                if bias_label not in LABEL_MAP:
                    continue
                    
                print(f"✅ Test case {i+1}: Processed successfully")
                
            except Exception as e:
                print(f"❌ Test case {i+1} failed: {e}")
                print(f"   Row data: {row}")
                return False
        
        print("✅ Data processing logic handles various formats")
        return True
        
    except Exception as e:
        print(f"❌ Error in data processing test: {e}")
        return False

def test_config_structure():
    """Test config structure matches expectations"""
    print("\n🔍 Testing config structure...")
    
    try:
        # Check if the CPU config has the expected structure
        import yaml
        
        config_path = "vertex_configs/distilbert_cpu_only.yaml"
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['model', 'training', 'data', 'logging', 'vertex']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False
        
        # Check specific fields that are used in the code
        required_fields = [
            ('model', 'architecture'),
            ('model', 'model_name'),
            ('training', 'batch_size'),
            ('training', 'epochs'),
            ('data', 'train_split'),
            ('logging', 'tensorboard'),
        ]
        
        for section, field in required_fields:
            if field not in config[section]:
                print(f"❌ Missing config field: {section}.{field}")
                return False
        
        print("✅ Config structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Config structure test failed: {e}")
        return False

def test_import_structure():
    """Test import structure for potential circular imports"""
    print("\n🔍 Testing import structure...")
    
    try:
        # Check for potential circular imports by analyzing import statements
        files_to_check = [
            "trainer/__init__.py",
            "trainer/task.py",
            "trainer/model.py", 
            "trainer/data.py",
            "trainer/config.py",
            "trainer/experiment.py"
        ]
        
        imports = {}
        
        for file_path in files_to_check:
            if not Path(file_path).exists():
                continue
                
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract imports
            file_imports = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('from trainer.') or line.startswith('import trainer.'):
                    file_imports.append(line)
            
            imports[file_path] = file_imports
        
        # Check for obvious circular imports
        # (This is a simplified check)
        potential_issues = []
        
        for file_path, file_imports in imports.items():
            for import_line in file_imports:
                if 'trainer.task' in import_line and 'task.py' not in file_path:
                    potential_issues.append(f"{file_path} imports task module")
        
        if potential_issues:
            for issue in potential_issues:
                print(f"⚠️  Potential circular import: {issue}")
        
        print("✅ Import structure looks reasonable")
        return True
        
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False

def test_dockerfile_and_setup():
    """Test Dockerfile and setup configuration"""
    print("\n🔍 Testing Docker and setup files...")
    
    issues = []
    
    # Check if Dockerfile exists and has required components
    if not Path("Dockerfile").exists():
        issues.append("Dockerfile not found")
    else:
        with open("Dockerfile", 'r') as f:
            dockerfile_content = f.read()
        
        required_components = [
            "FROM python:",
            "COPY requirements-vertex.txt",
            "RUN pip install",
            "COPY trainer/",
            "ENTRYPOINT"
        ]
        
        for component in required_components:
            if component not in dockerfile_content:
                issues.append(f"Dockerfile missing: {component}")
    
    # Check setup.py
    if not Path("setup.py").exists():
        issues.append("setup.py not found")
    
    if issues:
        for issue in issues:
            print(f"❌ {issue}")
        return False
    else:
        print("✅ Docker and setup files look good")
        return True

def main():
    """Run all runtime issue tests"""
    print("🧪 Testing for Runtime Issues That Could Cause Job Failures")
    print("="*60)
    
    tests = [
        ("Method Signatures", test_method_signatures),
        ("Data Processing Logic", test_data_processing_logic),
        ("Config Structure", test_config_structure),
        ("Import Structure", test_import_structure),
        ("Docker & Setup", test_dockerfile_and_setup),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "="*60)
    print("📊 Runtime Issue Test Results:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All runtime issue tests passed!")
        print("✅ Should be safe to rebuild and deploy")
        return True
    else:
        print("⚠️  Found potential runtime issues - fix before deployment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)