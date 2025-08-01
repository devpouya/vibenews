#!/usr/bin/env python3
"""
Master test runner for VibeNews test suite
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test_suite():
    """Run the complete VibeNews test suite"""
    
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    print("🧪 VibeNews Test Suite")
    print("=" * 50)
    
    # Test categories and their priorities
    test_categories = [
        {
            'name': 'Unit Tests',
            'path': test_dir / 'unit',
            'priority': 'high',
            'description': 'Fast syntax and pattern validation'
        },
        {
            'name': 'Validation Tests', 
            'path': test_dir / 'validation',
            'priority': 'high',
            'description': 'Deployment readiness checks'
        },
        {
            'name': 'Integration Tests',
            'path': test_dir / 'integration', 
            'priority': 'medium',
            'description': 'Complete workflow testing'
        }
    ]
    
    results = {}
    
    for category in test_categories:
        print(f"\n🔍 Running {category['name']}")
        print(f"   {category['description']}")
        print("-" * 40)
        
        category_results = []
        
        # Find all Python test files in the category
        test_files = list(category['path'].glob('*.py'))
        test_files = [f for f in test_files if f.name != '__init__.py']
        
        if not test_files:
            print(f"   ⚠️  No test files found in {category['path']}")
            continue
        
        for test_file in test_files:
            try:
                print(f"   Running {test_file.name}...")
                
                # Change to project root for relative imports
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    print(f"   ✅ {test_file.name}: PASSED")
                    category_results.append(('PASSED', test_file.name, ''))
                else:
                    print(f"   ❌ {test_file.name}: FAILED")
                    category_results.append(('FAILED', test_file.name, result.stderr))
                    
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {test_file.name}: TIMEOUT")
                category_results.append(('TIMEOUT', test_file.name, 'Test timed out after 60 seconds'))
            except Exception as e:
                print(f"   💥 {test_file.name}: ERROR - {e}")
                category_results.append(('ERROR', test_file.name, str(e)))
        
        results[category['name']] = category_results
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    total_tests = 0
    total_passed = 0
    critical_failures = []
    
    for category_name, category_results in results.items():
        passed = len([r for r in category_results if r[0] == 'PASSED'])
        total = len(category_results)
        
        total_tests += total
        total_passed += passed
        
        status = "✅" if passed == total else "❌"
        print(f"{status} {category_name}: {passed}/{total} passed")
        
        # Collect critical failures
        for status, test_name, error in category_results:
            if status in ['FAILED', 'ERROR'] and category_name in ['Unit Tests', 'Validation Tests']:
                critical_failures.append((category_name, test_name, status, error))
    
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    # Report critical failures
    if critical_failures:
        print(f"\n🚨 Critical Issues Found:")
        for category, test, status, error in critical_failures:
            print(f"   {category}/{test}: {status}")
            if error and len(error) < 200:
                print(f"      {error.strip()}")
    
    # Final recommendation
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Ready for deployment.")
        return True
    elif not critical_failures:
        print("\n⚠️  Some non-critical tests failed. Review before deployment.")
        return True
    else:
        print("\n❌ Critical tests failed. Fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)