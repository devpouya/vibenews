#!/usr/bin/env python3
"""
Test Vertex AI setup and container build
"""

import subprocess
import sys
import os
from pathlib import Path


def check_gcloud_setup():
    """Check if gcloud is set up correctly"""
    print("🔍 Checking gcloud setup...")
    
    try:
        # Check gcloud is installed
        result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ gcloud CLI not installed")
            return False
        print("✅ gcloud CLI installed")
        
        # Check project is set
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            print("❌ No project set. Run: gcloud config set project YOUR_PROJECT_ID")
            return False
        
        project_id = result.stdout.strip()
        print(f"✅ Project: {project_id}")
        
        # Check authentication
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE'], 
                              capture_output=True, text=True)
        if 'ACTIVE' not in result.stdout:
            print("❌ Not authenticated. Run: gcloud auth login")
            return False
        print("✅ Authenticated")
        
        return True
        
    except FileNotFoundError:
        print("❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk")
        return False


def check_required_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        "Dockerfile",
        "requirements-vertex.txt", 
        "setup.py",
        "trainer/__init__.py",
        "trainer/task.py",
        "trainer/config.py",
        "trainer/data.py",
        "trainer/model.py",
        "trainer/experiment.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True


def check_apis_enabled():
    """Check if required APIs are enabled"""
    print("🔌 Checking APIs...")
    
    required_apis = [
        "aiplatform.googleapis.com",
        "compute.googleapis.com",
        "storage.googleapis.com",
        "cloudbuild.googleapis.com"
    ]
    
    try:
        result = subprocess.run(['gcloud', 'services', 'list', '--enabled'], 
                              capture_output=True, text=True)
        enabled_apis = result.stdout
        
        missing_apis = []
        for api in required_apis:
            if api in enabled_apis:
                print(f"✅ {api}")
            else:
                missing_apis.append(api)
        
        if missing_apis:
            print(f"❌ Missing APIs: {missing_apis}")
            print("Enable with: gcloud services enable " + " ".join(missing_apis))
            return False
        
        print("✅ All required APIs enabled")
        return True
        
    except Exception as e:
        print(f"❌ Could not check APIs: {e}")
        return False


def check_buckets():
    """Check if storage buckets exist"""
    print("🪣 Checking storage buckets...")
    
    try:
        # Get project ID
        result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                              capture_output=True, text=True)
        project_id = result.stdout.strip()
        
        expected_buckets = [
            f"{project_id}-models",
            f"{project_id}-data"
        ]
        
        # List buckets
        result = subprocess.run(['gsutil', 'ls'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Could not list buckets. Make sure gsutil is configured.")
            return False
        
        existing_buckets = result.stdout
        
        missing_buckets = []
        for bucket in expected_buckets:
            bucket_path = f"gs://{bucket}/"
            if bucket_path in existing_buckets:
                print(f"✅ {bucket}")
            else:
                missing_buckets.append(bucket)
        
        if missing_buckets:
            print(f"❌ Missing buckets: {missing_buckets}")
            print("Create with:")
            for bucket in missing_buckets:
                print(f"  gsutil mb -l us-central1 gs://{bucket}")
            return False
        
        print("✅ All required buckets exist")
        return True
        
    except Exception as e:
        print(f"❌ Could not check buckets: {e}")
        return False


def test_docker_build():
    """Test Docker build locally"""
    print("🐳 Testing Docker build...")
    
    if not Path("Dockerfile").exists():
        print("❌ Dockerfile not found")
        return False
    
    # Build test container
    build_cmd = ["docker", "build", "-t", "bias-trainer:test", "."]
    
    try:
        print("Building container (this may take a few minutes)...")
        result = subprocess.run(build_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Docker build failed:")
            print(result.stderr)
            return False
        
        print("✅ Docker build successful")
        
        # Test container help
        help_cmd = ["docker", "run", "--rm", "bias-trainer:test", "--help"]
        result = subprocess.run(help_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Container test failed:")
            print(result.stderr)
            return False
        
        print("✅ Container runs correctly")
        return True
        
    except FileNotFoundError:
        print("❌ Docker not installed. Install from: https://docker.com")
        return False
    except Exception as e:
        print(f"❌ Docker test failed: {e}")
        return False


def show_cost_estimate():
    """Show cost estimates for different configs"""
    print("\n💰 Cost Estimates:")
    print("=================")
    
    configs = {
        "distilbert_ultra_cheap.yaml": {
            "machine": "e2-standard-2 + T4",
            "time": "1-1.5 hours",
            "cost": "$0.30-0.45"
        },
        "bert_cheap.yaml": {
            "machine": "e2-standard-4 + T4", 
            "time": "1.5-2 hours",
            "cost": "$0.60-0.80"
        }
    }
    
    for config, details in configs.items():
        print(f"📊 {config}:")
        print(f"   Machine: {details['machine']}")
        print(f"   Time: {details['time']}")
        print(f"   Cost: {details['cost']}")
        print()


def main():
    """Run all tests"""
    print("🧪 Testing Vertex AI Setup")
    print("=" * 40)
    
    all_checks = [
        check_gcloud_setup,
        check_required_files,
        check_apis_enabled,
        check_buckets,
    ]
    
    # Run all checks
    results = []
    for check in all_checks:
        try:
            result = check()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Check failed: {e}")
            results.append(False)
            print()
    
    # Docker test (optional)
    docker_ok = False
    try:
        docker_ok = test_docker_build()
        print()
    except Exception as e:
        print(f"⚠️ Docker test skipped: {e}")
        print()
    
    # Summary
    print("📋 Summary:")
    print("=" * 20)
    
    if all(results):
        print("✅ All checks passed! Ready for Vertex AI training.")
        
        if docker_ok:
            print("✅ Docker build test passed.")
            print("\n🚀 Ready to submit training jobs!")
        else:
            print("⚠️ Docker test failed, but you can still build on Cloud Build.")
        
        show_cost_estimate()
        
        print("\n📝 Next steps:")
        print("1. Submit ultra-cheap job: python scripts/submit_vertex_job.py --config vertex_configs/distilbert_ultra_cheap.yaml")
        print("2. Monitor at: https://console.cloud.google.com/vertex-ai/training")
        
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)