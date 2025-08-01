#!/usr/bin/env python3
"""
GCP setup helper for running experiments on Google Cloud Platform
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, Any


class GCPSetup:
    """Helper for setting up GCP training infrastructure"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.zone = f"{region}-a"
    
    def check_prerequisites(self) -> bool:
        """Check if gcloud CLI is installed and configured"""
        try:
            result = subprocess.run(['gcloud', 'config', 'get-value', 'project'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                current_project = result.stdout.strip()
                print(f"✅ gcloud CLI configured for project: {current_project}")
                return True
            else:
                print("❌ gcloud CLI not configured")
                return False
        except FileNotFoundError:
            print("❌ gcloud CLI not installed")
            print("Install from: https://cloud.google.com/sdk/docs/install")
            return False
    
    def enable_apis(self):
        """Enable required Google Cloud APIs"""
        apis = [
            "compute.googleapis.com",
            "aiplatform.googleapis.com",
            "storage.googleapis.com"
        ]
        
        print("🔧 Enabling required APIs...")
        for api in apis:
            cmd = ['gcloud', 'services', 'enable', api, '--project', self.project_id]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Enabled {api}")
            else:
                print(f"❌ Failed to enable {api}: {result.stderr}")
    
    def create_storage_bucket(self, bucket_name: str):
        """Create Cloud Storage bucket for storing models and data"""
        print(f"🪣 Creating storage bucket: {bucket_name}")
        
        cmd = [
            'gcloud', 'storage', 'buckets', 'create', 
            f'gs://{bucket_name}',
            '--location', self.region,
            '--project', self.project_id
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created bucket: gs://{bucket_name}")
        else:
            if "already exists" in result.stderr:
                print(f"✅ Bucket already exists: gs://{bucket_name}")
            else:
                print(f"❌ Failed to create bucket: {result.stderr}")
    
    def create_training_instance(self, instance_config: Dict[str, Any]) -> str:
        """Create VM instance for training"""
        instance_name = instance_config.get('name', 'bias-training-vm')
        machine_type = instance_config.get('machine_type', 'n1-standard-4')
        gpu_type = instance_config.get('gpu_type', 'nvidia-tesla-v100')
        gpu_count = instance_config.get('gpu_count', 1)
        
        print(f"🖥️ Creating training instance: {instance_name}")
        
        cmd = [
            'gcloud', 'compute', 'instances', 'create', instance_name,
            '--zone', self.zone,
            '--machine-type', machine_type,
            '--accelerator', f'type={gpu_type},count={gpu_count}',
            '--image-family', 'pytorch-latest-gpu',
            '--image-project', 'deeplearning-platform-release',
            '--boot-disk-size', '100GB',
            '--boot-disk-type', 'pd-ssd',
            '--maintenance-policy', 'TERMINATE',
            '--project', self.project_id
        ]
        
        if instance_config.get('preemptible', True):
            cmd.append('--preemptible')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created instance: {instance_name}")
            return instance_name
        else:
            print(f"❌ Failed to create instance: {result.stderr}")
            return None
    
    def upload_code(self, instance_name: str, local_path: str = "."):
        """Upload code to training instance"""
        print(f"📤 Uploading code to {instance_name}...")
        
        # Create archive of code (excluding large files)
        exclude_patterns = [
            "--exclude=venv",
            "--exclude=.git",
            "--exclude=*.pyc",
            "--exclude=__pycache__",
            "--exclude=runs",
            "--exclude=experiments/runs",
            "--exclude=*.log"
        ]
        
        # Use gcloud compute scp
        cmd = [
            'gcloud', 'compute', 'scp',
            '--recurse',
            '--zone', self.zone,
            local_path,
            f'{instance_name}:~/vibenews',
            '--project', self.project_id
        ] + exclude_patterns
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Code uploaded successfully")
        else:
            print(f"❌ Failed to upload code: {result.stderr}")
    
    def run_training_command(self, instance_name: str, config_file: str):
        """Execute training command on remote instance"""
        print(f"🚀 Starting training on {instance_name}...")
        
        # Setup and training commands
        setup_commands = [
            "cd ~/vibenews",
            "python -m venv venv",
            "source venv/bin/activate",
            "pip install -r requirements.txt",  # We'll need to create this
            f"python run_experiment.py {config_file}"
        ]
        
        cmd = [
            'gcloud', 'compute', 'ssh', instance_name,
            '--zone', self.zone,
            '--project', self.project_id,
            '--command', ' && '.join(setup_commands)
        ]
        
        # Run command (this will be interactive)
        subprocess.run(cmd)
    
    def download_results(self, instance_name: str, remote_path: str = "~/vibenews/experiments/runs"):
        """Download training results from instance"""
        print(f"📥 Downloading results from {instance_name}...")
        
        local_results_dir = Path("experiments/runs_from_gcp")
        local_results_dir.mkdir(exist_ok=True)
        
        cmd = [
            'gcloud', 'compute', 'scp',
            '--recurse',
            '--zone', self.zone,
            f'{instance_name}:{remote_path}/*',
            str(local_results_dir),
            '--project', self.project_id
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Results downloaded to {local_results_dir}")
        else:
            print(f"❌ Failed to download results: {result.stderr}")
    
    def cleanup_instance(self, instance_name: str):
        """Delete training instance to save costs"""
        print(f"🗑️ Deleting instance: {instance_name}")
        
        cmd = [
            'gcloud', 'compute', 'instances', 'delete', instance_name,
            '--zone', self.zone,
            '--project', self.project_id,
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Deleted instance: {instance_name}")
        else:
            print(f"❌ Failed to delete instance: {result.stderr}")
    
    def estimate_costs(self, instance_config: Dict[str, Any], hours: float = 2) -> Dict[str, float]:
        """Estimate training costs"""
        # Rough GCP pricing (as of 2025)
        pricing = {
            'n1-standard-4': 0.19,  # per hour
            'nvidia-tesla-v100': 2.55,  # per GPU per hour
            'nvidia-tesla-t4': 0.95,
            'pd-ssd': 0.17 / (24 * 30)  # per GB per hour
        }
        
        machine_cost = pricing.get(instance_config.get('machine_type', 'n1-standard-4'), 0.19)
        gpu_cost = pricing.get(instance_config.get('gpu_type', 'nvidia-tesla-v100'), 2.55)
        gpu_count = instance_config.get('gpu_count', 1)
        storage_cost = 100 * pricing['pd-ssd']  # 100GB disk
        
        total_hourly = machine_cost + (gpu_cost * gpu_count) + storage_cost
        
        if instance_config.get('preemptible', True):
            total_hourly *= 0.2  # 80% discount for preemptible
        
        total_cost = total_hourly * hours
        
        return {
            'hourly_cost': round(total_hourly, 2),
            'estimated_total': round(total_cost, 2),
            'machine_cost': round(machine_cost * hours, 2),
            'gpu_cost': round(gpu_cost * gpu_count * hours, 2),
            'storage_cost': round(storage_cost * hours, 2),
            'hours': hours
        }


def main():
    """Interactive GCP setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GCP setup for bias classification training")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--setup-only", action="store_true", help="Only setup APIs and buckets")
    
    args = parser.parse_args()
    
    setup = GCPSetup(args.project, args.region)
    
    # Check prerequisites
    if not setup.check_prerequisites():
        print("❌ Prerequisites not met")
        sys.exit(1)
    
    # Setup APIs and buckets
    setup.enable_apis()
    setup.create_storage_bucket(f"{args.project}-bias-models")
    
    if args.setup_only:
        print("✅ GCP setup complete!")
        return
    
    # Interactive training setup
    print("\n🖥️ Training Instance Configuration:")
    instance_config = {
        'name': 'bias-training-vm',
        'machine_type': 'n1-standard-4',
        'gpu_type': 'nvidia-tesla-v100',
        'gpu_count': 1,
        'preemptible': True
    }
    
    # Show cost estimate
    costs = setup.estimate_costs(instance_config)
    print(f"\n💰 Estimated costs:")
    print(f"  Hourly: ${costs['hourly_cost']}")
    print(f"  2-hour training: ${costs['estimated_total']}")
    
    confirm = input("\n🚀 Create training instance? (y/n): ")
    if confirm.lower() == 'y':
        instance_name = setup.create_training_instance(instance_config)
        if instance_name:
            print(f"\n✅ Instance ready: {instance_name}")
            print("\n📋 Next steps:")
            print(f"1. Upload code: python scripts/gcp_setup.py --project {args.project} --upload {instance_name}")
            print(f"2. Start training: python scripts/gcp_setup.py --project {args.project} --train {instance_name} experiments/configs/bert_baseline.yaml")
            print(f"3. Download results: python scripts/gcp_setup.py --project {args.project} --download {instance_name}")
            print(f"4. Cleanup: python scripts/gcp_setup.py --project {args.project} --cleanup {instance_name}")


if __name__ == "__main__":
    main()