#!/usr/bin/env python3
"""
Submit training job to Vertex AI
"""

import argparse
import yaml
import os
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage

# Your project configuration - WILL BE AUTO-DETECTED
PROJECT_ID = None  # Will be detected from gcloud config
REGION = "us-central1"
STAGING_BUCKET = None  # Will be set based on project
DATA_BUCKET = None  # Will be set based on project

# Container image (will be built)
CONTAINER_IMAGE_URI = None  # Will be set based on project


def upload_config_to_gcs(config_path: str, bucket_name: str) -> str:
    """Upload config to GCS and return GCS path"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Generate unique config name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = f"configs/config_{timestamp}.yaml"
    
    blob = bucket.blob(config_name)
    blob.upload_from_filename(config_path)
    
    gcs_path = f"gs://{bucket_name}/{config_name}"
    print(f"✅ Config uploaded to: {gcs_path}")
    return gcs_path


def build_and_push_container():
    """Build and push container to Google Container Registry"""
    detect_project_config()
    
    print("🐳 Building and pushing container...")
    print(f"Target image: {CONTAINER_IMAGE_URI}")
    
    # Find gcloud executable
    import shutil
    gcloud_path = shutil.which('gcloud')
    if not gcloud_path:
        gcloud_path = '/Users/pouyapourjafar/google-cloud-sdk/bin/gcloud'
    
    # Build container
    build_cmd = f"{gcloud_path} builds submit --tag {CONTAINER_IMAGE_URI} ."
    
    print(f"Running: {build_cmd}")
    print("This will take 5-10 minutes...")
    
    os.system(build_cmd)
    print("✅ Container built and pushed!")


def detect_project_config():
    """Auto-detect project configuration"""
    global PROJECT_ID, STAGING_BUCKET, DATA_BUCKET, CONTAINER_IMAGE_URI
    
    if PROJECT_ID is None:
        # Get project from gcloud config
        import subprocess
        import shutil
        
        # Find gcloud executable
        gcloud_path = shutil.which('gcloud')
        if not gcloud_path:
            # Try common locations
            possible_paths = [
                '/Users/pouyapourjafar/google-cloud-sdk/bin/gcloud',
                '/usr/local/bin/gcloud',
                '/opt/homebrew/bin/gcloud'
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    gcloud_path = path
                    break
        
        if not gcloud_path:
            raise ValueError("gcloud not found. Please ensure it's installed and in PATH")
        
        result = subprocess.run([gcloud_path, 'config', 'get-value', 'project'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            PROJECT_ID = result.stdout.strip()
            STAGING_BUCKET = f"{PROJECT_ID}-models"
            DATA_BUCKET = f"{PROJECT_ID}-data"
            CONTAINER_IMAGE_URI = f"gcr.io/{PROJECT_ID}/bias-trainer:latest"
            print(f"✅ Detected project: {PROJECT_ID}")
        else:
            raise ValueError("Could not detect project ID. Run 'gcloud config set project YOUR_PROJECT_ID'")


def submit_training_job(config_path: str, job_name: str = None):
    """Submit training job to Vertex AI"""
    
    # Auto-detect project configuration
    detect_project_config()
    
    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{STAGING_BUCKET}"
    )
    
    # Upload config to GCS
    config_gcs_path = upload_config_to_gcs(config_path, STAGING_BUCKET)
    
    # Load config to get job details
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate job name if not provided
    if not job_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"bert-bias-{timestamp}"
    
    # Prepare training arguments
    training_args = [
        f"--model-dir=gs://{STAGING_BUCKET}/models/{job_name}",
        f"--data-path=gs://{DATA_BUCKET}/datasets/babe_with_annotations_20250731.jsonl",
        f"--config-path={config_gcs_path}",
        f"--project-id={PROJECT_ID}",
        f"--region={REGION}",
        f"--staging-bucket=gs://{STAGING_BUCKET}",
        f"--tensorboard-log-dir=gs://{STAGING_BUCKET}/tensorboard/{job_name}",
        f"--job-name={job_name}",
    ]
    
    # Get machine configuration from config
    vertex_config = config.get('vertex', {})
    machine_type = vertex_config.get('machine_type', 'n1-standard-4')
    accelerator_type = vertex_config.get('accelerator_type')  # No default - None means CPU-only
    accelerator_count = vertex_config.get('accelerator_count', 1)
    
    print(f"🚀 Submitting job: {job_name}")
    print(f"📊 Config: {config_path}")
    if accelerator_type:
        print(f"🖥️ Machine: {machine_type} + {accelerator_count}x {accelerator_type}")
        print(f"💰 Estimated cost: ~$1.35 (2.5 hours)")
    else:
        print(f"🖥️ Machine: {machine_type} (CPU-only)")
        print(f"💰 Estimated cost: ~$0.25 (3 hours)")
    
    # Create and submit custom training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=CONTAINER_IMAGE_URI
    )
    
    # Prepare job.run arguments
    run_args = {
        "args": training_args,
        "replica_count": 1,
        "machine_type": machine_type,
        "sync": False,  # Don't wait for completion
    }
    
    # Only add accelerator if specified in config
    if accelerator_type:
        run_args["accelerator_type"] = accelerator_type
        run_args["accelerator_count"] = accelerator_count
    
    # Submit job
    model = job.run(**run_args)
    
    print(f"✅ Job submitted successfully!")
    print(f"📝 Job name: {job_name}")
    print(f"🔗 Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    print(f"📊 Tensorboard: gs://{STAGING_BUCKET}/tensorboard/{job_name}")
    print(f"💾 Model output: gs://{STAGING_BUCKET}/models/{job_name}")
    
    return job_name


def monitor_job(job_name: str):
    """Monitor training job progress"""
    print(f"📊 Monitoring job: {job_name}")
    print("You can monitor the job in several ways:")
    print(f"1. Web Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    print(f"2. Command line: gcloud ai custom-jobs describe {job_name} --region={REGION}")
    print(f"3. Logs: gcloud logging read 'resource.labels.job_id=\"{job_name}\"' --limit=50")


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI training job")
    parser.add_argument("--config", required=True, help="Path to job config YAML")
    parser.add_argument("--job-name", help="Custom job name")
    parser.add_argument("--build-container", action="store_true", help="Build and push container")
    parser.add_argument("--submit-only", action="store_true", help="Only submit job (skip container build)")
    
    args = parser.parse_args()
    
    try:
        # Build container if requested
        if args.build_container or not args.submit_only:
            build_and_push_container()
        
        # Submit job
        job_name = submit_training_job(args.config, args.job_name)
        
        # Show monitoring info
        monitor_job(job_name)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())