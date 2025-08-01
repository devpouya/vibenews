#!/usr/bin/env python3
"""
Check Vertex AI job status using Python API
"""

from google.cloud import aiplatform
import sys

def check_job_status():
    """Check all training jobs"""
    try:
        # Initialize Vertex AI
        aiplatform.init(
            project="vibenews-bias-detection",
            location="us-central1"
        )
        
        print("🔍 Checking Vertex AI training jobs...")
        
        # List all training jobs
        jobs = aiplatform.CustomContainerTrainingJob.list()
        
        if not jobs:
            print("❌ No custom training jobs found")
            
            # Check for any pipelines
            from google.cloud import aiplatform_v1
            client = aiplatform_v1.PipelineServiceClient()
            parent = f"projects/vibenews-bias-detection/locations/global"
            
            try:
                pipelines = client.list_training_pipelines(parent=parent)
                pipeline_list = list(pipelines)
                
                if pipeline_list:
                    print(f"📊 Found {len(pipeline_list)} training pipelines:")
                    for pipeline in pipeline_list[:3]:  # Show first 3
                        print(f"  - Name: {pipeline.display_name}")
                        print(f"  - State: {pipeline.state}")
                        print(f"  - Created: {pipeline.create_time}")
                        print()
                else:
                    print("❌ No training pipelines found either")
                    
            except Exception as e:
                print(f"Could not check pipelines: {e}")
            
            return False
        
        print(f"✅ Found {len(jobs)} training jobs:")
        
        for job in jobs:
            print(f"\n📊 Job: {job.display_name}")
            print(f"   Resource: {job.resource_name}")
            print(f"   State: {job.state}")
            print(f"   Created: {job.create_time}")
            
            if hasattr(job, '_gca_resource') and job._gca_resource:
                resource = job._gca_resource
                if hasattr(resource, 'job_spec') and resource.job_spec:
                    worker_spec = resource.job_spec.worker_pool_specs[0] if resource.job_spec.worker_pool_specs else None
                    if worker_spec:
                        print(f"   Machine: {worker_spec.machine_spec.machine_type if worker_spec.machine_spec else 'Unknown'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking jobs: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_job_status()
    sys.exit(0 if success else 1)