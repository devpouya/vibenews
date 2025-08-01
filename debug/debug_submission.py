#!/usr/bin/env python3
"""
Debug Vertex AI job submission issues
"""

from google.cloud import aiplatform
import sys

def debug_vertex_ai():
    """Debug Vertex AI setup and permissions"""
    print("🔍 Debugging Vertex AI setup...")
    
    try:
        # Initialize Vertex AI
        aiplatform.init(
            project="vibenews-bias-detection",
            location="us-central1"
        )
        print("✅ Vertex AI initialized successfully")
        
        # Check if we can list models (permission test)
        try:
            models = aiplatform.Model.list(limit=1)
            print("✅ Can access Vertex AI Model service")
        except Exception as e:
            print(f"❌ Cannot access Model service: {e}")
        
        # Check if we can list jobs
        try:
            jobs = aiplatform.CustomContainerTrainingJob.list(limit=1)
            print("✅ Can access CustomContainerTrainingJob service")
        except Exception as e:
            print(f"❌ Cannot access CustomContainerTrainingJob service: {e}")
        
        # Test creating a job object (but not submitting)
        try:
            job = aiplatform.CustomContainerTrainingJob(
                display_name="test-job-debug",
                container_uri="gcr.io/vibenews-bias-detection/bias-trainer:latest"
            )
            print("✅ Can create CustomContainerTrainingJob object")
            print(f"   Job resource name: {job.resource_name}")
        except Exception as e:
            print(f"❌ Cannot create CustomContainerTrainingJob: {e}")
        
        # Check quotas
        print("\n📊 Checking quotas...")
        try:
            from google.cloud import compute_v1
            client = compute_v1.RegionsClient()
            project = "vibenews-bias-detection"
            region = "us-central1"
            
            request = compute_v1.GetRegionRequest(
                project=project,
                region=region
            )
            response = client.get(request=request)
            
            print(f"✅ Can access Compute Engine API for region {region}")
            
            # Look for GPU quotas
            for quota in response.quotas:
                if 'GPU' in quota.metric or 'T4' in quota.metric:
                    print(f"   {quota.metric}: {quota.usage}/{quota.limit}")
                    
        except Exception as e:
            print(f"❌ Cannot check quotas: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vertex AI initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_vertex_ai()
    sys.exit(0 if success else 1)