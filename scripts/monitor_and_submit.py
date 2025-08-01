#!/usr/bin/env python3
"""
Monitor build and submit job when ready
"""

import subprocess
import time
import sys
import os

def check_build_status():
    """Check if the container build is complete"""
    try:
        gcloud_path = '/Users/pouyapourjafar/google-cloud-sdk/bin/gcloud'
        result = subprocess.run([
            gcloud_path, 'builds', 'list', '--limit=1', '--format=value(status)'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            status = result.stdout.strip()
            return status
        return "UNKNOWN"
    except Exception as e:
        print(f"Error checking build: {e}")
        return "ERROR"

def submit_job_after_build():
    """Submit training job after build completes"""
    print("🔍 Monitoring container build...")
    
    max_wait = 30  # Maximum wait time in minutes
    check_interval = 30  # Check every 30 seconds
    
    for attempt in range(max_wait * 2):  # max_wait * 60 / check_interval
        status = check_build_status()
        print(f"Build status: {status}")
        
        if status == "SUCCESS":
            print("✅ Container build completed successfully!")
            print("🚀 Submitting training job...")
            
            # Submit the job
            os.system("python scripts/submit_vertex_job.py --config vertex_configs/distilbert_ultra_cheap.yaml --submit-only")
            return True
            
        elif status == "FAILURE":
            print("❌ Container build failed!")
            return False
            
        elif status in ["WORKING", "QUEUED"]:
            print(f"⏳ Build still {status.lower()}... waiting {check_interval}s")
            time.sleep(check_interval)
            
        else:
            print(f"⚠️ Unknown status: {status}")
            time.sleep(check_interval)
    
    print("⏰ Build taking too long, please check manually")
    return False

if __name__ == "__main__":
    submit_job_after_build()