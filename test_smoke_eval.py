#!/usr/bin/env python3
"""
Quick smoke test for evaluation pipeline
Runs just 2 samples to test JSON generation
"""

import subprocess
import sys
import os

def run_smoke_test():
    """Run smoke test evaluation on server"""
    print("🧪 Running smoke test evaluation...")
    
    # Set environment variables for better JSON generation
    env = os.environ.copy()
    env.update({
        "GEN_TEMPERATURE": "0.1",
        "GEN_MAX_NEW_TOKENS": "512", 
        "GEN_TOP_P": "0.9",
        "GEN_DO_SAMPLE": "true"
    })
    
    # Run evaluation with smoke test manifest
    cmd = [
        "python", "src/evaluation/eval_batch_simple.py",
        "--manifest", "evaluation/demo_manifest_smoke.csv",
        "--output_dir", "evaluation/results",
        "--device", "cpu"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    print("\nSTDERR:")
    print(result.stderr)
    print(f"\nReturn code: {result.returncode}")
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_smoke_test()
    if success:
        print("✅ Smoke test completed successfully!")
    else:
        print("❌ Smoke test failed!")
        sys.exit(1)
