#!/usr/bin/env python3
"""
Deployment script for Mac Studio server
Transfers all necessary files and sets up the training environment
"""

import subprocess
import os
import sys
from pathlib import Path

# Server configuration
SERVER_IP = "100.77.217.18"
SERVER_USER = "bilbouser"
SERVER_PASSWORD = "fiveplusone6"
SERVER_PATH = "~/radiology_report"

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result

def transfer_files():
    """Transfer all necessary files to the server"""
    print("🚀 DEPLOYING TO MAC STUDIO SERVER")
    print("=" * 50)
    
    # Files to transfer
    files_to_transfer = [
        "data/processed/",
        "train/",
        "advanced_training_config.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    for file_path in files_to_transfer:
        if os.path.exists(file_path):
            print(f"📤 Transferring {file_path}...")
            cmd = f"scp -r {file_path} {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/"
            run_command(cmd)
        else:
            print(f"⚠️  Warning: {file_path} not found, skipping...")
    
    print("✅ File transfer complete!")

def setup_server_environment():
    """Setup the training environment on the server"""
    print("\n🔧 SETTING UP SERVER ENVIRONMENT")
    print("=" * 50)
    
    setup_commands = [
        f"ssh {SERVER_USER}@{SERVER_IP} 'cd {SERVER_PATH} && pwd'",
        f"ssh {SERVER_USER}@{SERVER_IP} 'cd {SERVER_PATH} && ls -la'",
        f"ssh {SERVER_USER}@{SERVER_IP} 'cd {SERVER_PATH} && python3 -m venv venv'",
        f"ssh {SERVER_USER}@{SERVER_IP} 'cd {SERVER_PATH} && source venv/bin/activate && pip install --upgrade pip'",
        f"ssh {SERVER_USER}@{SERVER_IP} 'cd {SERVER_PATH} && source venv/bin/activate && pip install -r requirements.txt'",
        f"ssh {SERVER_USER}@{SERVER_IP} 'cd {SERVER_PATH} && source venv/bin/activate && python -c \"import torch; print(f\\\"PyTorch: {{torch.__version__}}\\\"); print(f\\\"MPS available: {{torch.backends.mps.is_available()}}\\\")\"'"
    ]
    
    for cmd in setup_commands:
        print(f"Running: {cmd}")
        result = run_command(cmd, check=False)
        if result.returncode != 0:
            print(f"Warning: {result.stderr}")
        else:
            print(f"✅ Success: {result.stdout.strip()}")

def create_training_script():
    """Create a training script for the server"""
    training_script = """#!/bin/bash
# Advanced Training Script for Mac Studio Server

echo "🚀 STARTING ADVANCED RADIOLOGY TRAINING"
echo "======================================="

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Check GPU availability
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"

# Run training
echo "Starting advanced curriculum training..."
python train/advanced_trainer.py --config advanced_training_config.yaml --debug

echo "✅ Training complete!"
"""
    
    with open("start_training.sh", "w") as f:
        f.write(training_script)
    
    # Make it executable
    os.chmod("start_training.sh", 0o755)
    
    # Transfer to server
    print("📤 Transferring training script...")
    run_command(f"scp start_training.sh {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/")
    run_command(f"ssh {SERVER_USER}@{SERVER_IP} 'chmod +x {SERVER_PATH}/start_training.sh'")

def create_monitoring_script():
    """Create a monitoring script for training progress"""
    monitoring_script = """#!/bin/bash
# Training Monitoring Script

echo "📊 TRAINING MONITORING"
echo "====================="

# Check if training is running
if pgrep -f "advanced_trainer.py" > /dev/null; then
    echo "✅ Training is running"
    
    # Show GPU usage
    echo "\\n🔧 System Resources:"
    top -l 1 | grep "CPU usage"
    
    # Show disk usage
    echo "\\n💾 Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)"
    
    # Show training logs
    echo "\\n📝 Recent Training Logs:"
    if [ -f "logs/trainer_state.json" ]; then
        tail -20 logs/trainer_state.json
    fi
    
else
    echo "❌ Training is not running"
fi
"""
    
    with open("monitor_training.sh", "w") as f:
        f.write(monitoring_script)
    
    os.chmod("monitor_training.sh", 0o755)
    
    # Transfer to server
    print("📤 Transferring monitoring script...")
    run_command(f"scp monitor_training.sh {SERVER_USER}@{SERVER_IP}:{SERVER_PATH}/")
    run_command(f"ssh {SERVER_USER}@{SERVER_IP} 'chmod +x {SERVER_PATH}/monitor_training.sh'")

def main():
    """Main deployment function"""
    print("🚀 DEPLOYING RADIOLOGY TRAINING TO MAC STUDIO SERVER")
    print("=" * 60)
    print(f"Server: {SERVER_USER}@{SERVER_IP}")
    print(f"Path: {SERVER_PATH}")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("train/dataset.py"):
        print("❌ Error: Please run this script from the radiology_report directory")
        sys.exit(1)
    
    # Transfer files
    transfer_files()
    
    # Setup server environment
    setup_server_environment()
    
    # Create training script
    create_training_script()
    
    # Create monitoring script
    create_monitoring_script()
    
    print("\n🎉 DEPLOYMENT COMPLETE!")
    print("=" * 30)
    print("To start training on the server, run:")
    print(f"ssh {SERVER_USER}@{SERVER_IP}")
    print(f"cd {SERVER_PATH}")
    print("./start_training.sh")
    print("\nTo monitor training:")
    print("./monitor_training.sh")

if __name__ == "__main__":
    main()
