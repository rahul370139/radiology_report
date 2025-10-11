#!/bin/bash
# Training Monitoring Script

echo "📊 TRAINING MONITORING"
echo "====================="

# Check if training is running
if pgrep -f "advanced_trainer.py" > /dev/null; then
    echo "✅ Training is running"
    
    # Show GPU usage
    echo "\n🔧 System Resources:"
    top -l 1 | grep "CPU usage"
    
    # Show disk usage
    echo "\n💾 Disk Usage:"
    df -h | grep -E "(Filesystem|/dev/)"
    
    # Show training logs
    echo "\n📝 Recent Training Logs:"
    if [ -f "logs/trainer_state.json" ]; then
        tail -20 logs/trainer_state.json
    fi
    
else
    echo "❌ Training is not running"
fi
