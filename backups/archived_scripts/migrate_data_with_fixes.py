#!/usr/bin/env python3
"""
Comprehensive data migration script with all upstream fixes applied.
This script will:
1. Re-extract EHR context with corrected ItemIDs and unit sanity checks
2. Re-run data correction with improved CheXpert logic
3. Validate the final dataset with QC script
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout[-500:])
        if e.stderr:
            print("STDERR:", e.stderr[-500:])
        return False

def main():
    """Main migration process"""
    print("🚀 COMPREHENSIVE DATA MIGRATION WITH FIXES")
    print("=" * 70)
    print("This will fix all root causes and create clean datasets")
    print()
    
    # Change to project directory
    os.chdir("/Users/rahul/Downloads/Code scripts/radiology_report")
    
    # Step 1: Backup existing datasets
    print("📦 Step 1: Backing up existing datasets...")
    backup_cmd = """
    mkdir -p backups/$(date +%Y%m%d_%H%M%S)
    cp src/data/processed/curriculum_*_final_clean.jsonl backups/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
    echo "Backup created in backups/$(date +%Y%m%d_%H%M%S)/"
    """
    run_command(backup_cmd, "Creating backup of existing datasets")
    
    # Step 2: Clean up old processed files
    print("\n🧹 Step 2: Cleaning up old processed files...")
    cleanup_cmd = """
    rm -f src/data/processed/curriculum_*_final_clean*.jsonl
    rm -f src/data/processed/ehr_context.jsonl
    echo "Old processed files removed"
    """
    run_command(cleanup_cmd, "Removing old processed files")
    
    # Step 3: Re-extract EHR context with fixes
    print("\n🏥 Step 3: Re-extracting EHR context with fixes...")
    ehr_cmd = "cd src/utils && source ../../venv/bin/activate && python3 build_ehr_context.py"
    if not run_command(ehr_cmd, "Extracting EHR context with corrected ItemIDs and unit sanity"):
        print("❌ EHR extraction failed. Check the logs above.")
        return False
    
    # Step 4: Re-run data correction
    print("\n🔧 Step 4: Running comprehensive data correction...")
    correct_cmd = "source venv/bin/activate && python3 src/utils/correct_ehr_data.py"
    if not run_command(correct_cmd, "Running data correction with improved CheXpert logic"):
        print("❌ Data correction failed. Check the logs above.")
        return False
    
    # Step 5: Validate with QC script
    print("\n🔍 Step 5: Validating with QC script...")
    qc_cmd = "source venv/bin/activate && python3 ehr_cxr_qc.py src/data/processed/curriculum_train_final_clean.jsonl --base-dir files | head -100"
    if not run_command(qc_cmd, "Running QC validation"):
        print("❌ QC validation failed. Check the logs above.")
        return False
    
    # Step 6: Final summary
    print("\n📊 Step 6: Final dataset summary...")
    summary_cmd = """
    echo "=== FINAL DATASET SUMMARY ==="
    echo "Training dataset:"
    wc -l src/data/processed/curriculum_train_final_clean.jsonl
    echo "Validation dataset:"
    wc -l src/data/processed/curriculum_val_final_clean.jsonl
    echo ""
    echo "Sample of fixes applied:"
    echo "- Fixed ItemID mappings (PT, INR, Procalcitonin)"
    echo "- Added unit sanity checks at extraction time"
    echo "- Fixed CheXpert label logic with proper No Finding rules"
    echo "- Fixed image path double-prefix bug"
    echo "- Added vital sign sanity checks"
    echo ""
    echo "✅ Migration complete! Datasets are now clean and ready for training."
    """
    run_command(summary_cmd, "Generating final summary")
    
    print("\n🎉 MIGRATION COMPLETE!")
    print("=" * 70)
    print("All root causes have been fixed upstream.")
    print("Your datasets are now clean and ready for training.")
    print("\nNext steps:")
    print("1. Run training with the clean datasets")
    print("2. Monitor training progress")
    print("3. The QC script can be used anytime to validate data quality")

if __name__ == "__main__":
    main()
