#!/usr/bin/env python3.10
"""
Build CheXpert schema from MIMIC-CXR CheXpert labels.

This script reads the CheXpert labels CSV file and creates a fast lookup dictionary
for CheXpert labels keyed by study_id.

Usage:
    python3.10 02_build_chexpert_schema.py

Output:
    data/processed/chexpert_dict.json - JSON dictionary with study_id -> labels mapping
"""

import pandas as pd
import json
import pathlib as pl
from typing import Dict, Any

# Configuration
CHEXPERT_CSV = pl.Path("mimic-cxr-2.0.0-chexpert.csv")
OUTPUT_FILE = pl.Path("data/processed/chexpert_dict.json")

def build_chexpert_schema() -> None:
    """
    Build CheXpert schema dictionary from CSV file.
    """
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not CHEXPERT_CSV.exists():
        print(f"Error: CheXpert CSV file {CHEXPERT_CSV} not found!")
        return
    
    print(f"Reading CheXpert labels from: {CHEXPERT_CSV}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(CHEXPERT_CSV)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        # Identify label columns (skip first 4 columns which are IDs/splits)
        id_columns = df.columns[:4]  # subject_id, study_id, and other ID columns
        label_columns = df.columns[4:]  # CheXpert label columns
        
        print(f"ID columns: {list(id_columns)}")
        print(f"Label columns ({len(label_columns)}): {list(label_columns)}")
        
        # Build records dictionary
        records = {}
        missing_study_ids = 0
        
        for idx, row in df.iterrows():
            study_id = row['study_id']
            
            if pd.isna(study_id):
                missing_study_ids += 1
                continue
            
            # Convert study_id to string for consistency (remove .0 if present)
            study_id_str = str(int(study_id)) if isinstance(study_id, float) else str(study_id)
            
            # Build label dictionary for this study
            label_dict = {}
            for col in label_columns:
                value = row[col]
                
                # Handle different value types
                if pd.isna(value):
                    label_dict[col] = -1  # Unknown/uncertain
                elif isinstance(value, (int, float)):
                    # Ensure values are in expected range (-1, 0, 1, 2)
                    if value in [-1, 0, 1, 2]:
                        label_dict[col] = int(value)
                    else:
                        # Map other values to closest valid value
                        if value < 0:
                            label_dict[col] = -1
                        elif value == 0:
                            label_dict[col] = 0
                        elif value <= 1:
                            label_dict[col] = 1
                        else:
                            label_dict[col] = 2
                else:
                    # Handle string values
                    value_str = str(value).lower().strip()
                    if value_str in ['-1', 'uncertain', 'unknown']:
                        label_dict[col] = -1
                    elif value_str in ['0', 'negative', 'no']:
                        label_dict[col] = 0
                    elif value_str in ['1', 'positive', 'yes']:
                        label_dict[col] = 1
                    elif value_str in ['2', 'positive', 'yes']:
                        label_dict[col] = 2
                    else:
                        label_dict[col] = -1  # Default to uncertain
            
            records[study_id_str] = label_dict
        
        # Save to JSON file
        print(f"Saving {len(records)} records to: {OUTPUT_FILE}")
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        
        print(f"\nCheXpert schema built successfully!")
        print(f"Total records: {len(records)}")
        print(f"Missing study IDs: {missing_study_ids}")
        print(f"Label columns: {len(label_columns)}")
        print(f"Output saved to: {OUTPUT_FILE}")
        
        # Show sample record
        if records:
            sample_id = list(records.keys())[0]
            print(f"\nSample record (study_id: {sample_id}):")
            print(json.dumps(records[sample_id], indent=2))
        
        # Show label value distribution
        print(f"\nLabel value distribution:")
        for col in label_columns:
            values = [records[sid][col] for sid in records.keys()]
            value_counts = pd.Series(values).value_counts().sort_index()
            print(f"{col}: {dict(value_counts)}")
        
    except Exception as e:
        print(f"Error processing CheXpert CSV: {e}")
        raise

def validate_schema() -> None:
    """
    Validate the generated CheXpert schema.
    """
    if not OUTPUT_FILE.exists():
        print("Schema file not found for validation!")
        return
    
    print(f"\nValidating schema: {OUTPUT_FILE}")
    
    try:
        with OUTPUT_FILE.open("r", encoding="utf-8") as f:
            records = json.load(f)
        
        print(f"Loaded {len(records)} records for validation")
        
        # Check a few records
        sample_ids = list(records.keys())[:5]
        for study_id in sample_ids:
            record = records[study_id]
            print(f"\nStudy ID: {study_id}")
            print(f"Labels: {record}")
            
            # Validate label values
            invalid_values = []
            for label, value in record.items():
                if value not in [-1, 0, 1, 2]:
                    invalid_values.append((label, value))
            
            if invalid_values:
                print(f"  Invalid values: {invalid_values}")
            else:
                print(f"  All values valid")
        
        print(f"\nSchema validation complete!")
        
    except Exception as e:
        print(f"Error validating schema: {e}")

def main():
    """Main function."""
    print("MIMIC-CXR CheXpert Schema Builder")
    print("=" * 40)
    
    build_chexpert_schema()
    validate_schema()

if __name__ == "__main__":
    main()
