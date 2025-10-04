#!/usr/bin/env python3.10
"""
Build EHR context and ICD targets for MIMIC-CXR studies.

This script links MIMIC-CXR with MIMIC-IV to create compact EHR JSON
and ICD diagnosis flags for each chest X-ray study.

Usage:
    python3.10 12_build_ehr_context.py

Requirements:
    pip install duckdb pandas
    
Output:
    data/processed/ehr_context.jsonl - EHR data and ICD flags for each study
"""

import duckdb
import pandas as pd
import json
import pathlib as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import warnings

warnings.filterwarnings("ignore")

# Configuration
MIMIC_IV_DIR = pl.Path("mimic-iv-3.1")
CXR_STUDY_LIST = pl.Path("cxr-study-list.csv")
MANIFEST_FILE = pl.Path("data/processed/phaseA_manifest.jsonl")
OUTPUT_FILE = pl.Path("data/processed/ehr_context.jsonl")

# Acute ICD codes for pulmonary conditions
ACUTE_ICD_CODES = {
    "pneumonia": ["J18", "J18.0", "J18.1", "J18.2", "J18.8", "J18.9"],
    "pleural_effusion": ["J90", "J91", "J91.0", "J91.8"],
    "pneumothorax": ["J93", "J93.0", "J93.1", "J93.8", "J93.9"],
    "pulmonary_embolism": ["I26", "I26.0", "I26.9"],
    "chf_exacerbation": ["I50", "I50.1", "I50.2", "I50.3", "I50.4", "I50.9"],
    "pulmonary_edema": ["J81.0"],
    "rib_fracture": ["S22", "S22.0", "S22.1", "S22.2", "S22.3", "S22.4"],
    "lung_mass": ["C34", "R91"]
}

def initialize_duckdb() -> duckdb.DuckDBPyConnection:
    """Initialize DuckDB connection and load MIMIC-IV data."""
    print("🔧 Initializing DuckDB and loading MIMIC-IV data...")
    print("=" * 70)
    
    conn = duckdb.connect(database=':memory:')
    
    # Load MIMIC-IV tables
    tables_to_load = {
        "patients": f"{MIMIC_IV_DIR}/hosp/patients.csv.gz",
        "admissions": f"{MIMIC_IV_DIR}/hosp/admissions.csv.gz",
        "diagnoses_icd": f"{MIMIC_IV_DIR}/hosp/diagnoses_icd.csv.gz",
        "labevents": f"{MIMIC_IV_DIR}/hosp/labevents.csv.gz",
        "chartevents": f"{MIMIC_IV_DIR}/icu/chartevents.csv.gz",
        "d_icd_diagnoses": f"{MIMIC_IV_DIR}/hosp/d_icd_diagnoses.csv.gz"
    }
    
    for table_name, file_path in tables_to_load.items():
        if pl.Path(file_path).exists():
            print(f"Loading {table_name}...")
            conn.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT * FROM read_csv_auto('{file_path}')
            """)
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"  ✅ Loaded {count:,} rows")
        else:
            print(f"  ⚠️  {table_name} not found at {file_path}")
    
    return conn

def load_study_metadata() -> pd.DataFrame:
    """Load CXR study metadata with admission times."""
    print("\n📋 Loading CXR study metadata...")
    
    # Load study list
    studies_df = pd.read_csv(CXR_STUDY_LIST)
    print(f"Loaded {len(studies_df)} studies from CXR study list")
    
    # Load manifest to get only studies we're using
    manifest = []
    with MANIFEST_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))
    
    manifest_df = pd.DataFrame(manifest)
    print(f"Using {len(manifest_df)} studies from Phase A manifest")
    
    # Convert study_id to int for matching
    manifest_df['study_id'] = manifest_df['study_id'].astype(int)
    
    # Merge to get subject_id for each study
    merged = pd.merge(
        manifest_df[['study_id', 'subject_id']],
        studies_df,
        left_on='study_id',
        right_on='study_id',
        how='left',
        suffixes=('', '_y')
    )
    
    return merged

def extract_ehr_for_study(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    study_id: str,
    study_time: Optional[datetime] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract EHR data for a specific study.
    
    Args:
        conn: DuckDB connection
        subject_id: Patient subject ID
        study_id: Study ID
        study_time: Time of the chest X-ray study
        
    Returns:
        Dictionary with EHR data or None if not available
    """
    try:
        # Get patient demographics
        patient_query = f"""
            SELECT 
                gender,
                anchor_age
            FROM patients
            WHERE subject_id = {subject_id}
            LIMIT 1
        """
        patient_result = conn.execute(patient_query).fetchone()
        
        if not patient_result:
            return None
        
        gender, age = patient_result
        
        # Get admission for this study
        # Note: Without exact study timestamps, we'll use the most recent admission
        admission_query = f"""
            SELECT 
                hadm_id,
                admittime,
                dischtime,
                admission_type,
                admission_location
            FROM admissions
            WHERE subject_id = {subject_id}
            ORDER BY admittime DESC
            LIMIT 1
        """
        admission_result = conn.execute(admission_query).fetchone()
        
        if not admission_result:
            # Patient has no admissions, use basic demographics only
            return {
                "Age": int(age) if age else None,
                "Sex": "M" if gender == "M" else "F",
                "hadm_id": None,
                "Vitals": {},
                "Labs": {},
                "O2_device": None,
                "Chronic_conditions": []
            }
        
        hadm_id, admittime, dischtime, admission_type, admission_location = admission_result
        
        # Build EHR JSON
        ehr_json = {
            "Age": int(age) if age else None,
            "Sex": "M" if gender == "M" else "F",
            "hadm_id": int(hadm_id) if hadm_id else None,
            "admission_type": admission_type,
            "Vitals": {},
            "Labs": {},
            "O2_device": None,
            "Chronic_conditions": []
        }
        
        # Note: Vitals and labs would require chartevents/labevents with timestamps
        # For now, we'll create the structure and populate with available data
        
        return ehr_json
        
    except Exception as e:
        print(f"Error extracting EHR for study {study_id}: {e}")
        return None

def extract_icd_flags(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    hadm_id: Optional[int]
) -> Dict[str, bool]:
    """
    Extract ICD diagnosis flags for acute conditions.
    
    Args:
        conn: DuckDB connection
        subject_id: Patient subject ID
        hadm_id: Hospital admission ID
        
    Returns:
        Dictionary with boolean flags for each acute condition
    """
    icd_flags = {condition: False for condition in ACUTE_ICD_CODES.keys()}
    
    if not hadm_id:
        return icd_flags
    
    try:
        # Get all ICD codes for this admission
        icd_query = f"""
            SELECT icd_code, icd_version
            FROM diagnoses_icd
            WHERE subject_id = {subject_id}
            AND hadm_id = {hadm_id}
        """
        icd_results = conn.execute(icd_query).fetchall()
        
        if not icd_results:
            return icd_flags
        
        # Check each ICD code against our acute conditions
        for icd_code, icd_version in icd_results:
            if not icd_code:
                continue
                
            icd_code = str(icd_code).strip()
            
            # Check each condition
            for condition, code_prefixes in ACUTE_ICD_CODES.items():
                for prefix in code_prefixes:
                    if icd_code.startswith(prefix):
                        icd_flags[condition] = True
                        break
        
        return icd_flags
        
    except Exception as e:
        print(f"Error extracting ICD flags for hadm_id {hadm_id}: {e}")
        return icd_flags

def build_ehr_context() -> None:
    """Build EHR context for all studies."""
    print("🏥 BUILDING EHR CONTEXT")
    print("=" * 70)
    
    # Initialize DuckDB
    conn = initialize_duckdb()
    
    # Load study metadata
    studies_df = load_study_metadata()
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each study
    print(f"\n🔄 Processing {len(studies_df)} studies...")
    
    records_processed = 0
    records_with_ehr = 0
    records_with_icd = 0
    
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for idx, row in studies_df.iterrows():
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(studies_df)} studies...")
            
            subject_id = int(row['subject_id'])
            study_id = str(row['study_id'])
            
            # Extract EHR data
            ehr_json = extract_ehr_for_study(conn, subject_id, study_id)
            
            if not ehr_json:
                continue
            
            records_with_ehr += 1
            
            # Extract ICD flags
            hadm_id = ehr_json.get('hadm_id')
            icd_json = extract_icd_flags(conn, subject_id, hadm_id)
            
            if any(icd_json.values()):
                records_with_icd += 1
            
            # Write record
            record = {
                "study_id": study_id,
                "subject_id": subject_id,
                "ehr_json": ehr_json,
                "icd_json": icd_json
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_processed += 1
    
    # Close connection
    conn.close()
    
    print(f"\n✅ EHR Context Building Complete!")
    print(f"=" * 70)
    print(f"Total records processed: {records_processed}")
    print(f"Records with EHR data: {records_with_ehr}")
    print(f"Records with ICD flags: {records_with_icd}")
    print(f"Output saved to: {OUTPUT_FILE}")

def analyze_ehr_context() -> None:
    """Analyze the generated EHR context."""
    print(f"\n📊 ANALYZING EHR CONTEXT")
    print("=" * 70)
    
    if not OUTPUT_FILE.exists():
        print("EHR context file not found!")
        return
    
    # Load records
    records = []
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Total records: {len(records)}")
    
    # Analyze demographics
    ages = [r['ehr_json']['Age'] for r in records if r['ehr_json'].get('Age')]
    sexes = [r['ehr_json']['Sex'] for r in records if r['ehr_json'].get('Sex')]
    
    if ages:
        print(f"\nAge distribution:")
        print(f"  Mean: {sum(ages)/len(ages):.1f}")
        print(f"  Min: {min(ages)}, Max: {max(ages)}")
    
    if sexes:
        sex_counts = pd.Series(sexes).value_counts()
        print(f"\nSex distribution:")
        for sex, count in sex_counts.items():
            print(f"  {sex}: {count} ({count/len(sexes)*100:.1f}%)")
    
    # Analyze ICD flags
    print(f"\nICD Flag Distribution:")
    icd_counts = {}
    for record in records:
        for condition, flag in record['icd_json'].items():
            if flag:
                icd_counts[condition] = icd_counts.get(condition, 0) + 1
    
    for condition, count in sorted(icd_counts.items()):
        print(f"  {condition}: {count} ({count/len(records)*100:.1f}%)")
    
    # Analyze admissions
    with_admission = sum(1 for r in records if r['ehr_json'].get('hadm_id'))
    print(f"\nRecords with admission data: {with_admission} ({with_admission/len(records)*100:.1f}%)")

def main():
    """Main function."""
    print("MIMIC-CXR EHR Context Builder")
    print("=" * 70)
    
    try:
        # Build EHR context
        build_ehr_context()
        
        # Analyze results
        analyze_ehr_context()
        
        print(f"\n🎉 EHR context building complete!")
        print(f"📁 Output: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error building EHR context: {e}")
        raise

if __name__ == "__main__":
    main()
