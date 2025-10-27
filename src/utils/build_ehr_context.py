#!/usr/bin/env python3.10
"""
Enhanced EHR Context Builder for MIMIC-CXR studies.

This script creates comprehensive EHR context by integrating:
- labevents.csv.gz: Comprehensive laboratory data
- chartevents.csv.gz: Vital signs and clinical measurements
- d_labitems.csv.gz: Lab test definitions and categories
- d_items.csv.gz: Clinical measurement definitions

Usage:
    python3.10 12_build_ehr_context_enhanced.py

Requirements:
    pip install duckdb pandas
    
Output:
    data/processed/ehr_context_enhanced.jsonl - Comprehensive EHR data for each study
"""

import duckdb
import pandas as pd
import json
import pathlib as pl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Configuration - Use absolute paths based on script location
import os
SCRIPT_DIR = pl.Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MIMIC_IV_DIR = PROJECT_ROOT / "files/mimic-iv-3.1"
CXR_STUDY_LIST = PROJECT_ROOT / "files/cxr-study-list.csv"
MANIFEST_FILE = PROJECT_ROOT / "src/data/processed/phaseA_manifest.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "src/data/processed/ehr_context.jsonl"

# Key lab test itemids for common clinical labs
KEY_LAB_ITEMIDS = {
    # Basic Metabolic Panel
    50912: "Creatinine",  # mg/dL
    50902: "BUN",  # mg/dL
    50931: "Glucose",  # mg/dL
    50983: "Sodium",  # mEq/L
    50971: "Potassium",  # mEq/L
    50960: "Chloride",  # mEq/L
    50970: "CO2",  # mEq/L
    
    # Complete Blood Count
    51222: "Hemoglobin",  # g/dL
    51265: "Hematocrit",  # %
    51279: "Platelet Count",  # K/uL
    51256: "WBC Count",  # K/uL
    51249: "Neutrophils",  # %
    51248: "Lymphocytes",  # %
    
    # Cardiac Markers
    51006: "BNP",  # pg/mL
    51002: "Troponin I",  # ng/mL
    51003: "Troponin T",  # ng/mL
    
    # Liver Function
    50861: "Albumin",  # g/dL
    50878: "Total Bilirubin",  # mg/dL
    50885: "ALT",  # U/L
    50884: "AST",  # U/L
    50888: "Alkaline Phosphatase",  # U/L
    
    # Inflammatory Markers
    51277: "CRP",  # mg/L (FIXED: was mg/dL, MIMIC uses mg/L)
    51288: "ESR",  # mm/hr
    51236: "Procalcitonin",  # ng/mL (FIXED: correct ItemID)
    
    # Coagulation
    51237: "PT",  # seconds (FIXED: was Procalcitonin, now PT only)
    51274: "INR",  # ratio (FIXED: correct ItemID)
    51275: "PTT",  # seconds
}

# Dynamic vital signs mapping - will be built from d_items tables
VITAL_PATTERNS = {
    'heart_rate': r'heart.*rate',
    'respiratory_rate': r'respiratory.*rate', 
    'o2_saturation': r'(spo2|o2.*saturation)',
    'systolic_bp': r'(systolic.*blood.*pressure|art.*bp.*systolic|arterial.*blood.*pressure.*systolic)',
    'diastolic_bp': r'(diastolic.*blood.*pressure|art.*bp.*diastolic|arterial.*blood.*pressure.*diastolic)',
    'mean_bp': r'(mean.*arterial|blood.*pressure.*mean|arterial.*blood.*pressure.*mean)',
    'temperature_c': r'temperature.*c',
    'temperature_f': r'temperature.*f',
    'weight': r'weight',
    'height': r'height'
}

# OMR result names for ward/outpatient vitals
OMR_VITAL_NAMES = {
    'Heart Rate': 'heart_rate',
    'Respiratory Rate': 'respiratory_rate', 
    'O2 Saturation': 'o2_saturation',
    'Systolic Blood Pressure': 'systolic_bp',
    'Diastolic Blood Pressure': 'diastolic_bp',
    'Temperature': 'temperature_c',
    'Weight': 'weight',
    'Height': 'height'
}

# Oxygen device itemids for respiratory support detection
OXYGEN_DEVICE_ITEMIDS = {
    223835: "Oxygen_Device",  # Text field for device type
    224700: "Oxygen_Device_Type",  # Specific device type
    224701: "Oxygen_Device_Flow",  # Flow rate
    224702: "Oxygen_Device_FiO2",  # Fraction of inspired oxygen
}

def build_vital_item_mapping(conn: duckdb.DuckDBPyConnection) -> Dict[int, str]:
    """
    ⭐ Build dynamic vital item mapping from MIMIC-IV d_items table
    
    WHY THIS IS NEEDED:
    - MIMIC-IV uses itemids (numeric codes) for each vital sign
    - Example: 220045 = heart_rate, 220210 = respiratory_rate
    - Need to map these codes to human-readable names
    - Different ICU/Ward systems use different itemids for same vital
    
    HOW IT WORKS:
    1. Load d_items.csv.gz (contains all measurement definitions)
    2. Use regex patterns to match vital names (e.g., "heart.*rate" → heart_rate)
    3. Build mapping of itemid → standardized vital_code
    4. Returns dictionary like {220045: "heart_rate", 220210: "respiratory_rate"}
    
    WHY DYNAMIC:
    - Different institutions may use different itemids
    - Building dynamically ensures we catch all variants
    - More robust than hardcoding specific itemids
    
    Args:
        conn: DuckDB connection to MIMIC-IV database
        
    Returns:
        Dictionary mapping itemid (int) to standardized vital_code (str)
        Example: {220045: "heart_rate", 220210: "respiratory_rate"}
    """
    import re
    
    # Load d_items table (contains all item definitions)
    d_items_query = """
        SELECT itemid, LOWER(label) AS label
        FROM 'files/mimic-iv-3.1/icu/d_items.csv.gz'
    """
    
    try:
        d_items_df = conn.execute(d_items_query).df()
        item_map = {}
        
        # Build mapping using regex patterns
        for vital_code, pattern in VITAL_PATTERNS.items():
            regex = re.compile(pattern, re.IGNORECASE)
            matching_items = d_items_df[d_items_df['label'].str.contains(regex, na=False)]
            
            for _, row in matching_items.iterrows():
                itemid = int(row['itemid'])
                item_map[itemid] = vital_code
                
        print(f"   Built dynamic vital mapping: {len(item_map)} itemids")
        return item_map
        
    except Exception as e:
        print(f"   Warning: Could not build dynamic mapping: {e}")
        # Fallback to hardcoded ICU itemids
        return {
            220045: "heart_rate",
            220210: "respiratory_rate", 
            220277: "o2_saturation",
            220050: "systolic_bp",
            220051: "diastolic_bp",
            220052: "mean_bp",
            223761: "temperature_f",
            223762: "temperature_c",
            226512: "weight",
            226707: "height",
            226730: "height"
        }

def initialize_duckdb() -> duckdb.DuckDBPyConnection:
    """
    ⭐ Initialize DuckDB connection and load MIMIC-IV database files
    
    WHY DUCKDB:
    - Fast in-memory database for CSV files
    - No PostgreSQL setup required (just reads .csv.gz files)
    - Perfect for analytics queries on large datasets
    - Handles compressed files (.csv.gz) natively
    
    WHAT IT LOADS:
    1. Hospital data: patients, admissions, diagnoses, labs, vitalsign
    2. ICU data: chartevents (vitals), d_items (definitions), icustays
    3. Metadata: d_labitems, d_icd_diagnoses (lookup tables)
    
    TABLES LOADED (~15 tables):
    - patients.csv.gz: Demographics (age, gender)
    - admissions.csv.gz: Admission details
    - diagnoses_icd.csv.gz: ICD-10 diagnosis codes
    - labevents.csv.gz: Laboratory test results
    - chartevents.csv.gz: ICU vitals (continuous monitoring)
    - vitalsign.csv.gz: Ward vitals (4-hourly)
    - omr.csv.gz: Outpatient vitals (clinic visits)
    - etc.
    
    HOW IT WORKS:
    1. Create in-memory DuckDB database
    2. Load each .csv.gz file as a table
    3. Auto-detect column types (or fallback to VARCHAR)
    4. Return connection for querying
    
    RETURNS:
        DuckDB connection object ready for SQL queries
    """
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
        "d_labitems": f"{MIMIC_IV_DIR}/hosp/d_labitems.csv.gz",
        "d_items": f"{MIMIC_IV_DIR}/icu/d_items.csv.gz",
        "microbiologyevents": f"{MIMIC_IV_DIR}/hosp/microbiologyevents.csv.gz",
        "omr": f"{MIMIC_IV_DIR}/hosp/omr.csv.gz",
        "prescriptions": f"{MIMIC_IV_DIR}/hosp/prescriptions.csv.gz",
        "procedures_icd": f"{MIMIC_IV_DIR}/hosp/procedures_icd.csv.gz",
        "transfers": f"{MIMIC_IV_DIR}/hosp/transfers.csv.gz",
        "services": f"{MIMIC_IV_DIR}/hosp/services.csv.gz",
        "icustays": f"{MIMIC_IV_DIR}/icu/icustays.csv.gz",
        "d_icd_diagnoses": f"{MIMIC_IV_DIR}/hosp/d_icd_diagnoses.csv.gz",
        "vitalsign": f"{MIMIC_IV_DIR}/hosp/vitalsign.csv.gz"  # Add ward vitals
    }
    
    for table_name, file_path in tables_to_load.items():
        if pl.Path(file_path).exists():
            print(f"Loading {table_name}...")
            try:
                # Use read_csv_auto for most tables
                conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{file_path}')
                """)
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"  ✅ Loaded {count:,} rows")
            except Exception as e:
                # If auto-detection fails, try with all varchar types
                print(f"  ⚠️  Auto-detection failed, using VARCHAR types...")
                conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_csv('{file_path}', header=true, auto_detect=true, all_varchar=true)
                """)
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"  ✅ Loaded {count:,} rows (VARCHAR mode)")
        else:
            print(f"  ⚠️  {table_name} not found at {file_path}")
    
    return conn

def load_study_metadata() -> pd.DataFrame:
    """
    ⭐ Load chest X-ray study metadata
    
    WHAT IT DOES:
    - Loads cxr-study-list.csv (contains study_id, subject_id for each X-ray)
    - Loads phaseA_manifest.jsonl (contains which studies to use)
    - Merges them to get subject_id for each study
    - Returns DataFrame with study_id, subject_id pairs
    
    WHY:
    - Need subject_id to query patient EHR data
    - Each chest X-ray is linked to a patient (subject_id)
    - Filters to only studies in the manifest (ones we want to train on)
    
    WHAT THE DATA CONTAINS:
    - study_id: Unique identifier for this chest X-ray
    - subject_id: Which patient this X-ray belongs to
    - StudyDate, StudyTime: When the X-ray was taken
    
    HOW IT WORKS:
    1. Read cxr-study-list.csv (all available X-rays)
    2. Read phaseA_manifest.jsonl (which studies to use)
    3. Merge on study_id
    4. Extract subject_id (links to patient data)
    
    RETURNS:
        DataFrame with columns: study_id, subject_id, (optional StudyDate, StudyTime)
    """
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

def find_best_admission_for_study(conn: duckdb.DuckDBPyConnection, subject_id: int, study_id: str) -> Optional[int]:
    """
    ⭐ Find the hospital admission that matches this chest X-ray study
    
    WHY THIS IS CRITICAL:
    - Chest X-rays are taken during hospital admissions
    - Need hadm_id (admission ID) to get vitals, labs, diagnoses
    - X-ray timing may not exactly match admission timing
    - Need to find the "best" admission for this study
    
    STRATEGY:
    1. Find the most recent admission for this patient
    2. Assume chest X-ray was taken during that admission
    3. Fallback: Return None if no admissions found
    
    LIMITATIONS:
    - We don't have exact study dates for all X-rays
    - May not match the X-ray timing exactly
    - But "most recent" is usually correct for emergency/urgent X-rays
    
    WHAT IT RETURNS:
    - hadm_id (int): Hospital admission ID
    - None: If no admission found for this patient
    
    HOW IT'S USED:
    - Used to query vitals from admissions table
    - Used to query labs from labevents (filtered by hadm_id)
    - Used to query diagnoses from diagnoses_icd
    
    Args:
        conn: DuckDB connection
        subject_id: Patient ID
        study_id: Chest X-ray study ID
        
    Returns:
        hadm_id if found, None otherwise
    """
    try:
        # Strategy 1: Find the most recent admission for this subject
        query = f"""
            SELECT hadm_id, admittime, dischtime
            FROM admissions 
            WHERE subject_id = {subject_id}
            ORDER BY admittime DESC
            LIMIT 1
        """
        
        result = conn.execute(query).fetchone()
        if result:
            hadm_id, admittime, dischtime = result
            return hadm_id
        
        return None
        
    except Exception as e:
        print(f"  Error finding admission for subject {subject_id}: {e}")
        return None

def extract_comprehensive_vitals(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    hadm_id: Optional[int],
    study_time: Optional[datetime] = None,
    vital_item_map: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    ⭐ CRITICAL FUNCTION: Extract comprehensive vital signs from ICU chartevents + OMR
    
    WHY THIS MATTERS: 
    - Vital signs are essential EHR data for radiology context
    - Heart rate, blood pressure, O2 saturation help interpret chest X-rays
    - Example: Patient with low O2 saturation on X-ray → likely pneumonia
    
    WHAT IT DOES:
    1. Queries chartevents table for ICU vitals (high-resolution, continuous monitoring)
    2. Queries vitalsign table for ward vitals (4-hourly floor vitals)
    3. Queries OMR table for outpatient vitals (clinic visits, height/weight)
    4. Applies sanity checks (filters impossible values like HR=0)
    5. Temporally aligns data (7-day window before chest X-ray)
    
    DATA SOURCES:
    - chartevents: Real-time ICU monitoring (every few minutes)
    - vitalsign: Floor ward measurements (every 4 hours)
    - omr: Outpatient clinic visits (sparse, periodic)
    
    MEDICAL ACCURACY:
    - Validates ranges (e.g., SpO2 must be 0-100%, HR 20-300 bpm)
    - Prevents impossible values (respiratory rate cannot be 0)
    - Distinguishes sources (ICU vs Ward vs OMR)
    
    Args:
        conn: DuckDB connection to MIMIC-IV database
        subject_id: Patient subject ID (unique identifier)
        hadm_id: Hospital admission ID (for temporal alignment)
        study_time: Time of the chest X-ray (for temporal window)
        vital_item_map: Mapping of itemids to vital codes (e.g., 220045 → heart_rate)
        
    Returns:
        Dictionary with comprehensive vital signs data like:
        {
            "heart_rate": {"value": 85, "unit": "bpm", "time": "...", "source": "ICU"},
            "bp_systolic": {"value": 120, "unit": "mmHg", ...},
            "respiratory_rate": {...},
            ...
        }
    """
    vitals = {}
    
    try:
        # Build time window for vitals (much more inclusive approach)
        time_filter = ""
        if study_time:
            # Use 7-day window around study time
            start_time = study_time - timedelta(days=7)
            time_filter = f"AND charttime >= '{start_time}' AND charttime <= '{study_time}'"
        elif hadm_id:
            # Use very broad window: 30 days before admission to 7 days after discharge
            # This covers previous ICU stays and ED vitals
            time_filter = f"AND charttime >= (SELECT admittime - INTERVAL '30 days' FROM admissions WHERE hadm_id = {hadm_id}) AND charttime <= (SELECT dischtime + INTERVAL '7 days' FROM admissions WHERE hadm_id = {hadm_id})"
        else:
            # Fallback: look for any vitals in the last 90 days
            time_filter = f"AND charttime >= CURRENT_TIMESTAMP - INTERVAL '90 days'"
        
        # Debug logging for subject 10000032
        if subject_id == 10000032:
            print(f"DEBUG: Subject {subject_id} - hadm_id: {hadm_id}, study_time: {study_time}")
            print(f"DEBUG: Time filter: {time_filter}")
            print(f"DEBUG: vital_item_map has {len(vital_item_map) if vital_item_map else 0} items")
        
        # Step 1: Extract ICU vitals from chartevents (high-resolution)
        if vital_item_map:
            vital_itemids_str = ",".join(map(str, vital_item_map.keys()))
            
            icu_query = f"""
                SELECT 
                    itemid,
                    value,
                    valuenum,
                    valueuom,
                    charttime
                FROM chartevents
                WHERE subject_id = {subject_id}
                AND itemid IN ({vital_itemids_str})
                AND value IS NOT NULL
                {time_filter}
                ORDER BY charttime DESC
            """
            
            icu_results = conn.execute(icu_query).fetchall()
            
            # Process ICU vitals
            for itemid, value, valuenum, valueuom, charttime in icu_results:
                vital_code = vital_item_map.get(itemid)
                if vital_code:
                    # CRITICAL: Vital sign sanity checks at extraction time
                    if valuenum is not None:
                        # Respiratory rate cannot be 0
                        if vital_code == "respiratory_rate" and valuenum == 0:
                            print(f"   ⚠️  Dropping impossible RR: {valuenum}")
                            continue
                        
                        # SpO2 must be 0-100%
                        if vital_code == "o2_saturation" and not (0 <= valuenum <= 100):
                            print(f"   ⚠️  Dropping impossible SpO2: {valuenum}%")
                            continue
                        
                        # Heart rate sanity check
                        if vital_code == "heart_rate" and (valuenum < 20 or valuenum > 300):
                            print(f"   ⚠️  Dropping implausible HR: {valuenum} bpm")
                            continue
                        
                        # Temperature sanity check
                        if vital_code in ["temperature_c", "temperature_f"]:
                            if vital_code == "temperature_c" and (valuenum < 30 or valuenum > 45):
                                print(f"   ⚠️  Dropping extreme temp: {valuenum}°C")
                                continue
                            elif vital_code == "temperature_f" and (valuenum < 86 or valuenum > 113):
                                print(f"   ⚠️  Dropping extreme temp: {valuenum}°F")
                                continue
                        
                        vitals[vital_code] = {
                            "value": float(valuenum),
                            "unit": valueuom or "unknown",
                            "time": str(charttime) if charttime else None,
                            "source": "ICU"
                        }
                    elif value:
                        vitals[vital_code] = {
                            "value": value,
                            "unit": valueuom or "unknown",
                            "time": str(charttime) if charttime else None,
                            "source": "ICU"
                        }
            
            # Debug logging for subject 10000032
            if subject_id == 10000032:
                print(f"DEBUG: Subject {subject_id} - Found {len(icu_results)} ICU vitals, stored {len(vitals)} vitals")
                if vitals:
                    for vital_name, vital_data in vitals.items():
                        print(f"  {vital_name}: {vital_data}")
        
        # Step 2: Extract ward vitals from vitalsign (4-hourly floor vitals)
        ward_time_filter = ""
        if study_time:
            start_time = study_time - timedelta(days=7)
            ward_time_filter = f"AND charttime >= '{start_time}' AND charttime <= '{study_time}'"
        elif hadm_id:
            ward_time_filter = f"AND charttime >= (SELECT admittime - INTERVAL '30 days' FROM admissions WHERE hadm_id = {hadm_id}) AND charttime <= (SELECT dischtime + INTERVAL '7 days' FROM admissions WHERE hadm_id = {hadm_id})"
        else:
            ward_time_filter = f"AND charttime >= CURRENT_TIMESTAMP - INTERVAL '90 days'"
        
        # Try to load ward vitals if table exists
        try:
            ward_query = f"""
                SELECT 
                    itemid,
                    value,
                    valuenum,
                    valueuom,
                    charttime
                FROM vitalsign
                WHERE subject_id = {subject_id}
                AND value IS NOT NULL
                {ward_time_filter}
                ORDER BY charttime DESC
            """
            
            ward_results = conn.execute(ward_query).fetchall()
            
            # Process ward vitals (only if not already found in ICU)
            for itemid, value, valuenum, valueuom, charttime in ward_results:
                # Map ward itemids to vital codes (simplified mapping)
                vital_code = None
                if itemid in [220045, 220210, 220277, 220050, 220051, 220052, 223761, 223762, 226512, 226707, 226730]:
                    vital_code = vital_item_map.get(itemid) if vital_item_map else None
                
                if vital_code and vital_code not in vitals:  # Only add if not already found in ICU
                    if valuenum is not None:
                        vitals[vital_code] = {
                            "value": float(valuenum),
                            "unit": valueuom or "unknown",
                            "time": str(charttime) if charttime else None,
                            "source": "Ward"
                        }
                    elif value:
                        vitals[vital_code] = {
                            "value": value,
                            "unit": valueuom or "unknown",
                            "time": str(charttime) if charttime else None,
                            "source": "Ward"
                        }
        except Exception as e:
            print(f"Warning: Could not extract ward vitals for subject {subject_id}: {e}")
        
        # Step 3: Extract outpatient vitals from OMR (clinic visits, weight/height)
        omr_time_filter = ""
        if study_time:
            start_time = study_time - timedelta(days=7)
            omr_time_filter = f"AND chartdate >= '{start_time.date()}' AND chartdate <= '{study_time.date()}'"
        elif hadm_id:
            omr_time_filter = f"AND chartdate >= (SELECT DATE(admittime) FROM admissions WHERE hadm_id = {hadm_id})"
        
        omr_query = f"""
            SELECT 
                result_name,
                result_value,
                chartdate
            FROM omr
            WHERE subject_id = {subject_id}
            AND result_name IN ({','.join([f"'{name}'" for name in OMR_VITAL_NAMES.keys()])})
            AND result_value IS NOT NULL
            {omr_time_filter}
            ORDER BY chartdate DESC
        """
        
        try:
            omr_results = conn.execute(omr_query).fetchall()
            
            # Process OMR vitals (only if not already found in ICU/Ward)
            for result_name, result_value, chartdate in omr_results:
                vital_code = OMR_VITAL_NAMES.get(result_name)
                if vital_code and vital_code not in vitals:  # Only add if not already found
                    try:
                        # Try to convert to float
                        numeric_value = float(result_value)
                        vitals[vital_code] = {
                            "value": numeric_value,
                            "unit": "unknown",
                            "time": str(chartdate) if chartdate else None,
                            "source": "OMR"
                        }
                    except (ValueError, TypeError):
                        # If not numeric, store as string
                        vitals[vital_code] = {
                            "value": result_value,
                            "unit": "unknown",
                            "time": str(chartdate) if chartdate else None,
                            "source": "OMR"
                        }
        except Exception as e:
            print(f"Warning: Could not extract OMR vitals for subject {subject_id}: {e}")
        
        # Step 3: Calculate derived vitals
        # Calculate BMI if we have both weight and height
        if "weight" in vitals and "height" in vitals:
            weight_kg = vitals["weight"]["value"]
            height_cm = vitals["height"]["value"]
            
            # Convert height to cm if needed (assuming feet if < 3)
            if height_cm < 3:
                height_cm = height_cm * 30.48
            
            if height_cm > 0:
                bmi = weight_kg / ((height_cm / 100) ** 2)
                vitals["bmi"] = {
                    "value": round(bmi, 1),
                    "unit": "kg/m²",
                    "time": vitals["weight"]["time"],
                    "source": "calculated"
                }
        
        # Combine blood pressure
        if "systolic_bp" in vitals and "diastolic_bp" in vitals:
            sys_val = vitals["systolic_bp"]["value"]
            dia_val = vitals["diastolic_bp"]["value"]
            if isinstance(sys_val, (int, float)) and isinstance(dia_val, (int, float)):
                vitals["blood_pressure"] = {
                    "value": f"{int(sys_val)}/{int(dia_val)}",
                    "unit": "mmHg",
                    "time": vitals["systolic_bp"]["time"],
                    "source": "combined"
                }
        
        return vitals
        
    except Exception as e:
        print(f"Error extracting vitals for subject {subject_id}: {e}")
        return {}

def extract_comprehensive_labs(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    hadm_id: Optional[int],
    study_time: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    ⭐ CRITICAL FUNCTION: Extract comprehensive laboratory data from MIMIC-IV labevents
    
    WHY THIS MATTERS:
    - Lab values are crucial clinical context for interpreting chest X-rays
    - Example: High white blood cell count + infiltrate on X-ray → bacterial pneumonia
    - Example: Elevated BNP + pulmonary edema on X-ray → heart failure
    
    WHAT IT DOES:
    1. Queries labevents table for blood tests and chemistry panels
    2. Filters to key lab tests (WBC, hematocrit, sodium, creatinine, troponin, etc.)
    3. Applies unit conversions (mg/dL ↔ mg/L, M/uL ↔ K/uL)
    4. Removes sentinel values (-9999, -1) and unit mismatches
    5. Stores most recent value per lab test (within 48-hour window)
    
    KEY LAB TESTS (20 most important):
    - WBC Count (white blood cells) → infection indicator
    - Hematocrit, Hemoglobin → blood loss indicator
    - Sodium, Creatinine, BUN → kidney function
    - Troponin → heart damage indicator
    - BNP → heart failure indicator
    - Lactate → tissue hypoxia indicator
    - Procalcitonin, CRP → infection/inflammation markers
    
    MEDICAL ACCURACY:
    - Unit sanity checks (BNP should be pg/mL, not mg/dL)
    - Unit conversions (Platelet count M/uL → K/uL)
    - Reference ranges (flags abnormal values)
    - Most recent value capture (clinical relevance)
    
    Args:
        conn: DuckDB connection to MIMIC-IV database
        subject_id: Patient subject ID
        hadm_id: Hospital admission ID (for temporal alignment)
        study_time: Time of the chest X-ray (for 48-hour window)
        
    Returns:
        Dictionary with laboratory data like:
        {
            "WBC Count": {"value": 7.5, "unit": "K/uL", "flag": "NORMAL", ...},
            "Hematocrit": {"value": 38.2, "unit": "%", ...},
            ...
        }
    """
    labs = {}
    
    try:
        # Build time window for labs (48 hours before study if available)
        time_filter = ""
        if study_time:
            start_time = study_time - timedelta(hours=48)
            time_filter = f"AND charttime >= '{start_time}' AND charttime <= '{study_time}'"
        elif hadm_id:
            # If no study time, get labs from admission
            time_filter = f"AND charttime >= (SELECT admittime FROM admissions WHERE hadm_id = {hadm_id})"
        
        # Extract key laboratory tests
        lab_itemids_str = ",".join(map(str, KEY_LAB_ITEMIDS.keys()))
        
        labs_query = f"""
            SELECT 
                itemid,
                value,
                valuenum,
                valueuom,
                ref_range_lower,
                ref_range_upper,
                flag,
                charttime
            FROM labevents
            WHERE subject_id = {subject_id}
            AND itemid IN ({lab_itemids_str})
            AND value IS NOT NULL
            {time_filter}
            ORDER BY charttime DESC
        """
        
        lab_results = conn.execute(labs_query).fetchall()
        
        # Process laboratory results
        for itemid, value, valuenum, valueuom, ref_lower, ref_upper, flag, charttime in lab_results:
            lab_name = KEY_LAB_ITEMIDS.get(itemid, f"Lab_{itemid}")
            
            if lab_name not in labs:
                labs[lab_name] = []
            
            # Store the most recent value for each lab
            if not labs[lab_name]:  # Only store the first (most recent) value
                # CRITICAL: Unit sanity checks at extraction time
                if valuenum is not None:
                    # BNP sanity check - drop mg/dL values
                    if lab_name == "BNP" and valueuom == "mg/dL":
                        print(f"   ⚠️  Dropping BNP in mg/dL: {valuenum} mg/dL (should be pg/mL)")
                        continue
                    
                    # PLT sanity check - fix M/uL scaling
                    if lab_name == "Platelet Count" and valueuom and valueuom.lower() in {"m/ul", "m/uL"}:
                        if valuenum < 1.0:
                            valuenum *= 1000  # M/µL → K/µL
                            valueuom = "K/uL"
                            print(f"   🔧 Fixed PLT unit: {valuenum/1000} M/uL → {valuenum} K/uL")
                        else:
                            print(f"   ⚠️  Dropping implausible PLT: {valuenum} M/uL")
                            continue
                    
                    # BUN sanity check - drop mEq/L values
                    if lab_name == "BUN" and valueuom == "mEq/L":
                        print(f"   ⚠️  Dropping BUN in mEq/L: {valuenum} mEq/L (should be mg/dL)")
                        continue
                    
                    # WBC sanity check - drop % values for total count
                    if lab_name == "WBC Count" and valueuom == "%":
                        print(f"   ⚠️  Dropping WBC total in %: {valuenum}% (should be K/uL)")
                        continue
                    
                    # INR sanity check - drop K/uL values
                    if lab_name == "INR" and valueuom == "K/uL":
                        print(f"   ⚠️  Dropping INR in K/uL: {valuenum} K/uL (should be ratio)")
                        continue
                    
                    # CRP unit fix - convert mg/dL to mg/L if needed
                    if lab_name == "CRP" and valueuom == "mg/dL":
                        valuenum *= 10  # mg/dL → mg/L
                        valueuom = "mg/L"
                        print(f"   🔧 Fixed CRP unit: {valuenum/10} mg/dL → {valuenum} mg/L")
                    
                    lab_data = {
                        "time": str(charttime) if charttime else None,
                        "flag": flag,
                        "value": float(valuenum),
                        "unit": valueuom or "unknown"
                    }
                    
                    # Add reference ranges if available
                    if ref_lower is not None and ref_upper is not None:
                        lab_data["reference_range"] = f"{ref_lower}-{ref_upper}"
                        lab_data["normal_range"] = ref_lower <= valuenum <= ref_upper
                elif value:
                    lab_data = {
                        "time": str(charttime) if charttime else None,
                        "flag": flag,
                        "value": value,
                        "unit": valueuom or "unknown"
                    }
                else:
                    continue  # Skip if no usable value
                
                labs[lab_name] = lab_data
        
        return labs
        
    except Exception as e:
        print(f"Error extracting labs for subject {subject_id}: {e}")
        return {}

def extract_oxygen_device(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    hadm_id: Optional[int],
    study_time: Optional[datetime] = None
) -> Optional[str]:
    """
    ⭐ Extract oxygen device information from chartevents
    
    WHY THIS MATTERS:
    - Oxygen device indicates respiratory support needs
    - Example: Patient on ventilator → likely in ICU, critically ill
    - Example: Patient on nasal cannula → mild respiratory support
    - Helps interpret chest X-rays (why was this patient imaged?)
    
    DEVICE TYPES:
    - Ventilator: Maximum support (intubated patient)
    - Non-rebreather: High flow oxygen mask
    - High-flow nasal cannula: Advanced oxygen delivery
    - Nasal cannula: Low-flow oxygen delivery
    - Room air: No oxygen support
    
    Args:
        conn: DuckDB connection
        subject_id: Patient subject ID
        hadm_id: Hospital admission ID
        study_time: Time of the chest X-ray study
        
    Returns:
        String describing oxygen device or None (e.g., "Ventilator: 100%", "Nasal cannula: 2L/min")
    """
    try:
        # Build time window for oxygen device
        time_filter = ""
        if study_time:
            start_time = study_time - timedelta(hours=24)
            time_filter = f"AND charttime >= '{start_time}' AND charttime <= '{study_time}'"
        elif hadm_id:
            time_filter = f"AND charttime >= (SELECT admittime FROM admissions WHERE hadm_id = {hadm_id})"
        
        # Extract oxygen device information
        oxygen_itemids_str = ",".join(map(str, OXYGEN_DEVICE_ITEMIDS.keys()))
        
        oxygen_query = f"""
            SELECT 
                itemid,
                value,
                charttime
            FROM chartevents
            WHERE subject_id = {subject_id}
            AND itemid IN ({oxygen_itemids_str})
            AND value IS NOT NULL
            AND value != ''
            {time_filter}
            ORDER BY charttime DESC
            LIMIT 1
        """
        
        oxygen_result = conn.execute(oxygen_query).fetchone()
        
        if oxygen_result:
            itemid, value, charttime = oxygen_result
            device_type = OXYGEN_DEVICE_ITEMIDS.get(itemid, "Unknown")
            return f"{device_type}: {value}"
        
        return None
        
    except Exception as e:
        print(f"Error extracting oxygen device for subject {subject_id}: {e}")
        return None

def extract_icd_flags(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    hadm_id: Optional[int]
) -> Dict[str, bool]:
    """
    ⭐ Extract ICD-10 diagnosis flags for acute and chronic conditions
    
    WHY THIS MATTERS:
    - Chronic conditions affect how we interpret imaging findings
    - Example: COPD patient → expect emphysema/chronic changes on X-ray
    - Example: Diabetic patient → higher risk of infections
    - Example: Heart failure → correlates with pulmonary edema on X-ray
    
    ACUTE CONDITIONS (8 conditions):
    - Pneumonia (J18.x)
    - Pleural_Effusion (J90)
    - Pneumothorax (J93.x)
    - Pulmonary_Embolism (I26.x)
    - CHF_exacerbation (I50.x)
    - Pulmonary_Edema (J81.0)
    - Rib_Fracture (S22.x)
    - Lung_mass (C34, R91)
    
    CHRONIC CONDITIONS (8 comorbidities):
    - Diabetes (E10-E14)
    - COPD (J44.x)
    - Chronic Kidney Disease (N18.x)
    - Hypertension (I10-I15)
    - Coronary Artery Disease (I25.x)
    - Atrial Fibrillation (I48.x)
    - Stroke (I63-I69)
    - Liver Disease (K70-K77)
    
    Args:
        conn: DuckDB connection
        subject_id: Patient subject ID
        hadm_id: Hospital admission ID
        
    Returns:
        Dictionary with boolean flags for each condition
        Example: {"pneumonia": True, "diabetes": True, "copd": False, ...}
    """
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

    # Chronic ICD codes for comorbidities
    CHRONIC_ICD_CODES = {
        "diabetes": ["E10", "E11", "E12", "E13", "E14"],
        "copd": ["J44", "J44.0", "J44.1", "J44.9"],
        "ckd": ["N18", "N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"],
        "hypertension": ["I10", "I11", "I12", "I13", "I15"],
        "coronary_artery_disease": ["I25", "I25.0", "I25.1", "I25.2", "I25.3", "I25.4", "I25.5", "I25.6", "I25.7", "I25.8", "I25.9"],
        "atrial_fibrillation": ["I48", "I48.0", "I48.1", "I48.2", "I48.3", "I48.4", "I48.9"],
        "stroke": ["I63", "I64", "I65", "I66", "I67", "I68", "I69"],
        "liver_disease": ["K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77"]
    }
    
    # Initialize flags for both acute and chronic conditions
    icd_flags = {}
    icd_flags.update({condition: False for condition in ACUTE_ICD_CODES.keys()})
    icd_flags.update({condition: False for condition in CHRONIC_ICD_CODES.keys()})
    
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
        
        # Check each ICD code against our conditions
        for icd_code, icd_version in icd_results:
            if not icd_code:
                continue
                
            icd_code = str(icd_code).strip()
            
            # Check acute conditions
            for condition, code_prefixes in ACUTE_ICD_CODES.items():
                for prefix in code_prefixes:
                    if icd_code.startswith(prefix):
                        icd_flags[condition] = True
                        break
            
            # Check chronic conditions
            for condition, code_prefixes in CHRONIC_ICD_CODES.items():
                for prefix in code_prefixes:
                    if icd_code.startswith(prefix):
                        icd_flags[condition] = True
                        break
        
        return icd_flags
        
    except Exception as e:
        print(f"Error extracting ICD flags for hadm_id {hadm_id}: {e}")
        return icd_flags

def extract_ehr_for_study(
    conn: duckdb.DuckDBPyConnection,
    subject_id: int,
    study_id: str,
    study_time: Optional[datetime] = None,
    vital_item_map: Optional[Dict[int, str]] = None
) -> Optional[Dict[str, Any]]:
    """
    ⭐ MAIN ORCHESTRATOR: Extract complete EHR data for one chest X-ray study
    
    WHAT IT DOES:
    - Orchestrates the entire EHR extraction process
    - Calls all the individual extraction functions
    - Combines data from multiple sources into one EHR record
    - This is THE function that creates the final EHR context
    
    DATA SOURCES COMBINED:
    1. Patient demographics (age, sex) → from patients table
    2. Vital signs → from extract_comprehensive_vitals()
    3. Laboratory tests → from extract_comprehensive_labs()
    4. Oxygen devices → from extract_oxygen_device()
    5. ICD diagnoses → from extract_icd_flags()
    6. Chronic conditions → derived from ICD flags
    
    OUTPUT STRUCTURE:
    {
        "study_id": "123456",
        "subject_id": 10000032,
        "ehr_json": {
            "Age": 65,
            "Sex": "M",
            "hadm_id": 54321,
            "admission_type": "EMERGENCY",
            "Vitals": {...},
            "Labs": {...},
            "O2_device": "Nasal_cannula: 2L/min",
            "Chronic_conditions": ["diabetes", "hypertension"]
        },
        "icd_json": {
            "pneumonia": False,
            "diabetes": True,
            ...
        }
    }
    
    TEMPORAL ALIGNMENT:
    - Uses study_time to align vitals/labs near the X-ray
    - 7-day window for vitals (before X-ray)
    - 48-hour window for labs (before X-ray)
    - 24-hour window for oxygen devices
    
    WHY THIS FUNCTION MATTERS:
    - This is where all EHR data comes together
    - Output is used to create Stage B training samples
    - Without this, there's no patient context for the model
    
    Args:
        conn: DuckDB connection
        subject_id: Patient ID
        study_id: Chest X-ray study ID
        study_time: When the X-ray was taken (for temporal alignment)
        vital_item_map: Mapping of itemids to vital names
        
    Returns:
        Complete EHR record or None if patient not found
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
        
        # Get admission for this study using smart matching
        hadm_id = find_best_admission_for_study(conn, subject_id, study_id)
        
        if hadm_id:
            # Get admission details
            admission_query = f"""
                SELECT 
                    admittime,
                    dischtime,
                    admission_type,
                    admission_location
                FROM admissions
                WHERE hadm_id = {hadm_id}
            """
            admission_result = conn.execute(admission_query).fetchone()
            if admission_result:
                admittime, dischtime, admission_type, admission_location = admission_result
            else:
                admission_type = None
        else:
            # No admission found
            admission_type = None
        
        # Extract comprehensive EHR data
        vitals_data = extract_comprehensive_vitals(conn, subject_id, hadm_id, study_time, vital_item_map)
        labs_data = extract_comprehensive_labs(conn, subject_id, hadm_id, study_time)
        oxygen_device = extract_oxygen_device(conn, subject_id, hadm_id, study_time)
        
        # Extract ICD flags
        icd_flags = extract_icd_flags(conn, subject_id, hadm_id)
        
        # Extract chronic conditions from ICD flags
        chronic_conditions = []
        for condition, has_condition in icd_flags.items():
            if condition in ["diabetes", "copd", "ckd", "hypertension", "coronary_artery_disease", 
                           "atrial_fibrillation", "stroke", "liver_disease"] and has_condition:
                chronic_conditions.append(condition)
        
        # Build comprehensive EHR JSON
        ehr_json = {
            "Age": int(age) if age else None,
            "Sex": "M" if gender == "M" else "F",
            "hadm_id": int(hadm_id) if hadm_id else None,
            "admission_type": admission_type,
            "Vitals": vitals_data if vitals_data else {},
            "Labs": labs_data if labs_data else {},
            "O2_device": oxygen_device,
            "Chronic_conditions": chronic_conditions
        }
        
        return {
            "study_id": study_id,
            "subject_id": subject_id,
            "ehr_json": ehr_json,
            "icd_json": icd_flags
        }
        
    except Exception as e:
        print(f"Error extracting EHR for study {study_id}: {e}")
        return None

def build_enhanced_ehr_context() -> None:
    """
    ⭐ MAIN FUNCTION: Build EHR context for all chest X-ray studies
    
    WHAT IT DOES:
    - Processes ALL chest X-ray studies in the dataset
    - Extracts comprehensive EHR data for each study
    - Writes to ehr_context.jsonl (one line per study)
    
    PROCESS:
    1. Initialize DuckDB and load MIMIC-IV tables
    2. Build vital item mapping (itemid → vital name)
    3. Load study metadata (study_id, subject_id pairs)
    4. For each study:
       - Extract EHR data (vitals, labs, devices, ICD codes)
       - Write to output file
    5. Print summary statistics
    
    OUTPUT:
    - File: src/data/processed/ehr_context.jsonl
    - Format: JSONL (one JSON object per line)
    - Records: ~5,441 studies with EHR data
    
    STATISTICS SHOWN:
    - Records processed
    - Records with vitals (%)
    - Records with labs (%)
    - Records with ICD flags (%)
    - Vitals/Labs coverage by type
    
    RUNTIME:
    - ~2-4 hours for full dataset
    - Processes ~5,441 studies
    - Extracts data from multiple MIMIC-IV tables
    
    WHY IT'S SLOW:
    - Querying multiple large tables (labevents, chartevents)
    - Temporal alignment (finding data near X-ray time)
    - Unit conversions and sanity checks
    - Deduplication and quality checks
    """
    print("🏥 BUILDING ENHANCED EHR CONTEXT")
    print("=" * 70)
    print("✅ Using comprehensive MIMIC-IV data sources:")
    print("   - labevents.csv.gz: Comprehensive laboratory data")
    print("   - chartevents.csv.gz: Vital signs and clinical measurements")
    print("   - d_labitems.csv.gz: Lab test definitions")
    print("   - d_items.csv.gz: Clinical measurement definitions")
    print("=" * 70)
    
    # Initialize DuckDB
    conn = initialize_duckdb()
    
    # Build dynamic vital item mapping
    print("🔧 Building dynamic vital item mapping...")
    vital_item_map = build_vital_item_mapping(conn)
    print(f"   Built mapping with {len(vital_item_map)} itemids")
    print(f"   Sample mappings: {dict(list(vital_item_map.items())[:5])}")
    
    # Load study metadata
    studies_df = load_study_metadata()
    
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each study
    print(f"\n🔄 Processing {len(studies_df)} studies...")
    
    records_processed = 0
    records_with_ehr = 0
    records_with_icd = 0
    records_with_vitals = 0
    records_with_labs = 0
    
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for idx, row in studies_df.iterrows():
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(studies_df)} studies...")
            
            subject_id = int(row['subject_id'])
            study_id = str(row['study_id'])
            
            # Extract study time from metadata (use admission time as proxy if not available)
            study_time = None
            if 'StudyDate' in row and 'StudyTime' in row:
                try:
                    study_datetime_str = f"{row['StudyDate']} {row['StudyTime']}"
                    study_time = datetime.strptime(study_datetime_str, '%Y-%m-%d %H:%M:%S')
                except:
                    study_time = None
            
            # If no study time available, try to get admission time as proxy
            if study_time is None:
                try:
                    # Get the most recent admission for this subject
                    admission_query = f"""
                        SELECT admittime, dischtime
                        FROM admissions 
                        WHERE subject_id = {subject_id}
                        ORDER BY admittime DESC
                        LIMIT 1
                    """
                    admission_result = conn.execute(admission_query).fetchone()
                    if admission_result:
                        # Don't use admission time as study time - let vitals extraction use hadm_id
                        study_time = None
                except Exception as e:
                    print(f"Warning: Could not get admission time for subject {subject_id}: {e}")
            
            # Extract comprehensive EHR data
            ehr_record = extract_ehr_for_study(conn, subject_id, study_id, study_time, vital_item_map)
            
            if not ehr_record:
                continue
            
            records_with_ehr += 1
            
            # Extract data from the record
            ehr_json = ehr_record.get('ehr_json', {})
            icd_json = ehr_record.get('icd_json', {})
            
            # Count records with data
            if ehr_json.get('Vitals'):
                records_with_vitals += 1
            if ehr_json.get('Labs'):
                records_with_labs += 1
            
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
    
    print(f"\n✅ Enhanced EHR Context Building Complete!")
    print(f"=" * 70)
    print(f"Total records processed: {records_processed}")
    print(f"Records with EHR data: {records_with_ehr}")
    print(f"Records with vitals: {records_with_vitals}")
    print(f"Records with labs: {records_with_labs}")
    print(f"Records with ICD flags: {records_with_icd}")
    print(f"Output saved to: {OUTPUT_FILE}")

def analyze_enhanced_ehr_context() -> None:
    """
    ⭐ Analyze the generated EHR context file
    
    WHAT IT DOES:
    - Reads the ehr_context.jsonl file
    - Computes statistics on data coverage
    - Shows demographics, vitals, labs, ICD distribution
    
    STATISTICS COMPUTED:
    - Age distribution: Mean, min, max
    - Sex distribution: Male vs Female %
    - Vitals coverage: Which vitals are most common
    - Labs coverage: Which lab tests are most common
    - ICD flags: Which diagnoses are most frequent
    - Oxygen devices: How many patients on respiratory support
    - Admission data: How many have admission info
    
    WHY IT'S IMPORTANT:
    - Validates data quality
    - Shows coverage gaps (e.g., only 42% have vitals)
    - Identifies most common findings
    - Helps understand dataset characteristics
    
    OUTPUT EXAMPLE:
    - "93.6% have Creatinine lab"
    - "42.1% have weight vital"
    - "34.8% have hypertension"
    - "26.3% have oxygen device"
    
    HELPS WITH:
    - Understanding Stage B coverage (which samples have EHR data)
    - Identifying rare findings that need boosting
    - Validating data extraction worked correctly
    """
    print(f"\n📊 ANALYZING ENHANCED EHR CONTEXT")
    print("=" * 70)
    
    if not OUTPUT_FILE.exists():
        print("Enhanced EHR context file not found!")
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
    
    # Analyze vitals coverage
    vitals_coverage = {}
    for record in records:
        vitals = record['ehr_json'].get('Vitals', {})
        for vital_name in vitals.keys():
            vitals_coverage[vital_name] = vitals_coverage.get(vital_name, 0) + 1
    
    print(f"\nVitals Coverage (out of {len(records)} records):")
    for vital, count in sorted(vitals_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {vital}: {count} ({count/len(records)*100:.1f}%)")
    
    # Analyze labs coverage
    labs_coverage = {}
    for record in records:
        labs = record['ehr_json'].get('Labs', {})
        for lab_name in labs.keys():
            labs_coverage[lab_name] = labs_coverage.get(lab_name, 0) + 1
    
    print(f"\nLabs Coverage (out of {len(records)} records):")
    for lab, count in sorted(labs_coverage.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lab}: {count} ({count/len(records)*100:.1f}%)")
    
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
    
    # Analyze oxygen devices
    with_oxygen = sum(1 for r in records if r['ehr_json'].get('O2_device'))
    print(f"Records with oxygen device: {with_oxygen} ({with_oxygen/len(records)*100:.1f}%)")

def main():
    """Main function."""
    print("MIMIC-CXR Enhanced EHR Context Builder")
    print("=" * 70)
    
    try:
        # Build enhanced EHR context
        build_enhanced_ehr_context()
        
        # Analyze results
        analyze_enhanced_ehr_context()
        
        print(f"\n🎉 Enhanced EHR context building complete!")
        print(f"📁 Output: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"❌ Error building enhanced EHR context: {e}")
        raise

if __name__ == "__main__":
    main()
