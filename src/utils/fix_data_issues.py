#!/usr/bin/env python3
"""
PROPER data correction script that correctly classifies Stage A/B based on patient_data presence:
- If patient_data is present → Stage B
- If patient_data is NOT present → Stage A
Then applies all expert-level healthcare data corrections.
"""

import json
import re
import sys
from typing import Dict, Any, List, Tuple
from pathlib import Path

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def fix_stage_classification_proper(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Correctly classify Stage A/B based on patient_data presence."""
    print("Fixing Stage classification properly...")
    stage_a_to_b = 0
    stage_b_to_a = 0
    
    for item in data:
        has_patient_data = 'patient_data' in item and item['patient_data'] is not None
        
        if has_patient_data and item.get('stage') == 'A':
            # Stage A with patient_data → Stage B
            item['stage'] = 'B'
            stage_a_to_b += 1
            print(f"Corrected Stage A to Stage B (has patient_data): {item.get('image_path', 'unknown')}")
        elif not has_patient_data and item.get('stage') == 'B':
            # Stage B without patient_data → Stage A
            item['stage'] = 'A'
            stage_b_to_a += 1
            print(f"Corrected Stage B to Stage A (no patient_data): {item.get('image_path', 'unknown')}")
    
    print(f"Corrected {stage_a_to_b} Stage A → Stage B (had patient_data)")
    print(f"Corrected {stage_b_to_a} Stage B → Stage A (no patient_data)")
    return data

def fix_chexpert_contradictions(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix CheXpert label vs impression contradictions."""
    print("Fixing CheXpert contradictions...")
    fixed_count = 0
    
    # Define strong negation patterns
    negation_patterns = {
        'Pleural Effusion': [
            r'no pleural effusion', r'no pleural effusions', 
            r'no evidence of pleural effusion', r'no pleural fluid'
        ],
        'Pneumonia': [
            r'no pneumonia', r'no evidence of pneumonia', 
            r'pneumonia is doubtful', r'no acute pneumonia'
        ],
        'Pneumothorax': [
            r'no pneumothorax', r'no evidence of pneumothorax',
            r'no acute pneumothorax'
        ],
        'Edema': [
            r'no edema', r'no pulmonary edema', r'no vascular congestion',
            r'no evidence of edema'
        ]
    }
    
    for item in data:
        impression = item.get('impression', '').lower()
        chexpert_labels = item.get('chexpert_labels', {})
        
        for condition, patterns in negation_patterns.items():
            if condition in chexpert_labels:
                # Check if impression contains strong negation
                has_negation = any(re.search(pattern, impression) for pattern in patterns)
                
                if has_negation and chexpert_labels[condition] == 1:
                    # Set conflicting label to 0 (negative)
                    chexpert_labels[condition] = 0
                    fixed_count += 1
                    print(f"Fixed {condition} contradiction in {item.get('image_path', 'unknown')}")
    
    print(f"Fixed {fixed_count} CheXpert contradictions")
    return data

def fix_sentinel_lab_values(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix sentinel/implausible lab values with healthcare expertise."""
    print("Fixing sentinel lab values...")
    fixed_count = 0
    
    # Sentinel values that indicate "very high/unknown" or data artifacts
    sentinel_values = {
        'BNP': 100000,  # pg/mL - almost certainly sentinel
        'BUN': 200,     # mg/dL - common sentinel value
    }
    
    # Magic decimal patterns indicating conversion artifacts
    magic_decimal_patterns = {
        'ALT': [3.334, 5.001, 8.335, 13.336],  # Common conversion artifacts
    }
    
    for item in data:
        if 'patient_data' not in item or 'Labs' not in item['patient_data']:
            continue
            
        labs = item['patient_data']['Labs']
        
        # Create a list of lab names to avoid dictionary modification during iteration
        lab_names = list(labs.keys())
        
        for lab_name in lab_names:
            if lab_name not in labs:  # Skip if already deleted
                continue
                
            lab_data = labs[lab_name]
            if not isinstance(lab_data, dict) or 'value' not in lab_data:
                continue
                
            value = lab_data['value']
            
            # Check for sentinel values
            if lab_name in sentinel_values and value == sentinel_values[lab_name]:
                # Set to null (remove from EHR JSON)
                del labs[lab_name]
                fixed_count += 1
                print(f"Removed sentinel {lab_name}={value} from {item.get('image_path', 'unknown')}")
                continue
            
            # Check for magic decimal patterns
            if lab_name in magic_decimal_patterns:
                if value in magic_decimal_patterns[lab_name]:
                    # Set to null for conversion artifacts
                    del labs[lab_name]
                    fixed_count += 1
                    print(f"Removed magic decimal {lab_name}={value} from {item.get('image_path', 'unknown')}")
                    continue
                elif value < 5:  # ALT < 5 U/L is unusual with normal liver function
                    del labs[lab_name]
                    fixed_count += 1
                    print(f"Removed unusual low {lab_name}={value} from {item.get('image_path', 'unknown')}")
                    continue
            
            # Fix INR/PT inconsistency
            if lab_name == 'INR' and 'PT' in labs:
                inr_value = value
                pt_value = labs['PT'].get('value', 0)
                
                # Check if INR and PT are inconsistent
                if inr_value > 5 and pt_value > 30:
                    # Keep PT, remove INR as it's likely incorrect
                    del labs['INR']
                    fixed_count += 1
                    print(f"Removed inconsistent INR={inr_value} (PT={pt_value}) from {item.get('image_path', 'unknown')}")
                    continue
    
    print(f"Fixed {fixed_count} sentinel lab values")
    return data

def fix_normal_range_logic(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fix normal range logic and value_is_suspect flags."""
    print("Fixing normal range logic...")
    fixed_count = 0
    
    for item in data:
        if 'patient_data' not in item or 'Labs' not in item['patient_data']:
            continue
            
        labs = item['patient_data']['Labs']
        
        for lab_name, lab_data in labs.items():
            if not isinstance(lab_data, dict) or 'value' not in lab_data:
                continue
                
            value = lab_data['value']
            ref_range = lab_data.get('reference_range', [0, 1000])
            
            if len(ref_range) == 2:
                low, high = ref_range
                
                # Recompute normal_range correctly
                is_normal = low <= value <= high
                lab_data['normal_range'] = is_normal
                
                # Set value_is_suspect only for values outside range by wide margin
                # or for known outlier values
                outlier_values = {
                    'BNP': 100000,
                    'BUN': 200,
                    'PLT': 1000000,  # Extremely high platelet count
                }
                
                is_suspect = False
                if lab_name in outlier_values and value == outlier_values[lab_name]:
                    is_suspect = True
                elif not is_normal:
                    # Check if outside range by wide margin (>50% beyond range)
                    range_width = high - low
                    if value < low - 0.5 * range_width or value > high + 0.5 * range_width:
                        is_suspect = True
                
                if 'value_is_suspect' in lab_data:
                    if lab_data['value_is_suspect'] != is_suspect:
                        lab_data['value_is_suspect'] = is_suspect
                        fixed_count += 1
                else:
                    lab_data['value_is_suspect'] = is_suspect
                    if is_suspect:
                        fixed_count += 1
    
    print(f"Fixed {fixed_count} normal range logic issues")
    return data

def normalize_vitals_and_units(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize vitals and units with healthcare standards."""
    print("Normalizing vitals and units...")
    fixed_count = 0
    
    for item in data:
        if 'patient_data' not in item or 'Vitals' not in item['patient_data']:
            continue
            
        vitals = item['patient_data']['Vitals']
        
        # Convert temperature to Celsius and remove Fahrenheit
        if 'temperature_f' in vitals and 'temperature_c' in vitals:
            # Prefer Celsius, remove Fahrenheit
            del vitals['temperature_f']
            fixed_count += 1
        elif 'temperature_f' in vitals:
            # Convert Fahrenheit to Celsius
            temp_f = vitals['temperature_f']['value']
            temp_c = (temp_f - 32) * 5/9
            vitals['temperature_c'] = {
                'value': round(temp_c, 1),
                'unit': '°C',
                'delta_hours': vitals['temperature_f'].get('delta_hours', -2.0)
            }
            del vitals['temperature_f']
            fixed_count += 1
        
        # Remove string blood pressure, keep numeric fields
        if 'blood_pressure' in vitals and isinstance(vitals['blood_pressure'], str):
            del vitals['blood_pressure']
            fixed_count += 1
        
        # Ensure height is in meters and weight is in kg
        if 'height' in vitals:
            height_data = vitals['height']
            if height_data.get('unit') == 'cm':
                height_data['value'] = height_data['value'] / 100
                height_data['unit'] = 'm'
                fixed_count += 1
        
        if 'weight' in vitals:
            weight_data = vitals['weight']
            if weight_data.get('unit') == 'lbs':
                weight_data['value'] = weight_data['value'] * 0.453592
                weight_data['unit'] = 'kg'
                fixed_count += 1
        
        # Recalculate BMI if height and weight are present
        if 'height' in vitals and 'weight' in vitals:
            height_m = vitals['height']['value']
            weight_kg = vitals['weight']['value']
            if height_m > 0:
                bmi = weight_kg / (height_m ** 2)
                vitals['BMI'] = {
                    'value': round(bmi, 2),
                    'unit': 'kg/m²',
                    'source': 'recalculated',
                    'delta_hours': -2.0
                }
                fixed_count += 1
    
    print(f"Fixed {fixed_count} vitals and units issues")
    return data

def normalize_o2_device(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize O2 device to controlled vocabulary."""
    print("Normalizing O2 device...")
    fixed_count = 0
    
    # Controlled vocabulary for O2 devices
    device_mapping = {
        'none': 'none',
        'unknown': 'none',  # Map unknown to none
        'room air': 'none',
        'nasal cannula': 'NC',
        'non-rebreather': 'NRB',
        'high flow': 'HFNC',
        'bipap': 'BiPAP',
        'cpap': 'BiPAP',
        'endotracheal': 'ETT',
        'tracheostomy': 'Trach',
        'ventilator': 'ETT'
    }
    
    for item in data:
        if 'patient_data' not in item or 'O2_device' not in item['patient_data']:
            continue
            
        o2_device = item['patient_data']['O2_device']
        current_device = o2_device.get('device', 'unknown').lower()
        
        # Map to controlled vocabulary
        normalized_device = device_mapping.get(current_device, 'none')
        
        if current_device != normalized_device:
            o2_device['device'] = normalized_device
            fixed_count += 1
        
        # Clean up flow_lpm and fio2 if they're null or invalid
        if o2_device.get('flow_lpm') is None or o2_device.get('flow_lpm') == 0:
            o2_device['flow_lpm'] = None
        
        if o2_device.get('fio2') is None or o2_device.get('fio2') == 0:
            o2_device['fio2'] = None
        elif o2_device.get('fio2') and o2_device.get('fio2') > 1:
            # Convert percentage to decimal
            o2_device['fio2'] = o2_device['fio2'] / 100
            fixed_count += 1
    
    print(f"Fixed {fixed_count} O2 device issues")
    return data

def add_icd_labels_stage_b(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add placeholder ICD labels for Stage B rows."""
    print("Adding ICD labels for Stage B...")
    fixed_count = 0
    
    # Simple ICD-10 codes based on conditions (placeholder implementation)
    icd_mapping = {
        'hypertension': 'I10',
        'diabetes': 'E11.9',
        'coronary_artery_disease': 'I25.9',
        'atrial_fibrillation': 'I48.91',
        'copd': 'J44.9',
        'ckd': 'N18.9',
        'stroke': 'I69.9'
    }
    
    for item in data:
        if item.get('stage') == 'B' and 'icd_labels' not in item:
            icd_labels = []
            
            # Add ICD codes based on chronic conditions
            if 'patient_data' in item and 'Chronic_conditions' in item['patient_data']:
                chronic_conditions = item['patient_data']['Chronic_conditions']
                for condition in chronic_conditions:
                    if condition in icd_mapping:
                        icd_labels.append({
                            'code': icd_mapping[condition],
                            'description': condition,
                            'confidence': 0.8
                        })
            
            # Add ICD codes based on CheXpert findings
            chexpert_labels = item.get('chexpert_labels', {})
            if chexpert_labels.get('Pneumonia') == 1:
                icd_labels.append({'code': 'J18.9', 'description': 'Pneumonia', 'confidence': 0.9})
            if chexpert_labels.get('Pleural Effusion') == 1:
                icd_labels.append({'code': 'J94.8', 'description': 'Pleural effusion', 'confidence': 0.9})
            if chexpert_labels.get('Pneumothorax') == 1:
                icd_labels.append({'code': 'J93.9', 'description': 'Pneumothorax', 'confidence': 0.9})
            if chexpert_labels.get('Edema') == 1:
                icd_labels.append({'code': 'J81.1', 'description': 'Pulmonary edema', 'confidence': 0.8})
            
            item['icd_labels'] = icd_labels
            fixed_count += 1
    
    print(f"Added ICD labels to {fixed_count} Stage B rows")
    return data

def improve_ehr_completeness(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Improve EHR completeness and add delta_hours."""
    print("Improving EHR completeness...")
    fixed_count = 0
    
    for item in data:
        if 'patient_data' not in item:
            continue
            
        patient_data = item['patient_data']
        
        # Add delta_hours for vitals and labs (placeholder implementation)
        if 'Vitals' in patient_data:
            for vital_name, vital_data in patient_data['Vitals'].items():
                if isinstance(vital_data, dict) and 'value' in vital_data:
                    # Add delta_hours (negative means before CXR)
                    vital_data['delta_hours'] = -2.0  # 2 hours before CXR
                    fixed_count += 1
        
        if 'Labs' in patient_data:
            for lab_name, lab_data in patient_data['Labs'].items():
                if isinstance(lab_data, dict) and 'value' in lab_data:
                    # Add delta_hours (negative means before CXR)
                    lab_data['delta_hours'] = -4.0  # 4 hours before CXR
                    fixed_count += 1
        
        # Fix O2 device "none" to "unknown"
        if 'O2_device' in patient_data:
            o2_device = patient_data['O2_device']
            if o2_device.get('device') == 'none':
                o2_device['device'] = 'unknown'
                fixed_count += 1
    
    print(f"Improved EHR completeness for {fixed_count} fields")
    return data

def fix_chexpert_uncertainty_policy(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Define policy for CheXpert uncertainty (-1 vs 0)."""
    print("Applying CheXpert uncertainty policy...")
    fixed_count = 0
    
    for item in data:
        chexpert_labels = item.get('chexpert_labels', {})
        
        for condition, value in chexpert_labels.items():
            if value == -1:
                # Map -1 (uncertain) to 0 (negative) for training
                chexpert_labels[condition] = 0
                fixed_count += 1
    
    print(f"Applied uncertainty policy to {fixed_count} CheXpert labels")
    return data

def add_data_quality_metadata(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add data quality metadata to each record."""
    print("Adding data quality metadata...")
    
    for item in data:
        # Add data quality flags
        item['data_quality'] = {
            'has_ehr_data': 'patient_data' in item,
            'has_icd_labels': 'icd_labels' in item and len(item.get('icd_labels', [])) > 0,
            'has_suspect_values': False,
            'stage': item.get('stage', 'unknown')
        }
        
        # Check for suspect values
        if 'patient_data' in item and 'Labs' in item['patient_data']:
            for lab_name, lab_data in item['patient_data']['Labs'].items():
                if isinstance(lab_data, dict) and lab_data.get('value_is_suspect', False):
                    item['data_quality']['has_suspect_values'] = True
                    break
    
    return data

def main():
    """Main function to fix all data issues with proper Stage A/B classification."""
    # Process training data from backup
    train_input = "/Users/rahul/Downloads/Code scripts/radiology_report/backups/20251017_201834/curriculum_train_final_clean.jsonl"
    train_output = "/Users/rahul/Downloads/Code scripts/radiology_report/src/data/processed/curriculum_train_final_clean.jsonl"
    
    print(f"Loading training data from {train_input}")
    train_data = load_jsonl(train_input)
    print(f"Loaded {len(train_data)} training records")
    
    # Apply corrections to training data
    train_data = fix_stage_classification_proper(train_data)
    train_data = fix_chexpert_contradictions(train_data)
    train_data = fix_sentinel_lab_values(train_data)
    train_data = fix_normal_range_logic(train_data)
    train_data = normalize_vitals_and_units(train_data)
    train_data = normalize_o2_device(train_data)
    train_data = add_icd_labels_stage_b(train_data)
    train_data = improve_ehr_completeness(train_data)
    train_data = fix_chexpert_uncertainty_policy(train_data)
    train_data = add_data_quality_metadata(train_data)
    
    print(f"Saving corrected training data to {train_output}")
    save_jsonl(train_data, train_output)
    
    # Process validation data from backup
    val_input = "/Users/rahul/Downloads/Code scripts/radiology_report/backups/20251017_201834/curriculum_val_final_clean.jsonl"
    val_output = "/Users/rahul/Downloads/Code scripts/radiology_report/src/data/processed/curriculum_val_final_clean.jsonl"
    
    print(f"Loading validation data from {val_input}")
    val_data = load_jsonl(val_input)
    print(f"Loaded {len(val_data)} validation records")
    
    # Apply corrections to validation data
    val_data = fix_stage_classification_proper(val_data)
    val_data = fix_chexpert_contradictions(val_data)
    val_data = fix_sentinel_lab_values(val_data)
    val_data = fix_normal_range_logic(val_data)
    val_data = normalize_vitals_and_units(val_data)
    val_data = normalize_o2_device(val_data)
    val_data = add_icd_labels_stage_b(val_data)
    val_data = improve_ehr_completeness(val_data)
    val_data = fix_chexpert_uncertainty_policy(val_data)
    val_data = add_data_quality_metadata(val_data)
    
    print(f"Saving corrected validation data to {val_output}")
    save_jsonl(val_data, val_output)
    
    # Print summary statistics
    print(f"\nTraining Data Quality Summary:")
    total_train = len(train_data)
    stage_a_train = sum(1 for item in train_data if item.get('stage') == 'A')
    stage_b_train = sum(1 for item in train_data if item.get('stage') == 'B')
    ehr_train = sum(1 for item in train_data if item.get('data_quality', {}).get('has_ehr_data', False))
    suspect_train = sum(1 for item in train_data if item.get('data_quality', {}).get('has_suspect_values', False))
    
    print(f"Total records: {total_train}")
    print(f"Stage A (image+text only): {stage_a_train}")
    print(f"Stage B (image+text+EHR): {stage_b_train}")
    print(f"Records with EHR data: {ehr_train}")
    print(f"Records with suspect values: {suspect_train}")
    
    print(f"\nValidation Data Quality Summary:")
    total_val = len(val_data)
    stage_a_val = sum(1 for item in val_data if item.get('stage') == 'A')
    stage_b_val = sum(1 for item in val_data if item.get('stage') == 'B')
    ehr_val = sum(1 for item in val_data if item.get('data_quality', {}).get('has_ehr_data', False))
    suspect_val = sum(1 for item in val_data if item.get('data_quality', {}).get('has_suspect_values', False))
    
    print(f"Total records: {total_val}")
    print(f"Stage A (image+text only): {stage_a_val}")
    print(f"Stage B (image+text+EHR): {stage_b_val}")
    print(f"Records with EHR data: {ehr_val}")
    print(f"Records with suspect values: {suspect_val}")
    
    print("\nData correction completed successfully!")

if __name__ == "__main__":
    main()
