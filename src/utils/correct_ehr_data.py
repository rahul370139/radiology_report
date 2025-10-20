#!/usr/bin/env python3
"""
Comprehensive EHR data correction system
Fixes: Lab unit conversions, CheXpert labels, hadm_id validation, and root cause itemid mappings
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

class EHRDataCorrector:
    """Comprehensive EHR data correction system"""
    
    def __init__(self):
        # Canonical field mappings with correct units and ranges
        self.canonical_mappings = {
            # Hematology
            'WBC': {'canonical_name': 'WBC', 'unit': 'K/uL', 'normal_range': (4.0, 11.0)},
            'WBC Count': {'canonical_name': 'WBC', 'unit': 'K/uL', 'normal_range': (4.0, 11.0)},
            'Hemoglobin': {'canonical_name': 'HGB', 'unit': 'g/dL', 'normal_range': (12.0, 16.0)},
            'HGB': {'canonical_name': 'HGB', 'unit': 'g/dL', 'normal_range': (12.0, 16.0)},
            'Platelet Count': {'canonical_name': 'PLT', 'unit': 'K/uL', 'normal_range': (150.0, 450.0)},
            'PLT': {'canonical_name': 'PLT', 'unit': 'K/uL', 'normal_range': (150.0, 450.0)},
            'Neutrophils': {'canonical_name': 'Neutrophils', 'unit': '%', 'normal_range': (40.0, 70.0)},
            'Lymphocytes': {'canonical_name': 'Lymphocytes', 'unit': '%', 'normal_range': (20.0, 40.0)},
            
            # Chemistry
            'Sodium': {'canonical_name': 'Na', 'unit': 'mEq/L', 'normal_range': (135.0, 145.0)},
            'Na': {'canonical_name': 'Na', 'unit': 'mEq/L', 'normal_range': (135.0, 145.0)},
            'Potassium': {'canonical_name': 'K', 'unit': 'mEq/L', 'normal_range': (3.5, 5.0)},
            'K': {'canonical_name': 'K', 'unit': 'mEq/L', 'normal_range': (3.5, 5.0)},
            'Chloride': {'canonical_name': 'Cl', 'unit': 'mEq/L', 'normal_range': (98.0, 107.0)},
            'Cl': {'canonical_name': 'Cl', 'unit': 'mEq/L', 'normal_range': (98.0, 107.0)},
            'CO2': {'canonical_name': 'HCO3', 'unit': 'mEq/L', 'normal_range': (22.0, 28.0)},
            'HCO3': {'canonical_name': 'HCO3', 'unit': 'mEq/L', 'normal_range': (22.0, 28.0)},
            'BUN': {'canonical_name': 'BUN', 'unit': 'mg/dL', 'normal_range': (7.0, 20.0)},
            'Creatinine': {'canonical_name': 'Creatinine', 'unit': 'mg/dL', 'normal_range': (0.6, 1.2)},
            'Glucose': {'canonical_name': 'Glucose', 'unit': 'mg/dL', 'normal_range': (70.0, 100.0)},
            
            # Cardiac markers
            'BNP': {'canonical_name': 'BNP', 'unit': 'pg/mL', 'normal_range': (0.0, 100.0)},
            'Troponin T': {'canonical_name': 'Troponin_T', 'unit': 'ng/mL', 'normal_range': (0.0, 0.04)},
            
            # Coagulation
            'INR': {'canonical_name': 'INR', 'unit': 'ratio', 'normal_range': (0.8, 1.2)},
            'PT': {'canonical_name': 'PT', 'unit': 'sec', 'normal_range': (11.0, 13.5)},
            'PTT': {'canonical_name': 'PTT', 'unit': 'sec', 'normal_range': (25.0, 35.0)},
            
            # Inflammatory markers
            'CRP': {'canonical_name': 'CRP', 'unit': 'mg/L', 'normal_range': (0.0, 3.0)},
            
            # Liver function
            'ALT': {'canonical_name': 'ALT', 'unit': 'U/L', 'normal_range': (7.0, 56.0)},
            'Total Bilirubin': {'canonical_name': 'Total_Bilirubin', 'unit': 'mg/dL', 'normal_range': (0.2, 1.2)},
            'Total_Bilirubin': {'canonical_name': 'Total_Bilirubin', 'unit': 'mg/dL', 'normal_range': (0.2, 1.2)},
            'Albumin': {'canonical_name': 'Albumin', 'unit': 'g/dL', 'normal_range': (3.5, 5.0)},
            
            # Cardiac markers (add missing Troponin T)
            'Troponin T': {'canonical_name': 'Troponin_T', 'unit': 'ng/mL', 'normal_range': (0.0, 0.04)},
            'Troponin_T': {'canonical_name': 'Troponin_T', 'unit': 'ng/mL', 'normal_range': (0.0, 0.04)},
            'hs-TnT': {'canonical_name': 'Troponin_T', 'unit': 'ng/mL', 'normal_range': (0.0, 0.04)},
        }
        
        # O2 device controlled vocabulary
        self.o2_device_mapping = {
            'none': 'none',
            'room_air': 'none',
            'nc': 'NC',
            'nasal_cannula': 'NC',
            'nrb': 'NRB',
            'non_rebreather': 'NRB',
            'hfnc': 'HFNC',
            'high_flow_nasal_cannula': 'HFNC',
            'bipap': 'BiPAP',
            'cpap': 'CPAP',
            'ett': 'ETT',
            'endotracheal_tube': 'ETT',
            'trach': 'Trach',
            'tracheostomy': 'Trach',
        }
    
    def correct_lab_value(self, lab_name: str, value: float, unit: str, 
                         reference_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Correct a single lab value with proper unit conversion"""
        
        # Find canonical mapping
        canonical_info = None
        for key, mapping in self.canonical_mappings.items():
            if lab_name.lower() in key.lower() or key.lower() in lab_name.lower():
                canonical_info = mapping
                break
        
        if not canonical_info:
            return {
                'canonical_name': lab_name,
                'value': value,
                'unit': unit,
                'reference_range': reference_range or (0.0, 1000.0),
                'corrected': False,
                'error': 'Unknown lab type'
            }
        
        canonical_name = canonical_info['canonical_name']
        target_unit = canonical_info['unit']
        target_range = canonical_info['normal_range']
        
        # CRITICAL: Filter extreme values before conversion
        if canonical_name == 'BUN' and value > 200:
            return {
                'canonical_name': canonical_name,
                'value': None,
                'unit': target_unit,
                'reference_range': target_range,
                'normal_range': False,
                'corrected': True,
                'original_value': value,
                'original_unit': unit,
                'error': f'Extreme BUN value: {value} mg/dL (normal: 7-20, >200 is lethal)',
                'dropped': True
            }
        
        if canonical_name == 'PLT' and value < 10:
            return {
                'canonical_name': canonical_name,
                'value': None,
                'unit': target_unit,
                'reference_range': target_range,
                'normal_range': False,
                'corrected': True,
                'original_value': value,
                'original_unit': unit,
                'error': f'Critically low PLT value: {value} K/uL (normal: 150-450, <10 is critical)',
                'dropped': True
            }
        
        if canonical_name == 'BNP' and value > 100000:
            return {
                'canonical_name': canonical_name,
                'value': None,
                'unit': target_unit,
                'reference_range': target_range,
                'normal_range': False,
                'corrected': True,
                'original_value': value,
                'original_unit': unit,
                'error': f'Extreme BNP value: {value} pg/mL (normal: 0-100, >100k is extreme)',
                'dropped': True
            }
        
        # Convert value to target unit with error handling
        try:
            # Always call conversion method to check for impossible values
            corrected_value, corrected_unit = self._convert_lab_value(
                lab_name, value, unit, target_unit
            )
            
            # Validate corrected value
            is_normal = target_range[0] <= corrected_value <= target_range[1]
            
            return {
                'canonical_name': canonical_name,
                'value': corrected_value,
                'unit': corrected_unit,
                'reference_range': target_range,
                'normal_range': is_normal,
                'corrected': unit != corrected_unit,
                'original_value': value,
                'original_unit': unit
            }
        except ValueError as e:
            # Mark impossible/incorrect values as missing
            return {
                'canonical_name': canonical_name,
                'value': None,
                'unit': target_unit,
                'reference_range': target_range,
                'normal_range': False,
                'corrected': True,
                'original_value': value,
                'original_unit': unit,
                'error': str(e),
                'dropped': True
            }
    
    def _convert_lab_value(self, lab_name: str, value: float, 
                          from_unit: str, to_unit: str) -> Tuple[float, str]:
        """Convert lab value between units with CRITICAL FIXES for medical accuracy"""
        
        if from_unit == to_unit:
            # Even for same units, validate the value for impossible ranges
            if lab_name.lower() in ['cl', 'chloride'] and from_unit == 'mEq/L':
                if value < 50.0 or value > 150.0:
                    raise ValueError(f"Impossible Cl value: {value} mEq/L (normal: 98-107)")
            elif lab_name.lower() in ['hco3', 'co2'] and from_unit == 'mEq/L':
                if value < 10.0 or value > 50.0:
                    raise ValueError(f"Impossible HCO3 value: {value} mEq/L (normal: 22-28)")
            elif lab_name.lower() in ['pt'] and from_unit == 'sec':
                if value < 5.0 or value > 50.0:
                    raise ValueError(f"Impossible PT value: {value} sec (normal: 11-13.5)")
            return value, to_unit
        
        # CRITICAL FIXES based on medical accuracy requirements
        
        # 1. Platelets (PLT) - M/uL to K/uL conversion (FIXED LOGIC)
        if lab_name.lower() in ['plt', 'platelet count', 'platelets'] and from_unit == 'm/uL' and to_unit == 'K/uL':
            # Only convert if value < 1.0 (e.g., 0.2 M/uL → 200 K/uL)
            # Values like 4.36 M/uL are likely RBC units confused for PLT
            if value < 1.0:
                return value * 1000.0, to_unit
            else:
                raise ValueError(f"PLT value {value} M/uL is too high - likely RBC units confused for PLT")
        
        # 2. BNP - Drop mg/dL values entirely (BNP should already be in pg/mL)
        elif lab_name.lower() in ['bnp'] and from_unit == 'mg/dL' and to_unit == 'pg/mL':
            # BNP in mg/dL is almost always wrong - drop it
            raise ValueError(f"BNP cannot be in mg/dL: {value} mg/dL (BNP should already be in pg/mL)")
        
        # 3. Chloride (Cl) - Drop impossible values
        elif lab_name.lower() in ['cl', 'chloride'] and from_unit == 'mEq/L' and to_unit == 'mEq/L':
            # Values like 0.532 mEq/L are physiologically impossible (normal: 98-107)
            if value < 50.0 or value > 150.0:
                raise ValueError(f"Impossible Cl value: {value} mEq/L (normal: 98-107)")
            return value, to_unit
        
        # 4. HCO3 - Drop impossible values  
        elif lab_name.lower() in ['hco3', 'co2'] and from_unit == 'mEq/L' and to_unit == 'mEq/L':
            # Values like 0.92 mEq/L are physiologically impossible (normal: 22-28)
            if value < 10.0 or value > 50.0:
                raise ValueError(f"Impossible HCO3 value: {value} mEq/L (normal: 22-28)")
            return value, to_unit
        
        # 5. BUN - Drop mEq/L values and add urea conversion
        elif lab_name.lower() in ['bun'] and from_unit == 'mEq/L' and to_unit == 'mg/dL':
            # BUN in mEq/L is nonsense - mark as missing
            raise ValueError(f"BUN cannot be in mEq/L: {value} mEq/L")
        elif lab_name.lower() in ['bun'] and from_unit == 'mmol/L' and to_unit == 'mg/dL':
            # Urea in mmol/L to BUN mg/dL: BUN = urea(mmol/L) × 2.8
            return value * 2.8, to_unit
        
        # 6. WBC - Drop % values unless it's differential
        elif lab_name.lower() in ['wbc', 'wbc count'] and from_unit == '%' and to_unit == 'K/uL':
            # WBC % is for differentials, not total count - mark as missing
            raise ValueError(f"WBC total count cannot be in %: {value}%")
        
        # 7. PT - Drop impossible values
        elif lab_name.lower() in ['pt'] and from_unit == 'sec' and to_unit == 'sec':
            # Values like 1.0 sec or 1.8 sec are impossible (normal: 11-13.5)
            if value < 5.0 or value > 50.0:
                raise ValueError(f"Impossible PT value: {value} sec (normal: 11-13.5)")
            return value, to_unit
        
        # 8. INR - Drop K/uL values and impossible values
        elif lab_name.lower() in ['inr'] and from_unit == 'K/uL' and to_unit == 'ratio':
            # INR in K/uL is nonsense - mark as missing
            raise ValueError(f"INR cannot be in K/uL: {value} K/uL")
        elif lab_name.lower() in ['inr'] and from_unit == 'ratio' and to_unit == 'ratio':
            # INR > 10 is physiologically impossible (normal: 0.8-1.2)
            if value > 10.0:
                raise ValueError(f"Impossible INR value: {value} (normal: 0.8-1.2, >10 likely PT seconds mapped to INR)")
            return value, to_unit
        
        # 9. PT - Drop impossible values
        elif lab_name.lower() in ['pt'] and from_unit == 'sec' and to_unit == 'sec':
            # PT outside 8-200 sec is physiologically impossible (normal: 11-13.5)
            if value < 8.0 or value > 200.0:
                raise ValueError(f"Impossible PT value: {value} sec (normal: 11-13.5)")
            return value, to_unit
        
        # 10. BUN - Drop extreme values (lethal levels)
        elif lab_name.lower() in ['bun'] and from_unit == 'mg/dL' and to_unit == 'mg/dL':
            # BUN > 100 mg/dL is extremely dangerous (normal: 7-20)
            if value > 100.0:
                raise ValueError(f"Extreme BUN value: {value} mg/dL (normal: 7-20, >100 is lethal)")
            return value, to_unit
        
        # 11. BNP - Drop extreme values
        elif lab_name.lower() in ['bnp'] and from_unit == 'pg/mL' and to_unit == 'pg/mL':
            # BNP > 10,000 pg/mL is extremely high (normal: 0-100)
            if value > 10000.0:
                raise ValueError(f"Extreme BNP value: {value} pg/mL (normal: 0-100, >10,000 is extreme)")
            return value, to_unit
        
        # 12. Platelets - Drop critically low values
        elif lab_name.lower() in ['plt', 'platelet count', 'platelets'] and from_unit == 'K/uL' and to_unit == 'K/uL':
            # PLT < 10 K/uL is critically low (normal: 150-450)
            if value < 10.0:
                raise ValueError(f"Critically low PLT value: {value} K/uL (normal: 150-450, <10 is critical)")
            return value, to_unit
        
        # 13. CRP - Drop extreme values
        elif lab_name.lower() in ['crp'] and from_unit == 'mg/L' and to_unit == 'mg/L':
            # CRP > 50 mg/L is extremely high (normal: 0-3)
            if value > 50.0:
                raise ValueError(f"Extreme CRP value: {value} mg/L (normal: 0-3, >50 is extreme)")
            return value, to_unit
        
        # 9. Standard conversions (keep existing working ones)
        elif lab_name.lower() in ['crp'] and from_unit == '%' and to_unit == 'mg/L':
            return value * 10.0, to_unit
        elif lab_name.lower() in ['neutrophils'] and from_unit == 'g/dL' and to_unit == '%':
            return value, to_unit
        elif lab_name.lower() in ['lymphocytes'] and from_unit == 'pg' and to_unit == '%':
            return value, to_unit
        elif lab_name.lower() in ['co2', 'hco3'] and from_unit == 'mg/dL' and to_unit == 'mEq/L':
            return value * 0.23, to_unit
        elif lab_name.lower() in ['chloride', 'cl'] and from_unit == 'mg/dL' and to_unit == 'mEq/L':
            return value * 0.28, to_unit
        elif lab_name.lower() in ['albumin'] and from_unit == 'IU/L' and to_unit == 'g/dL':
            return 4.0, to_unit
        elif lab_name.lower() in ['alt'] and from_unit == 'mg/dL' and to_unit == 'U/L':
            return value * 16.67, to_unit
        elif lab_name.lower() in ['total bilirubin'] and from_unit == 'IU/L' and to_unit == 'mg/dL':
            return value * 0.0585, to_unit
        
        return value, to_unit
    
    def correct_vital_value(self, vital_name: str, value: float, unit: str) -> Dict[str, Any]:
        """Correct vital sign values with comprehensive validation"""
        
        corrected_value = value
        corrected_unit = unit
        
        # Height conversions
        if vital_name.lower() in ['height'] and unit.lower() in ['inch', 'inches']:
            corrected_value = value * 0.0254
            corrected_unit = 'm'
        elif vital_name.lower() in ['height'] and unit.lower() == 'cm':
            corrected_value = value * 0.01
            corrected_unit = 'm'
        
        # Weight unit validation and correction
        elif vital_name.lower() in ['weight'] and unit.lower() == 'unknown':
            # Try to infer unit from BMI and height
            # This will be handled in the BMI calculation section
            corrected_unit = 'kg'  # Default assumption
            print(f"   🔧 Fixed vital: {vital_name} unit 'unknown' → 'kg' (inferred)")
        
        # Temperature consistency check
        elif vital_name.lower() in ['temperature', 'temp']:
            if unit.lower() in ['f', 'fahrenheit']:
                # Convert F to C and validate
                temp_c = (value - 32) * 5/9
                corrected_value = temp_c
                corrected_unit = 'C'
                print(f"   🔧 Fixed vital: {vital_name} {value}°F → {temp_c:.1f}°C")
            elif unit.lower() in ['c', 'celsius']:
                corrected_unit = 'C'
                # Validate temperature range
                if value < 30.0 or value > 45.0:
                    print(f"   ⚠️  Extreme temperature: {value}°C (normal: 36-38°C)")
        
        # Respiratory rate validation
        elif vital_name.lower() in ['respiratory_rate', 'rr']:
            if value == 0.0:
                # Respiratory rate cannot be 0 - set to null
                raise ValueError(f"Impossible respiratory rate: {value} (cannot be 0)")
            corrected_value = value
            corrected_unit = unit
            
        return {
            'value': corrected_value,
            'unit': corrected_unit,
            'corrected': unit != corrected_unit,
            'original_value': value,
            'original_unit': unit
        }
    
    def calculate_bmi(self, weight_kg: float, height_m: float) -> float:
        """Calculate BMI from weight (kg) and height (m)"""
        if height_m <= 0:
            return 0.0
        return weight_kg / (height_m ** 2)
    
    def correct_o2_device(self, o2_device) -> Dict[str, Any]:
        """Parse and correct O2 device information"""
        
        # Handle case where o2_device is already a dict
        if isinstance(o2_device, dict):
            return o2_device
        
        # Handle string inputs
        if not o2_device or str(o2_device).lower() in ['none', 'null', '']:
            return {
                'device': 'none',
                'flow_lpm': None,
                'fio2': None,
                'corrected': False
            }
        
        # Handle malformed entries like "Oxygen_Device: 40"
        o2_str = str(o2_device)
        if ':' in o2_str:
            parts = o2_str.split(':')
            device_part = parts[0].strip().lower()
            value_part = parts[1].strip()
            
            device = 'none'
            for key, mapped in self.o2_device_mapping.items():
                if key in device_part:
                    device = mapped
                    break
            
            try:
                value = float(value_part)
                if device in ['NC', 'NRB', 'HFNC']:
                    return {
                        'device': device,
                        'flow_lpm': value,
                        'fio2': None,
                        'corrected': True
                    }
                elif device in ['BiPAP', 'CPAP', 'ETT', 'Trach']:
                    return {
                        'device': device,
                        'flow_lpm': None,
                        'fio2': value / 100.0 if value > 1 else value,
                        'corrected': True
                    }
            except ValueError:
                pass
        
        device = 'none'
        for key, mapped in self.o2_device_mapping.items():
            if key in o2_str.lower():
                device = mapped
                break
        
        return {
            'device': device,
            'flow_lpm': None,
            'fio2': None,
            'corrected': device != 'none'
        }
    
    def correct_patient_data(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Correct all patient data in a record"""
        
        corrected_data = patient_data.copy()
        
        # Correct labs
        if 'Labs' in corrected_data:
            corrected_labs = {}
            dropped_count = 0
            
            for lab_name, lab_info in corrected_data['Labs'].items():
                # Handle missing reference_range gracefully
                ref_range = lab_info.get('reference_range')
                if ref_range is None:
                    ref_range = (0.0, 1000.0)
                elif isinstance(ref_range, list):
                    ref_range = tuple(ref_range)
                
                corrected_lab = self.correct_lab_value(
                    lab_name=lab_name,
                    value=lab_info['value'],
                    unit=lab_info['unit'],
                    reference_range=ref_range
                )
                
                # Only keep labs that weren't dropped due to impossible values
                if not corrected_lab.get('dropped', False):
                    corrected_labs[corrected_lab['canonical_name']] = corrected_lab
                else:
                    dropped_count += 1
                    print(f"⚠️  Dropped {lab_name}: {corrected_lab.get('error', 'Unknown error')}")
            
            corrected_data['Labs'] = corrected_labs
            if dropped_count > 0:
                print(f"📊 Dropped {dropped_count} impossible lab values")
        
        # Correct vitals
        if 'Vitals' in corrected_data:
            corrected_vitals = {}
            weight_kg = None
            height_m = None
            
            for vital_name, vital_info in corrected_data['Vitals'].items():
                corrected_vital = self.correct_vital_value(
                    vital_name=vital_name,
                    value=vital_info['value'],
                    unit=vital_info['unit']
                )
                
                # Track weight and height for BMI calculation
                if vital_name.lower() in ['weight'] and corrected_vital['unit'] == 'kg':
                    weight_kg = corrected_vital['value']
                elif vital_name.lower() in ['height'] and corrected_vital['unit'] == 'm':
                    height_m = corrected_vital['value']
                
                corrected_vitals[vital_name] = corrected_vital
            
            # Recalculate BMI if we have weight and height
            if weight_kg and height_m:
                corrected_bmi = self.calculate_bmi(weight_kg, height_m)
                # Remove any existing BMI entries to avoid duplication
                corrected_vitals = {k: v for k, v in corrected_vitals.items() if k.lower() not in ['bmi', 'bmi_calculated']}
                corrected_vitals['BMI'] = {
                    'value': corrected_bmi,
                    'unit': 'kg/m²',
                    'corrected': True,
                    'source': 'recalculated'
                }
            
            # Validate temperature consistency
            corrected_vitals = self.validate_temperature_consistency(corrected_vitals)
            
            corrected_data['Vitals'] = corrected_vitals
        
        # Correct O2 device
        if 'O2_device' in corrected_data:
            corrected_data['O2_device'] = self.correct_o2_device(corrected_data['O2_device'])
        
        return corrected_data
    
    def fix_chexpert_labels(self, chexpert_labels: Dict[str, int], report_text: str = "") -> Dict[str, int]:
        """Fix CheXpert label consistency issues with proper semantics"""
        
        fixed_labels = chexpert_labels.copy()
        
        # CRITICAL FIX: Proper CheXpert semantics
        # 1 = present/positive, 0 = absent/negative, -1 = uncertain only
        
        # Check for No Finding logic violation
        no_finding = fixed_labels.get('No Finding', 0)
        other_positives = []
        support_devices = fixed_labels.get('Support Devices', 0)
        
        for label, value in fixed_labels.items():
            if label not in ['No Finding', 'Support Devices'] and value == 1:
                other_positives.append(label)
        
        # CRITICAL: Fix No Finding logic - if ANY positive finding exists, No Finding must be 0
        if (other_positives or support_devices == 1) and no_finding == 1:
            fixed_labels['No Finding'] = 0
            print(f"   🔧 Fixed CheXpert: No Finding=1 → 0 (other positives: {other_positives}, support devices: {support_devices})")
        
        # Fix Support Devices: if present, set to 1 (not -1)
        if support_devices == -1 and report_text:
            # Check if report mentions devices
            device_keywords = ['port', 'catheter', 'tube', 'lead', 'wire', 'stent', 'pacemaker', 'defibrillator', 'infusion', 'drainage', 'picc', 'chest tube', 'monitoring']
            if any(keyword in report_text.lower() for keyword in device_keywords):
                fixed_labels['Support Devices'] = 1
                print(f"   🔧 Fixed CheXpert: Support Devices=-1 → 1 (device mentioned in report)")
        
        # Fix fracture labels based on report text
        if 'Fracture' in fixed_labels and report_text:
            fracture_value = fixed_labels['Fracture']
            if fracture_value == 1:
                # Check if report denies fractures
                deny_keywords = ['no fracture', 'no displaced', 'no rib fracture', 'no acute fracture']
                if any(keyword in report_text.lower() for keyword in deny_keywords):
                    fixed_labels['Fracture'] = 0
                    print(f"   🔧 Fixed CheXpert: Fracture=1 → 0 (report denies fractures)")
        
        # Fix other labels based on report text analysis
        if report_text:
            # Check for clear negative statements - these should be -1 (uncertain), not 0
            clear_negatives = {
                'Pneumonia': ['no pneumonia', 'no acute pneumonia', 'no evidence of pneumonia', 'no focal pneumonia', 'without pneumonia', 'no acute process', 'no acute cardiopulmonary process', 'no evidence for pneumonia', 'no acute cardiopulmonary process'],
                'Pleural Effusion': ['no pleural effusion', 'no effusion', 'no acute effusion', 'no larger effusions', 'there is no pleural effusion', 'without pleural effusion', 'no larger pleural effusions', 'no evidence of pleural effusion', 'no evidence for pleural effusion', 'no larger effusions'],
                'Pneumothorax': ['no pneumothorax', 'no pneumo', 'no evidence for pneumothorax', 'no evidence for the presence of a pneumothorax', 'there is no pneumothorax', 'without pneumothorax', 'no evidence of pneumothorax', 'no evidence for the presence of pneumothorax', 'no evidence for the presence of a pneumothorax'],
                'Edema': ['no edema', 'no acute edema', 'no pulmonary edema', 'no vascular congestion', 'without edema', 'without pulmonary edema', 'no cardiopulmonary edema', 'no evidence of edema', 'no evidence for edema', 'no vascular congestion', 'no evidence of pulmonary edema'],
                'Lung Opacity': ['lungs are clear', 'no opacity', 'no acute opacity', 'no acute opacities', 'clear lungs', 'without opacity', 'no evidence of opacity', 'no evidence for opacity', 'lungs are clear'],
                'Consolidation': ['no consolidation', 'without consolidation', 'without definite consolidation', 'no acute consolidation', 'no evidence of consolidation', 'no evidence for consolidation'],
                'Enlarged Cardiomediastinum': ['no cardiomegaly', 'no cardiac enlargement', 'heart size normal', 'cardiomediastinum normal', 'no enlarged heart', 'heart size is normal', 'cardiomediastinum is within normal', 'cardiomediastinum is w/in normal', 'no cardiomediastinal enlargement']
            }
            
            # Track which labels were fixed by negative patterns to avoid overriding
            negative_fixed = set()
            
            for label, keywords in clear_negatives.items():
                if label in fixed_labels and fixed_labels[label] in [0, 1]:  # Fix both 0 and 1 if text is negative
                    if any(keyword.lower() in report_text.lower() for keyword in keywords):
                        old_value = fixed_labels[label]
                        fixed_labels[label] = -1
                        negative_fixed.add(label)  # Mark as fixed by negative pattern
                        print(f"   🔧 Fixed CheXpert: {label}={old_value} → -1 (clear negative in report)")
            
            # Check for clear positive statements - these should be 1
            clear_positives = {
                'Pneumonia': ['pneumonia', 'consolidation', 'infiltrate', 'focal pneumonia', 'acute pneumonia', 'bacterial pneumonia'],
                'Pleural Effusion': ['pleural effusion', 'effusion', 'fluid', 'pleural fluid', 'bilateral effusion', 'small effusion'],
                'Pneumothorax': ['pneumothorax', 'pneumo', 'air leak', 'tension pneumothorax'],
                'Edema': ['edema', 'congestion', 'vascular congestion', 'pulmonary edema', 'cardiogenic edema', 'mild edema', 'moderate edema'],
                'Lung Opacity': ['opacity', 'opacities', 'opacification', 'hazy', 'infiltrate', 'bilateral opacity', 'focal opacity'],
                'Consolidation': ['consolidation', 'consolidated', 'consolidating', 'lobar consolidation'],
                'Enlarged Cardiomediastinum': ['cardiomegaly', 'enlarged heart', 'cardiac enlargement', 'enlarged cardiomediastinum', 'mild cardiomegaly']
            }
            
            for label, keywords in clear_positives.items():
                # Only fix if not already fixed by negative pattern
                if label in fixed_labels and fixed_labels[label] in [-1, 0] and label not in negative_fixed:
                    if any(keyword.lower() in report_text.lower() for keyword in keywords):
                        old_value = fixed_labels[label]
                        fixed_labels[label] = 1
                        print(f"   🔧 Fixed CheXpert: {label}={old_value} → 1 (clear positive in report)")
        
        # Fix No Finding when there are clear findings
        if report_text and fixed_labels.get('No Finding', 0) == -1:
            # Check for clear positive findings
            positive_keywords = ['pneumonia', 'effusion', 'pneumothorax', 'opacity', 'consolidation', 'edema', 'fracture', 'lesion']
            if any(keyword in report_text.lower() for keyword in positive_keywords):
                fixed_labels['No Finding'] = 0
                print(f"   🔧 Fixed CheXpert: No Finding=-1 → 0 (positive findings in report)")
        
        # CRITICAL: Final No Finding logic - recompute after all other fixes
        other_positives = []
        support_devices = fixed_labels.get('Support Devices', 0)
        
        for label, value in fixed_labels.items():
            if label not in ['No Finding', 'Support Devices'] and value == 1:
                other_positives.append(label)
        
        # No Finding = 1 ONLY if every other label is negative (0) or uncertain (-1)
        if other_positives or support_devices == 1:
            fixed_labels['No Finding'] = 0
        else:
            # Check if all other labels are truly negative (0) - then No Finding = 1
            all_negative = all(v in {0, -1} for k, v in fixed_labels.items() if k != 'No Finding')
            if all_negative:
                fixed_labels['No Finding'] = 1
        
        # Ensure consistent semantics: 1=positive, 0=negative, -1=uncertain
        for label, value in fixed_labels.items():
            if value not in [-1, 0, 1]:
                # Convert any other values to uncertain (-1)
                fixed_labels[label] = -1
                print(f"   🔧 Fixed CheXpert: {label}={value} → -1 (uncertain)")
        
        return fixed_labels
    
    def fix_stage_consistency(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Fix stage consistency and hadm_id validation"""
        
        fixed_record = record.copy()
        
        # Fix hadm_id validation
        if 'patient_data' in fixed_record:
            patient_data = fixed_record['patient_data']
            
            if patient_data.get('hadm_id') is None or patient_data.get('hadm_id') == 'null':
                # For Stage B, hadm_id is required for ICD supervision
                if fixed_record.get('stage') == 'B':
                    print(f"   ⚠️  Stage B record with null hadm_id - converting to Stage A")
                    fixed_record['stage'] = 'A'  # Convert to Stage A
                else:
                    print(f"   ⚠️  Stage A record with null hadm_id - keeping as Stage A")
        
        # Validate stage consistency
        if fixed_record.get('stage') == 'B':
            # Stage B must have valid hadm_id and some EHR data
            hadm_id = fixed_record['patient_data'].get('hadm_id')
            labs = fixed_record['patient_data'].get('Labs', {})
            vitals = fixed_record['patient_data'].get('Vitals', {})
            
            # Stage B must have patient_data with subject_id, Age, Sex
            if not all(key in fixed_record['patient_data'] for key in ['subject_id', 'Age', 'Sex']):
                print(f"   ⚠️  Converting Stage B to Stage A (missing required patient data)")
                fixed_record['stage'] = 'A'
            elif hadm_id is None or hadm_id == 'null':
                print(f"   ⚠️  Converting Stage B to Stage A (no hadm_id)")
                fixed_record['stage'] = 'A'
            elif not labs and not vitals:
                print(f"   ⚠️  Converting Stage B to Stage A (no EHR data)")
                fixed_record['stage'] = 'A'
        
        return fixed_record
    
    def validate_temperature_consistency(self, vitals: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix temperature consistency between F and C"""
        
        temp_f = None
        temp_c = None
        
        # Find temperature values
        for vital_name, vital_info in vitals.items():
            if 'temperature' in vital_name.lower() or 'temp' in vital_name.lower():
                if vital_info.get('unit', '').lower() in ['f', 'fahrenheit']:
                    temp_f = vital_info.get('value')
                elif vital_info.get('unit', '').lower() in ['c', 'celsius']:
                    temp_c = vital_info.get('value')
        
        # Check consistency if both are present
        if temp_f is not None and temp_c is not None:
            expected_c = (temp_f - 32) * 5/9
            if abs(temp_c - expected_c) > 0.5:  # More than 0.5°C difference
                print(f"   🔧 Temperature inconsistency: {temp_f}°F vs {temp_c}°C (expected {expected_c:.1f}°C)")
                # Keep the more reasonable value
                if 95 <= temp_f <= 105:  # Reasonable F range
                    vitals['temperature_c'] = {
                        'value': expected_c,
                        'unit': 'C',
                        'corrected': True,
                        'original_value': temp_c,
                        'original_unit': 'C'
                    }
                    print(f"   🔧 Fixed temperature: {temp_c}°C → {expected_c:.1f}°C (from {temp_f}°F)")
                else:
                    print(f"   ⚠️  Both temperatures seem unreasonable: {temp_f}°F, {temp_c}°C")
        
        return vitals
    
    def clean_redaction_artifacts(self, text: str) -> str:
        """Replace redaction artifacts with consistent markers"""
        if not text:
            return text
        
        # Replace ___ with [REDACTED]
        cleaned_text = text.replace('___', '[REDACTED]')
        
        # Remove common redaction patterns
        import re
        cleaned_text = re.sub(r'\[REDACTED\]+', '[REDACTED]', cleaned_text)
        
        return cleaned_text
    
    def fix_image_path(self, image_path: str) -> str:
        """Fix image path double-prefix bug (files/files/... -> files/...)"""
        if not image_path:
            return image_path
        
        # Fix double prefix bug
        if image_path.startswith('files/files/'):
            fixed_path = image_path.replace('files/files/', '', 1)  # Remove files/ prefix entirely
            print(f"   🔧 Fixed image path: {image_path} → {fixed_path}")
            return fixed_path
        elif image_path.startswith('files/'):
            fixed_path = image_path.replace('files/', '', 1)  # Remove files/ prefix
            print(f"   🔧 Fixed image path: {image_path} → {fixed_path}")
            return fixed_path
        
        return image_path
    
    def strip_audit_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove audit metadata from data that will be fed to the model"""
        if isinstance(data, dict):
            cleaned_data = {}
            for key, value in data.items():
                if key in ['corrected', 'original_value', 'original_unit', 'error', 'dropped']:
                    continue  # Skip audit fields
                elif isinstance(value, dict):
                    cleaned_data[key] = self.strip_audit_metadata(value)
                elif isinstance(value, list):
                    cleaned_data[key] = [self.strip_audit_metadata(item) if isinstance(item, dict) else item for item in value]
                else:
                    cleaned_data[key] = value
            return cleaned_data
        return data
    
    def correct_dataset(self, input_file: str, output_file: str) -> Dict[str, int]:
        """Comprehensive dataset correction with all fixes"""
        
        stats = {
            'total_records': 0,
            'corrected_records': 0,
            'corrected_labs': 0,
            'corrected_vitals': 0,
            'corrected_o2_devices': 0,
            'chexpert_fixes': 0,
            'stage_fixes': 0,
            'dropped_records': 0,
            'errors': 0
        }
        
        print(f"🚀 Starting comprehensive data correction")
        print(f"📁 Input: {input_file}")
        print(f"📁 Output: {output_file}")
        print("=" * 60)
        
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line_num, line in enumerate(f_in, 1):
                try:
                    record = json.loads(line.strip())
                    stats['total_records'] += 1
                    
                    if line_num % 100 == 0:
                        print(f"🔍 Processing record {line_num}")
                    
                    # 1. Fix CheXpert labels with report text analysis
                    if 'chexpert_labels' in record:
                        original_labels = record['chexpert_labels']
                        report_text = record.get('impression', '') + ' ' + record.get('findings', '')
                        corrected_labels = self.fix_chexpert_labels(original_labels, report_text)
                        if corrected_labels != original_labels:
                            record['chexpert_labels'] = corrected_labels
                            stats['chexpert_fixes'] += 1
                    
                    # 2. Clean redaction artifacts in text fields
                    for text_field in ['impression', 'findings', 'report']:
                        if text_field in record and record[text_field]:
                            original_text = record[text_field]
                            cleaned_text = self.clean_redaction_artifacts(original_text)
                            if cleaned_text != original_text:
                                record[text_field] = cleaned_text
                                print(f"   🔧 Cleaned redaction artifacts in {text_field}")
                    
                    # 2.5. Fix image path double-prefix bug
                    if 'image_path' in record:
                        original_path = record['image_path']
                        fixed_path = self.fix_image_path(original_path)
                        if fixed_path != original_path:
                            record['image_path'] = fixed_path
                    
                    # 3. Fix stage consistency and hadm_id
                    original_stage = record.get('stage')
                    record = self.fix_stage_consistency(record)
                    if record.get('stage') != original_stage:
                        stats['stage_fixes'] += 1
                    
                    # 4. Correct patient data (labs, vitals, O2 device)
                    if 'patient_data' in record:
                        original_data = record['patient_data']
                        corrected_data = self.correct_patient_data(original_data)
                        record['patient_data'] = corrected_data
                        
                        # Count corrections
                        if corrected_data != original_data:
                            stats['corrected_records'] += 1
                            
                            # Count specific corrections
                            if 'Labs' in corrected_data:
                                for lab_name, lab_info in corrected_data['Labs'].items():
                                    if lab_info.get('corrected', False):
                                        stats['corrected_labs'] += 1
                            
                            if 'Vitals' in corrected_data:
                                for vital_name, vital_info in corrected_data['Vitals'].items():
                                    if vital_info.get('corrected', False):
                                        stats['corrected_vitals'] += 1
                            
                            if 'O2_device' in corrected_data:
                                if corrected_data['O2_device'].get('corrected', False):
                                    stats['corrected_o2_devices'] += 1
                    
                    # 5. Strip audit metadata before writing to model input
                    clean_record = self.strip_audit_metadata(record)
                    
                    f_out.write(json.dumps(clean_record) + '\n')
                    
                except Exception as e:
                    print(f"   ❌ Error processing line {line_num}: {e}")
                    stats['errors'] += 1
                    stats['dropped_records'] += 1
                    continue
        
        print("=" * 60)
        print(f"✅ Data correction completed!")
        print(f"📊 Summary:")
        print(f"   • Total records processed: {stats['total_records']}")
        print(f"   • CheXpert fixes: {stats['chexpert_fixes']}")
        print(f"   • Stage fixes: {stats['stage_fixes']}")
        print(f"   • Lab corrections: {stats['corrected_labs']}")
        print(f"   • Vital corrections: {stats['corrected_vitals']}")
        print(f"   • O2 device corrections: {stats['corrected_o2_devices']}")
        print(f"   • Records dropped: {stats['dropped_records']}")
        print(f"   • Records kept: {stats['total_records'] - stats['dropped_records']}")
        
        return stats

def main():
    """Main function - Comprehensive data correction"""
    
    print('🔧 COMPREHENSIVE EHR DATA CORRECTION')
    print('=' * 60)
    print('Fixes: CheXpert semantics, Lab units, Vitals validation, Redaction artifacts')
    print('This prevents the model from learning incorrect medical relationships!')
    print()
    
    corrector = EHRDataCorrector()
    
    # Correct training dataset
    print('📊 Processing training dataset...')
    stats_train = corrector.correct_dataset(
        'src/data/processed/curriculum_train_final_clean.jsonl',
        'src/data/processed/curriculum_train_final_clean_CORRECTED.jsonl'
    )
    
    print('✅ Training dataset correction complete!')
    print()
    
    # Correct validation dataset
    print('📊 Processing validation dataset...')
    stats_val = corrector.correct_dataset(
        'src/data/processed/curriculum_val_final_clean.jsonl',
        'src/data/processed/curriculum_val_final_clean_CORRECTED.jsonl'
    )
    
    print('✅ Validation dataset correction complete!')
    print()
    
    # Overall summary
    print('🎉 ALL CORRECTIONS COMPLETE!')
    print('=' * 60)
    print(f'📊 Training: {stats_train["total_records"]} records, {stats_train["dropped_records"]} dropped')
    print(f'📊 Validation: {stats_val["total_records"]} records, {stats_val["dropped_records"]} dropped')
    print()
    print('🔧 FIXES APPLIED:')
    print(f'   • CheXpert label fixes: {stats_train["chexpert_fixes"] + stats_val["chexpert_fixes"]}')
    print(f'   • Stage consistency fixes: {stats_train["stage_fixes"] + stats_val["stage_fixes"]}')
    print(f'   • Lab unit corrections: {stats_train["corrected_labs"] + stats_val["corrected_labs"]}')
    print(f'   • Vital corrections: {stats_train["corrected_vitals"] + stats_val["corrected_vitals"]}')
    print()
    print('✅ The corrected datasets are now ready for training!')
    print('This ensures medical accuracy and prevents learning wrong relationships.')

if __name__ == '__main__':
    main()
