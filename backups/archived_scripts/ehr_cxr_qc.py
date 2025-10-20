"""
EHR + CXR Dataset QC
--------------------
Usage:
    python ehr_cxr_qc.py /path/to/your_dataset.jsonl --base-dir /path/to/images

- Assumes one JSON object per line (like the samples you shared).
- Prints a concise report with WARN/ERROR items per record.
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

# --- Helpers -----------------------------------------------------------------

NEG_PATTERNS = {
    "pneumothorax": re.compile(r"\bno (?:evidence of |evidence for |)\s*pneumothorax\b|\bwithout pneumothorax\b", re.I),
    "pneumonia": re.compile(r"\bno (?:acute )?pneumonia\b|\bwithout (?:acute )?pneumonia\b", re.I),
    "edema": re.compile(r"\bno (?:pulmonary )?edema\b|\bwithout (?:pulmonary )?edema\b", re.I),
    "effusion": re.compile(r"\bno (?:pleural )?effusion\b|\bwithout (?:pleural )?effusion\b|\bno (?:larger )?pleural effusions\b", re.I),
    "opacity": re.compile(r"\bno (?:acute )?opacit(?:y|ies)\b|\bclear lungs?\b", re.I),
    "consolidation": re.compile(r"\bno consolidation\b|\bwithout definite consolidation\b", re.I),
    "cardiomegaly": re.compile(r"\bno (?:cardiac )?enlargement\b|\bheart size is normal\b|\bcardiomediastinum is (?:within|w\/in) normal\b", re.I)
}

POS_PATTERNS = {
    "pneumothorax": re.compile(r"\b(?:acute\s+)?pneumothorax\b", re.I),
    "pneumonia": re.compile(r"\b(?:acute\s+)?pneumonia\b", re.I),
    "edema": re.compile(r"\bpulmonary edema\b", re.I),
    "effusion": re.compile(r"\b(?:acute\s+)?pleural effusion\b|\bblunting of the .* costophrenic", re.I),
    "opacity": re.compile(r"\b(?:acute\s+)?(?:opacity|opacities)\b", re.I),
    "consolidation": re.compile(r"\b(?:acute\s+)?consolidation\b", re.I),
    "cardiomegaly": re.compile(r"\bcardiomegaly\b|\benlarged cardiomediastinum\b", re.I),
    "devices": re.compile(r"\b(?:picc|feeding tube|sternal wires|chest tube|endotracheal|dialysis catheter|pacer|icd|central line)\b", re.I)
}

CHEXPERT_KEYS = [
    "Consolidation","Edema","Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","No Finding","Pleural Effusion","Pleural Other","Pneumonia","Pneumothorax","Support Devices"
]

# Loose physiological sanity ranges (not for clinical use)
LAB_RANGES = {
    "Na": ("mEq/L", 110, 170),
    "K": ("mEq/L", 2.0, 7.0),
    "Creatinine": ("mg/dL", 0.2, 15),
    "BUN": ("mg/dL", 1, 200),
    "Glucose": ("mg/dL", 20, 800),
    "HGB": ("g/dL", 3, 22),
    "WBC": ("K/uL", 0.5, 60),
    "PLT": ("K/uL", 10, 1500),
    "CRP": ("mg/L", 0, 1000),
    "ALT": ("U/L", 1, 5000),
    "Total_Bilirubin": ("mg/dL", 0, 60),
    "Albumin": ("g/dL", 1, 6),
    "BNP": ("pg/mL", 0, 100000),  # BNP >100k pg/mL is exceedingly unlikely; NT-proBNP can be higher
    "PT": ("sec", 5, 60),
    "INR": ("ratio", 0.5, 10),
    "Troponin_T": ("ng/mL", 0, 100)
}

def warn(msg): return f"WARN: {msg}"
def err(msg): return f"ERROR: {msg}"

def label_name_map(key: str) -> str:
    # map CheXpert label to canonical feature
    mapping = {
        "Pneumothorax": "pneumothorax",
        "Pneumonia": "pneumonia",
        "Edema": "edema",
        "Pleural Effusion": "effusion",
        "Lung Opacity": "opacity",
        "Consolidation": "consolidation",
        "Enlarged Cardiomediastinum": "cardiomegaly",
        "Support Devices": "devices"
    }
    return mapping.get(key, key.lower())

def check_labels_against_text(rec, issues):
    text = (rec.get("impression") or "").lower()

    labels = rec.get("chexpert_labels", {})
    # 1) "No Finding" cannot co-exist with any positive finding
    if labels.get("No Finding", -1) == 1:
        any_pos = any(v == 1 for k,v in labels.items() if k != "No Finding")
        if any_pos:
            issues.append(err('"No Finding" is 1 but at least one other label is positive.'))

    # 2) Obvious text/label mismatches for a few key entities
    for key in ["Pneumothorax","Pneumonia","Edema","Pleural Effusion","Lung Opacity","Consolidation","Enlarged Cardiomediastinum","Support Devices"]:
        v = labels.get(key, None)
        canon = label_name_map(key)
        if v is None:
            continue

        # Check for explicit negatives (higher priority)
        is_negative = NEG_PATTERNS.get(canon) and NEG_PATTERNS[canon].search(text)
        
        if is_negative:
            if v != -1:
                issues.append(err(f'Label "{key}" should be -1 (text says no {canon}). Found {v}.'))
        
        # Note: Positive pattern matching disabled to avoid false positives
        # The correction script handles positive patterns more accurately

        # discourage uncertain 0 when text is a clean "no ..."
        if v == 0 and NEG_PATTERNS.get(canon) and NEG_PATTERNS[canon].search(text):
            issues.append(warn(f'Label "{key}" is 0 (uncertain) but text is definitive negative. Consider -1.'))

    return issues

def check_vitals(rec, issues):
    vit = rec.get("patient_data", {}).get("Vitals", {})
    rr = vit.get("respiratory_rate", {}).get("value")
    if rr is not None and rr <= 0:
        issues.append(err(f"respiratory_rate={rr} is impossible."))

    spo2 = vit.get("o2_saturation", {}).get("value")
    if spo2 is not None and not (0 <= spo2 <= 100):
        issues.append(err(f"o2_saturation={spo2} out of 0-100%."))

    hr = vit.get("heart_rate", {}).get("value")
    if hr is not None and (hr < 20 or hr > 220):
        issues.append(warn(f"heart_rate={hr} looks implausible."))

    bp = vit.get("mean_bp", {}).get("value")
    if bp is not None and (bp < 30 or bp > 200):
        issues.append(warn(f"mean_bp={bp} out of expected clinical range (30-200 mmHg)."))
    return issues

def check_labs(rec, issues):
    labs = rec.get("patient_data", {}).get("Labs", {})
    for cname, cfg in LAB_RANGES.items():
        expected_unit, lo, hi = cfg
        L = labs.get(cname)
        if not L:
            continue
        unit = L.get("unit")
        val = L.get("value")
        if unit and unit != expected_unit:
            # specific critical flags
            if cname == "BNP" and unit == "mg/dL":
                issues.append(err("BNP is in mg/dL — do not convert; source is likely wrong. Use pg/mL only or drop BNP."))
            elif cname == "BUN" and unit == "mEq/L":
                issues.append(err("BUN recorded in mEq/L — invalid. Accept mg/dL only. If urea in mmol/L, convert to BUN mg/dL using ×2.8."))
            else:
                issues.append(warn(f"{cname} unit is {unit} but expected {expected_unit}."))
        if isinstance(val, (int, float)):
            if cname == "BNP" and val > 100000:
                issues.append(err(f"BNP={val} pg/mL is implausibly high for BNP (maybe NT-proBNP or bad conversion)."))
            if cname == "PLT" and val < 10:
                issues.append(err(f"PLT={val} K/uL — near-zero platelets across many records usually indicates a scaling bug."))
            if val < lo or val > hi:
                issues.append(warn(f"{cname}={val} {unit or ''} is outside a broad sanity range [{lo}, {hi}]."))
    return issues

def check_structure(rec, issues, base_dir: Path|None):
    # Stage expectations
    stage = rec.get("stage")
    if stage not in {"A","B"}:
        issues.append(err(f'Unknown stage "{stage}". Expected "A" or "B".'))
    if stage == "B":
        pd = rec.get("patient_data")
        if not pd:
            issues.append(err("Stage B requires patient_data, but it's missing."))
        else:
            # basic fields
            for k in ["subject_id","Age","Sex"]:
                if k not in pd:
                    issues.append(err(f"patient_data.{k} is missing for Stage B."))
    # BMI duplication
    pd = rec.get("patient_data", {})
    vit = pd.get("Vitals", {})
    if "bmi" in vit and "BMI" in vit:
        issues.append(warn("Both 'bmi' and 'BMI' present in Vitals. Keep one canonical key (e.g., 'BMI')."))

    # Image path existence (optional check)
    img = rec.get("image_path")
    if img and base_dir:
        p = (base_dir / img).resolve()
        if not p.exists():
            issues.append(warn(f"image_path does not exist on disk: {p}"))
    return issues

def summarize_issues(all_issues):
    summary = defaultdict(int)
    for issues in all_issues:
        for item in issues:
            if item.startswith("ERROR"):
                summary["ERROR"] += 1
            elif item.startswith("WARN"):
                summary["WARN"] += 1
    return summary

# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="Path to JSONL file (one record per line).")
    ap.add_argument("--base-dir", help="Optional base dir to check image_path existence.", default=None)
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None

    all_issues = []
    with open(args.jsonl, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception as e:
                print(f"[Line {i}] ERROR: invalid JSON: {e}")
                continue

            issues = []
            check_labels_against_text(rec, issues)
            check_vitals(rec, issues)
            check_labs(rec, issues)
            check_structure(rec, issues, base_dir)

            if issues:
                print(f"--- Record {i} ---")
                for it in issues:
                    print(it)
                print()

            all_issues.append(issues)

    summary = summarize_issues(all_issues)
    print("Summary:", dict(summary))

if __name__ == "__main__":
    main()
