#!/usr/bin/env python3.10
"""
Extract Impression text from MIMIC-CXR radiology reports.

This script processes radiology report text files and extracts the IMPRESSION section,
with fallback to FINDINGS section if IMPRESSION is not found. It also performs
basic de-identification by removing common artifacts.

Usage:
    python3.10 01_extract_impression.py

Output:
    data/impressions.jsonl - JSONL file with study_id and impression text
"""

import re
import json
import pandas as pd
import pathlib as pl
from typing import Dict, Optional

# Configuration
REPORTS_DIR = pl.Path("files/reports/")  # MIMIC-CXR reports directory
OUTPUT_FILE = pl.Path("data/processed/impressions.jsonl")

def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace and common artifacts.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove de-identification artifacts
    text = re.sub(r'\[\*\*.*?\*\*\]', '', text)  # Remove [** **] patterns
    text = re.sub(r'Dictated by.*?\.', '', text, flags=re.IGNORECASE)  # Remove dictation info
    text = re.sub(r'Electronically signed by.*?\.', '', text, flags=re.IGNORECASE)  # Remove signature info
    text = re.sub(r'Date:.*?\.', '', text, flags=re.IGNORECASE)  # Remove date info
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = text.strip()
    
    return text

def extract_impression_from_text(text: str) -> Optional[str]:
    """
    Extract impression text from radiology report.
    
    Args:
        text: Full radiology report text
        
    Returns:
        Extracted impression text or None if not found
    """
    # Convert to uppercase for pattern matching
    text_upper = text.upper()
    
    # Pattern for IMPRESSION section (more flexible)
    impression_patterns = [
        re.compile(r'(?<=IMPRESSION:)(.*?)(?:\n[A-Z ]+:|$)', re.DOTALL),
        re.compile(r'(?<=IMPRESSION)(.*?)(?:\n[A-Z ]+:|$)', re.DOTALL),
        re.compile(r'(?<=IMPRESSION:)(.*?)(?:\n[A-Z]{2,}|$)', re.DOTALL),
    ]
    
    # Try to find IMPRESSION section
    for pattern in impression_patterns:
        match = pattern.search(text_upper)
        if match:
            impression_text = match.group(1).strip()
            if impression_text and len(impression_text) > 10:  # Minimum length check
                return clean_text(impression_text)
    
    # Fallback to FINDINGS section if no IMPRESSION found
    findings_patterns = [
        re.compile(r'(?<=FINDINGS:)(.*?)(?:\n[A-Z ]+:|$)', re.DOTALL),
        re.compile(r'(?<=FINDINGS)(.*?)(?:\n[A-Z ]+:|$)', re.DOTALL),
        re.compile(r'(?<=FINDINGS:)(.*?)(?:\n[A-Z]{2,}|$)', re.DOTALL),
    ]
    
    for pattern in findings_patterns:
        match = pattern.search(text_upper)
        if match:
            findings_text = match.group(1).strip()
            if findings_text and len(findings_text) > 10:  # Minimum length check
                return clean_text(findings_text)
    
    return None

def process_reports() -> None:
    """
    Process all radiology reports and extract impressions.
    """
    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if reports directory exists
    if not REPORTS_DIR.exists():
        print(f"Error: Reports directory {REPORTS_DIR} not found!")
        print("Please ensure the MIMIC-CXR reports are in the correct location.")
        return
    
    print(f"Processing reports from: {REPORTS_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Get all report files (including subdirectories)
    report_files = list(REPORTS_DIR.rglob("*.txt"))
    print(f"Found {len(report_files)} report files")
    
    if not report_files:
        print("No .txt files found in reports directory!")
        return
    
    # Process reports
    processed_count = 0
    extracted_count = 0
    
    with OUTPUT_FILE.open("w", encoding="utf-8") as outfile:
        for i, report_file in enumerate(report_files):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(report_files)} files...")
            
            try:
                # Read report text
                report_text = report_file.read_text(encoding="utf-8", errors="ignore")
                
                # Extract impression
                impression = extract_impression_from_text(report_text)
                
                if impression:
                    # Write to JSONL
                    record = {
                        "study_id": report_file.stem,
                        "impression": impression,
                        "source_file": str(report_file)
                    }
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    extracted_count += 1
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {report_file}: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total files processed: {processed_count}")
    print(f"Impressions extracted: {extracted_count}")
    print(f"Success rate: {extracted_count/processed_count*100:.1f}%")
    print(f"Output saved to: {OUTPUT_FILE}")

def main():
    """Main function."""
    print("MIMIC-CXR Impression Extraction")
    print("=" * 40)
    
    process_reports()

if __name__ == "__main__":
    main()
