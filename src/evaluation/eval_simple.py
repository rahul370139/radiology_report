#!/usr/bin/env python3
"""
Simple Evaluation Script for Radiology Report Model
Uses inference pipeline for real generation
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import sys
import os

# Add src to path and import pipeline
after_src = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.abspath(after_src))
from inference.pipeline import generate as pipeline_generate

class SimpleEvaluator:
    """Simple evaluator using inference pipeline"""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with config (kept for interface parity)"""
        self.config_path = config_path
    
    def load_model(self):
        """No-op: model is lazily loaded by the pipeline"""
        print("🤖 Using inference pipeline (lazy-loaded model)...")
    
    def evaluate_sample(self, image_path: str, ehr_data: Dict[str, Any] = None, device: str = "cpu") -> Dict[str, Any]:
        """Evaluate a single sample using the inference pipeline"""
        result = pipeline_generate(image_path, ehr_data, device=device)
        return {
            'image_path': image_path,
            'ehr_data': ehr_data,
            'response': json.dumps({
                'impression': result.get('impression', ''),
                'chexpert': result.get('chexpert', {}),
                'icd': result.get('icd', {})
            }),
            'parsed': result
        }


def main():
    parser = argparse.ArgumentParser(description='Simple Evaluation for Radiology Report Model')
    parser.add_argument('--image', required=True, help='Path to chest X-ray image')
    parser.add_argument('--ehr_json', help='Path to EHR JSON file (optional, for Stage B)')
    parser.add_argument('--config', default='configs/advanced_training_config.yaml', help='Config file path (unused)')
    parser.add_argument('--device', default='cpu', help='Device: cpu|cuda|mps')
    parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    print("🔬 SIMPLE EVALUATION")
    print("=" * 40)
    
    evaluator = SimpleEvaluator(args.config)
    evaluator.load_model()
    
    ehr_data = None
    if args.ehr_json:
        with open(args.ehr_json, 'r') as f:
            ehr_data = json.load(f)
        print(f"📋 Loaded EHR data from {args.ehr_json}")
    
    print(f"🖼️ Processing image: {args.image}")
    result = evaluator.evaluate_sample(args.image, ehr_data, device=args.device)
    
    print("\n📊 RESULTS:")
    print("=" * 40)
    print(f"IMPRESSION:\n{result['parsed']['impression']}\n")
    
    print("CHEXPERT LABELS:")
    for label, value in result['parsed']['chexpert'].items():
        print(f"  {label}: {value}")
    
    if result['parsed'].get('icd'):
        print("\nICD PREDICTIONS:")
        for label, value in result['parsed']['icd'].items():
            print(f"  {label}: {value}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    main()
