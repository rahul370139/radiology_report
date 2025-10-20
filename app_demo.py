"""
Streamlit Demo App for Radiology Report Generation
A/B Testing Interface: Demo A (image-only) vs Demo B (image+EHR)
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from PIL import Image
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from inference.pipeline import generate, get_pipeline

# Page config
st.set_page_config(
    page_title="Radiology Report Generator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stage-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .ehr-box {
        background-color: #fff2e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_demo_manifest():
    """Load the demo manifest CSV"""
    manifest_path = Path("evaluation/demo_manifest.csv")
    if not manifest_path.exists():
        st.error("❌ Demo manifest not found. Please run generate_demo_manifest.py first.")
        return None
    
    return pd.read_csv(manifest_path)

def load_ehr_data(ehr_path):
    """Load EHR data from JSON file"""
    if not ehr_path or not Path(ehr_path).exists():
        return None
    
    with open(ehr_path, 'r') as f:
        return json.load(f)

def display_chexpert_labels(labels_dict, title="CheXpert Labels"):
    """Display CheXpert labels in a nice format"""
    st.subheader(title)
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (label, value) in enumerate(list(labels_dict.items())[:6]):
            if value == 1:
                st.write(f"✅ **{label}**")
            elif value == -1:
                st.write(f"❌ **{label}**")
            else:
                st.write(f"⚪ {label}")
    
    with col2:
        for i, (label, value) in enumerate(list(labels_dict.items())[6:]):
            if value == 1:
                st.write(f"✅ **{label}**")
            elif value == -1:
                st.write(f"❌ **{label}**")
            else:
                st.write(f"⚪ {label}")

def display_icd_labels(labels_dict, title="ICD Predictions"):
    """Display ICD labels in a nice format"""
    st.subheader(title)
    
    # Only show labels with confidence > 0
    active_labels = {k: v for k, v in labels_dict.items() if v > 0}
    
    if active_labels:
        for label, confidence in active_labels.items():
            confidence_pct = confidence * 100
            st.write(f"🔍 **{label}**: {confidence_pct:.1f}%")
    else:
        st.write("No ICD predictions with confidence > 0")

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Radiology Report Generator</h1>', unsafe_allow_html=True)
    st.markdown("**A/B Testing Interface**: Compare image-only vs image+EHR inference")
    
    # Load demo manifest
    manifest_df = load_demo_manifest()
    if manifest_df is None:
        return
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Device selection
    device = st.sidebar.selectbox("Select Device", ["cpu", "cuda", "mps"], index=0)
    
    # Stage selection
    stage = st.sidebar.selectbox("Select Stage", ["A (Image Only)", "B (Image + EHR)"], index=0)
    is_stage_b = stage.startswith("B")
    
    # Sample selection
    if is_stage_b:
        available_samples = manifest_df[manifest_df['stage'] == 'B']
    else:
        available_samples = manifest_df[manifest_df['stage'] == 'A']
    
    if len(available_samples) == 0:
        st.error(f"No {stage} samples available in manifest")
        return
    
    sample_idx = st.sidebar.selectbox(
        f"Select Sample (0-{len(available_samples)-1})", 
        range(len(available_samples)),
        format_func=lambda x: f"Sample {x+1}: {available_samples.iloc[x]['study_id']}"
    )
    
    selected_sample = available_samples.iloc[sample_idx]
    
    # Load EHR data if Stage B
    ehr_data = None
    if is_stage_b and pd.notna(selected_sample['ehr_json_path']):
        ehr_data = load_ehr_data(selected_sample['ehr_json_path'])
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="stage-header">📸 Input</h2>', unsafe_allow_html=True)
        
        # Display image
        image_path = selected_sample['image_path']
        if Path(image_path).exists():
            image = Image.open(image_path)
            st.image(image, caption=f"Study ID: {selected_sample['study_id']}", use_column_width=True)
        else:
            st.error(f"❌ Image not found: {image_path}")
            return
        
        # Display EHR data if available
        if ehr_data:
            st.markdown('<h3 class="ehr-box">📋 Patient EHR Data</h3>', unsafe_allow_html=True)
            
            # Basic info
            st.write(f"**Age**: {ehr_data.get('Age', 'N/A')}")
            st.write(f"**Sex**: {ehr_data.get('Sex', 'N/A')}")
            
            # Vitals
            vitals = ehr_data.get('Vitals', {})
            if vitals:
                st.write("**Vitals**:")
                for key, value in vitals.items():
                    st.write(f"  - {key}: {value}")
            
            # Labs
            labs = ehr_data.get('Labs', {})
            if labs:
                st.write("**Key Labs**:")
                for key, value in list(labs.items())[:5]:  # Show first 5 labs
                    if isinstance(value, dict):
                        val = value.get('value', 'N/A')
                        unit = value.get('unit', '')
                        st.write(f"  - {key}: {val} {unit}")
            
            # Chronic conditions
            conditions = ehr_data.get('Chronic_conditions', [])
            if conditions:
                st.write(f"**Chronic Conditions**: {', '.join(conditions)}")
    
    with col2:
        st.markdown('<h2 class="stage-header">🤖 AI Prediction</h2>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🚀 Generate Report", type="primary"):
            with st.spinner("Generating radiology report..."):
                start_time = time.time()
                
                try:
                    # Generate prediction
                    result = generate(image_path, ehr_data, device=device)
                    generation_time = time.time() - start_time
                    
                    # Display results
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    
                    # Impression
                    st.subheader("📝 Impression")
                    st.write(result['impression'])
                    
                    # CheXpert labels
                    display_chexpert_labels(result['chexpert'])
                    
                    # ICD labels (only for Stage B)
                    if is_stage_b:
                        display_icd_labels(result['icd'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Performance metrics
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.write(f"⏱️ **Generation Time**: {generation_time:.2f} seconds")
                    st.write(f"🖥️ **Device**: {device}")
                    st.write(f"📊 **Stage**: {stage}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during generation: {str(e)}")
                    st.write("Please check that the model is properly loaded and the image path is correct.")
    
    # Ground truth section
    st.markdown('<h2 class="stage-header">📋 Ground Truth (for comparison)</h2>', unsafe_allow_html=True)
    
    col3, col4 = st.columns([1, 1])
    
    with col3:
        st.subheader("📝 Ground Truth Impression")
        st.write(selected_sample['ground_truth_impression'])
    
    with col4:
        st.subheader("🏷️ Ground Truth CheXpert")
        gt_chexpert = json.loads(selected_sample['ground_truth_chexpert'].replace("'", "\""))
        display_chexpert_labels(gt_chexpert, "Ground Truth CheXpert")
    
    # Ground truth ICD (only for Stage B)
    if is_stage_b and pd.notna(selected_sample['ground_truth_icd']):
        st.subheader("🏷️ Ground Truth ICD")
        gt_icd = json.loads(selected_sample['ground_truth_icd'].replace("'", "\""))
        # Convert list to dict if needed
        if isinstance(gt_icd, list):
            gt_icd = {item['code']: item.get('confidence', 0) for item in gt_icd}
        display_icd_labels(gt_icd, "Ground Truth ICD")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note**: This is a demo interface for the fine-tuned LLaVA-Med model. Results are for demonstration purposes only.")

if __name__ == "__main__":
    main()
