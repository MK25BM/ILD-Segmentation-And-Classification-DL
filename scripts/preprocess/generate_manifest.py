"""
Generate dataset manifest CSV with slice alignment validation.

Reads from raw DICOM directories and creates manifest tracking:
- Total slices per modality (CT, lung_mask, roi_mask)
- Aligned slice counts (slices present in all 3 modalities)
- Class distribution per patient
- Cohort classification (Core_Set vs Incomplete_Data_Exception)

Usage:
    python scripts/preprocess/generate_manifest.py \
        --dicom-root /path/to/ILD_DB \
        --output manifest.csv
"""

import os
import sys
import glob
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydicom

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CLASS_MAPPING = {
    1: "healthy",
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
    7: "bronchial_wall_thickening",
    8: "reticulation",
    9: "macronodules",
    10: "cysts",
    11: "peripheral_micronodules",
    12: "bronchiectasis",
    13: "air_trapping",
    14: "early_fibrosis",
    15: "increased_attenuation",
    16: "tuberculosis",
    17: "pcp",
}


def get_clean_dicoms(directory: str) -> List[str]:
    """Get sorted list of .dcm files (skip hidden files)."""
    if not os.path.exists(directory):
        return []
    files = glob.glob(os.path.join(directory, "*.dcm"))
    return sorted([f for f in files if not os.path.basename(f).startswith(".")])


def extract_trailing_suffix(filename: str) -> Optional[int]:
    """Extract trailing digits from filename via regex.
    
    Examples:
    - "000001.dcm" → 1
    - "000042.dcm" → 42
    """
    name_without_ext = os.path.splitext(filename)[0]
    match = re.search(r"(\d+)$", name_without_ext)
    return int(match.group(1)) if match else None


def build_suffix_map(file_list: List[str]) -> Dict[int, str]:
    """Build mapping from suffix (trailing digits) to full file path."""
    suffix_map = {}
    for f in file_list:
        idx = extract_trailing_suffix(os.path.basename(f))
        if idx is not None:
            suffix_map[idx] = f
    return suffix_map


def extract_class_counts(dcm_file: str) -> Dict[str, int]:
    """Extract class pixel counts from ROI DICOM file.
    
    Returns dict: {class_name: pixel_count, ...}
    """
    try:
        ds = pydicom.dcmread(dcm_file, stop_before_pixels=False)
        pixel_array = ds.pixel_array
        
        counts = {}
        for class_id, class_name in CLASS_MAPPING.items():
            counts[class_name] = int(np.sum(pixel_array == class_id))
        return counts
    except Exception as e:
        logger.warning(f"Failed to extract classes from {dcm_file}: {e}")
        return {c: 0 for c in CLASS_MAPPING.values()}


def generate_manifest(
    dicom_root: str,
    output_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate manifest CSV for ILD dataset.
    
    Args:
        dicom_root: Root directory containing ILD_DB_volumeROIs and ILD_DB_lungMasks
        output_path: Where to save manifest.csv
        verbose: Print progress
    
    Returns:
        DataFrame with manifest data
    """
    vol_dir = os.path.join(dicom_root, "ILD_DB_volumeROIs")
    mask_dir = os.path.join(dicom_root, "ILD_DB_lungMasks")
    
    if not os.path.exists(vol_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(
            f"Expected DICOM directories not found.\n"
            f"  {vol_dir}: {os.path.exists(vol_dir)}\n"
            f"  {mask_dir}: {os.path.exists(mask_dir)}"
        )
    
    manifest_data = []
    base_patients = sorted([
        d for d in os.listdir(vol_dir)
        if os.path.isdir(os.path.join(vol_dir, d))
    ])
    
    logger.info(f"Found {len(base_patients)} base patient folders")
    
    for p_id in base_patients:
        p_vol_root = os.path.join(vol_dir, p_id)
        p_mask_root = os.path.join(mask_dir, p_id)
        
        # Handle special "HRCT_pilot" folder with sessions
        if p_id == "HRCT_pilot":
            sessions = sorted([
                s for s in os.listdir(p_vol_root)
                if os.path.isdir(os.path.join(p_vol_root, s))
            ])
            session_pairs = [
                (f"pilot_{s}", os.path.join(p_vol_root, s), os.path.join(p_mask_root, s))
                for s in sessions
            ]
        else:
            # Check for sub-session folders
            subdirs = sorted([
                d for d in os.listdir(p_vol_root)
                if os.path.isdir(os.path.join(p_vol_root, d)) and d != "roi_mask"
            ])
            
            if subdirs:
                session_pairs = [
                    (f"{p_id}_{s}", os.path.join(p_vol_root, s), os.path.join(p_mask_root, s))
                    for s in subdirs
                ]
            else:
                session_pairs = [(p_id, p_vol_root, p_mask_root)]
        
        # Process each session
        for unique_id, vol_path, mask_path in session_pairs:
            # Get DICOM files
            ct_raw_list = get_clean_dicoms(vol_path)
            roi_raw_list = get_clean_dicoms(os.path.join(vol_path, "roi_mask"))
            lung_raw_list = get_clean_dicoms(mask_path)
            
            if not lung_raw_list:
                lung_raw_list = get_clean_dicoms(os.path.join(mask_path, "lung_mask"))
            
            ct_len = len(ct_raw_list)
            roi_len = len(roi_raw_list)
            lung_len = len(lung_raw_list)
            
            # Check alignment
            if ct_len == 0 or roi_len == 0 or lung_len == 0:
                cohort = "Incomplete_Data_Exception"
                aligned_count = 0
                class_counts = {c: 0 for c in CLASS_MAPPING.values()}
            else:
                # Build suffix maps and find common slices
                ct_map = build_suffix_map(ct_raw_list)
                roi_map = build_suffix_map(roi_raw_list)
                lung_map = build_suffix_map(lung_raw_list)
                
                common_suffixes = sorted(
                    list(set(ct_map.keys()) & roi_map.keys() & lung_map.keys())
                )
                aligned_count = len(common_suffixes)
                
                cohort = "Core_Set" if aligned_count > 0 else "Incomplete_Data_Exception"
                
                # Extract class distribution from first aligned ROI
                class_counts = {c: 0 for c in CLASS_MAPPING.values()}
                if common_suffixes:
                    first_roi = roi_map[common_suffixes[0]]
                    class_counts.update(extract_class_counts(first_roi))
            
            # Get total lung tissue pixels (approximate from first lung mask)
            total_lung_pixels = 0
            if lung_len > 0:
                try:
                    ds = pydicom.dcmread(lung_raw_list[0])
                    total_lung_pixels = int(np.sum(ds.pixel_array > 0))
                except:
                    pass
            
            # Build record
            record = {
                "Unique_Session_ID": unique_id,
                "Cohort": cohort,
                "Total_CT_Slices": ct_len,
                "Total_Lung_Mask_Slices": lung_len,
                "Total_ROI_Slices": roi_len,
                "Aligned_Slices_Count": aligned_count,
                "Total_Lung_Tissue_Pixels": total_lung_pixels,
            }
            
            # Add class columns
            for class_name in sorted(CLASS_MAPPING.values()):
                record[class_name] = class_counts.get(class_name, 0)
            
            manifest_data.append(record)
            
            if verbose and len(manifest_data) % 10 == 0:
                logger.info(f"Processed {len(manifest_data)} sessions...")
    
    # Create DataFrame
    df = pd.DataFrame(manifest_data)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"✓ Manifest saved to {output_path}")
    logger.info(f"\nCohort distribution:\n{df['Cohort'].value_counts()}")
    
    return df


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Generate ILD dataset manifest from DICOM files"
    )
    parser.add_argument(
        "--dicom-root",
        type=str,
        required=True,
        help="Root directory containing ILD_DB_volumeROIs and ILD_DB_lungMasks",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset_manifest.csv",
        help="Output manifest path (default: dataset_manifest.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )
    
    args = parser.parse_args()
    
    generate_manifest(
        dicom_root=args.dicom_root,
        output_path=args.output,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()