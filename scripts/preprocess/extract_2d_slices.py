"""
Extract 2D slices from raw DICOM files and save as .npy + .png.

Reads DICOM files using suffix-aware matching and saves aligned slices:
- CT images → normalized .npy (uint8) + .png
- Lung masks → binary .npy + .png
- ROI masks → class indices .npy + .png

Usage:
    python scripts/preprocess/extract_2d_slices.py \
        --manifest manifest.csv \
        --dicom-root /path/to/ILD_DB \
        --output-root /scratch/ild_dataset_processed
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
import numpy as np
import pydicom
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_clean_dicoms(directory: str) -> List[str]:
    """Get sorted list of .dcm files (skip hidden files)."""
    if not os.path.exists(directory):
        return []
    files = glob.glob(os.path.join(directory, "*.dcm"))
    return sorted([f for f in files if not os.path.basename(f).startswith(".")])


def extract_trailing_suffix(filename: str) -> Optional[int]:
    """Extract trailing digits from filename."""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.search(r"(\d+)$", name_without_ext)
    return int(match.group(1)) if match else None


def build_suffix_map(file_list: List[str]) -> Dict[int, str]:
    """Build mapping from suffix to full path."""
    suffix_map = {}
    for f in file_list:
        idx = extract_trailing_suffix(os.path.basename(f))
        if idx is not None:
            suffix_map[idx] = f
    return suffix_map


def normalize_ct(pixel_array: np.ndarray) -> np.ndarray:
    """Normalize CT Hounsfield units to [0, 255]."""
    # Typical HRCT range: -1000 (air) to 3000 (bone)
    # Clip to [-200, 400] for lung tissue
    clipped = np.clip(pixel_array, -200, 400)
    normalized = ((clipped + 200) / 600 * 255).astype(np.uint8)
    return normalized


def save_slice_images(
    array: np.ndarray,
    output_dir: str,
    slice_num: int,
    array_type: str = "image",  # "image", "mask", "binary"
):
    """Save array as .npy and .png."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename_base = f"slice_{slice_num}"
    
    # Save .npy
    npy_path = os.path.join(output_dir, f"{filename_base}.npy")
    np.save(npy_path, array)
    
    # Save .png
    png_path = os.path.join(output_dir, f"{filename_base}.png")
    
    if array_type == "image":
        # Normalized CT (0-255)
        img = Image.fromarray(array, mode="L")
    elif array_type == "binary":
        # Binary mask (0 or 1) → convert to 0 or 255
        img_array = (array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
    else:
        # Class indices → use values as-is
        img = Image.fromarray(array.astype(np.uint8), mode="L")
    
    img.save(png_path)


def extract_session_slices(
    session_id: str,
    vol_path: str,
    mask_path: str,
    output_root: str,
    dicom_root: str,
):
    """Extract aligned slices for a single session."""
    
    # Find DICOM files
    ct_files = get_clean_dicoms(vol_path)
    roi_files = get_clean_dicoms(os.path.join(vol_path, "roi_mask"))
    lung_files = get_clean_dicoms(mask_path)
    if not lung_files:
        lung_files = get_clean_dicoms(os.path.join(mask_path, "lung_mask"))
    
    if not all([ct_files, roi_files, lung_files]):
        logger.warning(f"⚠️  Skipping {session_id}: missing modalities")
        return 0
    
    # Build suffix maps
    ct_map = build_suffix_map(ct_files)
    roi_map = build_suffix_map(roi_files)
    lung_map = build_suffix_map(lung_files)
    
    # Find aligned slices
    common_suffixes = sorted(
        list(set(ct_map.keys()) & roi_map.keys() & lung_map.keys())
    )
    
    if not common_suffixes:
        logger.warning(f"⚠️  No aligned slices in {session_id}")
        return 0
    
    # Create output directories
    session_dir = os.path.join(output_root, f"patient_{session_id}")
    images_dir = os.path.join(session_dir, "images")
    roi_dir = os.path.join(session_dir, "roi_masks")
    lung_dir = os.path.join(session_dir, "lung_masks")
    
    for d in [images_dir, roi_dir, lung_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Extract slices
    for suffix in common_suffixes:
        try:
            # Load CT image
            ct_ds = pydicom.dcmread(ct_map[suffix])
            ct_array = ct_ds.pixel_array.astype(np.float32)
            ct_normalized = normalize_ct(ct_array)
            
            # Load ROI mask
            roi_ds = pydicom.dcmread(roi_map[suffix])
            roi_array = roi_ds.pixel_array.astype(np.int64)
            
            # Load lung mask
            lung_ds = pydicom.dcmread(lung_map[suffix])
            lung_array = (lung_ds.pixel_array > 0).astype(np.float32)
            
            # Save
            save_slice_images(ct_normalized, images_dir, suffix, array_type="image")
            save_slice_images(roi_array, roi_dir, suffix, array_type="mask")
            save_slice_images(lung_array, lung_dir, suffix, array_type="binary")
            
        except Exception as e:
            logger.error(f"Failed to extract slice {suffix} from {session_id}: {e}")
            continue
    
    return len(common_suffixes)


def extract_all_slices(
    manifest_path: str,
    dicom_root: str,
    output_root: str,
):
    """Extract all aligned slices for Core_Set sessions."""
    
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    core_set = manifest[manifest["Cohort"] == "Core_Set"].copy()
    
    logger.info(f"Extracting {len(core_set)} Core_Set sessions...")
    
    vol_dir = os.path.join(dicom_root, "ILD_DB_volumeROIs")
    mask_dir = os.path.join(dicom_root, "ILD_DB_lungMasks")
    
    total_slices = 0
    
    for _, row in tqdm(core_set.iterrows(), total=len(core_set)):
        session_id = row["Unique_Session_ID"]
        
        # Map session_id to folder paths
        if session_id.startswith("pilot_"):
            # pilot_200 → HRCT_pilot/200
            pilot_num = session_id.replace("pilot_", "")
            vol_path = os.path.join(vol_dir, "HRCT_pilot", pilot_num)
            mask_path = os.path.join(mask_dir, "HRCT_pilot", pilot_num)
        elif "_" in session_id:
            # 8_CT-INSPIRIUM-8871 → 8/CT-INSPIRIUM-8871
            patient_id, session_suffix = session_id.split("_", 1)
            vol_path = os.path.join(vol_dir, patient_id, session_suffix)
            mask_path = os.path.join(mask_dir, patient_id, session_suffix)
        else:
            # 101 → 101
            vol_path = os.path.join(vol_dir, session_id)
            mask_path = os.path.join(mask_dir, session_id)
        
        n_slices = extract_session_slices(
            session_id, vol_path, mask_path, output_root, dicom_root
        )
        total_slices += n_slices
    
    logger.info(f"✓ Extracted {total_slices} total aligned slices")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Extract 2D slices from ILD DICOM files"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest.csv",
    )
    parser.add_argument(
        "--dicom-root",
        type=str,
        required=True,
        help="Root directory containing ILD_DB_volumeROIs and ILD_DB_lungMasks",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output directory for extracted slices",
    )
    
    args = parser.parse_args()
    
    extract_all_slices(
        manifest_path=args.manifest,
        dicom_root=args.dicom_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()