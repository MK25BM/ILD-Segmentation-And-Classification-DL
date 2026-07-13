import os
import glob
import re
import pandas as pd
import numpy as np
import pydicom
from PIL import Image

MANIFEST_PATH = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/dataset_manifest.csv"
DATA_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/ILD_DB"
OUTPUT_ROOT = os.path.join(os.environ.get("SCRATCHDIR"), "ild_dataset_processed")

def get_clean_dicoms(directory):
    files = glob.glob(os.path.join(directory, "*.dcm"))
    return sorted([f for f in files if not os.path.basename(f).startswith(".")])

def extract_trailing_suffix(filename):
    """Extracts trailing digits from filename strings via regex."""
    name_without_ext = os.path.splitext(filename)[0]
    match = re.search(r'(\d+)$', name_without_ext)
    return int(match.group(1)) if match else None

def build_suffix_map(file_list):
    suffix_map = {}
    for f in file_list:
        idx = extract_trailing_suffix(os.path.basename(f))
        if idx is not None:
            suffix_map[idx] = f
    return suffix_map

# Load manifest configurations
df = pd.read_csv(MANIFEST_PATH)
core_sessions = df[df["Cohort"] == "Core_Set"]["Unique_Session_ID"].tolist()

print("=== Running Pure Pydicom True-Suffix Data Extractor ===")

base_patients = sorted([d for d in os.listdir(os.path.join(DATA_DIR, "ILD_DB_volumeROIs")) if os.path.isdir(os.path.join(DATA_DIR, "ILD_DB_volumeROIs", d))])
session_map = {}
for p_id in base_patients:
    p_vol_root = os.path.join(DATA_DIR, "ILD_DB_volumeROIs", p_id)
    p_mask_root = os.path.join(DATA_DIR, "ILD_DB_lungMasks", p_id)
    if p_id == "HRCT_pilot":
        for s in os.listdir(p_vol_root):
            if os.path.isdir(os.path.join(p_vol_root, s)):
                session_map[f"pilot_{s}"] = (os.path.join(p_vol_root, s), os.path.join(p_mask_root, s))
    else:
        subdirs = [d for d in os.listdir(p_vol_root) if os.path.isdir(os.path.join(p_vol_root, d)) and d != "roi_mask"]
        if subdirs:
            for s in subdirs:
                session_map[f"{p_id}_{s}"] = (os.path.join(p_vol_root, s), os.path.join(p_mask_root, s))
        else:
            session_map[p_id] = (p_vol_root, p_mask_root)

for session_id in core_sessions:
    vol_path, mask_path = session_map[session_id]
    
    ct_raw_list = get_clean_dicoms(vol_path)
    roi_raw_list = get_clean_dicoms(os.path.join(vol_path, "roi_mask"))
    lung_raw_list = get_clean_dicoms(os.path.join(mask_path, "lung_mask"))
    if not lung_raw_list:
        # Fall back to loose parent files only if the subfolder doesn't exist
        lung_raw_list = get_clean_dicoms(mask_path)
        
    if not ct_raw_list or not roi_raw_list or not lung_raw_list:
        continue

    ct_map = build_suffix_map(ct_raw_list)
    roi_map = build_suffix_map(roi_raw_list)
    lung_map = build_suffix_map(lung_raw_list)

    common_suffixes = sorted(list(set(ct_map.keys()).intersection(roi_map.keys()).intersection(lung_map.keys())))
    if not common_suffixes:
        continue

    p_folder = os.path.join(OUTPUT_ROOT, f"patient_{session_id}")
    img_out = os.path.join(p_folder, "images")
    lung_out = os.path.join(p_folder, "lung_masks")
    roi_out = os.path.join(p_folder, "roi_masks")
    
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lung_out, exist_ok=True)
    os.makedirs(roi_out, exist_ok=True)

    slices_saved = 0
    for suffix in common_suffixes:
        ct_dcm = pydicom.dcmread(ct_map[suffix])
        roi_dcm = pydicom.dcmread(roi_map[suffix])
        lung_dcm = pydicom.dcmread(lung_map[suffix])

        ct_arr = ct_dcm.pixel_array.astype(float)
        roi_arr = roi_dcm.pixel_array.astype(np.int16)
        lung_arr = lung_dcm.pixel_array.astype(np.int16)

        lung_binary = (lung_arr > 0).astype(np.float32)
        if np.sum(lung_binary > 0) < 500:
            continue

        slope = float(getattr(ct_dcm, 'RescaleSlope', 1))
        intercept = float(getattr(ct_dcm, 'RescaleIntercept', 0))
        ct_hu = ct_arr * slope + intercept

        window_min, window_max = -1200.0, 200.0
        ct_clipped = np.clip(ct_hu, window_min, window_max)

        lung_pixels = ct_clipped[lung_binary > 0]
        if len(lung_pixels) > 0:
            ct_norm = (ct_clipped - np.mean(lung_pixels)) / (np.std(lung_pixels) + 1e-8)
        else:
            ct_norm = (ct_clipped - window_min) / (window_max - window_min)

        ct_png = ((ct_clipped - window_min) / (window_max - window_min) * 255.0).astype(np.uint8)
        lung_png = (lung_binary * 255).astype(np.uint8)
        roi_png = (roi_arr * 14).astype(np.uint8)

        # FIX: Force the file prefix name to match the true raw suffix number directly!
        slice_prefix = f"slice_{suffix}"
        
        np.save(os.path.join(img_out, f"{slice_prefix}.npy"), ct_norm.astype(np.float32))
        np.save(os.path.join(lung_out, f"{slice_prefix}.npy"), lung_binary)
        np.save(os.path.join(roi_out, f"{slice_prefix}.npy"), roi_arr)
        
        Image.fromarray(ct_png).save(os.path.join(img_out, f"{slice_prefix}.png"))
        Image.fromarray(lung_png).save(os.path.join(lung_out, f"{slice_prefix}.png"))
        Image.fromarray(roi_png).save(os.path.join(roi_out, f"{slice_prefix}.png"))
        
        slices_saved += 1

    print(f"  Processed patient_{session_id}: {slices_saved} slices exported.")

print("\n=== Pipeline Complete. Structured Database Regenerated ===")
