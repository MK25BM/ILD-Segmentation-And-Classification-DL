import os
import glob
import re
import pandas as pd
import numpy as np
import pydicom

DATA_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/ILD_DB"
VOL_DIR = os.path.join(DATA_DIR, "ILD_DB_volumeROIs")
MASK_DIR = os.path.join(DATA_DIR, "ILD_DB_lungMasks")

CLASS_MAPPING = {
    1: "healthy", 2: "emphysema", 3: "ground_glass", 4: "fibrosis", 
    5: "micronodules", 6: "consolidation", 7: "bronchial_wall_thickening", 
    8: "reticulation", 9: "macronodules", 10: "cysts", 11: "peripheral_micronodules", 
    12: "bronchiectasis", 13: "air_trapping", 14: "early_fibrosis", 
    15: "increased_attenuation", 16: "tuberculosis", 17: "pcp"
}

def get_clean_dicoms(directory):
    if not os.path.exists(directory):
        return []
    files = glob.glob(os.path.join(directory, "*.dcm"))
    return sorted([f for f in files if not os.path.basename(f).startswith(".")])

def extract_trailing_suffix(filename):
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

print("=== Running Silent Suffix-Aware Manifest Generator (Pure Pydicom) ===")

manifest_data = []
exception_audit = []
base_patients = sorted([d for d in os.listdir(VOL_DIR) if os.path.isdir(os.path.join(VOL_DIR, d))])

for p_id in base_patients:
    p_vol_root = os.path.join(VOL_DIR, p_id)
    p_mask_root = os.path.join(MASK_DIR, p_id)
    
    if p_id == "HRCT_pilot":
        sessions = sorted([s for s in os.listdir(p_vol_root) if os.path.isdir(os.path.join(p_vol_root, s))])
        session_pairs = [(f"pilot_{s}", os.path.join(p_vol_root, s), os.path.join(p_mask_root, s)) for s in sessions]
    else:
        subdirs = sorted([d for d in os.listdir(p_vol_root) if os.path.isdir(os.path.join(p_vol_root, d)) and d != "roi_mask"])
        if subdirs:
            session_pairs = [(f"{p_id}_{s}", os.path.join(p_vol_root, s), os.path.join(p_mask_root, s)) for s in subdirs]
        else:
            session_pairs = [(p_id, p_vol_root, p_mask_root)]

    for unique_id, vol_path, mask_path in session_pairs:
        ct_raw_list = get_clean_dicoms(vol_path)
        roi_raw_list = get_clean_dicoms(os.path.join(vol_path, "roi_mask"))
        lung_raw_list = get_clean_dicoms(mask_path)
        if not lung_raw_list:
            lung_raw_list = get_clean_dicoms(os.path.join(mask_path, "lung_mask"))

        # Base evaluation counts
        ct_len, roi_len, lung_len = len(ct_raw_list), len(roi_raw_list), len(lung_raw_list)

        if ct_len == 0 or roi_len == 0 or lung_len == 0:
            cohort = "Incomplete_Data_Exception"
            reason = f"Missing Track completely -> CT: {ct_len} | ROI: {roi_len} | Lung: {lung_len}"
            exception_audit.append((unique_id, reason))
        else:
            ct_map = build_suffix_map(ct_raw_list)
            roi_map = build_suffix_map(roi_raw_list)
            lung_map = build_suffix_map(lung_raw_list)
            common_suffixes = sorted(list(set(ct_map.keys()).intersection(roi_map.keys()).intersection(lung_map.keys())))

            if len(common_suffixes) > 0:
                cohort = "Core_Set"
            else:
                cohort = "Incomplete_Data_Exception"
                reason = f"Suffix Intersection Mismatch -> Suffix Maps built: CT:{len(ct_map)} ROI:{len(roi_map)} Lung:{len(lung_map)}"
                exception_audit.append((unique_id, reason))

        record = {
            "Unique_Session_ID": unique_id,
            "Cohort": cohort,
            "Total_CT_Slices": ct_len,
            "Total_Lung_Mask_Slices": lung_len,
            "Total_ROI_Slices": roi_len,
            "Aligned_Slices_Count": len(common_suffixes) if cohort == "Core_Set" else 0,
            "Total_Lung_Tissue_Pixels": 0
        }
        
        for cls_name in CLASS_MAPPING.values():
            record[cls_name] = 0

        if cohort == "Core_Set":
            try:
                for suffix in common_suffixes:
                    lung_dcm = pydicom.dcmread(lung_map[suffix])
                    record["Total_Lung_Tissue_Pixels"] += int(np.sum(lung_dcm.pixel_array > 0))
                    
                    roi_dcm = pydicom.dcmread(roi_map[suffix])
                    roi_arr = roi_dcm.pixel_array
                    unique_labels, counts = np.unique(roi_arr, return_counts=True)
                    for label, count in zip(unique_labels, counts):
                        if label in CLASS_MAPPING:
                            record[CLASS_MAPPING[label]] += int(count)
            except Exception:
                record["Cohort"] = "Parsing_Error_Exception"

        manifest_data.append(record)

df = pd.DataFrame(manifest_data)
output_path = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/dataset_manifest.csv"
df.to_csv(output_path, index=False)

print(f"\nManifest successfully updated with session tracking at: {output_path}")
print("\n--- Cohort Distribution Summary ---")
print(df["Cohort"].value_counts())

# --- LIVE PATHOLOGY PATIENT COUNTS BLOCK ---
core_df = df[df["Cohort"] == "Core_Set"]
pathology_cols = [c for c in core_df.columns if c not in ["Unique_Session_ID", "Cohort", "Total_CT_Slices", "Total_Lung_Mask_Slices", "Total_ROI_Slices", "Aligned_Slices_Count", "Total_Lung_Tissue_Pixels", "healthy"]]
presence = (core_df[pathology_cols] > 0).sum().sort_values(ascending=False)
print("\n=== Patient Counts per Pathology ===")
print(presence.to_string())

# --- EXCEPTION AUDIT DISPLAY ---
if exception_audit:
    print("\n🕵️ Exception Directory Audit Report:")
    for eid, r in exception_audit:
        print(f"  • Session {eid}: {r}")
