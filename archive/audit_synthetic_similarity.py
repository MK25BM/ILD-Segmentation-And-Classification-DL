# audit_real_vs_synthetic_full.py
import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
REAL_DATA_ROOT = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
SYNTH_DATA_DIR = os.path.join(BASE_DIR, "synthetic_augmentations", "controlled_generation")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=================================================================")
print("🔬 INITIALIZING REAL-VS-SYNTHETIC FULL FIDELITY BENCHMARKER")
print("=================================================================")

DISEASE_MAPPING = {
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
    7: "other_rare_pathologies"
}

# ─── 📂 1. COLLECT AND PREPROCESS REAL PATIENT REFERENCE POOLS ───
print("📡 Scanning scratch disk to assemble true disease reference pools...")
real_pools = {name: [] for name in DISEASE_MAPPING.values()}

patient_folders = sorted(os.listdir(REAL_DATA_ROOT))
for p_folder in patient_folders:
    p_path = os.path.join(REAL_DATA_ROOT, p_folder)
    if not os.path.isdir(p_path): continue
    
    roi_paths = sorted(glob.glob(os.path.join(p_path, "roi_masks", "*.npy")))
    for roi_f in roi_paths:
        slice_name = os.path.basename(roi_f)
        img_f = os.path.join(p_path, "images", slice_name)
        if not os.path.exists(img_f): continue
        
        roi_mask = np.load(roi_f)
        roi_mask[roi_mask > 6] = 7  
        
        present_classes = np.unique(roi_mask)
        for class_idx, disease_name in DISEASE_MAPPING.items():
            if class_idx in present_classes:
                real_img = np.load(img_f).astype(np.float64)
                real_norm = (real_img - real_img.min()) / (real_img.max() - real_img.min() + 1e-8)
                real_resized = cv2.resize(real_norm, (256, 256), interpolation=cv2.INTER_LINEAR)
                real_pools[disease_name].append(real_resized)

# ─── 🧮 2. EXECUTE STRUCTURAL MATCHING MATRIX PASS ───
comparison_records = []
print("\n📡 Computing cross-distribution metrics (Real vs. Generated Slices)...")
print("─" * 110)

for class_idx, disease_name in DISEASE_MAPPING.items():
    synth_path = os.path.join(SYNTH_DATA_DIR, f"guided_synthesis_{disease_name}.png")
    if not os.path.exists(synth_path): continue
        
    synth_raw = cv2.imread(synth_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    # FIX: Force the synthetic image to 256x256 to completely neutralize plotting axis expansions!
    synth_img = cv2.resize(synth_raw, (256, 256), interpolation=cv2.INTER_AREA)
    
    real_slices = real_pools[disease_name]
    if len(real_slices) == 0: continue
        
    ssim_scores, psnr_scores, mse_scores, ncc_scores = [], [], [], []
    
    for real_slice in real_slices:
        mse = np.mean((real_slice - synth_img) ** 2)
        mse_scores.append(mse)
        
        ssim_score = ssim(real_slice, synth_img, data_range=1.0)
        ssim_scores.append(ssim_score)
        
        if mse > 0:
            psnr_score = psnr(real_slice, synth_img, data_range=1.0)
            psnr_scores.append(psnr_score)
            
        mean_r, mean_s = np.mean(real_slice), np.mean(synth_img)
        diff_r, diff_s = real_slice - mean_r, synth_img - mean_s
        denom = np.sqrt(np.sum(diff_r ** 2) * np.sum(diff_s ** 2))
        if denom > 0:
            ncc = np.sum(diff_r * diff_s) / denom
            ncc_scores.append(ncc)
            
    mean_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    mean_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    mean_mse = np.mean(mse_scores) if mse_scores else 0.0
    mean_ncc = np.mean(ncc_scores) if ncc_scores else 0.0
    
    comparison_records.append({
        "Pathology_Class": disease_name,
        "Real_Slices_Count": len(real_slices),
        "Mean_SSIM": mean_ssim,
        "Mean_PSNR_dB": mean_psnr,
        "Mean_MSE": mean_mse,
        "Mean_NCC": mean_ncc
    })
    
    print(f"   • {disease_name:<24} | Pool: {len(real_slices):3d} | SSIM: {mean_ssim:.4f} | PSNR: {mean_psnr:.2f}dB | MSE: {mean_mse:.5f} | NCC: {mean_ncc:.4f}")

df_report = pd.DataFrame(comparison_records)
df_report.to_csv(os.path.join(OUTPUT_DIR, "real_vs_synthetic_full_metrics.csv"), index=False)
print("─" * 110)
print("=================================================================")
