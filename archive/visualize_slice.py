import os
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_ROOT = os.path.expandvars("$SCRATCHDIR/ild_dataset_processed")

TARGET_PATIENT = "patient_168"
# FIX: Now pointing safely and explicitly to the real raw suffix file!
TARGET_SLICE = "slice_23.npy"

p_dir = os.path.join(PROCESSED_ROOT, TARGET_PATIENT)
img_path = os.path.join(p_dir, "images", TARGET_SLICE)
lung_path = os.path.join(p_dir, "lung_masks", TARGET_SLICE)
roi_path = os.path.join(p_dir, "roi_masks", TARGET_SLICE)

print(f"=== Loading True Raw Suffix-Matched Layers for {TARGET_PATIENT} ({TARGET_SLICE}) ===")

image_matrix = np.load(img_path)
lung_mask_matrix = np.load(lung_path)
roi_matrix = np.load(roi_path)

# --- Console Text Art Map ---
print(f"\nUnique numeric keys present in this slice matrix: {np.unique(roi_matrix)}")
print("\n--- Console Matrix Annotation Density Preview ---")
step = 512 // 40
for y in range(0, 512, step):
    row_chars = []
    for x in range(0, 512, step):
        val = roi_matrix[y, x]
        if val == 0:
            row_chars.append(".")
        elif val == 1:
            row_chars.append("H")
        else:
            row_chars.append(str(val))
    print("".join(row_chars))

# --- Build the 5-Panel Plot Layout ---
fig, axes = plt.subplots(1, 5, figsize=(25, 5))

# Pure Grayscale setting
axes[0].imshow(image_matrix, cmap="gray") #"bone"
axes[0].set_title("1. Preprocessed 2D CT\n(Input Channel 1)", fontsize=11, pad=10)
axes[0].axis("off")

axes[1].imshow(lung_mask_matrix, cmap="gray")
axes[1].set_title("2. Aligned Lung Mask\n(Input Channel 2)", fontsize=11, pad=10)
axes[1].axis("off")

axes[2].imshow(roi_matrix, cmap="tab20", vmin=0, vmax=17)
axes[2].set_title("3. Disease Patch ROI\n(Multi-Class Targets)", fontsize=11, pad=10)
axes[2].axis("off")

axes[3].imshow(image_matrix, cmap="gray") #"bone"
masked_roi = np.ma.masked_where(roi_matrix == 0, roi_matrix)
axes[3].imshow(masked_roi, cmap="autumn", alpha=0.4)
axes[3].set_title("4. Pathology Overlay\non Raw CT Scan", fontsize=11, pad=10)
axes[3].axis("off")

axes[4].imshow(lung_mask_matrix, cmap="gray")
masked_anomalies_only = np.ma.masked_where(roi_matrix <= 1, roi_matrix)
axes[4].imshow(masked_anomalies_only, cmap="autumn", alpha=0.6)
axes[4].set_title("5. Aligned Pathology\ninside Lung Frame", fontsize=11, pad=10)
axes[4].axis("off")

plt.tight_layout(pad=3.0)
output_img_path = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/five_panel_verification.png"
plt.savefig(output_img_path, dpi=200)

print(f"\n🎨 Clean 5-panel spatial map safely generated at: {output_img_path}")