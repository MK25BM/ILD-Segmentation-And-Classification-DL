import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
MODELS = ["standard_unet", "attention_unet", "r2_unet", "attention_residual_unet"]
FG_CLASSES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules", "consolidation"]

MODEL_STYLES = {
    "standard_unet": {"color": "crimson", "linestyle": "-", "label": "Standard U-Net (Baseline)"},
    "attention_unet": {"color": "royalblue", "linestyle": "--", "label": "Attention U-Net"},
    "r2_unet": {"color": "forestgreen", "linestyle": ":", "label": "R2U-Net (Recurrent Residual)"},
    "attention_residual_unet": {"color": "darkorange", "linestyle": "-.", "label": "Attention Residual U-Net"}
}

print("=== 🎨 COMPILING 5-FOLD CROSS-VALIDATION FOREGROUND OVERLAY CURVES ===")
plt.figure(figsize=(12, 6.5), facecolor="white")

for m_name in MODELS:
    history_files = sorted(glob.glob(os.path.join(BASE_DIR, m_name, "epoch_history_fold_*.csv")))
    if not history_files:
        continue
        
    # Read the first file to establish your exact active epoch count framework matrix
    df_sample = pd.read_csv(history_files[0])
    num_epochs = len(df_sample)
    epochs = df_sample["Epoch"].values
    
    # Matrix shape: [Num Folds, Num Epochs]
    fold_matrix = np.zeros((len(history_files), num_epochs))
    
    for f_idx, f_path in enumerate(history_files):
        df = pd.read_csv(f_path)
        fg_cols = [f"Dice_{c}" for c in FG_CLASSES if f"Dice_{c}" in df.columns]
        # Store the computed Foreground Dice row average for this specific epoch pass
        fold_matrix[f_idx] = df[fg_cols].mean(axis=1).values
        
    # Calculate cross-validation statistical boundaries across all 5 folds
    mean_fg_curve = np.mean(fold_matrix, axis=0)
    std_fg_curve = np.std(fold_matrix, axis=0)
    
    style = MODEL_STYLES[m_name]
    
    # Plot the solid mean performance trajectory line
    plt.plot(epochs, mean_fg_curve, color=style["color"], linestyle=style["linestyle"],
             label=style["label"], linewidth=2.5)
    
    # Enforce standard deviation variance shading to show cross-validation consistency
    plt.fill_between(epochs, mean_fg_curve - std_fg_curve, mean_fg_curve + std_fg_curve,
                     color=style["color"], alpha=0.10)
    
    peak_val = mean_fg_curve.max()
    peak_ep = epochs[mean_fg_curve.argmax()]
    plt.scatter(peak_ep, peak_val, color=style["color"], edgecolors="black", s=60, zorder=5)
    print(f"  • {m_name.upper():<25} | Loaded {len(history_files)} Folds. Mean Peak FG Dice: {peak_val:.4f} at Epoch {peak_ep}")

plt.title("MSc Thesis Unified Performance: 5-Fold Cross-Validation Foreground (FG) Learning Paradigm", fontsize=12, fontweight="bold", pad=15)
plt.xlabel("Training Epoch Count Track", fontsize=11, labelpad=8)
plt.ylabel("Foreground Mean Dice Coefficient Spectrum ($\mu \pm \sigma$)", fontsize=11, labelpad=8)
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
plt.xlim(1, num_epochs)
plt.ylim(0.0, 0.75)
plt.tight_layout()

output_curve_img = os.path.join(BASE_DIR, "unified_5fold_thesis_learning_curves.png")
plt.savefig(output_curve_img, dpi=200)
plt.close()
print(f"\n🏆 5-Fold Master comparative curve figure successfully exported to: {output_curve_img}")