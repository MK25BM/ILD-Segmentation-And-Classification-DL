import os
import pandas as pd
import matplotlib.pyplot as plt

# 📂 Define directory roadmaps matching your output paths
BENCHMARKS_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
OUTPUT_PLOT_PATH = os.path.join(BENCHMARKS_DIR, "ild_training_curves.png")

# 🔍 Paths to the CSV files created by your two runs
std_unet_csv = os.path.join(BENCHMARKS_DIR, "standard_unet", "history_fixed_split.csv")
att_unet_csv = os.path.join(BENCHMARKS_DIR, "attention_unet", "history_fixed_split.csv") # update folder name if different

plt.figure(figsize=(14, 5))

# 📉 LEFT PANEL: TRAINING LOSS SUBPLOT
plt.subplot(1, 2, 1)
if os.path.exists(std_unet_csv):
    df_std = pd.read_csv(std_unet_csv)
    plt.plot(df_std["Epoch"], df_std["Train_Loss"], 'b-', label="Standard U-Net", linewidth=2)
if os.path.exists(att_unet_csv):
    df_att = pd.read_csv(att_unet_csv)
    plt.plot(df_att["Epoch"], df_att["Train_Loss"], 'r-', label="Attention U-Net", linewidth=2)
plt.title("MSc Dissertation: Training Loss Convergence", fontsize=12, fontweight='bold')
plt.xlabel("Epochs", fontsize=10)
plt.ylabel("Loss Matrix Scale", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# 📈 RIGHT PANEL: VALIDATION FOREGROUND DICE SUBPLOT
plt.subplot(1, 2, 2)
if os.path.exists(std_unet_csv):
    plt.plot(df_std["Epoch"], df_std["Val_Mean_Dice"], 'b--', label="Standard U-Net", linewidth=2)
if os.path.exists(att_unet_csv):
    plt.plot(df_att["Epoch"], df_att["Val_Mean_Dice"], 'r--', label="Attention U-Net", linewidth=2)
plt.title("MSc Dissertation: Foreground Mean Dice", fontsize=12, fontweight='bold')
plt.xlabel("Epochs", fontsize=10)
plt.ylabel("Validation Dice Score", fontsize=10)
plt.ylim(0, 0.5)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT_PLOT_PATH, dpi=300)
print(f"📊 Training curves rendered and saved successfully to:\n   {OUTPUT_PLOT_PATH}")
