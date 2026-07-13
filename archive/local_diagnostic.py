import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force non-interactive headless backend execution
import matplotlib.pyplot as plt

# FIX A: Point directly to your active, running ATTENTION_UNET architecture directory
CSV_PATH = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv/standard_unet/epoch_history_fold_5.csv"

if not os.path.exists(CSV_PATH):
    print(f"❌ Error: Cannot find file at {CSV_PATH}. Fold 1 history file has not been written yet!")
    print("   💡 Quick check: Let's see what fold history files exist right now:")
    import glob
    parent_dir = os.path.dirname(CSV_PATH)
    print(f"   Files inside {parent_dir}: {glob.glob(os.path.join(parent_dir, '*.csv'))}")
    exit()

df = pd.read_csv(CSV_PATH)
print("=================================================================")
print("📊 LIVE ATTENTION U-NET HEALTH AND ANOMALY ENGINE 📊")
print("=================================================================")
print(f"Current Completed Epochs Logged: {len(df)}")

# Fix column naming handle variations dynamically
DICE_KEY = "Val_Global_Dice" if "Val_Global_Dice" in df.columns else "Val_Mean_Dice"

# 1. CHECK FOR GRADIENT PROFILE HEALTH
latest_loss = df["Train_Loss"].iloc[-1]
initial_loss = df["Train_Loss"].iloc[0]

print(f"\n📉 Loss Progression Tracker:")
print(f"   • Initial Epoch Loss: {initial_loss:.4f}")
print(f"   • Latest Epoch Loss:  {latest_loss:.4f}")

if np.isnan(latest_loss):
    print("   🚨 ANOMALY DETECTED: Loss has exploded to NaN! Learning rate too high.")
elif latest_loss > initial_loss * 1.5:
    print("   🚨 ANOMALY DETECTED: Loss is diverging/increasing! Check loss weights.")
else:
    print("   ✅ Success: Loss is trending downward stably.")

# 2. AUDIT CLASS INTERSTITIAL IMBALANCE SUPPORTS
print(f"\n🧬 Validation Pathology Support Snapshot (Last Recorded Epoch):")
support_cols = [c for c in df.columns if "Support_" in c]
last_row = df.iloc[-1]

has_imbalance_choke = False
for col in support_cols:
    class_name = col.replace("Support_", "").replace("_Pixels", "").upper()
    px_count = last_row[col]
    if px_count > 0:
        print(f"   • {class_name:<25}: {int(px_count):,} pixels evaluated.")
        dice_col = f"Dice_{class_name.lower()}"
        if dice_col in df.columns:
            class_dice = last_row[dice_col]
            if class_dice == 0.0 and px_count > 10000 and len(df) > 5:
                print(f"     ⚠️ ANOMALY: High pixel support but 0.0 Dice score! Model ignoring features.")
                has_imbalance_choke = True

if not has_imbalance_choke:
    print("   ✅ Success: Class mapping metrics are registering updates across features.")

# 3. GENERATE THE NEW 8-CLASS HEADLESS PROFILE VISUAL
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(df["Epoch"], df["Train_Loss"], color="crimson", marker="o", linewidth=2)
ax1.set_title("Training Loss Convergence Path")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss Scale")
ax1.grid(True, linestyle=":")

# Filter active disease tracking tags
ax2.set_title("Validation Dice Spectrum Profiles (8-Class Patch Grid)")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Dice Score")
ax2.set_ylim(0.0, 1.0)
ax2.grid(True, linestyle=":")

# 1. Plot the Global Mean line as a thick, authoritative black tracker
ax2.plot(df["Epoch"], df[DICE_KEY], color="black", linestyle="--", linewidth=2.5, label="Global Mean")

# 2. FIX: Dynamic addition of the Healthy class as a muted, light gray background line
if "Dice_healthy" in df.columns:
    ax2.plot(df["Epoch"], df["Dice_healthy"], color="lightgray", linestyle="-", linewidth=2.0, alpha=0.8, label="Healthy (Anchor)")

# 3. Layer all active foreground disease paths clearly on top
disease_cols = [c for c in df.columns if "Dice_" in c and c not in ["Dice_healthy", "Dice_background"]]
for col in disease_cols:
    # This automatically matches your exact folder palette sequence
    ax2.plot(df["Epoch"], df[col], label=col.replace("Dice_", "").capitalize(), alpha=0.85, linewidth=1.8)

ax2.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
plt.tight_layout()

# Save out to your workspace directory
output_image_path = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/local_diagnostic_curves.png"
plt.savefig(output_image_path, dpi=150)
plt.close()

print(f"\n📊 Convergence graphics successfully exported to your workspace directory:\n👉 {output_image_path}")

print("\n=================================================================")
print("📈 ADVANCED QUANTITATIVE THESIS METRICS 📊")
print("=================================================================")

recent_df = df.tail(10)
loss_variance = recent_df["Train_Loss"].std()
dice_variance = recent_df[DICE_KEY].std()

print(f"📊 Training Stability Metrics (Last 10 Epochs):")
print(f"   • Train Loss Standard Deviation: {loss_variance:.6f}")
print(f"   • Val Global Mean Dice Variance: {dice_variance:.6f}")

print(f"\n🏆 Individual Consolidated Pathology Performance Spectrum:")
high_support_classes = ["emphysema", "ground_glass", "fibrosis", "micronodules", "consolidation", "other_rare_pathologies"]

for cls in high_support_classes:
    dice_col = f"Dice_{cls}"
    support_col = f"Support_{cls}_Pixels"
    
    if dice_col in df.columns:
        latest_dice = df[dice_col].iloc[-1]
        pixel_count = df[support_col].iloc[-1]
        print(f"   • {cls.upper():<25} | Latest Dice: {latest_dice:.4f} | Voxel Support: {int(pixel_count):,} px")
