# utils/helpers.py
import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BENCHMARKS_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
SCRATCH_DIR = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
OUTPUT_ROOT = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
CLASS_MAPPING = {0: "background", 1: "healthy_control", 2: "emphysema", 3: "ground_glass", 4: "fibrosis", 5: "micronodules", 6: "consolidation", 7: "other_rare_pathologies"}

# ─── ⚖️ 1. UNIFIED THE SINGLE SOURCE OF TRUTH: STRATIFIED SPLITS ENGINE ───
def get_leakage_proof_splits(manifest_path=None, n_splits=5, seed=42):
    """Isolates a 20% holdout test set and stratifies the remaining patients by class."""
    np.random.seed(seed)
    all_patients = sorted([d for d in os.listdir(SCRATCH_DIR) if os.path.isdir(os.path.join(SCRATCH_DIR, d))])
    
    # --- 🔒 STEP 1: PERMANENT HOLDOUT TEST SET ISOLATION ---
    shuffled_patients = all_patients.copy()
    np.random.shuffle(shuffled_patients)
    
    num_test = int(len(shuffled_patients) * 0.20)  # Carves out exactly 21 patients
    holdout_test_patients = sorted(shuffled_patients[:num_test])
    cross_val_patients = sorted(shuffled_patients[num_test:])    # 87 patients remain
    
    test_manifest_path = os.path.join(os.path.dirname(OUTPUT_ROOT), "permanent_holdout_test_set.csv")
    pd.DataFrame({"Patient_ID": holdout_test_patients}).to_csv(test_manifest_path, index=False)
    
    print("=================================================================")
    print(f"🔒 HOLDOUT TEST SET SECURED: {len(holdout_test_patients)} patients permanently banned from training.")
    print(f"   Manifest saved to: {test_manifest_path}")
    print("=================================================================")

    # --- ⚖️ STEP 2: MULTI-LABEL STRATIFICATION ON REMAINING PATIENTS ---
    cv_matrix = np.zeros((len(cross_val_patients), 6), dtype=int)
    
    for idx, p_id in enumerate(cross_val_patients):
        roi_paths = glob.glob(os.path.join(SCRATCH_DIR, p_id, "roi_masks", "*.npy"))
        for r_path in roi_paths:
            mask = np.load(r_path)
            mask[mask > 6] = 7  # 8-class consolidation
            for c_idx in range(2, 8):
                if np.any(mask == c_idx):
                    cv_matrix[idx, c_idx - 2] = 1

    folds = [[] for _ in range(n_splits)]
    fold_class_counts = np.zeros((n_splits, 6))
    
    patient_indices = np.argsort(np.sum(cv_matrix, axis=1))[::-1]
    
    for p_idx in patient_indices:
        p_id = cross_val_patients[p_idx]
        p_labels = cv_matrix[p_idx]
        
        scores = []
        for f in range(n_splits):
            temp_counts = fold_class_counts[f] + p_labels
            scores.append(np.std(temp_counts))
            
        best_fold = np.argmin(scores)
        folds[best_fold].append(p_id)
        fold_class_counts[best_fold] += p_labels

    print(f"\n⚖️ Cross-Validation Fold Stratification Profiles ({len(cross_val_patients)} Active Patients):")
    for f in range(n_splits):
        print(f"  • Fold {f+1}: Total Patients = {len(folds[f]):<2} | Unique Diseases Covered = {int(np.sum(fold_class_counts[f] > 0))}/6")
    print("=================================================================\n")
        
    splits = []
    for f in range(n_splits):
        val_pts = folds[f]
        train_pts = []
        for sub_f in range(n_splits):
            if sub_f != f:
                train_pts.extend(folds[sub_f])
        splits.append((train_pts, val_pts))
        
    return splits

def get_fixed_stratified_splits(scratch_root, seed=42):
    """
    Splits 108 patients into Train (70%), Val (15%), and Test (15%).
    Dynamically finds and anchors rare pathology classes (> 6) to Train.
    """
    patient_list = sorted([p for p in os.listdir(scratch_root) if os.path.isdir(os.path.join(scratch_root, p))])
    
    # 🧬 YOUR EXACT VERIFIED RARE PATHOLOGY PATIENT FOLDERS
    rare_pts = [
        'patient_105', 'patient_107', 'patient_108', 'patient_112', 'patient_118', 
        'patient_119', 'patient_12', 'patient_123', 'patient_124', 'patient_126', 
        'patient_128', 'patient_129', 'patient_130', 'patient_137', 
        'patient_142_CT-INSPIRIUM-2951', 'patient_150', 'patient_152', 'patient_155', 
        'patient_157', 'patient_165', 'patient_181', 'patient_21', 'patient_34', 
        'patient_38', 'patient_41', 'patient_46', 'patient_57_CT-INSPIRIUM-3550', 
        'patient_66', 'patient_70', 'patient_8_CT-INSPIRIUM-8871', 
        'patient_8_CT-INSPIRIUM-8873', 'patient_90', 'patient_pilot_204', 
        'patient_pilot_205', 'patient_pilot_207', 'patient_pilot_209'
    ]
    
    # Isolate common pathology patients
    common_pts = [p for p in patient_list if p not in rare_pts]
    
    train_pts, val_pts, test_pts = [], [], []
    rng = np.random.default_rng(seed)
    
    # Proportionally distribute your rare patients (70 / 15 / 15)
    rng.shuffle(rare_pts)
    n_rare = len(rare_pts)
    tr_r = int(0.70 * n_rare)
    vl_r = int(0.15 * n_rare)
    
    train_pts.extend(rare_pts[:tr_r])
    val_pts.extend(rare_pts[tr_r:tr_r+vl_r])
    test_pts.extend(rare_pts[tr_r+vl_r:])
    
    # Proportionally distribute your remaining common patients (70 / 15 / 15)
    rng.shuffle(common_pts)
    n_common = len(common_pts)
    tr_c = int(0.70 * n_common)
    vl_c = int(0.15 * n_common)
    
    train_pts.extend(common_pts[:tr_c])
    val_pts.extend(common_pts[tr_c:tr_c+vl_c])
    test_pts.extend(common_pts[tr_c+vl_c:])
    
    print(f"\n🧬 Stratification Context Successfully Established:")
    print(f"   • Train Cohort: {len(train_pts)} Patients")
    print(f"   • Val Cohort:   {len(val_pts)} Patients")
    print(f"   • Test Cohort:  {len(test_pts)} Patients")
    
    return sorted(train_pts), sorted(val_pts), sorted(test_pts)

# ─── 📊 2. THE AD-HOC MASTER PERFORMANCE CURVE PLOTTER ───
def plot_learning_curves(model_name, fold_num):
    history_file = os.path.join(BENCHMARKS_DIR, model_name, f"history_fold_{fold_num}.csv")
    if not os.path.exists(history_file): return
    df_history = pd.read_csv(history_file)
    vis_dir = os.path.join(BENCHMARKS_DIR, model_name, f"fold_{fold_num}_visuals")
    os.makedirs(vis_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor="white")
    epochs = df_history["Epoch"].values
    
    color = "crimson"
    ax1.set_xlabel("Training Epochs Count", fontweight="bold", labelpad=8)
    ax1.set_ylabel("Loss Scale (Cross-Entropy + Dice)", color=color, fontweight="bold", labelpad=8)
    ax1.plot(epochs, df_history["Train_Loss"].values, color=color, linestyle="-", linewidth=2.5, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2 = ax1.twinx()
    color = "darkblue"
    ax2.set_ylabel("Global Validation Mean Dice Coefficient", color=color, fontweight="bold", labelpad=8)
    ax2.plot(epochs, df_history["Val_Mean_Dice"].values, color=color, linestyle="--", linewidth=2.5, label="Val Dice")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0.0, 1.0)
    
    plt.title(f"MSc Thesis Performance Curve Profile — {model_name.upper()} (Fold {fold_num})", fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(os.path.join(vis_dir, "fold_learning_curve.png"), dpi=150)
    plt.close()

# ─── 🎨 3. THE FULL-FIDELITY AD-HOC PREDICTION OVERLAY ENGINE ───
def save_visual_prediction(epoch, fold, inputs, labels, preds, model_name):
    vis_dir = os.path.join(BENCHMARKS_DIR, model_name, f"fold_{fold}_visuals")
    os.makedirs(vis_dir, exist_ok=True)
    
    inputs_np = inputs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    preds_np = torch.argmax(preds, dim=1).detach().cpu().numpy() if preds.ndim == 4 else preds.detach().cpu().numpy()
    
    best_batch_idx = 0
    max_disease_pixels = -1
    for b in range(labels_np.shape[0]):
        disease_pixel_count = np.sum(labels_np[b] > 1)
        if disease_pixel_count > max_disease_pixels:
            max_disease_pixels = disease_pixel_count
            best_batch_idx = b
            
    raw_ct = inputs_np[best_batch_idx, 0]
    true_mask = labels_np[best_batch_idx]
    pred_mask = preds_np[best_batch_idx]
    
    unique_classes, counts = np.unique(true_mask, return_counts=True)
    support_strings = []
    for cls_id, count in zip(unique_classes, counts):
        if cls_id in CLASS_MAPPING and cls_id > 1:
            support_strings.append(f"{CLASS_MAPPING[cls_id].upper()}: {count}px")
    support_text = " | ".join(support_strings) if support_strings else "No Active Disease Pathologies"
    
    COLOR_MAP_RGB = {0: (0, 0, 0), 1: (40, 40, 40), 2: (255, 128, 0), 3: (255, 255, 0), 4: (255, 0, 128), 5: (0, 191, 255), 6: (128, 0, 128), 7: (40, 40, 40)}
    gt_rgb = np.zeros((512, 512, 3), dtype=np.uint8) + 40
    pred_rgb = np.zeros((512, 512, 3), dtype=np.uint8) + 40
    
    gt_rgb[true_mask == 0] = (0, 0, 0)
    pred_rgb[pred_mask == 0] = (0, 0, 0)
    
    for cls_idx in [1, 2, 3, 4, 5, 6, 7]:
        gt_rgb[true_mask == cls_idx] = COLOR_MAP_RGB.get(cls_idx, (40, 40, 40))
        pred_rgb[pred_mask == cls_idx] = COLOR_MAP_RGB.get(cls_idx, (40, 40, 40))
        
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    axes[0].imshow(raw_ct, cmap="gray")
    axes[0].set_title("1. Preprocessed Input CT Scan", fontsize=12, fontweight="bold", pad=12)
    axes[0].axis("off")
    
    axes[1].imshow(gt_rgb)
    axes[1].set_title("2. Ground Truth Mask (Black Background)", fontsize=12, fontweight="bold", pad=12)
    axes[1].axis("off")
    
    axes[2].imshow(raw_ct, cmap="gray")
    axes[2].imshow(pred_rgb, alpha=0.45 if np.max(pred_rgb) > 0 else 0.0)
    axes[2].set_title("3. Network Prediction Map Overlay", fontsize=12, fontweight="bold", pad=12)
    axes[2].axis("off")
    
    plt.suptitle(f"Qualitative Evaluation (Epoch {epoch}) — Slice Support Profile: [{support_text}]", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=3.0)
    plt.savefig(os.path.join(vis_dir, f"epoch_{epoch}.png"), dpi=200)
    plt.close()
