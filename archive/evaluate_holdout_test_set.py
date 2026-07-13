import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Pull constants directly from your master configuration script
sys.path.append("/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL")
from config import SCRATCH_DIR, OUTPUT_DIR, CLASS_MAPPING
from dataset import RobustILDDataset
from run_leakage_proof_experiment import build_architecture

print("=================================================================")
print("🏆 MASTER HOLDOUT MULTI-MODEL REPORT SUITE (ISOLATED TRACKING) 🏆")
print("=================================================================")

test_manifest_path = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/permanent_holdout_test_set.csv"
if not os.path.exists(test_manifest_path):
    print(f"❌ Error: Missing locked test manifest at {test_manifest_path}")
    sys.exit(1)

test_patients = pd.read_csv(test_manifest_path)["Patient_ID"].tolist()
print(f"🔒 Isolated Holdout Test Set Loaded: {len(test_patients)} patients isolated.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MONAI discrete transformation layers to force perfect 8-channel one-hot formatting
post_pred = AsDiscrete(argmax=True, to_onehot=8)
post_true = AsDiscrete(to_onehot=8)

def compute_model_metrics(m_name):
    """Evaluates an architecture checkpoint using correct one-hot tensor layouts."""
    weight_path = os.path.join(OUTPUT_DIR, m_name, f"best_weights_{m_name}.pt")
    if not os.path.exists(weight_path):
        return None
        
    print(f"⚙️ Running holdout inference matrix for model: {m_name.upper()}...")
    model = build_architecture(m_name).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    # Isolated evaluator: 7 foreground channels (excludes background class 0)
    evaluator = DiceMetric(include_background=False, reduction="mean_batch")
    patient_volume_scores = []
    
    for p_id in test_patients:
        p_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=[p_id], skip_empty_masks=False)
        if len(p_ds) == 0: continue
        p_loader = DataLoader(p_ds, batch_size=4, shuffle=False)
        
        evaluator.reset()
        with torch.no_grad():
            for inputs, targets in p_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Convert un-activated continuous logits into normalized probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Add a singleton channel axis via unsqueeze(0) to satisfy MONAI's shape constraints
                discrete_preds = [post_pred(p) for p in probs]
                discrete_targets = [post_true(t.unsqueeze(0)) for t in targets]
                
                # Stack back into batch tensors safely
                y_pred_tensor = torch.stack(discrete_preds)
                y_true_tensor = torch.stack(discrete_targets)
                
                evaluator(y_pred=y_pred_tensor, y=y_true_tensor)
                
        class_dices = evaluator.aggregate().cpu().numpy()
        evaluator.reset()
        patient_volume_scores.append(class_dices)
        
    if len(patient_volume_scores) == 0: return None
    patient_volume_scores = np.array(patient_volume_scores) # Shape: [Num_Patients, 7_Classes]
    mean_class_dices = np.mean(patient_volume_scores, axis=0) # Average along the patient axis
    
    return {
        "with_healthy": float(np.mean(mean_class_dices)),
        "pure_disease": float(np.mean(mean_class_dices[1:])),
        "median_volume": float(np.median(np.mean(patient_volume_scores, axis=1))),
        "classes": mean_class_dices.tolist() # Convert array to clean list to lock scope bindings
    }

# Execute evaluations independently to extract unique metrics
results_dictionary = {}
for model_key in ["standard_unet", "r2_unet", "attention_unet"]:
    metrics_snapshot = compute_model_metrics(model_key)
    if metrics_snapshot is not None:
        results_dictionary[model_key] = metrics_snapshot

# ─── 🚀 PRINT DEFINITIVE DISSERTATION HOLDOUT REPORT MATRIX ───
print("\n" + "="*80)
print("🏆 DEFINITIVE DISSERTATION HOLDOUT MULTI-MODEL REPORT MATRIX")
print("="*80)

for m_key in ["standard_unet", "r2_unet", "attention_unet"]:
    if m_key not in results_dictionary:
        continue
        
    m_data = results_dictionary[m_key]
    scores_list = m_data["classes"]
    
    print(f"\n📈 Performance Profile: {m_key.upper()}")
    print(f"  • Mean Foreground Volume Dice (With Healthy): {m_data['with_healthy']:.4f}")
    print(f"  • 🚨 PURE DISEASE MEAN DICE (EXCLUDING HEALTHY): {m_data['pure_disease']:.4f}")
    print(f"  • Volume Median Foreground Dice Score        : {m_data['median_volume']:.4f}")
    
    print("   🔬 Itemized 8-Class Pathology Performance Spectrum Breakdown:")
    # Direct explicit list index calls ensure true unique channel printing
    print(f"     • HEALTHY                  | Holdout Test Dice: {scores_list[0]:.4f}")
    print(f"     • EMPHYSEMA                | Holdout Test Dice: {scores_list[1]:.4f}")
    print(f"     • GROUND_GLASS             | Holdout Test Dice: {scores_list[2]:.4f}")
    print(f"     • FIBROSIS                 | Holdout Test Dice: {scores_list[3]:.4f}")
    print(f"     • MICRONODULES             | Holdout Test Dice: {scores_list[4]:.4f}")
    print(f"     • CONSOLIDATION            | Holdout Test Dice: {scores_list[5]:.4f}")
    print(f"     • OTHER_RARE_PATHOLOGIES   | Holdout Test Dice: {scores_list[6]:.4f}")
    print("-"*70)
print("=================================================================")
