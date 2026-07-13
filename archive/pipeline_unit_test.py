import os
import sys
import torch
import numpy as np
import pandas as pd
import hashlib
from torch.utils.data import DataLoader
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

# Integrate explicit system repository path variables
sys.path.append("/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL")
from config import SCRATCH_DIR, OUTPUT_DIR, CLASS_MAPPING
from dataset import RobustILDDataset
from run_leakage_proof_experiment import build_architecture

print("=================================================================")
print("🛡️ DISPATCHING FULL AUDIT-GRADE PIPELINE FORENSIC SUITE")
print("=================================================================")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 🔍 PHASE 1: CRYPTOGRAPHIC CHECK-SUM HASHING OF WEIGHT FILES ───
print("\n[PHASE 1] Hashing Raw Checkpoint Weight Files on Disk...")
for m_name in ["standard_unet", "r2_unet", "attention_unet"]:
    weight_path = os.path.join(OUTPUT_DIR, m_name, f"best_weights_{m_name}.pt")
    if os.path.exists(weight_path):
        with open(weight_path, "rb") as f:
            file_bytes = f.read()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
        print(f"   • Model: {m_name.upper():<15} | File Path Size: {len(file_bytes)/1024/1024:.2f} MB | SHA-256: {file_hash[:24]}...")
    else:
        print(f"   • Model: {m_name.upper():<15} | ❌ MISSING FROM DISK STORAGE")

# ─── 🔍 PHASE 2: RIGOROUS MONAI DICEMETRIC BOUNDARY VERIFICATION ───
print("\n[PHASE 2] Verifying MONAI DiceMetric Channel Specificity...")
metric_validator = DiceMetric(include_background=False, reduction="mean_batch")

# Construct crisp 8-channel one-hot tensors: [Batch=1, Classes=8, H=4, W=4]
y_true = torch.zeros(1, 8, 4, 4)
y_true[0, 3, :, :] = 1  # Populate index position 3 exclusively

# Scenario A: Bit-Identical Arrays -> Only index position 3 (channel 2 in foreground) must be 1.0
metric_validator.reset()
metric_validator(y_pred=y_true, y=y_true)
result_identical = metric_validator.aggregate().cpu().numpy()
print(f"   • Same-Tensor Multi-Channel Output Vector  ➔ {result_identical}")

# Scenario B: Mismatched Arrays -> Should evaluate to 0.0 or handle NaN channels cleanly
y_mismatched = torch.zeros(1, 8, 4, 4)
y_mismatched[0, 5, :, :] = 1  # Populate a completely separate channel
metric_validator.reset()
metric_validator(y_pred=y_mismatched, y=y_true)
result_mismatch = metric_validator.aggregate().cpu().numpy()
print(f"   • Cross-Tensor Multi-Channel Output Vector ➔ {result_mismatch}")

# ─── 🔍 PHASE 3: MULTI-PATIENT FULL-VOLUME RAW LOGIT & ARGMAX HASHING ───
print("\n[PHASE 3] Compiling Full-Volume Raw Logit & Discrete Hashes...")
test_manifest_path = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/permanent_holdout_test_set.csv"
test_patients = pd.read_csv(test_manifest_path)["Patient_ID"].tolist()[:3] # Audit a sub-grid sample of 3 separate patients
print(f"   • Target Sample Focus Group: {test_patients}")

for m_name in ["standard_unet", "r2_unet", "attention_unet"]:
    weight_path = os.path.join(OUTPUT_DIR, m_name, f"best_weights_{m_name}.pt")
    if not os.path.exists(weight_path):
        continue
        
    print(f"\n⚙️ Initializing Full Inference Graph for: {m_name.upper()}...")
    model = build_architecture(m_name).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    for p_id in test_patients:
        p_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=[p_id], skip_empty_masks=False)
        if len(p_ds) == 0: continue
        p_loader = DataLoader(p_ds, batch_size=4, shuffle=False)
        
        # Accumulators to build entire multi-slice 3D volume grids for checking
        all_raw_logits = []
        all_argmax_maps = []
        
        with torch.no_grad():
            for inputs, _ in p_loader:
                inputs = inputs.to(device)
                outputs = model(inputs) # Raw network logits output array
                
                all_raw_logits.append(outputs.cpu().numpy())
                all_argmax_maps.append(torch.argmax(outputs, dim=1).cpu().numpy())
                
        # Consolidate into full continuous multi-dimensional matrix arrays
        full_volume_logits = np.concatenate(all_raw_logits, axis=0)
        full_volume_argmax = np.concatenate(all_argmax_maps, axis=0)
        
        # Generate clear cryptographic check-sums of the absolute raw continuous data values
        logit_hash = hashlib.sha256(full_volume_logits.tobytes()).hexdigest()
        argmax_hash = hashlib.sha256(full_volume_argmax.tobytes()).hexdigest()
        
        print(f"     ➔ Patient: {p_id:<20} | Shape: {full_volume_logits.shape} | Logit Hash: {logit_hash[:16]}... | Argmax Hash: {argmax_hash[:16]}...")
print("=================================================================")
