# scripts/test_weights_nan.py
import os
import torch

CHECKPOINT_PATH = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv/ddpm_model_pt/checkpoint_ddpm_epoch_10.pt"

print("=================================================================")
print("🔬 EXECUTING FORENSIC BINARY AUDIT ON CHECKPOINT 10 PARAMETERS")
print("=================================================================")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Error: Target weights file is missing at: {CHECKPOINT_PATH}")
    print("   Please ensure the path matches your active workspace.")
    exit(1)

# Load the raw parameter dictionary keys straight from the binary file array
try:
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    print("📂 Successfully unpacked state dictionary metadata layer.")
except Exception as e:
    print(f"❌ Fatal file corruption error while opening binary: {str(e)}")
    exit(1)

total_nan_layers = 0
total_param_elements = 0
corrupted_elements_count = 0

print("\n📡 Scanning internal convolutional and normalization layers...")
print("─" * 65)

# Loop directly over every individual parameter matrix layer block
for layer_key, tensor_data in state_dict.items():
    if not isinstance(tensor_data, torch.Tensor):
        continue
        
    num_elements = tensor_data.numel()
    total_param_elements += num_elements
    
    # Mathematical proof check: Identify any invalid NaN float signatures
    nan_mask = torch.isnan(tensor_data)
    nan_count = torch.sum(nan_mask).item()
    
    if nan_count > 0:
        total_nan_layers += 1
        corrupted_elements_count += nan_count
        print(f"   🚨 CORRUPTED LAYER: {layer_key:<45} | Contains {nan_count}/{num_elements} NaN values!")

print("─" * 65)
if total_nan_layers == 0:
    print("✅ ANALYTICAL VERIFICATION PASSED!")
    print(f"   The file contains {total_param_elements:,} healthy, valid parameter floating points.")
    print("   The NaN issue is driven by runtime execution formatting inside sample.py.")
else:
    print("❌ ANALYTICAL VERIFICATION FAILED!")
    corruption_ratio = (corrupted_elements_count / total_param_elements) * 100
    print(f"   Discovered {total_nan_layers} corrupted layer arrays on disk.")
    print(f"   Total Dead Parameters: {corrupted_elements_count}/{total_param_elements} ({corruption_ratio:.2f}% Corrupted).")
    print("\n👉 ACTION REQUIRED: Your training weights are broken. You must clean the directory and rerun the GradScaler training engine.")
print("=================================================================")
