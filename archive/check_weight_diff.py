import torch
import os

OUTPUT_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"

w_r2 = os.path.join(OUTPUT_DIR, "r2_unet", "best_weights_r2_unet.pt")
w_attn = os.path.join(OUTPUT_DIR, "attention_unet", "best_weights_attention_unet.pt")

if not os.path.exists(w_r2) or not os.path.exists(w_attn):
    print("❌ Error: One or both checkpoint files are missing from disk.")
    sys.exit(1)

# Load the raw parameter dictionaries directly from disk
state_r2 = torch.load(w_r2, map_location="cpu")
state_attn = torch.load(w_attn, map_location="cpu")

print("=================================================================")
print("🔍 STRUCTURAL PARAMETER WEIGHT CHECK INSPECTION SUITE")
print("=================================================================")
print(f"📦 R2 U-Net State Dict Keys   : {len(state_r2.keys())} parameter layers")
print(f"📦 Attention U-Net State Keys : {len(state_attn.keys())} parameter layers")

# Compare shared layer parameters
shared_keys = set(state_r2.keys()).intersection(set(state_attn.keys()))
print(f"📊 Total Intersecting Overlapping Layer Keys: {len(shared_keys)}")

# Calculate the mean absolute parameter weight delta across a primary convolution layer
sample_key = "model.inc.conv.0.weight" if "model.inc.conv.0.weight" in state_r2 else list(state_r2.keys())[0]

print(f"\n🔬 Sampling Weights Matrix Layer: '{sample_key}'")
weights_1 = state_r2[sample_key].float()
weights_2 = state_attn[sample_key].float()

if weights_1.shape == weights_2.shape:
    mean_abs_diff = torch.mean(torch.abs(weights_1 - weights_2)).item()
    print(f"   • Mean Absolute Weight Difference Delta: {mean_abs_diff:.6f}")
    if mean_abs_diff == 0:
        print("   ⚠️ WARNING: The numeric values inside these weights layers are IDENTICAL.")
    else:
        print("   ✅ SUCCESS: The layers contain completely distinct numeric parameter variables!")
else:
    print("   ✅ SUCCESS: The tensor architectures feature entirely different shape layers!")
print("=================================================================")
