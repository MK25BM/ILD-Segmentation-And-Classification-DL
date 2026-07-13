# debug_sampling_tensors.py
import os
import sys
import torch

BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "lung_ddpm_plus_src"))

from diffusion_model.unet import UNetModel

CHECKPOINT_PATH = os.path.join(BASE_DIR, "benchmarks_cv", "ddpm_model_pt", "checkpoint_ddpm_epoch_10.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=================================================================")
print("🕵️‍♂️ RUNNING LIVE TEN_SOR FORENSIC DIAGNOSTIC SUITE")
print("=================================================================")

# 1. Instantiate the network architecture
model = UNetModel(
    image_size=256, in_channels=10, model_channels=32, out_channels=1,
    num_res_blocks=2, attention_resolutions=(16, 8), channel_mult=(1, 2, 4, 8), dims=3
).to(device)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("   ✅ Trained checkpoint weights successfully parsed.")
model.eval()

# 2. Build mock inputs to simulate the generation pass
semantic_layout_guide = torch.zeros((1, 9, 256, 256), device=device)
semantic_layout_guide[0, 0, :, :] = 1.0  # Set mock lung mask
semantic_layout_guide[0, 3, :, :] = 1.0  # Set mock target disease track (Ground Glass)

# Initialize standard random noise
torch.manual_seed(42)
generated_sample = torch.randn((1, 1, 256, 256), device=device)

print("\n🔍 TRACKING TENSOR VALUE RANGES ACROSS TIMESTEP MILESTONES:")
print("─" * 65)

# Simulate 3 quick backward steps to watch the values change
with torch.no_grad():
    for t_idx in [999, 750, 500, 250, 0]:
        unet_input = torch.cat([generated_sample, semantic_layout_guide], dim=1)
        unet_input_3d = unet_input.unsqueeze(2).repeat(1, 1, 8, 1, 1)
        t_tensor = torch.tensor([t_idx], device=device).long()
        
        with torch.amp.autocast(device_type="cuda", enabled=True):
            noise_pred_3d = model(unet_input_3d, timesteps=t_tensor)
            noise_pred = noise_pred_3d[:, :, 4, :, :]
            
        print(f"⏱️ Timestep {t_idx} | Input Canvas Min: {generated_sample.min().item():.3f} | Max: {generated_sample.max().item():.3f}")
        print(f"             | Model Pred   Min: {noise_pred.min().item():.3f} | Max: {noise_pred.max().item():.3f} | Mean: {noise_pred.mean().item():.5f}")
        
        # Check for zeroed-out model outputs
        if torch.allclose(noise_pred, torch.zeros_like(noise_pred), atol=1e-5):
            print("   🚨 CRITICAL WARNING: Model output has collapsed entirely to 0.000!")
            
        # Update your canvas array using standard subtraction steps
        generated_sample = generated_sample - 0.01 * noise_pred

print("─" * 65)
print("=================================================================")
