# sample.py
import os
import sys
import glob
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Bind project repository paths natively
BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "lung_ddpm_plus_src"))

from diffusion_model.unet import UNetModel

# Target your healthy 10-channel checkpoint file explicitly
CHECKPOINT_PATH = os.path.join(BASE_DIR, "benchmarks_cv", "ddpm_model_pt", "checkpoint_ddpm_epoch_100.pt")
PROCESSED_ROOT = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
OUTPUT_SYNTH_DIR = os.path.join(BASE_DIR, "synthetic_augmentations", "controlled_generation")
os.makedirs(OUTPUT_SYNTH_DIR, exist_ok=True)

class SimpleDDPMScheduler:
    def __init__(self, timesteps=1000, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0).to(device)

    def step_backward(self, model_output, t, sample):
        beta_t = self.beta[t]
        alpha_t = self.alpha[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        mean = (1.0 / torch.sqrt(alpha_t)) * (sample - (beta_t / torch.sqrt(1.0 - alpha_cumprod_t)) * model_output)
        if t > 0:
            noise = torch.randn_like(sample)
            variance = torch.sqrt(beta_t)
            return mean + variance * noise
        return mean

print("=================================================================")
print("🚀 LUNG-DDPM+ CONTROLLED CLASS-CONDITIONAL GENERATION ENGINE")
print("=================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the architecture matching your 10-channel configuration exactly
model = UNetModel(
    image_size=256, 
    in_channels=10,       # 2 baseline structural channels + 8 one-hot pathology tracks
    model_channels=32, 
    out_channels=1,
    num_res_blocks=2,
    attention_resolutions=(16, 8), 
    channel_mult=(1, 2, 4, 8),
    dims=3
).to(device)

if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print(f"   ✅ Successfully loaded trained parameters from: {CHECKPOINT_PATH}")
else:
    raise FileNotFoundError(f"❌ Error: Checkpoint weights file missing at {CHECKPOINT_PATH}")

model.eval()
scheduler = SimpleDDPMScheduler(timesteps=1000, device=device)

# Load a real lung silhouette from scratch to act as our anatomical spatial guide
patient_folders = sorted(os.listdir(PROCESSED_ROOT))
first_patient = [p for p in patient_folders if os.path.isdir(os.path.join(PROCESSED_ROOT, p))][0]
mask_files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, first_patient, "lung_masks", "*.npy")))
sample_lung_mask = np.load(mask_files[len(mask_files)//2])

lung_condition = torch.from_numpy(sample_lung_mask).unsqueeze(0).unsqueeze(0).float()
lung_condition_256 = F.interpolate(lung_condition, size=(256, 256), mode="nearest").to(device)

# Define our active foreground mapping tracks
DISEASE_TRACKS = {
    "emphysema": 2,
    "ground_glass": 3,
    "fibrosis": 4,
    "micronodules": 5,
    "consolidation": 6
}

# ─── 🚀 THE CONTROLLED GENERATION LOOP ───
for disease_name, channel_idx in DISEASE_TRACKS.items():
    print(f"\n🎨 Forcing synthesis for pathology class: {disease_name.upper()}...")
    
    # Construct a clean 9-channel layout semantic guide matrix canvas
    # Channel 0: Lung binary envelope mask
    # Channels 1-8: One-hot disease selection switches
    semantic_layout_guide = torch.zeros((1, 9, 256, 256), device=device)
    semantic_layout_guide[0, 0, :, :] = lung_condition_256.squeeze() # Inject spatial outer wall boundary
    
    # THE GUIDING KEY OVERRIDE: Set the target disease layer channel to exactly 1.0!
    # This instructs the network's layers to generate this specific micro-texture inside the lung space.
    semantic_layout_guide[0, channel_idx, :, :] = lung_condition_256.squeeze()
    
    # Initialize a fresh starting noise tensor canvas (Stable FP32 Gaussian Distribution)
    torch.manual_seed(42)
    generated_sample = torch.randn((1, 1, 256, 256), device=device)
    
    with torch.no_grad():
        for t_idx in reversed(range(1000)):
            unet_input = torch.cat([generated_sample, semantic_layout_guide], dim=1)
            unet_input_3d = unet_input.unsqueeze(2).repeat(1, 1, 8, 1, 1) # Symmetrical downsampling depth
            t_tensor = torch.tensor([t_idx], device=device).long()
            
            # Execute the forward pass directly without autocast interference
            noise_pred_3d = model(unet_input_3d, timesteps=t_tensor)
            noise_pred = noise_pred_3d[:, :, 4, :, :]
                
            generated_sample = scheduler.step_backward(noise_pred, t_idx, generated_sample)
            
    # Export the completed synthetic slice straight to disk storage
    synth_np = generated_sample.squeeze().cpu().numpy()
    
    plt.figure(figsize=(6, 6), facecolor="black")
    plt.imshow(synth_np, cmap="gray")
    plt.title(f"Synthesized Pathology: {disease_name.upper()}", color="white", fontsize=12, pad=10)
    plt.axis("off")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_SYNTH_DIR, f"guided_synthesis_{disease_name}.png")
    plt.savefig(output_path, dpi=150, facecolor="black", edgecolor='none')
    plt.close()
    print(f"   🎉 Successful! Pristine augmented matrix sheet saved: guided_synthesis_{disease_name}.png")

print("\n🏆 CLASS-CONDITIONAL COHORT GENERATION PASS COMPLETE.")
print("=================================================================")
