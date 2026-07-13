# generate_all_trajectories.py
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

# Bind project repository paths natively
BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "lung_ddpm_plus_src"))

from diffusion_model.unet import UNetModel

# Setup clear path roadmaps
CHECKPOINT_DIR = os.path.join(BASE_DIR, "benchmarks_cv", "ddpm_model_pt")
PROCESSED_ROOT = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
OUTPUT_TRAJ_DIR = os.path.join(BASE_DIR, "synthetic_augmentations", "trajectory_study_all_epochs")
os.makedirs(OUTPUT_TRAJ_DIR, exist_ok=True)

class SimpleDDPMScheduler:
    """Mathematical linear variance inverse denoising scheduler."""
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
print("🔬 DISPATCHING ALL-EPOCH TRAJECTORY GRID GENERATION CORE")
print("=================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Systematically locate all available checkpoints on your disk storage drive
ckpt_files = sorted([
    f for f in os.listdir(CHECKPOINT_DIR) 
    if f.startswith("checkpoint_ddpm_epoch_") and f.endswith(".pt")
], key=lambda x: int(x.split("_")[-1].split(".")[0]))

if not ckpt_files:
    print(f"❌ Error: No valid checkpoint files discovered inside: {CHECKPOINT_DIR}")
    sys.exit(1)

print(f"📦 Discovered {len(ckpt_files)} trained epoch models. Compiling grids for each model...")

# 2. Extract an uncorrupted real lung silhouette framework matrix to serve as our anchor condition
patient_folders = sorted(os.listdir(PROCESSED_ROOT))
first_patient = [p for p in patient_folders if os.path.isdir(os.path.join(PROCESSED_ROOT, p))][0]
mask_files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, first_patient, "lung_masks", "*.npy")))
sample_lung_mask = np.load(mask_files[len(mask_files)//2])

# Interpolate to 256x256 inside the active workspace memory channels
lung_condition = torch.from_numpy(sample_lung_mask).unsqueeze(0).unsqueeze(0).float()
lung_condition_256 = torch.nn.functional.interpolate(lung_condition, size=(256, 256), mode="nearest").to(device)

# Instantiate the exact 3D structural model matching your training parameters
model = UNetModel(
    image_size=256, in_channels=2, model_channels=32, out_channels=1,
    num_res_blocks=2, attention_resolutions=(16, 8), channel_mult=(1, 2, 4, 8), dims=3
).to(device)

scheduler = SimpleDDPMScheduler(timesteps=1000, device=device)

# Enforce a frozen baseline noise vector canvas seed context
torch.manual_seed(101)
frozen_initial_noise = torch.randn((1, 1, 256, 256), device=device)
target_milestones = [999, 750, 500, 250, 0]

# ─── 🚀 THE COMPREHENSIVE GENERATION LOOP MATRIX ───
for ckpt_name in ckpt_files:
    epoch_num = ckpt_name.split("_")[-1].split(".")[0]
    print(f"\n▶️ Generating 5-panel denoising trajectory for Model: {ckpt_name} (Epoch {epoch_num})...")
    
    # Reload model weights layers dynamically for the target checkpoint iteration
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, ckpt_name), map_location=device))
    model.eval()
    
    # Reset accumulator storage and clone the exact same noise vector
    trajectory_panels = {}
    generated_sample = frozen_initial_noise.clone()
    
    with torch.no_grad():
        for t_idx in reversed(range(1000)):
            unet_input = torch.cat([generated_sample, lung_condition_256], dim=1)
            unet_input_3d = unet_input.unsqueeze(2).repeat(1, 1, 8, 1, 1)
            t_tensor = torch.tensor([t_idx], device=device).long()
            
            with torch.amp.autocast(device_type="cuda", enabled=True):
                noise_pred_3d = model(unet_input_3d, timesteps=t_tensor)
                noise_pred = noise_pred_3d[:, :, 4, :, :]
                
            generated_sample = scheduler.step_backward(noise_pred, t_idx, generated_sample)
            
            if t_idx in target_milestones:
                trajectory_panels[t_idx] = generated_sample.squeeze().cpu().numpy()
                
    # ─── 📊 PLOT INDEPENDENT 5-PANEL HORIZONTAL CANVAS FOR THIS FOCUSED EPOCH ───
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), facecolor="black")
    
    for i, t_val in enumerate(target_milestones):
        axes[i].imshow(trajectory_panels[t_val], cmap="gray")
        axes[i].set_title(f"Timestep {t_val}", color="white", fontsize=12, fontweight="bold", pad=8)
        axes[i].axis("off")
        
    plt.suptitle(f"DDPM Reverse Denoising Trajectory Spectrum — Trained at Epoch {epoch_num}", 
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    output_grid_path = os.path.join(OUTPUT_TRAJ_DIR, f"trajectory_grid_epoch_{epoch_num}.png")
    plt.savefig(output_grid_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"   ✅ Finished! Matrix grid file committed safely: trajectory_grid_epoch_{epoch_num}.png")

print("\n🏆 ALL TRAJECTORY MO_DEL SNAPSHOTS COMPILED AND SECURED NATIVELY.")
print("=================================================================")
