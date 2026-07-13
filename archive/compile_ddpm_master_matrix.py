# compile_master_matrix.py
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

# Bind project repository layouts natively
BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "lung_ddpm_plus_src"))

from diffusion_model.unet import UNetModel

# Setup clear path roadmaps
CHECKPOINT_DIR = os.path.join(BASE_DIR, "benchmarks_cv", "ddpm_model_pt")
PROCESSED_ROOT = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
OUTPUT_TRAJ_DIR = os.path.join(BASE_DIR, "synthetic_augmentations", "trajectory_study_all_epochs")
FINAL_MATRIX_DIR = os.path.join(BASE_DIR, "synthetic_augmentations")

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
print("📊 INITIALIZING COMPREHENSIVE GENERATIVE PROGRESSION MATRIX")
print("=================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Systematically locate all available checkpoints
ckpt_files = sorted([
    f for f in os.listdir(CHECKPOINT_DIR) 
    if f.startswith("checkpoint_ddpm_epoch_") and f.endswith(".pt")
], key=lambda x: int(x.split("_")[-1].split(".")[0]))

# Restrict explicitly to your 10 matching milestone checkpoints
target_epochs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ckpt_files = [f for f in ckpt_files if int(f.split("_")[-1].split(".")[0]) in target_epochs]

print(f"📦 Compiling a unified 10x5 grid matrix for {len(ckpt_files)} milestone models...")

# 2. Extract real lung silhouette framework
patient_folders = sorted(os.listdir(PROCESSED_ROOT))
first_patient = [p for p in patient_folders if os.path.isdir(os.path.join(PROCESSED_ROOT, p))][0]
mask_files = sorted(glob.glob(os.path.join(PROCESSED_ROOT, first_patient, "lung_masks", "*.npy")))
sample_lung_mask = np.load(mask_files[len(mask_files)//2])

lung_condition = torch.from_numpy(sample_lung_mask).unsqueeze(0).unsqueeze(0).float()
lung_condition_256 = torch.nn.functional.interpolate(lung_condition, size=(256, 256), mode="nearest").to(device)

model = UNetModel(
    image_size=256, in_channels=2, model_channels=32, out_channels=1,
    num_res_blocks=2, attention_resolutions=(16, 8), channel_mult=(1, 2, 4, 8), dims=3
).to(device)

scheduler = SimpleDDPMScheduler(timesteps=1000, device=device)

torch.manual_seed(101)
frozen_initial_noise = torch.randn((1, 1, 256, 256), device=device)
target_milestones = [999, 750, 500, 250, 0]

# Initialize global layout figure canvas: 10 rows (epochs) x 5 columns (timesteps)
fig, axes = plt.subplots(10, 5, figsize=(16, 32), facecolor="black")

# ─── 🚀 THE COMPREHENSIVE COMPILATION LOOP MATRIX ───
for row_idx, ckpt_name in enumerate(ckpt_files):
    epoch_num = int(ckpt_name.split("_")[-1].split(".")[0])
    print(f"▶️ Generating timeline sequence for Row {row_idx+1}/10: Epoch {epoch_num}...")
    
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, ckpt_name), map_location=device))
    model.eval()
    
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
                
    # Plot this epoch's panels horizontally across the current row
    for col_idx, t_val in enumerate(target_milestones):
        ax = axes[row_idx, col_idx]
        ax.imshow(trajectory_panels[t_val], cmap="gray")
        ax.axis("off")
        
        # Inject column headers only on the very first row
        if row_idx == 0:
            ax.set_title(f"Timestep {t_val}", color="white", fontsize=12, fontweight="bold", pad=8)
            
    # Inject row headers on the leftmost edge
    axes[row_idx, 0].text(-25, 128, f"Epoch {epoch_num}", color="white", 
                          fontsize=12, fontweight="bold", ha="right", va="center")

plt.suptitle("MSc Dissertation Definitive DDPM Generative Progression Matrix", 
             color="white", fontsize=16, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0.05, 0, 1, 0.98], pad=1.5)

output_grid_path = os.path.join(FINAL_MATRIX_DIR, "definitive_generative_master_matrix.png")
plt.savefig(output_grid_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"\n🎉 MASTER MATRIX COMPLETE: Mega-grid figure committed safely to disk:\n👉 {output_grid_path}")
print("=================================================================")
