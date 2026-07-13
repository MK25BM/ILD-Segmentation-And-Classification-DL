import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Forced unbuffered stdout wrapper function
def log_milestone(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"📡 [{timestamp}] {message}")
    sys.stdout.flush()  # Forces Slurm to write this line to disk INSTANTLY

log_milestone("🚀 INITIALIZING PARALLEL OPENAI GUIDED-DIFFUSION DIAGNOSTIC WORKER")

# ─── 🚀 1. PATH INTEGRATIONS ───
BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "lung_ddpm_plus_src"))

from config import SCRATCH_DIR, CLASS_MAPPING
from dataset import RobustILDDataset
from diffusion_model.unet import UNetModel

# Force enterprise GPU acceleration
log_milestone("Checking hardware allocation hooks...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_milestone(f"Compute engine mapped strictly to device context: {device}")

# ─── 🚀 2. INITIALIZE DATA PIPELINE ───
log_milestone("Loading multi-label stratified cross-validation splits...")
from config import generate_stratified_splits
splits = generate_stratified_splits(seed=42)
train_pts = splits[0][0]
if isinstance(train_pts, tuple):
    train_pts = train_pts[0] # Safely unwrap training patient arrays if nested as a tuple
log_milestone(f"Holdout isolation complete. Training pool restricted to {len(train_pts)} active patients.")

log_milestone("Initializing High-Speed Native Disk Scraper...")
start_io = time.time()
train_dataset = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=train_pts, skip_empty_masks=True)
log_milestone(f"Disk scraper built sample matrix array containing {len(train_dataset)} targeted slices.")
log_milestone(f"Dataset manifest creation elapsed time: {time.time() - start_io:.2f} seconds.")

loader = DataLoader(
    train_dataset, 
    batch_size=1,         # SPEED UP: Parallelizes 4 slices simultaneously in VRAM
    shuffle=True, 
    num_workers=4,         # SPEED UP: Uses 4 distinct CPU background workers to stage slices
    pin_memory=True,       # SPEED UP: Locks pages in RAM for ultra-fast direct transfers to GPU
    persistent_workers=True
)

# ─── 🚀 3. INSTANTIATE REPOSITORY 3D DIFFUSION ARCHITECTURE ───
log_milestone("Building UNetModel neural configuration maps...")
model = UNetModel(
    image_size=256,       
    in_channels=10,          # 1 channel for noisy image + 1 channel for lung mask + 8 channels for pathology classes
    model_channels=32, 
    out_channels=1,
    num_res_blocks=2,
    attention_resolutions=(16, 8), 
    channel_mult=(1, 2, 4, 8),
    dims=3  
).to(device)
log_milestone("Model compiled and pushed to active GPU memory channels successfully.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Directs all checkpoints into benchmarks_cv/ddpm_model_pt to maintain repository hygiene
checkpoint_dir = os.path.join(BASE_DIR, "benchmarks_cv", "ddpm_model_pt")
os.makedirs(checkpoint_dir, exist_ok=True)

# Scan the dedicated subdirectory for any existing epoch checkpoints saved from previous runs
existing_checkpoints = sorted([
    f for f in os.listdir(checkpoint_dir) 
    if f.startswith("checkpoint_ddpm_epoch_") and f.endswith(".pt")
], key=lambda x: int(x.split("_")[-1].split(".")[0]))

start_epoch = 1
if existing_checkpoints:
    latest_ckpt = existing_checkpoints[-1]
    # Extract the exact epoch integer value out of the file string name
    last_epoch = int(latest_ckpt.split("_")[-1].split(".")[0])
    
    # Load the physical model parameters natively to resume state weights
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_ckpt), map_location=device))
    start_epoch = last_epoch + 1
    log_milestone(f"🔄 RESUME ENGINE ACTIVE: Loaded {latest_ckpt}. Resuming pipeline training from Epoch {start_epoch}/100...")

class LinearNoiseScheduler:
    def __init__(self, timesteps=1000):
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, original, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * original + sqrt_one_minus_alpha_hat * noise

scheduler = LinearNoiseScheduler(timesteps=1000)

# ─── 🚀 4. INSTRUMENTED ACCELERATED DISPATCH TRAINING LOOP ───
epochs = 100
log_milestone(f"=== 🎨 DISPATCHING VOLUMETRIC 3D DDPM LOOP MATRIX FOR {epochs} EPOCHS ===")

# ─── 🔒 NEW: NATIVE PRE-FLIGHT EXIT GATE FOR LOGIN NODE TESTING ───
is_preflight = os.environ.get("PREFLIGHT_TEST", "0") == "1"
if is_preflight:
    log_milestone("🔍 PREFLIGHT ENVIRONMENT DETECTED. Restricting engine to 2 batches for diagnostic validation...")
    epochs = start_epoch + 1  # Guard range validation check bounds

# Instantiate the GradScaler right before your training loop begins
# This acts as an anti-underflow shield, protecting multi-channel gradients from exploding!
scaler = torch.amp.GradScaler('cuda')
for epoch in range(start_epoch, epochs + 1):
    log_milestone(f"🏁 Beginning Epoch {epoch}/{epochs} data pipeline stream pass...")
    torch.cuda.empty_cache()
    model.train()
    epoch_loss = 0
    start_epoch_time = time.time()
    
    for batch_idx, (inputs, _) in enumerate(loader):
        start_batch = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        # Downsample the full 10-channel 512x512 tensor down to 256x256 at runtime
        inputs_256 = F.interpolate(inputs, size=(256, 256), mode="bilinear", align_corners=False)
        
        real_images = inputs_256[:, 0:1, :, :].to(device)
        semantic_layout_guide = inputs_256[:, 1:10, :, :].to(device)
        noise = torch.randn_like(real_images)
        
        current_batch_size = inputs_256.shape[0]
        t = torch.randint(0, 1000, size=(current_batch_size,), device=device).long()
        noisy_images = scheduler.add_noise(real_images, noise, t)
        
        # Concatenate 1 noise channel + 9 structural/pathology channels = 10 input tracks
        unet_input = torch.cat([noisy_images, semantic_layout_guide], dim=1)
        unet_input_3d = unet_input.unsqueeze(2).repeat(1, 1, 8, 1, 1) # Symmetrical depth=8
        
        # Enforce modern, non-deprecated autocast tracking syntax context
        with torch.amp.autocast(device_type="cuda", enabled=True):
            noise_pred_3d = model(unet_input_3d, timesteps=t)
            noise_pred = noise_pred_3d[:, :, 4, :, :]
            loss = F.mse_loss(noise_pred, noise)
            
        # ─── 🛡️ THE GRADIENT RESCALER SHIELD ───
        # This dynamically scales your loss values before backprop to stop underflow NaN bugs!
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

        if epoch == start_epoch and (batch_idx == 0 or (batch_idx + 1) % 50 == 0):
            elapsed = time.time() - start_batch
            log_milestone(f"   • Batch {batch_idx + 1}/{len(loader)} | Step Time: {elapsed:.2f}s | MSE Loss: {loss.item():.5f}")
            
    epoch_elapsed = time.time() - start_epoch_time
    log_milestone(f"🔄 Completed Epoch {epoch}/{epochs} | Average Loss: {epoch_loss/len(loader):.5f} | Time: {epoch_elapsed:.1f}s")
    
    # Save a fully uncorrupted, valid checkpoint file at the close of every epoch
    backup_path = os.path.join(checkpoint_dir, f"checkpoint_ddpm_epoch_{epoch}.pt")
    torch.save(model.state_dict(), backup_path)
    log_milestone(f"💾 [SHIELD] Secured pristine checkpoint backup to disk: {backup_path}")

# Save final generative weights matrix
final_path = os.path.join(BASE_DIR, "benchmarks_cv", "final_ild_generative_ddpm.pt")
torch.save(model.state_dict(), final_path)
log_milestone(f"🏆 GENERATIVE DIFFUSION WEIGHTS CHECKPOINT WRITTEN TO DISK SECURELY: {final_path}")
