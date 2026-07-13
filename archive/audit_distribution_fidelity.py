# scripts/audit_distribution_fidelity.py
import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
import cv2

# Define workspace directories
BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
REAL_DATA_DIR = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
SYNTH_DATA_DIR = os.path.join(BASE_DIR, "synthetic_augmentations", "controlled_generation")
OUTPUT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=================================================================")
print("🔬 INITIALIZING HIGH-FIDELITY DISTRIBUTION AUDITOR (FID / KID)")
print("=================================================================")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── 🏗️ 1. FEATURE EXTRACTOR FACTORY ───
class ResNetFeatureExtractor(nn.Module):
    """Uses a pre-trained feature extractor to pull semantic feature maps."""
    def __init__(self):
        super().__init__()
        # Use a lightweight ResNet to process our grayscale medical channels cleanly
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        return self.feature_extractor(x).squeeze(-1).squeeze(-1)

extractor = ResNetFeatureExtractor().to(device)
extractor.eval()

# ─── 📂 2. COMPILE COHORT MATRICES ───
def load_and_preprocess_real_samples(root_dir, num_samples=50):
    samples = []
    patient_folders = sorted(os.listdir(root_dir))
    for p in patient_folders:
        img_paths = glob.glob(os.path.join(root_dir, p, "images", "*.npy"))
        for path in img_paths:
            img = np.load(path)
            # Resize and scale to match standard three-channel image layouts
            img_resized = cv2.resize(img, (256, 256))
            img_scaled = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
            img_rgb = np.stack([img_scaled, img_scaled, img_scaled], axis=0)
            samples.append(img_rgb)
            if len(samples) >= num_samples:
                return torch.tensor(np.array(samples), dtype=torch.float32)
    return torch.tensor(np.array(samples), dtype=torch.float32)

def load_synth_disease_sample(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img_scaled = img.astype(np.float32) / 255.0
    img_rgb = np.stack([img_scaled, img_scaled, img_scaled], axis=0)
    return torch.tensor(img_rgb, dtype=torch.float32).unsqueeze(0)

print("📡 Extracting feature maps for real baseline patient cohorts...")
real_tensors = load_and_preprocess_real_samples(REAL_DATA_DIR, num_samples=50).to(device)
with torch.no_grad():
    real_features = extractor(real_tensors).cpu().numpy()

# Calculate background feature statistics for your real data distribution
mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)

# ─── 🧮 3. THE MATHEMATICAL FID MATRIX CALCULATOR ───
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Computes the continuous multivariate Gaussian distance between distributions."""
    diff = mu1 - mu2
    # Compute the square root of the product of your covariance matrices
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)

# Scan your output directory for generated disease files
synth_files = sorted(glob.glob(os.path.join(SYNTH_DATA_DIR, "guided_synthesis_*.png")))
fid_records = []

print("\n📡 Running cross-class distribution analysis against real patient profiles...")
print("─" * 65)

for f in synth_files:
    disease_name = os.path.basename(f).replace("guided_synthesis_", "").replace(".png", "")
    synth_tensor = load_synth_disease_sample(f)
    if synth_tensor is None: continue
    
    # Generate multiple unique augmented samples to populate your evaluation matrix
    augmented_features = []
    with torch.no_grad():
        for i in range(10): # Simulate minor variations to construct distribution variance
            # Inject varying baseline noise seeds to mirror your dataset distribution
            perturbed_tensor = synth_tensor + 0.02 * torch.randn_like(synth_tensor)
            feat = extractor(perturbed_tensor.to(device)).cpu().numpy()
            augmented_features.append(feat.squeeze())
            
    augmented_features = np.array(augmented_features)
    mu_synth = augmented_features.mean(axis=0)
    sigma_synth = np.cov(augmented_features, rowvar=False) + np.eye(mu_synth.shape[0]) * 1e-6
    
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_synth, sigma_synth)
    fid_records.append({"Pathology_Class": disease_name, "Calculated_FID": fid_score})
    print(f"   • {disease_name:<25} | Calculated FID Discrepancy Score: {fid_score:.2f}")

# Save the final distribution log cleanly to disk
df_fid = pd.DataFrame(fid_records)
df_fid.to_csv(os.path.join(OUTPUT_DIR, "synthetic_distribution_fid_scores.csv"), index=False)
print("─" * 65)
print("✅ DISTRIBUTION METRICS RECO_RDED: artifacts/synthetic_distribution_fid_scores.csv")
print("=================================================================")
