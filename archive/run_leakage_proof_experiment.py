import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

BASE_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL"
sys.path.append(BASE_DIR)

from core.config import (
    SCRATCH_DIR, OUTPUT_ROOT, GLOBAL_SEED, enforce_system_determinism, 
    get_device_context, CLASS_MAPPING, NUM_CLASSES,
    BACKGROUND_WEIGHT_SCALE, MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT
)
from core.dataset import RobustILDDataset
from core.models import build_architecture
from utils.helpers import get_fixed_stratified_splits

enforce_system_determinism(GLOBAL_SEED)
device = get_device_context()
BENCHMARKS_DIR = os.path.join(BASE_DIR, "benchmarks_cv")
os.makedirs(BENCHMARKS_DIR, exist_ok=True)

print("=================================================================")
print("🚀 PRODUCTION SEGMENTATION ENGINE: FIXED 70/15/15 COHORT SPLIT")
print("=================================================================")


class SoftDiceFocalLoss(nn.Module):
    def __init__(self, class_weights, focal_gamma=2.0, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, weight=self.class_weights, reduction="mean")
        pt = torch.exp(-ce_loss)
        focal_ce = ((1.0 - pt) ** self.focal_gamma) * ce_loss

        num_classes = logits.size(1)
        spatial_dims = list(range(2, logits.dim()))
        reduce_dims = [0] + spatial_dims

        probs = torch.softmax(logits, dim=1)[:, 1:]
        target_one_hot = F.one_hot(target, num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, -1, *range(1, target.dim())).float()[:, 1:]

        intersection = torch.sum(probs * target_one_hot, dim=reduce_dims)
        denominator = torch.sum(probs + target_one_hot, dim=reduce_dims)
        dice_per_class = (2.0 * intersection + 1e-6) / (denominator + 1e-6)

        class_present = torch.sum(target_one_hot, dim=reduce_dims) > 0
        if torch.any(class_present):
            dice_loss = 1.0 - dice_per_class[class_present].mean()
        else:
            dice_loss = logits.new_tensor(0.0)

        return (self.ce_weight * focal_ce) + (self.dice_weight * dice_loss)


class RobustSegmentationTracker:
    def __init__(self, num_classes=7):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.dice_by_class = [[] for _ in range(self.num_classes)]

    @torch.no_grad()
    def update(self, y_pred, y_true):
        preds_discrete = torch.argmax(y_pred, dim=1)
        batch_size = y_true.size(0)
        
        for b in range(batch_size):
            p_slice = preds_discrete[b]
            t_slice = y_true[b]
            
            for c in range(self.num_classes):
                p_mask = (p_slice == c)
                t_mask = (t_slice == c)
                
                if not torch.any(t_mask):
                    continue
                    
                inter = (p_mask & t_mask).sum().item()
                denom = p_mask.sum().item() + t_mask.sum().item()
                
                slice_dice = (2.0 * inter) / denom if denom > 0 else 1.0
                self.dice_by_class[c].append(slice_dice)

    def get_foreground_mean_dice(self):
        fg_averages = []
        for c in range(1, self.num_classes):
            scores = self.dice_by_class[c]
            if len(scores) > 0:
                fg_averages.append(np.mean(scores))
        return float(np.mean(fg_averages)) if fg_averages else 0.0

    def print_comprehensive_report(self, class_mapping, split_title="VAL"):
        print(f"\n📋 MULTI-CLASS PERFORMANCE PROFILE ({split_title}):")
        print("─" * 75)
        for idx in range(self.num_classes):
            cls_name = class_mapping.get(idx, f"Label {idx}")
            scores = self.dice_by_class[idx]
            mean_score = np.mean(scores) if len(scores) > 0 else 0.0
            print(f"   • {cls_name:<28} | Dice Score: {mean_score:.4f} | Seen in {len(scores)} slices")
        print("─" * 75)
        print(f"🌟 COMPUTED FOREGROUND DISEASE MEAN DICE: {self.get_foreground_mean_dice():.4f}")
        print("─" * 75)


def run_fixed_stratified_experiment(model_name, epochs=50):
    train_pts, val_pts, test_pts = get_fixed_stratified_splits(SCRATCH_DIR, seed=GLOBAL_SEED)
    
    model_root_dir = os.path.join(BENCHMARKS_DIR, model_name)
    os.makedirs(model_root_dir, exist_ok=True)
    patient_final_scores = []

    print(f"\n⚡ [{model_name.upper()}] Initializing Stratified Clean-Run Optimization Context...")
    print(f"   • Split Cohorts: Train={len(train_pts)} | Val={len(val_pts)} | Test={len(test_pts)}")
    
    train_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=train_pts, skip_empty_masks=True)
    val_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=val_pts, skip_empty_masks=False)
    test_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=test_pts, skip_empty_masks=False)
    
    sample_weights = []
    class_counts = np.zeros(7, dtype=np.int64)
    
    print("🛠️ Prefetching train manifest statistics...")
    for idx in range(len(train_ds)):
        _, target_mask = train_ds[idx]
        
        # Safe CPU remapping path maps messy indices (7-17) cleanly to background
        target_mask[target_mask > 6] = 0
        
        unique_classes = torch.unique(target_mask).numpy()
        class_counts += np.bincount(target_mask.numpy().flatten(), minlength=7)

        # Class 1: healthy_control, Class 2: emphysema, Class 5: micronodules, Class 6: consolidation
        if np.any(np.isin(unique_classes, [1, 2, 5, 6])):
            sample_weights.append(50.0)  # Forces healthy control slices into high-frequency rotation
        elif np.any(unique_classes > 1):
            sample_weights.append(15.0) 
        else:
            sample_weights.append(1.0)
        
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    # FIX: in_channels=2 preserves spatial lung field context alongside continuous density Z-scores
    model = build_architecture(model_name, in_channels=2, out_channels=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    pixel_counts_float = class_counts.astype(np.float64) # Keep all 8 classes
    class_probs = pixel_counts_float / pixel_counts_float.sum()
    class_weights = np.zeros_like(class_probs, dtype=np.float32)
    present_mask = class_counts > 0
    
    # Power root normalization balances foreground representations
    class_weights[present_mask] = 1.0 / (class_probs[present_mask] + 1e-12) ** 0.5
    if np.any(present_mask):
        class_weights[present_mask] /= class_weights[present_mask].mean()
        
    # FORCE BACKGROUND DE-WEIGHTING: Minimize empty space gradient scale
    class_weights[0] *= 0.10 
    
    class_weights = np.clip(class_weights, 0.1, 20.0)
    class_weights[~present_mask] = 0.0
    
    # Set up loss heads matching the 8 output classes
    criterion = SoftDiceFocalLoss(
        class_weights=torch.from_numpy(class_weights).float().to(device),
        ce_weight=0.5,
        dice_weight=0.5
    )
    
    epoch_history_log = []
    best_val_dice = -1.0

    print(f"\n🔮 Optimization active across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            targets[targets > 6] = 0
            
            # Extract 2 anatomical paths safely (CT density + Lung mask boundary guide)
            inputs = inputs[:, 0:2, :, :].to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                loss = criterion(model(inputs), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        model.eval()
        tracker = RobustSegmentationTracker(num_classes=7)
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_labels[val_labels > 6] = 0
                val_outputs = model(val_inputs[:, 0:2, :, :].to(device))
                tracker.update(val_outputs, val_labels.to(device))
        
        val_mean_dice = tracker.get_foreground_mean_dice()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            tracker.print_comprehensive_report(CLASS_MAPPING, split_title=f"VAL | EPOCH {epoch}")
            
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(model_root_dir, f"checkpoint_epoch_{epoch}.pt"))
            
        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            torch.save({
                "model_state_dict": model.state_dict(), 
                "metadata": {"architecture": model_name, "achieved_fg_dice": float(val_mean_dice), "epoch_captured": epoch}
            }, os.path.join(model_root_dir, f"best_weights_{model_name}_fixed_split.pt"))

        history_row = {"Epoch": epoch, "Train_Loss": epoch_loss / len(train_loader), "Val_Mean_Dice": val_mean_dice}
        epoch_history_log.append(history_row)

        pd.DataFrame(epoch_history_log).to_csv(os.path.join(model_root_dir, "history_fixed_split.csv"), index=False)
        
        print(f"  • Epoch {epoch}/{epochs} | Loss: {history_row['Train_Loss']:.4f} | Foreground Val Dice: {val_mean_dice:.4f}")
        sys.stdout.flush()
        scheduler.step()

    print("\n🔒 EVALUATING UNSEEN PATIENT SPLIT TEST HOLDOUT...")
    best_checkpoint = torch.load(os.path.join(model_root_dir, f"best_weights_{model_name}_fixed_split.pt"))
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    
    test_tracker = RobustSegmentationTracker(num_classes=7)
    with torch.no_grad():
        for val_inputs, val_labels in test_loader:
            # val_labels = torch.clamp(val_labels, max=6)
            val_labels[val_labels > 6] = 0
            
            test_outputs = model(val_inputs[:, 0:2, :, :].to(device))
            test_tracker.update(test_outputs, val_labels.to(device))
            
    test_tracker.print_comprehensive_report(CLASS_MAPPING, split_title="FINAL SECURE TEST HOLDOUT")
    
    # Compute and save patient-level metrics for your publication write-up
    with torch.no_grad():
        for p_id in test_pts:
            p_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, allowed_patients=[p_id], skip_empty_masks=False)
            if len(p_ds) == 0: continue
            p_tracker = RobustSegmentationTracker(num_classes=7)
            for inputs, labels in DataLoader(p_ds, batch_size=4, shuffle=False):
                #labels = torch.clamp(labels, max=6)
                labels[labels > 6] = 0
                p_tracker.update(model(inputs[:, 0:2, :, :].to(device)), labels.to(device))
            patient_final_scores.append({"Architecture": model_name, "Patient_ID": p_id, "Achieved_Dice": p_tracker.get_foreground_mean_dice()})
            
    pd.DataFrame(patient_final_scores).to_csv(os.path.join(BENCHMARKS_DIR, f"final_test_patient_scores_{model_name}.csv"), index=False)
    print(f"✅ Fixed Stratified Single Experiment Complete for: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
   # run_cross_validation(args.model, epochs=args.epochs)
    run_fixed_stratified_experiment(args.model, epochs=args.epochs)
