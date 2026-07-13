"""
Optimized training pipeline (ported from archive/run_leakage_proof_experiment.py)

Key improvements:
- 2-channel input (CT + lung mask)
- SoftDiceFocalLoss
- WeightedRandomSampler (disease-heavy sampling)
- Mixed precision (AMP)
- Foreground-only metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import numpy as np
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from core import (
    enforce_system_determinism,
    get_device_context,
    get_model,
    ILDDatasetSplit,
    load_yaml_config,
)
from core.metrics import MetricsLogger
from core.utils import (
    TrainingVisualizer,
    CheckpointManager,
    HistoryTracker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "background",
    1: "healthy_control",
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
    7: "other_rare_pathologies",
}


class SoftDiceFocalLoss(nn.Module):
    """Soft Dice + Focal CE loss with class weighting (from archive)."""
    
    def __init__(
        self,
        class_weights: torch.Tensor,
        focal_gamma: float = 2.0,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.register_buffer("class_weights", class_weights)
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, H, W)
            target: (B, H, W)
        """
        # Focal CE loss
        ce_loss = F.cross_entropy(logits, target, weight=self.class_weights, reduction="mean")
        pt = torch.exp(-ce_loss)
        focal_ce = ((1.0 - pt) ** self.focal_gamma) * ce_loss
        
        # Soft Dice loss
        num_classes = logits.size(1)
        spatial_dims = list(range(2, logits.dim()))
        reduce_dims = [0] + spatial_dims
        
        probs = torch.softmax(logits, dim=1)[:, 1:]  # Exclude background
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


class OptimizedTrainer:
    """Optimized training with archive pipeline improvements."""
    
    def __init__(self, config_path: str, data_root: str, device: str):
        self.config = load_yaml_config(config_path)
        self.data_root = data_root
        self.device = device
        
        # Setup directories
        self.log_dir = Path(self.config.logging.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.plot_dir = self.log_dir / "plots"
        
        for d in [self.log_dir, self.checkpoint_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self._setup_file_logging()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Optimized Training: {self.config.experiment.name}")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info(f"{'='*70}\n")
        
        # Load data
        logger.info("Loading dataset...")
        
        self.train_dataset = ILDDatasetSplit(
            split="train",
            seed=self.config.training.seed,
            empty_mask_strategy=self.config.dataset.empty_mask_strategy,
            verbose=False,
            return_metadata=True,  # ← Need metadata for WeightedRandomSampler
        )
        self.val_dataset = ILDDatasetSplit(
            split="val",
            seed=self.config.training.seed,
            empty_mask_strategy=self.config.dataset.empty_mask_strategy,
            verbose=False,
        )
        self.test_dataset = ILDDatasetSplit(
            split="test",
            seed=self.config.training.seed,
            empty_mask_strategy=self.config.dataset.empty_mask_strategy,
            verbose=False,
        )
        
        logger.info(f"Train: {len(self.train_dataset)} slices")
        logger.info(f"Val:   {len(self.val_dataset)} slices")
        logger.info(f"Test:  {len(self.test_dataset)} slices")
        
        # ← BUILD WEIGHTED SAMPLER (disease-heavy)
        logger.info("Building weighted sampler...")
        sample_weights = self._compute_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            sampler=sampler,  # ← Use weighted sampler
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        # Model (2-channel input: CT + lung mask)
        logger.info(f"Building model: {self.config.model.name}")
        self.model = get_model(
            self.config.model.name,
            in_channels=2,  # ← CT + lung mask
            out_channels=8,
        )
        self.model.to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        
        # Loss (SoftDiceFocalLoss)
        class_weights = torch.tensor(
            [self.config.training.class_weights.get(i, 1.0) for i in range(8)],
            dtype=torch.float32,
            device=self.device,
        )
        logger.info(f"Class weights: {dict(enumerate(class_weights.cpu().tolist()))}")
        self.loss_fn = SoftDiceFocalLoss(class_weights=class_weights)
        
        # Optimizer (AdamW)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.epochs,
            eta_min=1e-6,
        )
        
        # AMP scaler (mixed precision)
        self.scaler = torch.amp.GradScaler('cuda')
        
        # Utils
        self.history_tracker = HistoryTracker(num_classes=8)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.visualizer = TrainingVisualizer(self.plot_dir)
        
        self.best_val_dice = -np.inf
        self.best_epoch = 0
        
        # ← LOG FULL EXPERIMENT CONFIG (pass SimpleNamespace directly)
        self.history_tracker.set_metadata(
            config=self.config,  # ← Pass SimpleNamespace as-is
            
            model_name="standard_unet",
            model_architecture={
                "in_channels": 2,  # CT + lung mask
                "out_channels": 8,
            },
            
            loss_name="SoftDiceFocalLoss",
            loss_params={
                "focal_gamma": 2.0,
                "ce_weight": 0.5,
                "dice_weight": 0.5,
                "class_weights": class_weights.tolist(),
            },
            
            optimizer_name="AdamW",
            optimizer_params={
                "lr": float(self.config.training.learning_rate),
                "weight_decay": float(self.config.training.weight_decay),
                "betas": [0.9, 0.999],
            },
            
            scheduler_name="CosineAnnealingLR",
            scheduler_params={
                "T_max": self.config.training.epochs,
                "eta_min": 1e-6,
            },
            
            sampler_name="WeightedRandomSampler",
            sampler_params={
                "strategy": "disease_heavy",
                "background_weight": 1.0,
                "mild_disease_weight": 15.0,
                "severe_disease_weight": 50.0,
            },
            
            notes="2-channel input (CT+lung), foreground Dice optimization, mixed precision AMP",
        )
        
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT CONFIGURATION")
        logger.info("="*70)
        logger.info(json.dumps(self.history_tracker.metadata, indent=2, default=str))
        logger.info("="*70 + "\n")
        
        logger.info("\n" + "="*70)
        logger.info("WEIGHTED SAMPLER STATISTICS")
        logger.info("="*70)

        # Count weight distribution
        weight_counts = {}
        for w in sample_weights:
            w_key = f"{w:.1f}"
            weight_counts[w_key] = weight_counts.get(w_key, 0) + 1

        for weight, count in sorted(weight_counts.items()):
            pct = 100 * count / len(sample_weights)
            logger.info(f"Weight {weight:>5s}: {count:4d} samples ({pct:5.1f}%)")

        logger.info(f"Total weight sum: {sum(sample_weights):.0f}")
        logger.info("="*70 + "\n")
    
    def _setup_file_logging(self):
        """Setup file logging."""
        log_file = self.log_dir / "training.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def _compute_sample_weights(self) -> list:
        """Compute sample weights (disease-heavy sampling)."""
        sample_weights = []
        
        for idx in range(len(self.train_dataset)):
            sample = self.train_dataset[idx]
            mask = sample["mask"] if isinstance(sample, dict) else sample[1]
            
            # ← REMOVE THIS: Don't remap
            # mask_remapped = mask.clone()
            # mask_remapped[mask_remapped > 6] = 0
            
            unique_classes = torch.unique(mask).numpy()
            
            # Weight by disease presence (any class > 0 is disease)
            if np.any(unique_classes > 0):
                # Check for high-priority diseases
                if np.any(np.isin(unique_classes, [3, 4])):
                    # Ground glass or fibrosis
                    sample_weights.append(50.0)
                elif np.any(np.isin(unique_classes, [2, 5, 6])):
                    # Emphysema, micronodules, consolidation
                    sample_weights.append(25.0)
                else:
                    # Other diseases
                    sample_weights.append(10.0)
            else:
                # Background only
                sample_weights.append(1.0)
        
        return sample_weights
    
    def train_epoch(self, epoch: int, log_freq: int) -> Dict:
        """Train one epoch with mixed precision."""
        self.model.train()
        metrics = MetricsLogger(8, CLASS_NAMES)
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # Unpack batch
            if isinstance(batch_data, dict):
                images = batch_data["image"]
                masks = batch_data["mask"]
                lungs = batch_data["lung"]
            else:
                images, masks, lungs = batch_data
            
            # ← REMOVE THIS: Don't remap classes
            # masks[masks > 6] = 0  # ← DELETE THIS LINE
            
            # ← INSTEAD: Clamp any invalid values to class 7
            masks = torch.clamp(masks, 0, 7)
            
            # Concatenate CT + lung mask (2 channels)
            images_2ch = torch.cat([images, lungs.unsqueeze(1)], dim=1)
            
            images_2ch = images_2ch.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = self.model(images_2ch)
                loss = self.loss_fn(logits, masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            metrics.update(loss.item(), logits.detach(), masks.detach())
            
            if (batch_idx + 1) % log_freq == 0:
                logger.info(
                    f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(self.train_loader):4d} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        summary = metrics.get_summary()
        return summary
    
    def val_epoch(self, epoch: int) -> Dict:
        """Validate with foreground Dice metric."""
        self.model.eval()
        metrics = MetricsLogger(8, CLASS_NAMES)
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                # Unpack batch
                if isinstance(batch_data, dict):
                    images = batch_data["image"]
                    masks = batch_data["mask"]
                    lungs = batch_data["lung"]
                else:
                    images, masks, lungs = batch_data
                
                # ← REMOVE THIS: Don't remap
                # masks[masks > 6] = 0  # ← DELETE THIS LINE
                
                # ← INSTEAD: Clamp to valid range
                masks = torch.clamp(masks, 0, 7)
                
                # Concatenate CT + lung mask
                images_2ch = torch.cat([images, lungs.unsqueeze(1)], dim=1)
                
                images_2ch = images_2ch.to(self.device)
                masks = masks.to(self.device)
                
                logits = self.model(images_2ch)
                loss = self.loss_fn(logits, masks)
                
                metrics.update(loss.item(), logits.detach(), masks.detach())
        
        summary = metrics.get_summary()
        return summary
    
    def train(self):
        """Full training loop."""
        epochs = self.config.training.epochs
        patience = self.config.training.early_stopping.patience
        min_delta = self.config.training.early_stopping.min_delta
        log_freq = self.config.logging.log_frequency
        val_freq = self.config.logging.val_frequency
        ckpt_freq = self.config.logging.checkpoint_frequency
        
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train
            train_summary = self.train_epoch(epoch, log_freq)
            self.history_tracker.update_train_metrics(train_summary)
            
            # Validate
            if (epoch + 1) % val_freq == 0:
                val_summary = self.val_epoch(epoch)
                self.history_tracker.update_val_metrics(val_summary)
                
                logger.info(f"Val Dice: {val_summary['dice']:.4f}")
                
                # ← EARLY STOPPING LOGIC
                if val_summary["dice"] > self.best_val_dice + min_delta:
                    improvement = val_summary["dice"] - self.best_val_dice
                    self.best_val_dice = val_summary["dice"]
                    self.best_epoch = epoch
                    epochs_without_improvement = 0  # ← RESET COUNTER
                    
                    logger.info(f"✓ Best model! (+{improvement:.4f})\n")
                    
                    self.checkpoint_manager.save_checkpoint(
                        epoch,
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        self.scheduler.state_dict(),
                        self.history_tracker.get(),
                        is_best=True,
                    )
                else:
                    epochs_without_improvement += 1  # ← INCREMENT COUNTER
                    logger.info(f"No improvement ({epochs_without_improvement}/{patience})\n")
                
                # ← STOP IF NO IMPROVEMENT FOR `patience` EPOCHS
                if epochs_without_improvement >= patience:
                    logger.info(
                        f"\n⏹️  Early stopping at epoch {epoch} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break
            
            # Checkpoint
            if (epoch + 1) % ckpt_freq == 0:
                self.checkpoint_manager.save_checkpoint(
                    epoch,
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    self.scheduler.state_dict(),
                    self.history_tracker.get(),
                    is_best=False,
                )
            
            # Scheduler
            self.scheduler.step()
        
        # Final checkpoint
        self.checkpoint_manager.save_checkpoint(
            epoch,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            self.history_tracker.get(),
            is_best=False,
            is_final=True,
        )
        
        logger.info(f"\n✓ Training complete! Best epoch: {self.best_epoch}")
        logger.info(f"Best Dice: {self.best_val_dice:.4f}")
        logger.info("\n" + "="*70)
        logger.info("Generating plots...")
        logger.info("="*70)
        self.visualizer.plot_training_curves(self.history_tracker.get())
        self.visualizer.plot_class_distribution(self.history_tracker.get(), CLASS_NAMES)
        
        history_file = self.log_dir / "history.json"
        self.history_tracker.save(history_file)
        
        logger.info(f"\n✅ Training finished! Logs: {self.log_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="Train optimized U-Net")
    parser.add_argument(
        "--config",
        type=str,
        default="training.yaml",
        help="Config file (relative to configs/ dir)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
    )
    
    args = parser.parse_args()
    
    # Determinism
    enforce_system_determinism(42)
    
    # Device
    device = get_device_context()
    
    # Train
    trainer = OptimizedTrainer(args.config, args.data_root, device)
    trainer.train()


if __name__ == "__main__":
    main()