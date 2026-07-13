"""
Baseline U-Net training.

Usage:
    python scripts/train_baseline.py \
        --config configs/training.yaml \
        --data-root /scratch/u6dm/mk25bm.u6dm/ild_dataset_processed
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from core import (
    enforce_system_determinism,
    get_device_context,
    get_model,
    ILDDatasetSplit,
    load_yaml_config,
)
from core.metrics import MetricsLogger
from core.utils import (  # ← IMPORT FROM UTILS
    TrainingVisualizer,
    CheckpointManager,
    HistoryTracker,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CLASS_NAMES = {
    0: "background",
    1: "healthy",
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
    7: "other_rare",
}


class DiceCELoss(nn.Module):
    """Dice + Cross-Entropy loss with class weighting."""
    
    def __init__(self, class_weights: torch.Tensor, smooth: float = 1e-6):
        super().__init__()
        self.class_weights = class_weights
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) class indices
        """
        # Cross-entropy
        ce = self.ce_loss(pred, target)
        
        # Dice
        pred_soft = torch.softmax(pred, dim=1)  # (B, C, H, W)
        
        # Convert target to one-hot
        target_one_hot = torch.zeros_like(pred_soft)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        
        # Dice per class
        intersection = torch.sum(pred_soft * target_one_hot, dim=(0, 2, 3))
        union = torch.sum(pred_soft, dim=(0, 2, 3)) + torch.sum(target_one_hot, dim=(0, 2, 3))
        
        dice_per_class = 2.0 * intersection / (union + self.smooth)
        
        # Weighted dice
        dice = 1.0 - torch.mean(dice_per_class)
        
        # Combined loss
        loss = 0.5 * ce + 0.5 * dice
        return loss


class BaselineTrainer:
    """Training pipeline."""
    
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
        
        # Setup logging
        self._setup_file_logging()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Baseline Training: {self.config.experiment.name}")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info(f"{'='*70}\n")
        
        # Load data
        logger.info("Loading dataset...")
        
        self.train_dataset = ILDDatasetSplit(
            split="train",
            seed=self.config.training.seed,
            empty_mask_strategy=self.config.dataset.empty_mask_strategy,
            verbose=False
        )
        self.val_dataset = ILDDatasetSplit(
            split="val",
            seed=self.config.training.seed,
            empty_mask_strategy=self.config.dataset.empty_mask_strategy,
            verbose=False
        )
        self.test_dataset = ILDDatasetSplit(
            split="test",
            seed=self.config.training.seed,
            empty_mask_strategy=self.config.dataset.empty_mask_strategy,
            verbose=False
        )
        
        logger.info(f"Train: {len(self.train_dataset)} slices")
        logger.info(f"Val:   {len(self.val_dataset)} slices")
        logger.info(f"Test:  {len(self.test_dataset)} slices")
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
        )
        
        # Model
        logger.info(f"Building model: {self.config.model.name}")
        self.model = get_model(
            self.config.model.name,
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
        )
        self.model.to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        
        # Loss
        class_weights = torch.tensor(
            [self.config.training.class_weights.get(i, 1.0) for i in range(8)],
            dtype=torch.float32,
            device=self.device,
        )
        logger.info(f"Class weights: {dict(enumerate(class_weights.cpu().tolist()))}")
        self.loss_fn = DiceCELoss(class_weights=class_weights)
        
        # Optimizer
        self.optimizer = Adam(
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
        
        # Utils (from core/utils)
        self.history_tracker = HistoryTracker(num_classes=8)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_dir)
        self.visualizer = TrainingVisualizer(self.plot_dir)
        
        self.best_val_dice = -np.inf
        self.best_epoch = 0
    
    def _setup_file_logging(self):
        """Add file handler to logger."""
        log_file = self.log_dir / "training.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch."""
        self.model.train()
        metrics = MetricsLogger(8, CLASS_NAMES)
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            metrics.update(loss.item(), logits.detach(), masks.detach())
            
            # Logging
            if (batch_idx + 1) % self.config.logging.log_frequency == 0:
                logger.info(
                    f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(self.train_loader):4d} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        summary = metrics.get_summary()
        metrics.print_summary(phase="Train", epoch=epoch)
        
        return summary
    
    def val_epoch(self, epoch: int) -> Dict:
        """Validate one epoch."""
        self.model.eval()
        metrics = MetricsLogger(8, CLASS_NAMES)
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
                
                metrics.update(loss.item(), logits, masks)
        
        summary = metrics.get_summary()
        metrics.print_summary(phase="Val", epoch=epoch)
        
        return summary
    
    def train(self):
        """Full training loop."""
        epochs = self.config.training.epochs
        patience = self.config.training.early_stopping.patience  # ← From config
        min_delta = self.config.training.early_stopping.min_delta  # ← From config
        log_freq = self.config.logging.log_frequency
        val_freq = self.config.logging.val_frequency
        ckpt_freq = self.config.logging.checkpoint_frequency
        
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Train
            train_summary = self.train_epoch(epoch, log_freq)
            self.history_tracker.update_train_metrics(train_summary)  # ← Pass dict
            
            # Validate
            if (epoch + 1) % val_freq == 0:
                val_summary = self.val_epoch(epoch)
                self.history_tracker.update_val_metrics(val_summary)  # ← Pass dict
                
                # ← MINIMAL LOGGING: Just the key metric
                logger.info(f"Val Dice: {val_summary['dice']:.4f}")
                
                # Early stopping
                if val_summary["dice"] > self.best_val_dice + min_delta:
                    improvement = val_summary["dice"] - self.best_val_dice
                    self.best_val_dice = val_summary["dice"]
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                    
                    # ← ONE LINE: Show why we saved
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
                    epochs_without_improvement += 1
                    logger.info(f"No improvement ({epochs_without_improvement}/{patience})\n")
                
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
        logger.info("\n" + "="*70)
        logger.info("Generating plots...")
        logger.info("="*70)
        self.visualizer.plot_training_curves(self.history_tracker.get())
        self.visualizer.plot_class_distribution(self.history_tracker.get(), CLASS_NAMES)
        
        # Save history
        history_file = self.log_dir / "history.json"
        self.history_tracker.save(history_file)
        
        logger.info(f"\n✅ Training finished! Logs: {self.log_dir}\n")
    
    def train_epoch(self, epoch: int, log_freq: int) -> Dict:
        """Train one epoch with lung mask masking."""
        self.model.train()
        metrics = MetricsLogger(8, CLASS_NAMES)
        
        for batch_idx, (images, masks, lungs) in enumerate(self.train_loader):  # ← ADD lungs
            images = images.to(self.device)
            masks = masks.to(self.device)
            lungs = lungs.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images)
            
            # ← MASK OUT NON-LUNG REGIONS
            # Only compute loss where lung mask is 1
            masked_logits = logits * lungs.unsqueeze(1)  # (B, C, H, W) * (B, 1, H, W)
            masked_masks = masks * lungs.long()           # (B, H, W) * (B, H, W)
            
            loss = self.loss_fn(masked_logits, masked_masks)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics (only on lung region)
            metrics.update(loss.item(), masked_logits.detach(), masked_masks.detach())
            
            if (batch_idx + 1) % log_freq == 0:
                logger.info(
                    f"Epoch {epoch:3d} | Batch {batch_idx+1:4d}/{len(self.train_loader):4d} | "
                    f"Loss: {loss.item():.4f}"
                )
        
        summary = metrics.get_summary()
        return summary


    def val_epoch(self, epoch: int) -> Dict:
        """Validate with lung masking."""
        self.model.eval()
        metrics = MetricsLogger(8, CLASS_NAMES)
        
        with torch.no_grad():
            for images, masks, lungs in self.val_loader:  # ← ADD lungs
                images = images.to(self.device)
                masks = masks.to(self.device)
                lungs = lungs.to(self.device)
                
                logits = self.model(images)
                
                # ← MASK OUT NON-LUNG REGIONS
                masked_logits = logits * lungs.unsqueeze(1)
                masked_masks = masks * lungs.long()
                
                loss = self.loss_fn(masked_logits, masked_masks)
                metrics.update(loss.item(), masked_logits.detach(), masked_masks.detach())
        
        summary = metrics.get_summary()
        return summary
    
def main():
    parser = argparse.ArgumentParser(description="Train baseline U-Net")
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
    trainer = BaselineTrainer(args.config, args.data_root, device)
    trainer.train()


if __name__ == "__main__":
    main()