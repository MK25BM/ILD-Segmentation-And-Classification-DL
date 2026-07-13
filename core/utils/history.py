"""Training history tracking."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class HistoryTracker:
    """Track training history with full experiment metadata."""
    
    def __init__(self, num_classes: int = 8):
        self.num_classes = num_classes
        
        # Training metrics
        self.train_loss = []
        self.train_dice = []
        self.train_iou = []
        self.train_class_dice = {}
        
        # Validation metrics
        self.val_loss = []
        self.val_dice = []
        self.val_iou = []
        self.val_class_dice = {}
        
        # ← Experiment metadata
        self.metadata = {}
    
    def set_metadata(
        self,
        config,  # SimpleNamespace or dict
        model_name: str,
        model_architecture: Dict[str, Any],
        loss_name: str,
        loss_params: Dict[str, Any],
        optimizer_name: str,
        optimizer_params: Dict[str, Any],
        scheduler_name: str,
        scheduler_params: Dict[str, Any],
        sampler_name: str,
        sampler_params: Dict[str, Any],
        notes: str = "",
    ):
        """Store experiment configuration for reproducibility."""
        
        # Helper for SimpleNamespace or dict
        def get_val(obj, *keys, default=None):
            """Get nested value from SimpleNamespace or dict."""
            for key in keys:
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    obj = getattr(obj, key, None)
                if obj is None:
                    return default
            return obj
        
        self.metadata = {
            "experiment_timestamp": datetime.now().isoformat(),
            
            # Model
            "model_name": model_name,
            "model_architecture": model_architecture,
            "model_in_channels": model_architecture.get("in_channels"),
            "model_out_channels": model_architecture.get("out_channels"),
            
            # Loss
            "loss_name": loss_name,
            "loss_params": loss_params,
            
            # Optimizer
            "optimizer_name": optimizer_name,
            "optimizer_params": optimizer_params,
            
            # Scheduler
            "scheduler_name": scheduler_name,
            "scheduler_params": scheduler_params,
            
            # Sampling strategy
            "sampler_name": sampler_name,
            "sampler_params": sampler_params,
            
            # Full config
            "training_config": {
                "epochs": get_val(config, "training", "epochs"),
                "batch_size": get_val(config, "training", "batch_size"),
                "learning_rate": get_val(config, "training", "learning_rate"),
                "weight_decay": get_val(config, "training", "weight_decay"),
                "seed": get_val(config, "training", "seed"),
            },
            
            "early_stopping_config": {
                "patience": get_val(config, "training", "early_stopping", "patience"),
                "min_delta": get_val(config, "training", "early_stopping", "min_delta"),
            },
            
            "class_weights": get_val(config, "training", "class_weights"),
            
            # Dataset info
            "dataset_config": {
                "empty_mask_strategy": get_val(config, "dataset", "empty_mask_strategy"),
                "train_split": get_val(config, "dataset", "train_split"),
                "val_split": get_val(config, "dataset", "val_split"),
            },
            
            # Notes
            "experiment_notes": notes,
        }
    
    def update_train_metrics(self, metrics: Dict):
        """Update training metrics."""
        self.train_loss.append(metrics["loss"])
        self.train_dice.append(metrics["dice"])
        self.train_iou.append(metrics["iou"])
        
        for cls, dice in metrics["class_dice"].items():
            if cls not in self.train_class_dice:
                self.train_class_dice[cls] = []
            self.train_class_dice[cls].append(dice)
    
    def update_val_metrics(self, metrics: Dict):
        """Update validation metrics."""
        self.val_loss.append(metrics["loss"])
        self.val_dice.append(metrics["dice"])
        self.val_iou.append(metrics["iou"])
        
        for cls, dice in metrics["class_dice"].items():
            if cls not in self.val_class_dice:
                self.val_class_dice[cls] = []
            self.val_class_dice[cls].append(dice)
    
    def get(self) -> Dict:
        """Get all tracked data."""
        return {
            "metadata": self.metadata,
            "train": {
                "loss": self.train_loss,
                "dice": self.train_dice,
                "iou": self.train_iou,
                "class_dice": self.train_class_dice,
            },
            "val": {
                "loss": self.val_loss,
                "dice": self.val_dice,
                "iou": self.val_iou,
                "class_dice": self.val_class_dice,
            },
        }
    
    def save(self, filepath: Path):
        """Save history to JSON."""
        data = self.get()
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✓ History saved: {filepath}")