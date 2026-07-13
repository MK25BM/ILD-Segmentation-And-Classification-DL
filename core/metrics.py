"""
Class-wise metrics for sparse ROI segmentation.
"""

import numpy as np
import torch
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DiceMetric:
    """Dice coefficient with class-wise tracking."""
    
    @staticmethod
    def compute(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int = -1,
    ) -> Tuple[float, Dict[int, float]]:
        """
        Compute macro Dice and per-class Dice.
        
        Args:
            pred: (B, C, H, W) logits or probabilities
            target: (B, H, W) class indices
            num_classes: Number of classes
            ignore_index: Class to ignore (e.g., unlabeled)
        
        Returns:
            (macro_dice, class_wise_dice_dict)
        """
        if pred.dim() == 4:
            # pred is logits/probabilities (B, C, H, W)
            pred = torch.argmax(pred, dim=1)  # → (B, H, W)
        
        pred = pred.cpu().numpy().astype(np.int64)
        target = target.cpu().numpy().astype(np.int64)
        
        class_dice = {}
        valid_classes = []
        
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            
            pred_mask = (pred == cls).astype(np.float32)
            target_mask = (target == cls).astype(np.float32)
            
            intersection = np.sum(pred_mask * target_mask)
            union = np.sum(pred_mask) + np.sum(target_mask)
            
            if union == 0:
                # No pixels of this class in batch
                if np.sum(target_mask) == 0:
                    # Class not present in ground truth—skip
                    continue
                else:
                    # False negative
                    class_dice[cls] = 0.0
            else:
                dice = 2.0 * intersection / union
                class_dice[cls] = float(dice)
            
            valid_classes.append(cls)
        
        # Macro Dice (average over classes present in batch)
        if valid_classes:
            macro_dice = np.mean([class_dice[c] for c in valid_classes])
        else:
            macro_dice = 0.0
        
        return float(macro_dice), class_dice
    
    @staticmethod
    def compute_confusion_matrix(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> np.ndarray:
        """Compute confusion matrix."""
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for i in range(num_classes):
            for j in range(num_classes):
                cm[i, j] = np.sum((pred == i) & (target == j))
        
        return cm


class IoUMetric:
    """Intersection over Union with class-wise tracking."""
    
    @staticmethod
    def compute(
        pred: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ignore_index: int = -1,
    ) -> Tuple[float, Dict[int, float]]:
        """Compute macro IoU and per-class IoU."""
        if pred.dim() == 4:
            pred = torch.argmax(pred, dim=1)
        
        pred = pred.cpu().numpy().astype(np.int64)
        target = target.cpu().numpy().astype(np.int64)
        
        class_iou = {}
        valid_classes = []
        
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            
            pred_mask = (pred == cls).astype(np.float32)
            target_mask = (target == cls).astype(np.float32)
            
            intersection = np.sum(pred_mask * target_mask)
            union = np.sum(pred_mask) + np.sum(target_mask) - intersection
            
            if union == 0:
                if np.sum(target_mask) == 0:
                    continue
                else:
                    class_iou[cls] = 0.0
            else:
                iou = intersection / union
                class_iou[cls] = float(iou)
            
            valid_classes.append(cls)
        
        if valid_classes:
            macro_iou = np.mean([class_iou[c] for c in valid_classes])
        else:
            macro_iou = 0.0
        
        return float(macro_iou), class_iou


class MetricsLogger:
    """Log class-wise metrics with formatting."""
    
    def __init__(self, num_classes: int, class_names: Dict[int, str]):
        self.num_classes = num_classes
        self.class_names = class_names
        
        # Running averages
        self.reset()
    
    def reset(self):
        """Reset accumulators."""
        self.loss_sum = 0.0
        self.loss_count = 0
        self.dice_sum = 0.0
        self.dice_count = 0
        self.iou_sum = 0.0
        self.iou_count = 0
        
        self.class_dice = {c: {"sum": 0.0, "count": 0} for c in range(self.num_classes)}
        self.class_iou = {c: {"sum": 0.0, "count": 0} for c in range(self.num_classes)}
        
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def update(
        self,
        loss: float,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Update metrics."""
        # Loss
        self.loss_sum += loss
        self.loss_count += 1
        
        # Dice
        macro_dice, class_dice = DiceMetric.compute(pred, target, self.num_classes)
        self.dice_sum += macro_dice
        self.dice_count += 1
        
        for cls, dice in class_dice.items():
            self.class_dice[cls]["sum"] += dice
            self.class_dice[cls]["count"] += 1
        
        # IoU
        macro_iou, class_iou = IoUMetric.compute(pred, target, self.num_classes)
        self.iou_sum += macro_iou
        self.iou_count += 1
        
        for cls, iou in class_iou.items():
            self.class_iou[cls]["sum"] += iou
            self.class_iou[cls]["count"] += 1
        
        # Confusion matrix
        cm = DiceMetric.compute_confusion_matrix(pred, target, self.num_classes)
        self.confusion_matrix += cm
    
    def get_summary(self) -> Dict:
        """Get averaged metrics."""
        avg_loss = self.loss_sum / max(self.loss_count, 1)
        avg_dice = self.dice_sum / max(self.dice_count, 1)
        avg_iou = self.iou_sum / max(self.iou_count, 1)
        
        class_dice_avg = {}
        class_iou_avg = {}
        
        for cls in range(self.num_classes):
            count = self.class_dice[cls]["count"]
            if count > 0:
                class_dice_avg[cls] = self.class_dice[cls]["sum"] / count
                class_iou_avg[cls] = self.class_iou[cls]["sum"] / count
        
        return {
            "loss": avg_loss,
            "dice": avg_dice,
            "iou": avg_iou,
            "class_dice": class_dice_avg,
            "class_iou": class_iou_avg,
            "confusion_matrix": self.confusion_matrix,
        }
    
    def print_summary(self, phase: str = "Train", epoch: int = 0):
        """Pretty-print metrics."""
        summary = self.get_summary()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch:3d} | {phase:5s} Metrics")
        logger.info(f"{'='*70}")
        
        logger.info(f"Loss: {summary['loss']:.4f}")
        logger.info(f"Dice (macro): {summary['dice']:.4f}")
        logger.info(f"IoU (macro):  {summary['iou']:.4f}")
        
        logger.info(f"\n{'Class':20s} {'Name':20s} {'Dice':>8s} {'IoU':>8s}")
        logger.info(f"{'-'*70}")
        
        for cls in range(self.num_classes):
            name = self.class_names.get(cls, f"class_{cls}")
            dice = summary["class_dice"].get(cls, np.nan)
            iou = summary["class_iou"].get(cls, np.nan)
            
            dice_str = f"{dice:.4f}" if not np.isnan(dice) else "N/A"
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            
            logger.info(f"  {cls:<2d}           {name:20s} {dice_str:>8s} {iou_str:>8s}")
        
        logger.info(f"{'='*70}\n")