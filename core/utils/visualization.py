"""
Visualization utilities for training analysis.

Independent of training loop—can be called anytime.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class TrainingVisualizer:
    """Generate training plots from history JSON."""
    
    def __init__(self, plot_dir: Path):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(
        self,
        history: Dict,
        save_name: str = "training_curves.png",
        title: str = "Training Progress"
    ):
        """
        Plot loss, Dice, IoU curves.
        
        Args:
            history: Dict with keys: train_loss, val_loss, train_dice, val_dice, etc.
            save_name: Output filename
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")
        
        # Loss
        if history.get("train_loss") or history.get("val_loss"):
            axes[0, 0].plot(
                history.get("train_loss", []),
                label="Train",
                linewidth=2,
                marker="o",
                markersize=3
            )
            axes[0, 0].plot(
                history.get("val_loss", []),
                label="Val",
                linewidth=2,
                marker="s",
                markersize=3
            )
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].set_title("Loss Curve")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Dice
        if history.get("train_dice") or history.get("val_dice"):
            axes[0, 1].plot(
                history.get("train_dice", []),
                label="Train",
                linewidth=2,
                marker="o",
                markersize=3
            )
            axes[0, 1].plot(
                history.get("val_dice", []),
                label="Val",
                linewidth=2,
                marker="s",
                markersize=3
            )
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Dice")
            axes[0, 1].set_title("Dice Coefficient (Macro)")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        
        # IoU
        if history.get("train_iou") or history.get("val_iou"):
            axes[1, 0].plot(
                history.get("train_iou", []),
                label="Train",
                linewidth=2,
                marker="o",
                markersize=3
            )
            axes[1, 0].plot(
                history.get("val_iou", []),
                label="Val",
                linewidth=2,
                marker="s",
                markersize=3
            )
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("IoU")
            axes[1, 0].set_title("Intersection over Union (Macro)")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0, 1])
        
        # Class-wise Dice (Val only)
        class_dice = history.get("class_dice", {})
        if class_dice:
            for cls in sorted(class_dice.keys()):
                val_curve = class_dice[cls].get("val", [])
                if val_curve:
                    axes[1, 1].plot(
                        val_curve,
                        label=f"Class {cls}",
                        linewidth=2,
                        marker="o",
                        markersize=3
                    )
            
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Dice")
            axes[1, 1].set_title("Class-wise Dice (Validation)")
            axes[1, 1].legend(fontsize=8, loc="best")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        plot_file = self.plot_dir / save_name
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Plot saved: {plot_file}")
        plt.close()
    
    def plot_class_distribution(
        self,
        history: Dict,
        class_names: Dict[int, str] = None,
        save_name: str = "class_performance.png"
    ):
        """
        Plot per-class Dice and IoU.
        
        Args:
            history: Training history
            class_names: Mapping class index → name
            save_name: Output filename
        """
        if class_names is None:
            class_names = {i: f"Class {i}" for i in range(8)}
        
        class_dice = history.get("class_dice", {})
        
        if not class_dice:
            logger.warning("No class-wise metrics in history")
            return
        
        # Get final val metrics per class
        final_dice = {}
        for cls in sorted(class_dice.keys()):
            val_curve = class_dice[cls].get("val", [])
            if val_curve:
                final_dice[cls] = val_curve[-1]
        
        if not final_dice:
            logger.warning("No validation metrics available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Final Class-wise Performance", fontsize=16, fontweight="bold")
        
        # Dice bar plot
        classes = sorted(final_dice.keys())
        dice_vals = [final_dice[c] for c in classes]
        class_labels = [class_names.get(c, f"Class {c}") for c in classes]
        
        colors = plt.cm.RdYlGn(np.array(dice_vals))
        axes[0].barh(class_labels, dice_vals, color=colors)
        axes[0].set_xlabel("Dice Score")
        axes[0].set_title("Dice per Class (Validation)")
        axes[0].set_xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(dice_vals):
            axes[0].text(v + 0.02, i, f"{v:.3f}", va="center")
        
        # Learning curves per class
        for cls in classes:
            val_curve = class_dice[cls].get("val", [])
            if val_curve:
                axes[1].plot(
                    val_curve,
                    label=class_names.get(cls, f"Class {cls}"),
                    linewidth=2,
                    marker="o",
                    markersize=4
                )
        
        axes[1].set_xlabel("Validation Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].set_title("Class-wise Learning Curves (Validation)")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        plot_file = self.plot_dir / save_name
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Class performance plot saved: {plot_file}")
        plt.close()
    
    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: Dict[int, str] = None,
        save_name: str = "confusion_matrix.png"
    ):
        """Plot confusion matrix."""
        if class_names is None:
            class_names = {i: f"Class {i}" for i in range(confusion_matrix.shape[0])}
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Normalize for visualization
        cm_norm = confusion_matrix.astype(np.float32)
        cm_norm = cm_norm / (cm_norm.sum(axis=1, keepdims=True) + 1e-8)
        
        im = ax.imshow(cm_norm, cmap="Blues", aspect="auto")
        
        # Labels
        n_classes = confusion_matrix.shape[0]
        class_labels = [class_names.get(i, f"C{i}") for i in range(n_classes)]
        
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)
        
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Ground Truth Class")
        ax.set_title("Normalized Confusion Matrix")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion")
        
        plt.tight_layout()
        
        plot_file = self.plot_dir / save_name
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        logger.info(f"✓ Confusion matrix saved: {plot_file}")
        plt.close()