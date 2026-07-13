"""Checkpoint management utilities."""

import torch
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        epoch: int,
        model_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Dict,
        history: Dict,
        is_best: bool = False,
        is_final: bool = False,
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            history: Training history dict
            is_best: Whether this is best model
            is_final: Whether this is final model
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "history": history,
        }
        
        if is_final:
            filename = self.checkpoint_dir / "final_model.pt"
        elif is_best:
            filename = self.checkpoint_dir / "best_model.pt"
        else:
            filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        
        torch.save(checkpoint, filename)
        logger.info(f"✓ Checkpoint saved: {filename}")
        
        return filename
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dict
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"✓ Checkpoint loaded: {checkpoint_path}")
        return checkpoint