import torch
import torch.nn.functional as F
import numpy as np

class RobustSegmentationTracker:
    """
    Corrected publication-grade evaluation tracker. Handles slice-by-slice 
    individual arithmetic validation averages across 8 distinct classes.
    """
    def __init__(self, num_classes=8):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        # Store individual slice scores as lists to compute true sample means
        self.dice_by_class = [[] for _ in range(self.num_classes)]

    @torch.no_grad()
    def update(self, y_pred, y_true):
        """
        Updates matrix statistics across an active evaluation batch loop.
        """
        # Convert raw network logits into discrete class integer predictions
        preds_discrete = torch.argmax(y_pred, dim=1) # Shape: [Batch, H, W]
        
        batch_size = y_true.size(0)
        
        # Calculate isolated sample statistics explicitly to bypass ratio metrics corruption
        for b in range(batch_size):
            p_slice = preds_discrete[b]
            t_slice = y_true[b]
            
            for c in range(self.num_classes):
                p_mask = (p_slice == c)
                t_mask = (t_slice == c)
                
                # If class doesn't exist in ground truth, skip it to avoid zero-biasing the mean
                if not torch.any(t_mask):
                    continue
                    
                inter = (p_mask & t_mask).sum().item()
                denom = p_mask.sum().item() + t_mask.sum().item()
                
                # Compute arithmetic Dice for this specific slice
                slice_dice = (2.0 * inter) / denom if denom > 0 else 1.0
                self.dice_by_class[c].append(slice_dice)

    def get_foreground_mean_dice(self):
        fg_averages = []
        # Dynamically tracks up to whatever num_classes is provided (e.g., 7)
        for c in range(1, self.num_classes):
            scores = self.dice_by_class[c]
            if len(scores) > 0:
                fg_averages.append(np.mean(scores))
        return float(np.mean(fg_averages)) if fg_averages else 0.0

    def print_comprehensive_report(self, class_mapping):
        """Prints a clean, summary of your validation performance."""
        print("\n📋 MSc DISSERTATION MULTI-CLASS SEGMENTATION SCORE VAL PROFILE:")
        print("─" * 75)
        for idx in range(self.num_classes):
            cls_name = class_mapping.get(idx, f"Label {idx}")
            scores = self.dice_by_class[idx]
            
            mean_score = np.mean(scores) if len(scores) > 0 else 0.0
            print(f"   • {cls_name:<28} | Dice Score: {mean_score:.4f} | Seen in {len(scores)} slices")
        print("─" * 75)
        print(f"🌟 COMPUTED FOREGROUND DISEASE MEAN DICE: {self.get_foreground_mean_dice():.4f}")
        print("─" * 75)
