import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import MONAI's industrial-grade network portfolio natively
from monai.networks.nets import UNet, AttentionUnet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

# Direct imports from your custom workspace module
from dataset import RobustILDDataset

# Establish persistent cluster file directories
SCRATCH_DIR = os.path.expandvars("$SCRATCHDIR/ild_dataset_processed")
OUTPUT_DIR = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_MAPPING = {
    0: "background", 1: "healthy", 2: "emphysema", 3: "ground_glass", 4: "fibrosis", 
    5: "micronodules", 6: "consolidation", 7: "bronchial_wall_thickening", 
    8: "reticulation", 9: "macronodules", 10: "cysts", 11: "peripheral_micronodules", 
    12: "bronchiectasis", 13: "air_trapping", 14: "early_fibrosis", 
    15: "increased_attenuation", 16: "tuberculosis", 17: "pcp"
}

def build_model(model_name):
    """Dynamically instantiates and returns the requested network topology."""
    if model_name == "standard_unet":
        return UNet(
            spatial_dims=2, in_channels=2, out_channels=18,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=0
        )
    elif model_name == "attention_unet":
        return AttentionUnet(
            spatial_dims=2, in_channels=2, out_channels=18,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2)
        )
    elif model_name == "r2_unet":
        # Setting num_res_units > 0 inside MONAI's UNet enables internal residual/recurrent paths
        return UNet(
            spatial_dims=2, in_channels=2, out_channels=18,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2
        )
    elif model_name == "attention_residual_unet":
        # Combines the internal recurrent residual blocks with top-level attention mechanisms
        return AttentionUnet(
            spatial_dims=2, in_channels=2, out_channels=18,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2)
        )
    else:
        raise ValueError(f"Unknown architectural configuration requested: {model_name}")

def train_and_evaluate(model_name, epochs=20):
    print(f"\n🚀 Starting Evaluation Pipeline for Architecture: {model_name.upper()}")
    
    # Isolate data pools (Focusing train on dense annotated targets to bypass lazy background learning)
    train_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, skip_empty_masks=True)
    val_ds = RobustILDDataset(scratch_root=SCRATCH_DIR, skip_empty_masks=False)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    
    # Hybrid Clinical Loss: Cross-Entropy handles pixel probabilities, Dice scales class boundaries
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Initialise separate evaluation objects for overall and class-wise tracking
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric_per_class = DiceMetric(include_background=False, reduction="mean_batch")
    
    best_metric = -1
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            # Add singleton channel dimension to targets for matching loss inputs
            labels_onehot = labels.unsqueeze(1) 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation Pass
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data[0].to(device), val_data[1].to(device)
                val_outputs = model(val_inputs)
                
                # Convert logit outputs to discrete prediction masks via argmax
                val_outputs_discrete = torch.argmax(val_outputs, dim=1, keepdim=True)
                val_labels_onehot = val_labels.unsqueeze(1)
                
                # Perform MONAI one-hot conversions for pure metric scoring
                from monai.networks.utils import one_hot
                val_outputs_onehot = one_hot(val_outputs_discrete, num_classes=18)
                val_labels_onehot_converted = one_hot(val_labels_onehot, num_classes=18)
                
                dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot_converted)
                dice_metric_per_class(y_pred=val_outputs_onehot, y=val_labels_onehot_converted)
                
            mean_dice = dice_metric.aggregate().item()
            class_dices = dice_metric_per_class.aggregate().cpu().numpy()
            
            dice_metric.reset()
            dice_metric_per_class.reset()
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss/len(train_loader):.4f} | Val Mean Dice: {mean_dice:.4f}")
            
            if mean_dice > best_metric:
                best_metric = mean_dice
                # Save the elite weights file for this specific network model
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_weights_{model_name}.pt"))
                
                # Construct final class-wise data metrics row
                report_row = {"Architecture": model_name, "Global_Mean_Dice": mean_dice}
                for cls_idx in range(1, 18): # Ignore index 0 background tissue
                    cls_name = CLASS_MAPPING[cls_idx]
                    # Subtract 1 because include_background=False maps array indices 0-16 to classes 1-17
                    report_row[f"Dice_{cls_name}"] = class_dices[cls_idx - 1]
                    
                df_report = pd.DataFrame([report_row])
                df_report.to_csv(os.path.join(OUTPUT_DIR, f"results_{model_name}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Network architecture label")
    args = parser.parse_args()
    train_and_evaluate(args.model)
