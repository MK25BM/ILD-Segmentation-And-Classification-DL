import torch

PRETRAINED_PATH = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/pretrained_checkpoints/best_model.pt"
checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu")

print("=== 🔍 OPENING NESTED TENSOR STRUCTURES ===")
# Isolate the nested model weights dictionary
if "model" in checkpoint:
    model_weights = checkpoint["model"]
    keys = list(model_weights.keys())
    print(f"Total Model Parameter Layers Found: {len(keys)}")
    
    print("\nFirst 20 Layer Keys inside the weights tree:")
    for k in keys[:20]:
        print(f"  • {k}")
        
    # Check if there are explicit cross-attention channels
    conditioning_keys = [k for k in keys if "encoder" in k or "attn" in k or "class" in k]
    print("\n🎯 Target Layers:")
    for ck in conditioning_keys[:15]:
        print(f"  • {ck} | Shape: {model_weights[ck].shape}")
else:
    print("Could not locate a nested 'model' key inside this file.")
