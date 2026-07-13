"""Quick test of core modules with manifest-aware dataset."""

import sys
sys.path.insert(0, "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL")

from core import (
    enforce_system_determinism,
    get_device_context,
    ILDDatasetSplit,
    list_available_models,
    get_model,
)

print("=" * 70)
print("Testing Core Modules (Manifest-Aware Dataset)")
print("=" * 70)

# Test determinism
enforce_system_determinism(42)
print("✓ Determinism enforced")

# Test device
device = get_device_context()
print(f"✓ Device: {device}")

# Test models
print(f"\n✓ Available models: {list_available_models()}")

for model_name in list_available_models():
    try:
        model = get_model(model_name, in_channels=1, out_channels=8)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ {model_name:20s} → {n_params:,} parameters")
    except Exception as e:
        print(f"  ✗ {model_name:20s} → ERROR: {e}")

# Test dataset with manifest
print(f"\n--- Testing Manifest-Aware Dataset ---")
try:
    print(f"\nLoading train split...")
    ds_train = ILDDatasetSplit(split="train", seed=42, verbose=True)
    print(f"✓ Train split: {len(ds_train)} slices")
    
    sample = ds_train[0]
    print(f"✓ Sample shapes: image={sample[0].shape}, mask={sample[1].shape}")
    
    print(f"\nLoading val split...")
    ds_val = ILDDatasetSplit(split="val", seed=42, verbose=True)
    print(f"✓ Val split: {len(ds_val)} slices")
    
    print(f"\nLoading test split...")
    ds_test = ILDDatasetSplit(split="test", seed=42, verbose=True)
    print(f"✓ Test split: {len(ds_test)} slices")
    
    # Test metadata (FIXED: no slicing with DataLoader)
    print(f"\n--- Testing Individual Samples ---")
    for i in range(min(3, len(ds_test))):
        img, msk = ds_test[i]
        print(f"Sample {i}: image={img.shape}, mask={msk.shape}")
    
except Exception as e:
    print(f"✗ Dataset test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ Core Module Tests Complete!")
print("=" * 70)