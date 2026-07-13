# core/config.py
import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any

# ─── 📂 GLOBAL REPOSITORY CORE ROADMAPS ───
SCRATCH_DIR = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
OUTPUT_ROOT = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/artifacts"
REPO_ROOT = Path(__file__).parent.parent
GLOBAL_SEED = 42

CLASS_MAPPING = {
    0: "background", 1: "healthy_control", 2: "emphysema", 3: "ground_glass", 
    4: "fibrosis", 5: "micronodules", 6: "consolidation", 7: "other_rare_pathologies"
}
NUM_CLASSES = len(CLASS_MAPPING)  # 8 classes

# Loss tuning constants
BACKGROUND_WEIGHT_SCALE = 0.1
MIN_CLASS_WEIGHT = 0.5
MAX_CLASS_WEIGHT = 20.0

def enforce_system_determinism(seed: int = GLOBAL_SEED) -> None:
    """Enforce reproducibility across random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_context() -> torch.device:
    """Return available device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dict_to_namespace_recursive(data):
    """
    Recursively convert dict to SimpleNamespace.
    
    Keeps dicts with non-string keys as dicts (e.g., class_weights: {0: 0.01, 1: 10.0})
    """
    if not isinstance(data, dict):
        return data
    
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Check if this dict has non-string keys
            has_non_string_keys = any(not isinstance(k, str) for k in value.keys())
            
            if has_non_string_keys:
                # Keep as dict, recurse on values only
                result[key] = {
                    k: _dict_to_namespace_recursive(v) 
                    for k, v in value.items()
                }
            else:
                # All keys are strings - convert to SimpleNamespace
                converted_value = _dict_to_namespace_recursive(value)
                if isinstance(converted_value, dict):
                    result[key] = SimpleNamespace(**converted_value)
                else:
                    result[key] = converted_value
        else:
            result[key] = value
    
    return result

def load_yaml_config(config_path: str) -> SimpleNamespace:
    """
    Load YAML configuration file and return as SimpleNamespace.
    
    Args:
        config_path: Path to YAML config file (relative to repo root or absolute)
    
    Returns:
        SimpleNamespace object with config keys as attributes
    """
    if not os.path.isabs(config_path):
        config_path = REPO_ROOT / "configs" / config_path
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    
    if cfg_dict is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    # Recursively convert to SimpleNamespace
    converted = _dict_to_namespace_recursive(cfg_dict)
    
    return SimpleNamespace(**converted)


# Verify key directories exist
os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(REPO_ROOT / "configs", exist_ok=True)
os.makedirs(REPO_ROOT / "artifacts" / "logs", exist_ok=True)
os.makedirs(REPO_ROOT / "artifacts" / "checkpoints", exist_ok=True)
os.makedirs(REPO_ROOT / "artifacts" / "synthetic", exist_ok=True)
