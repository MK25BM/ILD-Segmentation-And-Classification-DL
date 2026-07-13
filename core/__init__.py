"""Core module for ILD segmentation pipeline."""

from .config import (
    load_yaml_config,
    enforce_system_determinism,
    get_device_context,
    SCRATCH_DIR,
    OUTPUT_ROOT,
    REPO_ROOT,
    NUM_CLASSES,
    CLASS_MAPPING,
    GLOBAL_SEED,
)

from .models import (
    build_architecture,
    get_model,
    list_available_models,
)

from .dataset import ILDDataset, ILDDatasetSplit

__all__ = [
    # Config
    "load_yaml_config",
    "enforce_system_determinism",
    "get_device_context",
    "SCRATCH_DIR",
    "OUTPUT_ROOT",
    "REPO_ROOT",
    "NUM_CLASSES",
    "CLASS_MAPPING",
    "GLOBAL_SEED",
    # Models
    "build_architecture",
    "get_model",
    "list_available_models",
    # Dataset
    "ILDDataset",
    "ILDDatasetSplit",
]