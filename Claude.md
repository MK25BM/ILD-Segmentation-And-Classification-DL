# Claude.md — ILD Segmentation & DDPM Augmentation Pipeline

## Project Overview
**Goal**: Develop a modular deep learning pipeline for **Interstitial Lung Disease (ILD) HRCT image segmentation** using U-Net variants, augmented with **DDPM synthetic data generation** to improve Dice scores.

**Dataset**: HUG ILD HRCT (processed)  
**Data Location**: `/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed`  
**Repository Root**: `/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL`

---

## Target Repository Structure

```
ILD-Segmentation-And-Classification-DL/
├── core/                          # Core modular pipeline
│   ├── __init__.py
│   ├── config.py                  # Config + system setup
│   ├── dataset.py                 # ILD HRCT dataset loader
│   ├── models.py                  # U-Net factory & variants
│   ├── preprocess.py              # Image preprocessing
│   └── augmentation/
│       └── ddpm.py                # DDPM augmentor wrapper
│
├── utils/
│   ├── __init__.py
│   ├── logging.py                 # Logging utilities
│   ├── metrics.py                 # Dice, IoU, sensitivity, specificity
│   ├── visualization.py           # Segmentation visualization
│   └── stitching.py               # Patch stitching for large images
│
├── configs/                       # YAML configuration files
│   ├── unet.yaml
│   ├── attention_unet.yaml
│   ├── r2unet.yaml
│   ├── ddpm.yaml
│   ├── training.yaml
│   └── evaluation.yaml
│
├── scripts/                       # Orchestration scripts
│   ├── run_training.py            # Train segmentation model
│   ├── run_ddpm.py                # Generate synthetic data via DDPM
│   └── evaluate_holdout.py        # Evaluate on test set
│
├── experiments/                   # Experiment runners
│   ├── cv_experiments.py          # Cross-validation pipeline
│   ├── ablation_studies.py        # Model variant comparisons
│   └── synthetic_experiments.py   # DDPM augmentation impact
│
├── tests/                         # Unit & integration tests
│   ├── test_metrics.py
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_pipeline_integrity.py
│
├── artifacts/                     # Outputs & results
│   ├── logs/                      # Training logs
│   ├── figures/                   # Visualizations
│   ├── checkpoints/               # Model weights
│   ├── synthetic/                 # DDPM-generated images
│   ├── manifests/                 # Reproducibility manifests
│   └── evaluations/               # Metric results
│
├── docs/                          # Documentation
│   ├── architecture.md
│   ├── dataset.md
│   ├── training_pipeline.md
│   ├── evaluation_pipeline.md
│   └── ddpm_pipeline.md
│
├── archive/                       # Legacy reference code
│   ├── lung_ddpm_plus_src/        # DDPM backbone (reference)
│   ├── Lung_Segmentation/         # U-Net implementations (reference)
│   └── ...                        # Other legacy files
│
├── Claude.md                      # This file
├── README.md
└── submit_job.sh                  # HPC job submission
```

---

## Key Integration Points

### 1. DDPM Backbone (Reference: `/archive/lung_ddpm_plus_src/`)
- **Diffusion Model**: `diffusion_model/unet.py` → `UNetModel`, `create_model()`
- **Trainer**: `diffusion_model/trainer.py` → `Trainer`, `GaussianDiffusion`
- **Sampling**: `sample.py` → reuse sampling logic for synthetic generation
- **Integration**: `core/augmentation/ddpm.py` wraps these as a callable augmentor

### 2. Segmentation Models (Reference: `/archive/Lung_Segmentation/`)
- **U-Net variants**: `U_Net`, `AttU_Net`, `R2U_Net`, `R2AttU_Net`
- **Integration**: `core/models.py` imports & exposes factory: `get_model(name, **kwargs)`

### 3. Dataset (Custom: Your ILD HRCT)
- **Location**: `/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed`
- **Classes** (8 classes, defined in `core/config.py`):
  ```python
  {0: "background", 1: "healthy_control", 2: "emphysema", 
   3: "ground_glass", 4: "fibrosis", 5: "micronodules", 
   6: "consolidation", 7: "other_rare_pathologies"}
  ```
- **Integration**: `core/dataset.py` → `ILDDataset` class with train/val/test split

---

## Development Roadmap

### Phase 1: Modularization ✓ (in progress)
- [ ] Scaffold `core/`, `utils/`, `configs/`, `scripts/`, `tests/`
- [ ] Create `core/dataset.py` → load ILD HRCT from `SCRATCH_DIR`
- [ ] Create `core/models.py` → factory for U-Net variants
- [ ] Create `core/augmentation/ddpm.py` → wraps diffusion sampling

### Phase 2: Configuration & Orchestration
- [ ] Write YAML configs: `configs/unet.yaml`, `configs/ddpm.yaml`, etc.
- [ ] Implement `core/config.py` YAML loader (typed configs)
- [ ] Build `scripts/run_training.py` → config-driven training loop
- [ ] Build `scripts/run_ddpm.py` → config-driven DDPM sampling

### Phase 3: Testing & Utilities
- [ ] Implement `utils/metrics.py` → Dice, IoU, sensitivity, specificity
- [ ] Implement `utils/visualization.py` → overlay segmentation on HRCT
- [ ] Write unit tests in `tests/`
- [ ] Add logging, checkpointing, manifest generation

### Phase 4: Experiments & Evaluation
- [ ] Implement `experiments/cv_experiments.py` → k-fold cross-validation
- [ ] Implement `experiments/synthetic_experiments.py` → compare Dice with/without DDPM
- [ ] Implement `scripts/evaluate_holdout.py` → final evaluation
- [ ] Generate artifacts (logs, figures, checkpoints, manifests)

### Phase 5: Documentation & Reproducibility
- [ ] Write `docs/architecture.md`, `docs/dataset.md`, etc.
- [ ] Add README with setup, usage, results
- [ ] Archive old code reference links in this file

---

## Current Config Values (from `core/config.py`)

```python
SCRATCH_DIR = "/scratch/u6dm/mk25bm.u6dm/ild_dataset_processed"
OUTPUT_ROOT = "/projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL/benchmarks_cv"
GLOBAL_SEED = 42

CLASS_MAPPING = {
    0: "background", 1: "healthy_control", 2: "emphysema", 
    3: "ground_glass", 4: "fibrosis", 5: "micronodules", 
    6: "consolidation", 7: "other_rare_pathologies"
}
NUM_CLASSES = 8

# Loss tuning
BACKGROUND_WEIGHT_SCALE = 0.1
MIN_CLASS_WEIGHT = 0.5
MAX_CLASS_WEIGHT = 20.0
```

---

## Key Commands

```bash
# Archive legacy code (one-time setup)
cd /projects/u6dm/mk25bm.u6dm/ILD-Segmentation-And-Classification-DL
mkdir -p archive
# (move legacy folders)

# Run training
python scripts/run_training.py --config configs/training.yaml

# Generate synthetic data via DDPM
python scripts/run_ddpm.py --config configs/ddpm.yaml --num_samples 500

# Evaluate on holdout test set
python scripts/evaluate_holdout.py --config configs/evaluation.yaml

# Run cross-validation experiment
python experiments/cv_experiments.py --config configs/training.yaml --folds 5

# Run tests
pytest tests/
```

---

## AI Assistant Notes

**Claude (GitHub Copilot)**: Use this file as your project context. Before making changes:
1. Verify the target file location and phase
2. Reference `/archive/` for legacy implementations (read-only)
3. Ensure new code follows the modular structure
4. Update this file if roadmap changes or new integration points emerge

**Last Updated**: July 8, 2026