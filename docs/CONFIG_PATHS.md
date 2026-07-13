# Configuration Management

## Config File Locations

### Training Configuration
- **Location**: `configs/training.yaml`
- **Loaded by**: `core/config.py:load_yaml_config()`
- **Resolution**: Relative paths are resolved from `REPO_ROOT/configs/`

### How to Use

```python
from core import load_yaml_config

# Relative path (recommended)
config = load_yaml_config("training.yaml")

# Absolute path
config = load_yaml_config("/full/path/to/training.yaml")
```

### Default Config Search Path

```
REPO_ROOT / "configs" / {config_name}.yaml
```

Where `REPO_ROOT` = parent directory of `core/`

### Command Line Usage

```bash
# Uses default: configs/training.yaml
python scripts/train_baseline.py

# Custom config
python scripts/train_baseline.py --config my_custom_config.yaml
```

---

## Config Path Resolution Logic

| Input | Resolved To |
|-------|-------------|
| `training.yaml` | `REPO_ROOT/configs/training.yaml` |
| `experiments/exp1.yaml` | `REPO_ROOT/configs/experiments/exp1.yaml` |
| `/absolute/path/config.yaml` | `/absolute/path/config.yaml` (unchanged) |

---

## Files Involved

- `core/config.py`: Defines `load_yaml_config()` and `REPO_ROOT`
- `configs/`: Directory for all YAML config files
- `scripts/train_baseline.py`: Example usage with `--config` argument

---

## Troubleshooting

**Error**: `FileNotFoundError: Config file not found: .../configs/configs/training.yaml`

**Cause**: Path is being doubled (usually from `--config configs/training.yaml`)

**Fix**: Use just the filename: `--config training.yaml`

---

## Future: Config Registry

For managing multiple configs, consider:

```python
CONFIG_REGISTRY = {
    "baseline": "training.yaml",
    "augmented": "training_augmented.yaml",
    "large": "training_large_batch.yaml",
}

# Usage
config = load_yaml_config(CONFIG_REGISTRY["baseline"])
```