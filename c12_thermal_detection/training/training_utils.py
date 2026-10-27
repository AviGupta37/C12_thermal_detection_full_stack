"""
Training utilities: configuration loading, hardware checks, and dataset verification.
"""

from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import yaml
try:
    import torch
except ImportError:
    torch = None


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def check_cuda_environment() -> Tuple[bool, str, Optional[float]]:
    """
    Check CUDA availability, device name, and total VRAM in GB.
    Returns: (is_available, description, vram_gb)
    """
    if torch is None or not torch.cuda.is_available():
        return False, "CPU only (PyTorch / CUDA unavailable)", None
    
    dev_idx = 0
    dev_name = torch.cuda.get_device_name(dev_idx)
    props = torch.cuda.get_device_properties(dev_idx)
    vram_gb = props.total_memory / (1024 ** 3)
    desc = f"{dev_name} ({vram_gb:.1f} GB VRAM)"
    return True, desc, vram_gb


def validate_training_dataset(data_yaml_path: Path) -> Tuple[bool, str]:
    """
    Validate that data.yaml exists and points to non-empty train and val directories.
    Prevents silent failures or training on empty datasets.
    """
    if not data_yaml_path.exists():
        return False, f"Dataset definition YAML does not exist: {data_yaml_path}"

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    if not data_cfg or "train" not in data_cfg or "val" not in data_cfg:
        return False, "data.yaml must define both 'train' and 'val' split paths"

    base_dir = data_yaml_path.parent
    if "path" in data_cfg and data_cfg["path"]:
        cfg_path = Path(data_cfg["path"])
        if cfg_path.is_absolute():
            base_dir = cfg_path
        else:
            base_dir = (data_yaml_path.parent / cfg_path).resolve()
    
    # Resolve train images path
    train_path = Path(data_cfg["train"])
    if not train_path.is_absolute():
        train_path = (base_dir / train_path).resolve()
        
    val_path = Path(data_cfg["val"])
    if not val_path.is_absolute():
        val_path = (base_dir / val_path).resolve()

    if not train_path.exists():
        return False, f"Training images directory not found: {train_path}"

    if not val_path.exists():
        return False, f"Validation images directory not found: {val_path}"

    # Verify train directory contains at least some images
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    train_images = [p for p in train_path.iterdir() if p.suffix.lower() in valid_extensions]
    val_images = [p for p in val_path.iterdir() if p.suffix.lower() in valid_extensions]

    if len(train_images) == 0:
        return False, f"Training directory is empty: {train_path}. Cannot train on 0 samples."

    if len(val_images) == 0:
        return False, f"Validation directory is empty: {val_path}. Cannot evaluate on 0 samples."

    return True, f"Valid dataset: {len(train_images)} train images, {len(val_images)} val images."
