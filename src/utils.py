"""
Utility helpers: reproducibility, device selection, logging, GPU monitoring.
"""

import json
import logging
import os
import random
from datetime import datetime, timezone

import numpy as np
import torch

from . import config


def seed_everything(seed: int = config.SEED) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Prefer throughput-oriented cuDNN kernels for fixed-size training inputs.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# Backward-compatible alias (train_pipeline.py imports set_seed)
set_seed = seed_everything


def get_device() -> torch.device:
    """Return the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage (useful for batch-size tuning)."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    msg = f"GPU Memory — Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Total: {total:.1f}GB"
    if prefix:
        msg = f"{prefix} | {msg}"
    print(msg)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create a configured logger with console and optional file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def write_run_manifest(path: str, payload: dict, merge: bool = False) -> None:
    """
    Write a JSON run manifest (hyperparameters, stage markers, eval metrics).

    When ``merge`` is True, shallow-merge ``payload`` into an existing JSON object
    at ``path`` (non-dict existing content is replaced). Parent directories are created.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    base: dict = {}
    if merge and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                base = loaded
        except (json.JSONDecodeError, OSError):
            base = {}

    merged = {**base, **payload}
    merged["_updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
