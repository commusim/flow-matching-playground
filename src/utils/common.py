import json
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import torch
import yaml


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def project_root():
    return Path(__file__).resolve().parents[2]


def load_config(path=None):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def create_run_dir(pipeline, seed, output_root=None):
    root = Path(output_root) if output_root else project_root() / "outputs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / pipeline / f"{stamp}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_metadata(run_dir, config, metrics):
    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
