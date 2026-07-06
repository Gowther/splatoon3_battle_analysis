from __future__ import annotations

import datetime as dt
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def configure_environment() -> None:
    os.environ.setdefault("UV_CACHE_DIR", str(ROOT / ".cache" / "uv"))
    os.environ.setdefault("PIP_CACHE_DIR", str(ROOT / ".cache" / "pip"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
    os.environ.setdefault("PYTHONPYCACHEPREFIX", str(ROOT / ".cache" / "pycache"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else ROOT / resolved


def default_output_path(input_path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"{input_path.stem}_{timestamp}.csv"


def model_path(name: str) -> Path:
    return ROOT / "models" / name
