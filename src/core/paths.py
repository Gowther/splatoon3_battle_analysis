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


def resolve_project_path(value: str | Path | None, root: Path = ROOT) -> Path | None:
    if value in (None, ""):
        return None
    resolved = Path(value).expanduser()
    return resolved if resolved.is_absolute() else root / resolved


def display_path(path: str | Path | None, root: Path = ROOT) -> str:
    if path is None:
        return ""
    resolved = Path(path).expanduser()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def relative_asset_path(
    value: str | Path | None,
    document_path: str | Path,
    root: Path = ROOT,
) -> str:
    target = resolve_project_path(value, root)
    if target is None:
        return ""
    document = resolve_project_path(document_path, root)
    if document is None:
        raise ValueError("document_path must not be empty")
    target = target.resolve()
    document_parent = document.resolve().parent
    try:
        return Path(os.path.relpath(target, start=document_parent)).as_posix()
    except ValueError:
        return target.as_uri()


def default_output_path(input_path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "outputs" / f"{input_path.stem}_{timestamp}.csv"


def model_path(name: str) -> Path:
    return ROOT / "models" / name
