from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from src.core.paths import project_path


def read_json(path: str | Path, default: Any) -> Any:
    if not str(path):
        return default
    resolved = project_path(path)
    if not resolved.exists() or resolved.is_dir():
        return default
    with resolved.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = project_path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=resolved.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return resolved


@contextmanager
def json_transaction(
    path: str | Path,
    loader: Callable[[Path], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    resolved = project_path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    # Lock a stable sidecar: replacing the JSON file must not replace its lock.
    with resolved.with_name(resolved.name + ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            payload = loader(resolved)
            yield payload
            write_json(resolved, payload)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
