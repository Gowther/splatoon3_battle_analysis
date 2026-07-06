from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUPS = {
    "models": ROOT / "models",
    "footages": ROOT / "footages",
    "weapon_icons": ROOT / "main_icons",
    "weapon_dataset": ROOT / "main_training_dataset",
    "outputs": ROOT / "outputs",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file())


def file_record(path: Path, include_hash: bool) -> dict[str, object]:
    stat = path.stat()
    record: dict[str, object] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
    }
    if include_hash:
        record["sha256"] = sha256(path)
    return record


def build_inventory(include_hash: bool = False) -> dict[str, object]:
    groups = {}
    for name, path in DEFAULT_GROUPS.items():
        files = [file_record(item, include_hash) for item in iter_files(path)]
        groups[name] = {
            "root": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(int(item["bytes"]) for item in files),
        }
    return {"project_root": str(ROOT), "groups": groups}


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory local project assets.")
    parser.add_argument("--hash", action="store_true", help="Include SHA-256 hashes for all files.")
    parser.add_argument("--output", type=Path, help="Write inventory JSON to this path.")
    args = parser.parse_args()

    inventory = build_inventory(args.hash)
    payload = json.dumps(inventory, indent=2, ensure_ascii=False)
    if args.output:
        output = args.output.expanduser()
        if not output.is_absolute():
            output = ROOT / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote inventory: {output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
