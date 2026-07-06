from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, load_registry, resolve_project_path
from src.heatmap.annotation_samples import export_annotation_package


DEFAULT_CONFIG = ROOT / "config" / "annotation_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a manual heatmap annotation package.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "annotation_samples")
    parser.add_argument("--match-id", action="append", default=[], help="Registry match id to include. May be repeated.")
    parser.add_argument("--frames-per-match", type=int, help="Override sampled frames per match.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = resolve_project_path(path) or path.expanduser()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    match_ids = args.match_id or config.get("heatmap_matches", [])
    frames_per_match = args.frames_per_match or int(defaults.get("frames_per_match", 5))
    manifest = export_annotation_package(
        registry,
        args.output_dir.expanduser(),
        match_ids=match_ids,
        frames_per_match=frames_per_match,
    )
    print(f"wrote annotation package: {args.output_dir.expanduser()}")
    print(f"annotation rows: {sum(match['annotation_rows'] for match in manifest['matches'])}")
    for match in manifest["matches"]:
        print(f"- {match['match_id']}: frames={len(match['frames'])} rows={match['annotation_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
