from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, load_registry, resolve_project_path
from src.heatmap.anomaly_export import export_anomalies


DEFAULT_CONFIG = ROOT / "config" / "annotation_samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export heatmap anomaly frames for manual review.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "heatmap_anomalies")
    parser.add_argument("--match-id", action="append", default=[], help="Registry match id to include. May be repeated.")
    parser.add_argument("--low-confidence", type=float)
    parser.add_argument("--large-step-px", type=float)
    parser.add_argument("--max-items-per-match", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    path = resolve_project_path(path) or path.expanduser()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    defaults = config.get("defaults", {})
    registry = load_registry(args.registry)
    match_ids = args.match_id or config.get("heatmap_matches", [])
    summary = export_anomalies(
        registry,
        args.output_dir.expanduser(),
        match_ids=match_ids,
        low_confidence=args.low_confidence
        if args.low_confidence is not None
        else float(defaults.get("anomaly_low_confidence", 0.56)),
        large_step_px=args.large_step_px
        if args.large_step_px is not None
        else float(defaults.get("anomaly_large_step_px", 420.0)),
        max_items_per_match=args.max_items_per_match
        if args.max_items_per_match is not None
        else int(defaults.get("anomaly_max_items_per_match", 24)),
    )
    print(f"wrote anomaly package: {args.output_dir.expanduser()}")
    print(f"total exported: {summary['total_exported']}")
    for match in summary["matches"]:
        print(f"- {match['match_id']}: {match['exported']} {match['by_type']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
