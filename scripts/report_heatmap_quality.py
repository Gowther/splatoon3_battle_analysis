from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, display_path, iter_heatmap_matches, load_registry, resolve_project_path
from src.heatmap.trajectory_quality import (
    quality_from_registry_heatmap,
    status_from_checks,
    write_json,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report heatmap player-track quality from the data registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--match-id", action="append", default=[], help="Registry match id to report. May be repeated.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "heatmap_quality")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every selected heatmap passes.")
    return parser.parse_args()


def selected(match_id: str, selected_ids: list[str]) -> bool:
    return not selected_ids or match_id in selected_ids


def main() -> int:
    args = parse_args()
    registry_path = resolve_project_path(args.registry) or args.registry.expanduser()
    output_dir = args.output_dir.expanduser()
    registry = load_registry(registry_path)
    results = []

    for match, heatmap in iter_heatmap_matches(registry):
        if not selected(match["id"], args.match_id):
            continue
        metrics, checks = quality_from_registry_heatmap(heatmap)
        status = status_from_checks(checks)
        match_dir = output_dir / match["id"]
        quality_json = match_dir / "trajectory_quality.json"
        quality_report = match_dir / "trajectory_quality.md"
        payload = {
            "match_id": match["id"],
            "heatmap_id": heatmap.get("id", ""),
            "status": status,
            "metrics": metrics,
            "checks": checks,
        }
        write_json(quality_json, payload)
        write_markdown_report(quality_report, f"{match['id']} Trajectory Quality", metrics, checks)
        results.append(
            {
                "match_id": match["id"],
                "heatmap_id": heatmap.get("id", ""),
                "status": status,
                "quality_json": display_path(quality_json),
                "quality_report": display_path(quality_report),
                "track_rows": metrics["track_rows"],
                "gap_ratio": metrics["gap_ratio"],
                "jump_reset_ratio": metrics["jump_reset_ratio"],
            }
        )

    summary = {
        "registry": display_path(registry_path),
        "status": "failed" if any(result["status"] != "passed" for result in results) else "passed",
        "results": results,
    }
    summary_json = output_dir / "heatmap_quality_results.json"
    write_json(summary_json, summary)
    print(f"wrote heatmap quality summary: {summary_json}")
    for result in results:
        print(
            "- {match_id}: {status} rows={track_rows} gap_ratio={gap_ratio} jump_reset_ratio={jump_reset_ratio}".format(
                **result
            )
        )

    if args.strict and summary["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
