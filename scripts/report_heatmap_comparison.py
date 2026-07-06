from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, display_path, load_registry, resolve_project_path
from src.heatmap.comparison_report import build_comparison_report, render_markdown, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare heatmap trajectory quality across registered matches.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--match-id", action="append", default=[], help="Registry match id to include. May be repeated.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "heatmap_comparison.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "heatmap_comparison.json")
    parser.add_argument("--low-confidence", type=float, default=0.56)
    parser.add_argument("--large-step-px", type=float, default=420.0)
    parser.add_argument("--max-anomaly-samples", type=int, default=8)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every selected heatmap passes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = resolve_project_path(args.registry) or args.registry.expanduser()
    registry = load_registry(registry_path)
    report = build_comparison_report(
        registry,
        match_ids=args.match_id or None,
        low_confidence=args.low_confidence,
        large_step_px=args.large_step_px,
        max_anomaly_samples=args.max_anomaly_samples,
    )

    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    write_json(args.json_output.expanduser(), report)
    print(f"wrote heatmap comparison: {output}")
    print(f"wrote heatmap comparison json: {args.json_output}")
    print(f"heatmap comparison status: {report['status']}")
    for match in report["matches"]:
        metrics = match["metrics"]
        print(
            "- {match_id}: {status} teams={teams} rows={rows} gap={gap} jump={jump} anomalies={anomalies}".format(
                match_id=match["match_id"],
                status=match["status"],
                teams=",".join(match.get("teams", [])),
                rows=metrics["track_rows"],
                gap=metrics["gap_ratio"],
                jump=metrics["jump_reset_ratio"],
                anomalies=match["anomalies"]["total"],
            )
        )

    if args.strict and report["status"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
