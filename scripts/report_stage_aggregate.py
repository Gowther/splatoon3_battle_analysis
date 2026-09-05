from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.config_loader import load_config, resolve_path
from src.heatmap.render_stage_space import DEFAULT_CANVAS_SIZE, DEFAULT_MARGIN
from src.heatmap.run_pipeline import stage_tracks_csv_path
from src.heatmap.stage_aggregate import build_stage_aggregate, render_markdown
from src.heatmap.stage_registry import DEFAULT_REGISTRY_PATH, load_stage_registry, stage_entry
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate stage-space heatmaps across every match registered on one stage."
    )
    parser.add_argument("--stage-id", required=True, help="Stage id from the stage registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=[],
        help="Heatmap config YAML for a registered match. May be repeated. Defaults to every src/heatmap/config_*.yaml.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "stage_aggregate")
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--output", type=Path, help="Optional Markdown report.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON report.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the aggregate is comparable.")
    return parser.parse_args()


def default_config_paths() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "src" / "heatmap").glob("config_*.yaml")
        if path.name != "config_overhead_default.yaml"
    )


def main() -> int:
    args = parse_args()
    registry = load_stage_registry(args.registry)
    entry = stage_entry(registry, args.stage_id)
    if entry is None:
        print(f"stage not registered: {args.stage_id}")
        print("register matches first: scripts/report_stage_registry.py --register <stage_id> <match_id>")
        return 1 if args.strict else 0

    wanted = set(entry.get("matches", []))
    paths = [resolve_path(item) for item in args.configs] if args.configs else default_config_paths()
    stage_csv_paths: dict[str, Path] = {}
    reference_config: dict | None = None
    for path in paths:
        config = load_config(path)
        match_id = str(config.get("match", {}).get("id", "")).strip()
        if match_id in wanted:
            stage_csv_paths[match_id] = stage_tracks_csv_path(config)
            reference_config = reference_config or config

    if reference_config is None:
        print(f"no heatmap config found for any match on stage {args.stage_id}")
        return 1 if args.strict else 0

    report = build_stage_aggregate(
        args.stage_id,
        registry,
        stage_csv_paths,
        reference_config,
        args.output_dir / args.stage_id,
        canvas_size=args.canvas_size,
        margin=args.margin,
    )

    if args.output:
        write_text_report(args.output.expanduser(), render_markdown(report))
    if args.json_output:
        write_json_report(args.json_output.expanduser(), report)

    print(f"stage aggregate: {report['status']} ({report['match_count']} matches)")
    if report["missing_matches"]:
        print(f"missing stage coordinates: {', '.join(report['missing_matches'])}")
    for match_id, reason in report["rejected_matches"].items():
        print(f"excluded {match_id}: {reason}")
    if report["rendered"]:
        print(f"rendered {len(report['rendered'])} images -> {report['output_dir']}")
    elif report["status"] != "ready":
        print("label and normalize at least two matches on this stage to compare them")
    return strict_exit_code(report["status"], args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
