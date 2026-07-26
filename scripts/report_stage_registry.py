from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.config_loader import load_config, resolve_path
from src.heatmap.stage_registry import (
    DEFAULT_MAX_DISAGREEMENT,
    DEFAULT_REGISTRY_PATH,
    build_stage_registry_report,
    load_stage_registry,
    register_match,
    render_registry_markdown,
    write_stage_registry,
)
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report stage control-point reuse and cross-validate labelings of the same stage."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=[],
        help="Heatmap config YAML to include. May be repeated. Defaults to every src/heatmap/config_*.yaml with a match id.",
    )
    parser.add_argument("--register", nargs=2, metavar=("STAGE_ID", "MATCH_ID"), help="Attach a match to a stage and exit.")
    parser.add_argument("--asset", default="", help="Control-point asset to record when using --register.")
    parser.add_argument("--max-disagreement", type=float, default=DEFAULT_MAX_DISAGREEMENT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "stage_registry.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "stage_registry.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless every stage passes.")
    return parser.parse_args()


def default_config_paths() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "src" / "heatmap").glob("config_*.yaml")
        if path.name != "config_overhead_default.yaml"
    )


def load_configs(config_args: list[str]) -> dict[str, dict]:
    paths = [resolve_path(item) for item in config_args] if config_args else default_config_paths()
    configs: dict[str, dict] = {}
    for path in paths:
        config = load_config(path)
        match_id = str(config.get("match", {}).get("id", "")).strip()
        if match_id:
            configs[match_id] = config
    return configs


def main() -> int:
    args = parse_args()
    registry = load_stage_registry(args.registry)

    if args.register:
        stage_id, match_id = args.register
        registry = register_match(registry, stage_id, match_id, asset_path=args.asset)
        written = write_stage_registry(registry, args.registry)
        print(f"registered {match_id} under stage {stage_id}: {written}")
        return 0

    configs = load_configs(args.configs)
    report = build_stage_registry_report(registry, configs, max_disagreement=args.max_disagreement)

    write_text_report(args.output.expanduser(), render_registry_markdown(report))
    write_json_report(args.json_output.expanduser(), report)

    print(f"stage registry: {report['status']} ({report['stage_count']} stages)")
    if report["unregistered_matches"]:
        print(f"unregistered matches: {', '.join(report['unregistered_matches'])}")
    for stage in report["stages"]:
        if stage["status"] != "ready":
            print(f"- {stage['stage_id']}: {stage['status']} {stage['disagreeing_matches'] or ''}".rstrip())
    return strict_exit_code(report["status"], args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
