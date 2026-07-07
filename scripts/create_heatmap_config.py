from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY
from src.heatmap.config_template import build_heatmap_config_override, load_default_registry, render_yaml, write_json, write_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small heatmap config override from a registry match.")
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--base-config", default="src/heatmap/config_overhead_default.yaml")
    parser.add_argument("--output", type=Path, help="YAML output path. Defaults under outputs/heatmap_config_templates/.")
    parser.add_argument("--json-output", type=Path, help="Optional JSON copy of the generated config.")
    parser.add_argument("--output-dir", help="Override heatmap output_dir.")
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--stop-seconds", type=float)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Print YAML without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or ROOT / "outputs" / "heatmap_config_templates" / f"config_{args.match_id}.yaml"
    config = build_heatmap_config_override(
        load_default_registry(args.registry),
        args.match_id,
        base_config=args.base_config,
        output_dir=args.output_dir,
        start_seconds=args.start_seconds,
        stop_seconds=args.stop_seconds,
        sample_fps=args.sample_fps,
        duration_seconds=args.duration_seconds,
    )
    if args.dry_run:
        print(render_yaml(config), end="")
        return 0
    write_yaml(output.expanduser(), config)
    if args.json_output:
        write_json(args.json_output.expanduser(), config)
    print(f"wrote heatmap config: {output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
