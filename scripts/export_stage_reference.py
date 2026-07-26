from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.config_loader import load_config, resolve_path
from src.heatmap.stage_reference import DEFAULT_GRID_DIVISIONS, DEFAULT_OUTPUT_ROOT, build_reference_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export grid-annotated reference frames and a control-point draft for stage landmark labeling."
    )
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml", help="Heatmap config YAML.")
    parser.add_argument("--stage-id", required=True, help="Stage identifier for the package and draft asset.")
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--times",
        help="Extra comma-separated reference times in seconds, added to the config reference time.",
    )
    parser.add_argument("--grid-divisions", type=int, default=DEFAULT_GRID_DIVISIONS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_times = [float(item) for item in args.times.split(",") if item.strip()] if args.times else None
    manifest = build_reference_package(
        load_config(resolve_path(args.config)),
        config_path=args.config,
        stage_id=args.stage_id,
        output_root=args.output_root,
        extra_times=extra_times,
        divisions=args.grid_divisions,
    )
    print(f"stage reference status: {manifest['status']}")
    print(f"exported frames: {manifest['exported_frames']}")
    print(f"draft asset: {manifest['draft_asset']}")
    print(f"guide: {manifest['guide']}")
    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
