from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_registry import DEFAULT_REGISTRY, load_registry, resolve_project_path
from src.heatmap.config_validation import config_paths_from_registry, validate_heatmap_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate heatmap YAML configs after base_config expansion.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", action="append", default=[], help="Specific heatmap config path. May be repeated.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any config fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.config:
        paths = [resolve_project_path(path) or Path(path).expanduser() for path in args.config]
    else:
        paths = config_paths_from_registry(load_registry(args.registry))
    report = validate_heatmap_configs(paths)
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote heatmap config validation: {output}")
    print(f"heatmap config validation status: {report['status']}")
    for item in report["configs"]:
        print(f"- {item['path']}: {item['status']}")
        for problem in item["problems"]:
            print(f"  - {problem}")
    return 1 if args.strict and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
