from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heatmap.annotation_ui import build_annotation_ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML helper for heatmap point annotation.")
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=ROOT / "outputs" / "heatmap_annotation_round1" / "annotation_template.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "heatmap_annotation_round1" / "annotation_ui.html",
    )
    parser.add_argument("--title", default="Heatmap Annotation Round")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_annotation_ui(args.annotation_csv.expanduser(), args.output.expanduser(), title=args.title)
    if args.json_output:
        args.json_output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.json_output.expanduser().write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"annotation UI status: {report['status']}")
    print(f"annotation UI: {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
