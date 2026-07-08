from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.paths import configure_environment
from src.report_io import strict_exit_code, write_json_report, write_text_report
from src.yolov5_vendor import DEFAULT_YOLOV5_ROOT, build_vendor_report, render_markdown


configure_environment()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the local YOLOv5 vendor/runtime boundary.")
    parser.add_argument("--root", type=Path, default=DEFAULT_YOLOV5_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "yolov5_vendor.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "yolov5_vendor.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless the vendor boundary passes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_vendor_report(args.root)
    write_text_report(args.output, render_markdown(report))
    write_json_report(args.json_output, report)
    print(f"yolov5 vendor status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"passed"})


if __name__ == "__main__":
    raise SystemExit(main())
