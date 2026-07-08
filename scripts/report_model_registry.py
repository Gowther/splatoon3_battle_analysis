from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_registry import DEFAULT_MODEL_REGISTRY, build_model_registry_report, load_model_registry, render_markdown
from src.report_io import strict_exit_code, write_json_report, write_text_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report canonical runtime model registry status.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_MODEL_REGISTRY)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "model_registry.md")
    parser.add_argument("--json-output", type=Path, default=ROOT / "outputs" / "model_registry.json")
    parser.add_argument("--hash", action="store_true", help="Compute and verify model SHA-256 hashes.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero unless all registered models pass.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_model_registry_report(load_model_registry(args.registry), verify_hash=args.hash)
    write_text_report(args.output.expanduser(), render_markdown(report))
    write_json_report(args.json_output.expanduser(), report)
    print(f"model registry status: {report['status']}")
    return strict_exit_code(report["status"], args.strict, passing_statuses={"passed"})


if __name__ == "__main__":
    raise SystemExit(main())
