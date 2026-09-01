from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.core.paths import ROOT, default_output_path, project_path
from src.csv_contracts import ANALYSIS_CSV_CONTRACT


CSV_HEADER = list(ANALYSIS_CSV_CONTRACT.fields)


@dataclass
class AnalysisRunResult:
    rows: List[List[object]]
    analyzed: int
    final_weapons: Optional[List[str]]


def resolve_io_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = project_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = project_path(args.output) if args.output else default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return input_path, output_path


def preview_dir_from_arg(value: str | None) -> Path | None:
    if not value:
        return None
    save_preview_dir = Path(value).expanduser()
    if not save_preview_dir.is_absolute():
        save_preview_dir = ROOT / save_preview_dir
    save_preview_dir.mkdir(parents=True, exist_ok=True)
    return save_preview_dir


def write_analysis_csv(output_path: Path, rows: List[List[object]], include_header: bool = True) -> None:
    for row_number, row in enumerate(rows, start=1):
        ANALYSIS_CSV_CONTRACT.validate_row(row, row_number=row_number)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)
