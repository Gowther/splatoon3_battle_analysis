from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.core.paths import ROOT, default_output_path, project_path


CSV_HEADER = [
    "elapsed_time",
    "player_state_1",
    "player_state_2",
    "player_state_3",
    "player_state_4",
    "player_state_5",
    "player_state_6",
    "player_state_7",
    "player_state_8",
    "count_left",
    "count_right",
    "penalty_left",
    "penalty_right",
    "weapon_1",
    "weapon_2",
    "weapon_3",
    "weapon_4",
    "weapon_5",
    "weapon_6",
    "weapon_7",
    "weapon_8",
    "stage",
    "asari_count",
    "hoko_count",
    "area_count",
    "yagura_count",
    "message",
    "player_detected",
    "reserved_28",
    "timestamp",
    "reserved_30",
    "reserved_31",
    "reserved_32",
]


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
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)
