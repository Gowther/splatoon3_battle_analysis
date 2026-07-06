from __future__ import annotations

import argparse
import bisect
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.heatmap.extract_frames import load_config, resolve_path


Row = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join heatmap points with nearest UI-state CSV rows.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    return parser.parse_args()


def read_csv(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_time(row: Row, field: str) -> Optional[float]:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return None


def sorted_state_rows(rows: Sequence[Row]) -> Tuple[List[float], List[Row]]:
    parsed: List[Tuple[float, Row]] = []
    for row in rows:
        time_value = parse_time(row, "elapsed_time")
        if time_value is None:
            continue
        parsed.append((time_value, row))
    parsed.sort(key=lambda item: item[0])
    return [item[0] for item in parsed], [item[1] for item in parsed]


def nearest_state(time_value: float, state_times: Sequence[float], state_rows: Sequence[Row]) -> Tuple[Optional[Row], Optional[float]]:
    if not state_times:
        return None, None
    index = bisect.bisect_left(state_times, time_value)
    candidates = []
    if index < len(state_times):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best_index = min(candidates, key=lambda item: abs(state_times[item] - time_value))
    return state_rows[best_index], abs(state_times[best_index] - time_value)


def join_points(config: Dict) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    point_rows = read_csv(resolve_path(config["outputs"]["clean_points_csv"]))
    state_rows_raw = read_csv(resolve_path(config["state_join"]["state_csv"]))
    state_times, state_rows = sorted_state_rows(state_rows_raw)
    max_delta = float(config["state_join"]["max_time_delta_seconds"])
    join_fields = list(config["state_join"]["fields"])
    output_rows: List[Dict[str, object]] = []
    matched = 0
    unmatched = 0
    deltas: List[float] = []

    for point in point_rows:
        point_time = parse_time(point, "time")
        output = dict(point)
        if point_time is None:
            output["join_status"] = "point_time_parse_error"
            output["ui_elapsed_time"] = ""
            output["ui_time_delta"] = ""
            for field in join_fields:
                output[f"ui_{field}"] = ""
            unmatched += 1
            output_rows.append(output)
            continue

        state_row, delta = nearest_state(point_time, state_times, state_rows)
        if state_row is None or delta is None or delta > max_delta:
            output["join_status"] = "no_state_within_tolerance"
            output["ui_elapsed_time"] = ""
            output["ui_time_delta"] = "" if delta is None else round(delta, 3)
            for field in join_fields:
                output[f"ui_{field}"] = ""
            unmatched += 1
        else:
            output["join_status"] = "matched"
            output["ui_elapsed_time"] = state_row.get("elapsed_time", "")
            output["ui_time_delta"] = round(delta, 3)
            for field in join_fields:
                output[f"ui_{field}"] = state_row.get(field, "")
            matched += 1
            deltas.append(delta)
        output_rows.append(output)

    report_rows = [
        {"metric": "point_rows", "value": len(point_rows)},
        {"metric": "state_rows", "value": len(state_rows)},
        {"metric": "matched_rows", "value": matched},
        {"metric": "unmatched_rows", "value": unmatched},
        {"metric": "max_allowed_delta_seconds", "value": max_delta},
        {"metric": "max_observed_delta_seconds", "value": round(max(deltas), 3) if deltas else ""},
        {"metric": "state_start_seconds", "value": state_times[0] if state_times else ""},
        {"metric": "state_stop_seconds", "value": state_times[-1] if state_times else ""},
    ]
    return output_rows, report_rows


def output_fieldnames(point_rows: Sequence[Row], config: Dict) -> List[str]:
    base_fields = list(point_rows[0].keys()) if point_rows else []
    join_fields = ["join_status", "ui_elapsed_time", "ui_time_delta"]
    join_fields.extend(f"ui_{field}" for field in config["state_join"]["fields"])
    return base_fields + [field for field in join_fields if field not in base_fields]


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    point_rows = read_csv(resolve_path(config["outputs"]["clean_points_csv"]))
    enriched_rows, report_rows = join_points(config)
    write_csv(resolve_path(config["outputs"]["enriched_points_csv"]), output_fieldnames(point_rows, config), enriched_rows)
    write_csv(resolve_path(config["outputs"]["state_join_report_csv"]), ["metric", "value"], report_rows)
    print(f"points: {len(point_rows)}")
    print(f"enriched points: {len(enriched_rows)}")
    print(f"enriched csv: {resolve_path(config['outputs']['enriched_points_csv'])}")
    print(f"join report: {resolve_path(config['outputs']['state_join_report_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
