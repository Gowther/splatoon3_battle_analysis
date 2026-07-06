from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from src.heatmap.extract_frames import ROOT, apply_excludes, clip_box, resolve_path, team_mask


HueRange = Dict[str, List[int]]


COLOR_PRESETS: Dict[str, Dict[str, object]] = {
    "red": {
        "hue": 178,
        "hsv_ranges": [
            {"lower": [0, 80, 80], "upper": [4, 255, 255]},
            {"lower": [172, 80, 80], "upper": [179, 255, 255]},
        ],
    },
    "orange": {
        "hue": 17,
        "hsv_ranges": [{"lower": [5, 80, 80], "upper": [30, 255, 255]}],
    },
    "yellow": {
        "hue": 32,
        "hsv_ranges": [{"lower": [20, 80, 80], "upper": [45, 255, 255]}],
    },
    "green": {
        "hue": 67,
        "hsv_ranges": [{"lower": [50, 80, 80], "upper": [85, 255, 255]}],
    },
    "cyan": {
        "hue": 95,
        "hsv_ranges": [{"lower": [85, 80, 80], "upper": [105, 255, 255]}],
    },
    "blue": {
        "hue": 122,
        "hsv_ranges": [{"lower": [105, 80, 80], "upper": [140, 255, 255]}],
    },
    "purple": {
        "hue": 139,
        "hsv_ranges": [{"lower": [132, 80, 70], "upper": [146, 255, 255]}],
    },
    "pink": {
        "hue": 160,
        "hsv_ranges": [{"lower": [145, 80, 80], "upper": [175, 255, 255]}],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve or auto-calibrate heatmap team colors.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    parser.add_argument("--output", help="Optional path for resolved YAML config.")
    parser.add_argument("--teams", help="Comma-separated preset names, for example orange,purple.")
    parser.add_argument("--disable-auto", action="store_true", help="Keep config teams without automatic calibration.")
    return parser.parse_args()


def circular_distance(a: int, b: int) -> int:
    distance = abs(int(a) - int(b)) % 180
    return min(distance, 180 - distance)


def circular_range(center: int, margin: int, min_saturation: int, min_value: int) -> List[HueRange]:
    lo = int(center) - int(margin)
    hi = int(center) + int(margin)
    if lo < 0:
        return [
            {"lower": [0, min_saturation, min_value], "upper": [hi, 255, 255]},
            {"lower": [180 + lo, min_saturation, min_value], "upper": [179, 255, 255]},
        ]
    if hi > 179:
        return [
            {"lower": [lo, min_saturation, min_value], "upper": [179, 255, 255]},
            {"lower": [0, min_saturation, min_value], "upper": [hi - 180, 255, 255]},
        ]
    return [{"lower": [lo, min_saturation, min_value], "upper": [hi, 255, 255]}]


def resolve_preset(name: str) -> Dict[str, object]:
    if name not in COLOR_PRESETS:
        raise ValueError(f"Unknown color preset: {name}")
    preset = COLOR_PRESETS[name]
    return {"hsv_ranges": copy.deepcopy(preset["hsv_ranges"])}


def manual_teams(names: Sequence[str]) -> Dict[str, Dict[str, object]]:
    teams: Dict[str, Dict[str, object]] = {}
    for raw_name in names:
        name = raw_name.strip()
        if not name:
            continue
        teams[name] = resolve_preset(name)
    if len(teams) != 2:
        raise ValueError("--teams must resolve exactly two color presets")
    return teams


def calibration_config(config: Dict) -> Dict[str, object]:
    return {
        "enabled": True,
        "reference_time_seconds": config.get("map_view", {}).get("reference_time_seconds", config["sampling"]["start_seconds"]),
        "sample_offsets_seconds": [-4, -2, 0, 2, 4],
        "min_saturation": 90,
        "min_value": 90,
        "smoothing_window": 4,
        "min_peak_distance": 18,
        "max_preset_distance": 11,
        "dynamic_hue_margin": 10,
        "sample_region": "top_icon_regions",
        "top_icon_region_ratios": [
            {"name": "left_team_icons", "x1": 0.146, "y1": 0.019, "x2": 0.469, "y2": 0.120},
            {"name": "right_team_icons", "x1": 0.521, "y1": 0.019, "x2": 0.781, "y2": 0.120},
        ],
        "order_by": "top_bar_x",
        **config.get("color_calibration", {}),
    }


def video_frame_at(video_path: Path, time_seconds: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_seconds)) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def color_sample_mask(hsv_roi: np.ndarray, min_saturation: int, min_value: int) -> np.ndarray:
    return ((hsv_roi[:, :, 1] >= min_saturation) & (hsv_roi[:, :, 2] >= min_value)).astype(np.uint8) * 255


def roi_hue_histogram(config: Dict, frame: np.ndarray, min_saturation: int, min_value: int) -> np.ndarray:
    height, width = frame.shape[:2]
    roi = clip_box(config["map_view"]["roi"], width, height)
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = color_sample_mask(hsv, min_saturation, min_value)
    apply_excludes(mask, roi, config["map_view"].get("exclude_regions", []))
    hue_values = hsv[:, :, 0][mask > 0]
    return np.bincount(hue_values.ravel(), minlength=180).astype(float)


def smooth_histogram(hist: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return hist
    output = hist.astype(float).copy()
    for offset in range(1, window + 1):
        output += np.roll(hist, offset)
        output += np.roll(hist, -offset)
    return output / float(window * 2 + 1)


def box_from_ratio(frame: np.ndarray, box: Dict[str, float]) -> Tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    return (
        int(round(float(box["x1"]) * width)),
        int(round(float(box["y1"]) * height)),
        int(round(float(box["x2"]) * width)),
        int(round(float(box["y2"]) * height)),
    )


def crop_hue_histogram(frame: np.ndarray, box: Tuple[int, int, int, int], min_saturation: int, min_value: int) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])
    if x2 <= x1 or y2 <= y1:
        return np.zeros(180, dtype=float)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = color_sample_mask(hsv, min_saturation, min_value)
    hue_values = hsv[:, :, 0][mask > 0]
    return np.bincount(hue_values.ravel(), minlength=180).astype(float)


def top_hue_peaks(hist: np.ndarray, count: int, min_distance: int) -> List[Tuple[int, int]]:
    peaks: List[Tuple[int, int]] = []
    for hue in np.argsort(hist)[::-1]:
        hue_int = int(hue)
        value = int(hist[hue_int])
        if value <= 0:
            break
        if any(circular_distance(hue_int, existing) < min_distance for existing, _ in peaks):
            continue
        peaks.append((hue_int, value))
        if len(peaks) >= count:
            break
    return peaks


def preset_for_hue(hue: int, max_distance: int) -> Optional[str]:
    best_name = None
    best_distance = 999
    for name, preset in COLOR_PRESETS.items():
        distance = circular_distance(hue, int(preset["hue"]))
        if distance < best_distance:
            best_name = name
            best_distance = distance
    return best_name if best_name is not None and best_distance <= max_distance else None


def dynamic_team_name(hue: int, used_names: Sequence[str]) -> str:
    base = f"hue_{hue:03d}"
    if base not in used_names:
        return base
    index = 2
    while f"{base}_{index}" in used_names:
        index += 1
    return f"{base}_{index}"


def team_from_hue(hue: int, used_names: Sequence[str], calib: Dict[str, object]) -> Tuple[str, Dict[str, object], str]:
    preset_name = preset_for_hue(hue, int(calib["max_preset_distance"]))
    if preset_name and preset_name not in used_names:
        return preset_name, resolve_preset(preset_name), "preset"

    team_name = dynamic_team_name(hue, used_names)
    team_config = {
        "hsv_ranges": circular_range(
            hue,
            int(calib["dynamic_hue_margin"]),
            int(calib["min_saturation"]),
            int(calib["min_value"]),
        )
    }
    return team_name, team_config, "dynamic"


def top_bar_score(config: Dict, frame: np.ndarray, hsv_ranges: Sequence[HueRange]) -> float:
    height, width = frame.shape[:2]
    top_regions = [
        region
        for region in config["map_view"].get("exclude_regions", [])
        if region.get("name") == "top_status_bar"
    ]
    if top_regions:
        x1, y1, x2, y2 = clip_box(top_regions[0], width, height)
    else:
        x1, y1, x2, y2 = 0, 0, width, min(180, height)
    if x2 <= x1 or y2 <= y1:
        return float(width)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    mask = team_mask(hsv, hsv_ranges)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float(width)
    return float(np.median(xs + x1))


def order_teams(config: Dict, frame: np.ndarray, teams: Dict[str, Dict[str, object]], order_by: str) -> Dict[str, Dict[str, object]]:
    if order_by != "top_bar_x":
        return teams
    scored = [
        (top_bar_score(config, frame, team_config["hsv_ranges"]), name, team_config)
        for name, team_config in teams.items()
    ]
    scored.sort(key=lambda item: item[0])
    return {name: team_config for _, name, team_config in scored}


def auto_teams(config: Dict, calib: Dict[str, object], frames: Sequence[np.ndarray]) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    if str(calib.get("sample_region", "top_icon_regions")) == "top_icon_regions":
        return auto_teams_from_top_icons(config, calib, frames)

    hist = np.zeros(180, dtype=float)
    for frame in frames:
        hist += roi_hue_histogram(config, frame, int(calib["min_saturation"]), int(calib["min_value"]))
    smoothed = smooth_histogram(hist, int(calib["smoothing_window"]))
    peaks = top_hue_peaks(smoothed, 2, int(calib["min_peak_distance"]))
    if len(peaks) < 2:
        raise RuntimeError("Could not detect two team color peaks from the reference frame(s).")

    teams: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    for order, (hue, count) in enumerate(peaks, start=1):
        team_name, team_config, source = team_from_hue(hue, list(teams), calib)
        teams[team_name] = team_config
        rows.append(
            {
                "order": order,
                "team": team_name,
                "detected_hue": hue,
                "histogram_count": count,
                "source": source,
                "hsv_ranges": yaml.safe_dump(team_config["hsv_ranges"], default_flow_style=True).strip(),
            }
        )
    return teams, rows


def auto_teams_from_top_icons(
    config: Dict,
    calib: Dict[str, object],
    frames: Sequence[np.ndarray],
) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    region_ratios = calib.get("top_icon_region_ratios", [])
    if len(region_ratios) < 2:
        raise RuntimeError("top_icon_region_ratios must define left and right team regions.")

    teams: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    for order, region in enumerate(region_ratios[:2], start=1):
        hist = np.zeros(180, dtype=float)
        for frame in frames:
            hist += crop_hue_histogram(
                frame,
                box_from_ratio(frame, region),
                int(calib["min_saturation"]),
                int(calib["min_value"]),
            )
        smoothed = smooth_histogram(hist, int(calib["smoothing_window"]))
        peaks = top_hue_peaks(smoothed, 5, int(calib["min_peak_distance"]))
        if not peaks:
            raise RuntimeError(f"Could not detect color peak for {region.get('name', f'team_{order}')}.")

        selected_hue, selected_count = peaks[0]
        selected_name = ""
        selected_config: Dict[str, object] = {}
        selected_source = ""
        for hue, count in peaks:
            team_name, team_config, source = team_from_hue(hue, list(teams), calib)
            if team_name not in teams:
                selected_hue, selected_count = hue, count
                selected_name, selected_config, selected_source = team_name, team_config, source
                break
        if not selected_name:
            selected_name, selected_config, selected_source = team_from_hue(selected_hue, list(teams), calib)

        teams[selected_name] = selected_config
        rows.append(
            {
                "order": order,
                "team": selected_name,
                "detected_hue": selected_hue,
                "histogram_count": selected_count,
                "source": selected_source,
                "hsv_ranges": yaml.safe_dump(selected_config["hsv_ranges"], default_flow_style=True).strip(),
            }
        )
    return teams, rows


def reference_frames(config: Dict, calib: Dict[str, object]) -> List[np.ndarray]:
    video_path = resolve_path(config["match"]["input_video"])
    base_time = float(calib["reference_time_seconds"])
    offsets = calib.get("sample_offsets_seconds", [0])
    frames: List[np.ndarray] = []
    for offset in offsets:
        frame = video_frame_at(video_path, base_time + float(offset))
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"Could not read calibration frames from {video_path}")
    return frames


def report_paths(config: Dict, output_path: Optional[Path]) -> Tuple[Path, Path]:
    output_dir = resolve_path(config["match"]["output_dir"])
    resolved_config_path = output_path or output_dir / "resolved_config.yaml"
    report_path = output_dir / "color_calibration_report.csv"
    return resolved_config_path, report_path


def write_report(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["order", "team", "detected_hue", "histogram_count", "source", "hsv_ranges"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_config(
    config: Dict,
    output_path: Optional[Path] = None,
    team_override: Optional[Sequence[str]] = None,
    disable_auto: bool = False,
) -> Tuple[Dict, Path, Path]:
    resolved = copy.deepcopy(config)
    calib = calibration_config(resolved)
    resolved_config_path, report_path = report_paths(resolved, output_path)

    if team_override:
        teams = manual_teams(team_override)
        rows = [
            {
                "order": index,
                "team": name,
                "detected_hue": "",
                "histogram_count": "",
                "source": "manual_preset",
                "hsv_ranges": yaml.safe_dump(team_config["hsv_ranges"], default_flow_style=True).strip(),
            }
            for index, (name, team_config) in enumerate(teams.items(), start=1)
        ]
        resolved["teams"] = teams
    elif disable_auto or not bool(calib.get("enabled", True)):
        rows = [
            {
                "order": index,
                "team": name,
                "detected_hue": "",
                "histogram_count": "",
                "source": "config",
                "hsv_ranges": yaml.safe_dump(team_config["hsv_ranges"], default_flow_style=True).strip(),
            }
            for index, (name, team_config) in enumerate(resolved["teams"].items(), start=1)
        ]
    else:
        frames = reference_frames(resolved, calib)
        teams, rows = auto_teams(resolved, calib, frames)
        resolved["teams"] = order_teams(resolved, frames[0], teams, str(calib.get("order_by", "top_bar_x")))
        ordered_rows = []
        for index, (name, team_config) in enumerate(resolved["teams"].items(), start=1):
            row = next(item for item in rows if item["team"] == name)
            row = dict(row)
            row["order"] = index
            row["hsv_ranges"] = yaml.safe_dump(team_config["hsv_ranges"], default_flow_style=True).strip()
            ordered_rows.append(row)
        rows = ordered_rows

    resolved["color_calibration"] = {**calib, "resolved_config": str(resolved_config_path.relative_to(ROOT)), "report_csv": str(report_path.relative_to(ROOT))}
    resolved["outputs"]["color_calibration_report_csv"] = str(report_path.relative_to(ROOT))
    resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, allow_unicode=True, sort_keys=False)
    write_report(report_path, rows)
    return resolved, resolved_config_path, report_path


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    output = resolve_path(args.output) if args.output else None
    team_override = [item.strip() for item in args.teams.split(",")] if args.teams else None
    _, resolved_config_path, report_path = resolve_config(config, output, team_override, args.disable_auto)
    print(f"resolved config: {resolved_config_path}")
    print(f"color report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
