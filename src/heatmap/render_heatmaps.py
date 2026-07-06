from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.heatmap.detect_markers import load_mask
from src.heatmap.extract_frames import load_config, resolve_path


Point = Dict[str, object]


TEAM_COLORS = {
    "yellow": (0, 245, 255),
    "blue": (255, 70, 0),
    "orange": (0, 105, 255),
    "purple": (255, 0, 180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render overhead-map heatmaps and team routes.")
    parser.add_argument("--config", default="src/heatmap/config_match9.yaml")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def read_video_frame(config: Dict) -> np.ndarray:
    input_video = resolve_path(config["match"]["input_video"])
    reference_time = float(config["rendering"]["reference_time_seconds"])
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, reference_time * 1000.0)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Could not read reference frame at {reference_time:.3f}s")
    return frame


def dim_outside_mask(frame: np.ndarray, mask: np.ndarray, dim: float) -> np.ndarray:
    output = frame.copy().astype(np.float32)
    outside = mask == 0
    output[outside] *= float(dim)
    return np.clip(output, 0, 255).astype(np.uint8)


def prepare_render_base(frame: np.ndarray, mask: np.ndarray, config: Dict) -> np.ndarray:
    rendering = config["rendering"]
    saturation = float(rendering.get("map_saturation", 1.0))
    brightness = float(rendering.get("map_brightness", 1.0))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation
    muted = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    muted = np.clip(muted.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return dim_outside_mask(muted, mask, float(rendering["outside_mask_dim"]))


def parse_points(rows: Sequence[Dict[str, str]]) -> List[Point]:
    points: List[Point] = []
    for row in rows:
        try:
            points.append(
                {
                    "match_id": row.get("match_id", ""),
                    "time": f"{float(row['time']):.3f}",
                    "frame_index": row.get("frame_index", ""),
                    "team": row.get("team", ""),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "confidence": float(row.get("confidence", 1.0)),
                    "frame_path": row.get("frame_path", ""),
                    "track_slot": row.get("track_slot", ""),
                    "track_status": row.get("track_status", ""),
                    "step_distance": row.get("step_distance", ""),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return points


def build_heat(points: Sequence[Point], shape: Tuple[int, int], mask: np.ndarray, config: Dict) -> np.ndarray:
    heat = np.zeros(shape, dtype=np.float32)
    radius = int(config["rendering"]["heat_point_radius_px"])
    sigma = float(config["rendering"]["heat_blur_sigma_px"])
    for point in points:
        x = int(round(float(point["x"])))
        y = int(round(float(point["y"])))
        if x < 0 or y < 0 or y >= shape[0] or x >= shape[1]:
            continue
        weight = max(0.05, float(point.get("confidence", 1.0)))
        cv2.circle(heat, (x, y), radius, weight, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    heat[mask == 0] = 0.0
    active = heat[heat > 0]
    if active.size == 0:
        return heat
    scale = float(np.percentile(active, 99.0))
    if scale <= 0:
        return heat
    return np.clip(heat / scale, 0.0, 1.0)


def adjust_heat_for_display(heat: np.ndarray, config: Dict) -> np.ndarray:
    rendering = config["rendering"]
    cutoff = float(rendering.get("heat_low_cutoff", 0.0))
    gamma = float(rendering.get("heat_contrast_gamma", 1.0))
    boost = float(rendering.get("heat_peak_boost", 1.0))
    if cutoff > 0:
        heat = np.clip((heat - cutoff) / max(1e-6, 1.0 - cutoff), 0.0, 1.0)
    heat = np.clip(heat * boost, 0.0, 1.0)
    return np.power(heat, gamma)


def blend_heat(
    base: np.ndarray,
    heat: np.ndarray,
    color: Tuple[int, int, int],
    max_alpha: float,
    config: Dict,
) -> np.ndarray:
    output = base.astype(np.float32)
    color_layer = np.zeros_like(output)
    color_layer[:, :] = np.array(color, dtype=np.float32)
    display_heat = adjust_heat_for_display(heat, config)
    alpha = (display_heat * float(max_alpha))[:, :, None]
    output = output * (1.0 - alpha) + color_layer * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def colormap_id(name: str) -> int:
    return int(getattr(cv2, f"COLORMAP_{name.upper()}", getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)))


def blend_colormap_heat(base: np.ndarray, heat: np.ndarray, config: Dict) -> np.ndarray:
    rendering = config["rendering"]
    display_heat = adjust_heat_for_display(heat, config)
    color_layer = cv2.applyColorMap(
        np.clip(display_heat * 255.0, 0, 255).astype(np.uint8),
        colormap_id(str(rendering.get("all_heat_colormap", "turbo"))),
    ).astype(np.float32)
    min_alpha = float(rendering.get("all_heat_min_alpha", 0.12))
    max_alpha = float(rendering.get("all_heat_max_alpha", 0.68))
    alpha = np.where(display_heat > 0, min_alpha + display_heat * (max_alpha - min_alpha), 0.0)
    output = base.astype(np.float32) * (1.0 - alpha[:, :, None]) + color_layer * alpha[:, :, None]
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_heat_legend(image: np.ndarray, config: Dict) -> np.ndarray:
    if not config["rendering"].get("all_heat_legend", True):
        return image
    output = image.copy()
    height, width = output.shape[:2]
    bar_w = 28
    bar_h = 260
    x1 = width - 82
    y1 = 250
    x2 = x1 + bar_w
    y2 = y1 + bar_h

    legend_background = output.copy()
    cv2.rectangle(legend_background, (x1 - 18, y1 - 44), (x2 + 42, y2 + 38), (0, 0, 0), -1)
    output = cv2.addWeighted(legend_background, 0.38, output, 0.62, 0)

    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    gradient = np.repeat(gradient, bar_w, axis=1)
    colorbar = cv2.applyColorMap(gradient, colormap_id(str(config["rendering"].get("all_heat_colormap", "turbo"))))
    output[y1:y2, x1:x2] = colorbar
    cv2.rectangle(output, (x1, y1), (x2, y2), (230, 230, 230), 1)
    cv2.putText(output, "HOT", (x1 - 7, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(output, "LOW", (x1 - 7, y2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return output


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def render_all_heatmap(
    base: np.ndarray,
    mask: np.ndarray,
    points: Sequence[Point],
    config: Dict,
) -> Tuple[str, Path]:
    output_dir = resolve_path(config["outputs"]["rendered_dir"])
    heat = build_heat(points, base.shape[:2], mask, config)
    image = draw_heat_legend(blend_colormap_heat(base, heat, config), config)
    path = output_dir / "heatmap_all.png"
    save_image(path, image)
    return "heatmap_all", path


def render_team_heatmaps(
    base: np.ndarray,
    mask: np.ndarray,
    points: Sequence[Point],
    config: Dict,
) -> Dict[str, Path]:
    output_dir = resolve_path(config["outputs"]["rendered_dir"])
    max_alpha = float(config["rendering"]["heat_max_alpha"])
    rendered: Dict[str, Path] = {}
    by_team: DefaultDict[str, List[Point]] = defaultdict(list)
    for point in points:
        by_team[str(point["team"])].append(point)

    key, all_path = render_all_heatmap(base, mask, points, config)
    rendered[key] = all_path

    combined = base.copy()
    for team in config["teams"]:
        heat = build_heat(by_team.get(team, []), base.shape[:2], mask, config)
        image = blend_heat(base, heat, TEAM_COLORS.get(team, (255, 255, 255)), max_alpha, config)
        path = output_dir / f"heatmap_{team}.png"
        save_image(path, image)
        rendered[f"heatmap_{team}"] = path
        combined = blend_heat(combined, heat, TEAM_COLORS.get(team, (255, 255, 255)), max_alpha * 0.9, config)

    combined_path = output_dir / "heatmap_combined.png"
    save_image(combined_path, combined)
    rendered["heatmap_combined"] = combined_path
    return rendered


def grouped_tracks(track_points: Sequence[Point]) -> DefaultDict[Tuple[str, str], List[Point]]:
    output: DefaultDict[Tuple[str, str], List[Point]] = defaultdict(list)
    for point in track_points:
        output[(str(point["team"]), str(point["track_slot"]))].append(point)
    for points in output.values():
        points.sort(key=lambda row: float(row["time"]))
    return output


def render_routes(base: np.ndarray, track_points: Sequence[Point], config: Dict) -> Path:
    output_dir = resolve_path(config["outputs"]["rendered_dir"])
    output_path = output_dir / "team_routes.png"
    overlay = base.copy()
    thickness = int(config["rendering"]["route_line_thickness_px"])
    point_radius = int(config["rendering"]["route_point_radius_px"])
    max_draw_step = float(config["rendering"]["route_max_draw_step_px"])

    for (team, _), points in grouped_tracks(track_points).items():
        color = TEAM_COLORS.get(team, (255, 255, 255))
        previous: Optional[Point] = None
        for point in points:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
            status = str(point.get("track_status", ""))
            step_distance = str(point.get("step_distance", ""))
            can_draw_step = step_distance != "" and float(step_distance) <= max_draw_step
            if previous is not None and status == "matched" and can_draw_step:
                px = int(round(float(previous["x"])))
                py = int(round(float(previous["y"])))
                cv2.line(overlay, (px, py), (x, y), color, thickness, cv2.LINE_AA)
            cv2.circle(overlay, (x, y), point_radius, color, -1, cv2.LINE_AA)
            previous = point

    save_image(output_path, overlay)
    return output_path


def report_rows(clean_points: Sequence[Point], track_points: Sequence[Point], rendered: Dict[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = [
        {"metric": "clean_points", "value": len(clean_points)},
        {"metric": "track_points", "value": len(track_points)},
    ]
    for team, count in sorted(Counter(str(point["team"]) for point in clean_points).items()):
        rows.append({"metric": f"clean_team_{team}", "value": count})
    for key, path in sorted(rendered.items()):
        rows.append({"metric": key, "value": path})
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    mask = load_mask(config)
    base = prepare_render_base(read_video_frame(config), mask, config)
    clean_points = parse_points(read_csv(resolve_path(config["outputs"]["clean_points_csv"])))
    track_points = parse_points(read_csv(resolve_path(config["outputs"]["tracks_csv"])))

    rendered = render_team_heatmaps(base, mask, clean_points, config)
    routes_path = render_routes(base, track_points, config)
    rendered["team_routes"] = routes_path
    write_csv(resolve_path(config["outputs"]["render_report_csv"]), report_rows(clean_points, track_points, rendered))

    print(f"clean points: {len(clean_points)}")
    print(f"track points: {len(track_points)}")
    for key, path in sorted(rendered.items()):
        print(f"{key}: {path}")
    print(f"render report: {resolve_path(config['outputs']['render_report_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
