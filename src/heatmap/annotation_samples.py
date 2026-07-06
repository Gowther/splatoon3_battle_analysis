from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from src.data_registry import display_path, iter_heatmap_matches, resolve_project_path


ANNOTATION_FIELDS = [
    "match_id",
    "heatmap_id",
    "time",
    "frame_index",
    "team",
    "slot_hint",
    "annotation_id",
    "x",
    "y",
    "visibility",
    "frame_complete",
    "notes",
    "frame_path",
    "preview_path",
    "source_prediction_x",
    "source_prediction_y",
    "source_confidence",
    "source_track_status",
    "source_player_id",
]

PREDICTION_FIELDS = [
    "match_id",
    "heatmap_id",
    "time",
    "frame_index",
    "team",
    "track_slot",
    "player_id",
    "x",
    "y",
    "confidence",
    "identity_confidence",
    "track_status",
    "step_distance",
    "frame_path",
    "preview_path",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def float_or_zero(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def frame_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("time", ""), row.get("frame_index", ""), row.get("frame_path", ""))


def grouped_by_frame(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(frame_key(row), []).append(row)
    return grouped


def select_frame_keys(rows: list[dict[str, str]], frames_per_match: int) -> list[tuple[str, str, str]]:
    grouped = grouped_by_frame(rows)
    keys = sorted(grouped, key=lambda key: float_or_zero(key[0]))
    if frames_per_match <= 0 or len(keys) <= frames_per_match:
        return keys

    selected: list[tuple[str, str, str]] = []
    jump_keys = sorted(
        keys,
        key=lambda key: (
            -sum(1 for row in grouped[key] if row.get("track_status") == "jump_reset"),
            float_or_zero(key[0]),
        ),
    )
    for key in jump_keys:
        if any(row.get("track_status") == "jump_reset" for row in grouped[key]):
            selected.append(key)
        if len(selected) >= max(1, frames_per_match // 2):
            break

    if len(selected) < frames_per_match:
        denominator = max(1, frames_per_match - len(selected) - 1)
        for offset in range(frames_per_match - len(selected)):
            index = round(offset * (len(keys) - 1) / denominator) if denominator else 0
            key = keys[index]
            if key not in selected:
                selected.append(key)

    for key in keys:
        if len(selected) >= frames_per_match:
            break
        if key not in selected:
            selected.append(key)

    return sorted(selected[:frames_per_match], key=lambda key: float_or_zero(key[0]))


def try_draw_preview(frame_path: Path, preview_path: Path, rows: list[dict[str, str]]) -> bool:
    try:
        import cv2
    except ImportError:
        return False

    image = cv2.imread(str(frame_path))
    if image is None:
        return False

    palette = {
        "yellow": (0, 220, 255),
        "blue": (255, 120, 0),
        "orange": (0, 140, 255),
        "purple": (220, 80, 220),
        "pink": (220, 80, 255),
        "green": (80, 220, 80),
    }
    for row in rows:
        x = int(round(float_or_zero(row.get("x"))))
        y = int(round(float_or_zero(row.get("y"))))
        team = row.get("team", "")
        color = palette.get(team, (255, 255, 255))
        label = f"{team}:{row.get('track_slot', '')}"
        cv2.circle(image, (x, y), 10, color, 2)
        cv2.putText(image, label, (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(preview_path), image))


def copy_frame_and_preview(
    output_dir: Path,
    registry_match_id: str,
    heatmap_id: str,
    key: tuple[str, str, str],
    rows: list[dict[str, str]],
) -> tuple[Path, Path | None]:
    source = resolve_project_path(key[2])
    if source is None or not source.exists():
        raise FileNotFoundError(f"Missing frame for annotation sample: {key[2]}")

    frame_name = f"{registry_match_id}_{source.name}"
    frame_dest = output_dir / "frames" / frame_name
    frame_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, frame_dest)

    preview_dest = output_dir / "previews" / frame_name
    if try_draw_preview(source, preview_dest, rows):
        return frame_dest, preview_dest
    return frame_dest, None


def annotation_id(match_id: str, row: dict[str, str]) -> str:
    time_key = row.get("time", "").replace(".", "_")
    return f"{match_id}_{time_key}_{row.get('team', '')}_{row.get('track_slot', '')}"


def build_annotation_rows(
    output_dir: Path,
    registry_match_id: str,
    heatmap: dict[str, Any],
    frames_per_match: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    heatmap_id = heatmap.get("id", registry_match_id)
    tracks_path = resolve_project_path(heatmap.get("player_tracks"))
    if tracks_path is None or not tracks_path.exists():
        raise FileNotFoundError(f"Missing player tracks for {registry_match_id}: {heatmap.get('player_tracks')}")

    rows = read_csv_rows(tracks_path)
    grouped = grouped_by_frame(rows)
    selected_keys = select_frame_keys(rows, frames_per_match)
    annotation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    manifest_frames: list[dict[str, Any]] = []

    for key in selected_keys:
        frame_rows = sorted(grouped[key], key=lambda row: (row.get("team", ""), row.get("track_slot", "")))
        frame_dest, preview_dest = copy_frame_and_preview(output_dir, registry_match_id, heatmap_id, key, frame_rows)
        frame_path = display_path(frame_dest)
        preview_path = display_path(preview_dest)
        manifest_frames.append(
            {
                "match_id": registry_match_id,
                "heatmap_id": heatmap_id,
                "time": key[0],
                "frame_index": key[1],
                "frame_path": frame_path,
                "preview_path": preview_path,
                "prediction_rows": len(frame_rows),
                "jump_reset_rows": sum(1 for row in frame_rows if row.get("track_status") == "jump_reset"),
            }
        )

        for row in frame_rows:
            common = {
                "match_id": registry_match_id,
                "heatmap_id": heatmap_id,
                "time": row.get("time", ""),
                "frame_index": row.get("frame_index", ""),
                "team": row.get("team", ""),
            }
            annotation_rows.append(
                {
                    **common,
                    "slot_hint": row.get("track_slot", ""),
                    "annotation_id": annotation_id(registry_match_id, row),
                    "x": "",
                    "y": "",
                    "visibility": "visible",
                    "frame_complete": "false",
                    "notes": "",
                    "frame_path": frame_path,
                    "preview_path": preview_path,
                    "source_prediction_x": row.get("x", ""),
                    "source_prediction_y": row.get("y", ""),
                    "source_confidence": row.get("confidence", ""),
                    "source_track_status": row.get("track_status", ""),
                    "source_player_id": row.get("player_id", ""),
                }
            )
            prediction_rows.append(
                {
                    **common,
                    "track_slot": row.get("track_slot", ""),
                    "player_id": row.get("player_id", ""),
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "confidence": row.get("confidence", ""),
                    "identity_confidence": row.get("identity_confidence", ""),
                    "track_status": row.get("track_status", ""),
                    "step_distance": row.get("step_distance", ""),
                    "frame_path": frame_path,
                    "preview_path": preview_path,
                }
            )

    return annotation_rows, prediction_rows, manifest_frames


def write_readme(output_dir: Path) -> None:
    lines = [
        "# Heatmap Annotation Package",
        "",
        "Fill `annotation_template.csv` by writing manual player positions into `x` and `y`.",
        "Prediction columns are reference hints only; do not copy them blindly as labels.",
        "",
        "Fields to edit:",
        "",
        "- `x`, `y`: manual player center in source-video pixel coordinates.",
        "- `visibility`: use `visible`, `occluded`, `absent`, or `uncertain`.",
        "- `frame_complete`: set `true` only when every visible player for that team/frame has been labeled.",
        "- `notes`: free-form annotation notes.",
        "",
        "Use `previews/` to see model predictions overlaid on the copied frames.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def export_annotation_package(
    registry: dict[str, Any],
    output_dir: Path,
    match_ids: list[str] | None = None,
    frames_per_match: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(match_ids or [])
    annotation_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    manifest_matches: list[dict[str, Any]] = []

    for match, heatmap in iter_heatmap_matches(registry):
        if selected and match["id"] not in selected:
            continue
        rows, predictions, frames = build_annotation_rows(output_dir, match["id"], heatmap, frames_per_match)
        annotation_rows.extend(rows)
        prediction_rows.extend(predictions)
        manifest_matches.append(
            {
                "match_id": match["id"],
                "heatmap_id": heatmap.get("id", ""),
                "frames": frames,
                "annotation_rows": len(rows),
            }
        )

    write_csv(output_dir / "annotation_template.csv", ANNOTATION_FIELDS, annotation_rows)
    write_csv(output_dir / "prediction_reference.csv", PREDICTION_FIELDS, prediction_rows)
    write_readme(output_dir)

    manifest = {
        "output_dir": display_path(output_dir),
        "frames_per_match": frames_per_match,
        "matches": manifest_matches,
        "annotation_template": display_path(output_dir / "annotation_template.csv"),
        "prediction_reference": display_path(output_dir / "prediction_reference.csv"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
