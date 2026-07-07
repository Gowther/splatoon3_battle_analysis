from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import cv2

from src.heatmap.detect_markers import load_mask
from src.heatmap.extract_frames import load_config, resolve_path
from src.heatmap.render_heatmaps import prepare_render_base, read_video_frame, save_image, team_display_color


Row = Dict[str, str]


PLAYER_TRACK_FIELDS = [
    "match_id",
    "time",
    "frame_index",
    "team",
    "track_slot",
    "player_id",
    "weapon_hint",
    "x",
    "y",
    "confidence",
    "identity_confidence",
    "track_status",
    "step_distance",
    "identity_method",
    "identity_note",
    "frame_path",
]

GAP_FIELDS = [
    "match_id",
    "time",
    "frame_index",
    "team",
    "track_slot",
    "player_id",
    "track_status",
    "step_distance",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer experimental slot-level player tracks from team tracks.")
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


def weapon_hints_by_team(config: Dict) -> Dict[str, Dict[str, str]]:
    state_rows = read_csv(resolve_path(config["state_join"]["state_csv"]))
    teams = list(config["teams"].keys())
    if not state_rows:
        return {team: {} for team in teams}
    selected = next((row for row in state_rows if row.get("weapon_1") or row.get("weapon_5")), state_rows[-1])
    output: Dict[str, Dict[str, str]] = {}
    for team_index, team in enumerate(teams):
        first_weapon_index = team_index * 4 + 1
        output[team] = {
            str(slot): selected.get(f"weapon_{first_weapon_index + slot - 1}", "") for slot in range(1, 5)
        }
    return output


def identity_confidence(row: Row, weapon_hint: str, config: Dict) -> float:
    identity = config["identity_tracking"]
    score = float(identity["base_identity_confidence"])
    if row.get("track_status") == "matched":
        score += float(identity["matched_bonus"])
    if weapon_hint:
        score += float(identity["weapon_hint_bonus"])
    if row.get("track_status") == "jump_reset":
        score -= float(identity["jump_reset_penalty"])
    return round(max(0.0, min(1.0, score)), 3)


def build_player_tracks(config: Dict) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    track_rows = read_csv(resolve_path(config["outputs"]["tracks_csv"]))
    weapon_hints = weapon_hints_by_team(config)
    method = config["identity_tracking"]["method"]
    player_rows: List[Dict[str, object]] = []
    gap_rows: List[Dict[str, object]] = []

    for row in track_rows:
        team = row.get("team", "")
        slot = row.get("track_slot", "")
        player_id = f"{team}_slot_{slot}"
        weapon_hint = weapon_hints.get(team, {}).get(slot, "")
        status = row.get("track_status", "")
        note = "experimental_slot_not_verified_player_identity"
        output = {
            "match_id": row.get("match_id", ""),
            "time": row.get("time", ""),
            "frame_index": row.get("frame_index", ""),
            "team": team,
            "track_slot": slot,
            "player_id": player_id,
            "weapon_hint": weapon_hint,
            "x": row.get("x", ""),
            "y": row.get("y", ""),
            "confidence": row.get("confidence", ""),
            "identity_confidence": identity_confidence(row, weapon_hint, config),
            "track_status": status,
            "step_distance": row.get("step_distance", ""),
            "identity_method": method,
            "identity_note": note,
            "frame_path": row.get("frame_path", ""),
        }
        player_rows.append(output)
        if status in {"new", "jump_reset"}:
            gap_rows.append(
                {
                    "match_id": output["match_id"],
                    "time": output["time"],
                    "frame_index": output["frame_index"],
                    "team": team,
                    "track_slot": slot,
                    "player_id": player_id,
                    "track_status": status,
                    "step_distance": output["step_distance"],
                    "note": note,
                }
            )

    return player_rows, gap_rows


def group_by_player(rows: Sequence[Dict[str, object]]) -> DefaultDict[str, List[Dict[str, object]]]:
    grouped: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["player_id"])].append(row)
    for player_rows in grouped.values():
        player_rows.sort(key=lambda item: float(item["time"]))
    return grouped


def clean_route_images(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    removed = 0
    for path in output_dir.glob("*.png"):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def render_player_routes(config: Dict, player_rows: Sequence[Dict[str, object]]) -> List[Path]:
    mask = load_mask(config)
    base = prepare_render_base(read_video_frame(config), mask, config)
    output_dir = resolve_path(config["outputs"]["player_routes_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_route_images(output_dir)
    max_draw_step = float(config["identity_tracking"]["route_max_draw_step_px"])
    thickness = int(config["identity_tracking"]["route_line_thickness_px"])
    point_radius = int(config["identity_tracking"]["route_point_radius_px"])
    paths: List[Path] = []

    for player_id, rows in sorted(group_by_player(player_rows).items()):
        image = base.copy()
        previous = None
        for row in rows:
            x = int(round(float(row["x"])))
            y = int(round(float(row["y"])))
            color = team_display_color(str(row["team"]), config)
            status = str(row.get("track_status", ""))
            step_distance = str(row.get("step_distance", ""))
            if previous is not None and status == "matched" and step_distance and float(step_distance) <= max_draw_step:
                px = int(round(float(previous["x"])))
                py = int(round(float(previous["y"])))
                cv2.line(image, (px, py), (x, y), color, thickness, cv2.LINE_AA)
            cv2.circle(image, (x, y), point_radius, color, -1, cv2.LINE_AA)
            previous = row
        cv2.putText(
            image,
            player_id,
            (28, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            player_id,
            (28, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        output_path = output_dir / f"{player_id}.png"
        save_image(output_path, image)
        paths.append(output_path)
    return paths


def report_rows(player_rows: Sequence[Dict[str, object]], gap_rows: Sequence[Dict[str, object]], route_paths: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = [
        {"metric": "player_track_rows", "value": len(player_rows)},
        {"metric": "gap_rows", "value": len(gap_rows)},
        {"metric": "route_images", "value": len(route_paths)},
        {"metric": "identity_method", "value": "team_slot_weapon_hint"},
        {"metric": "identity_warning", "value": "slot labels are not verified player identities"},
    ]
    for player_id, count in sorted(Counter(str(row["player_id"]) for row in player_rows).items()):
        rows.append({"metric": f"rows_{player_id}", "value": count})
    for status, count in sorted(Counter(str(row["track_status"]) for row in player_rows).items()):
        rows.append({"metric": f"track_status_{status}", "value": count})
    return rows


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    player_rows, gap_rows = build_player_tracks(config)
    route_paths = render_player_routes(config, player_rows)
    write_csv(resolve_path(config["outputs"]["player_tracks_csv"]), PLAYER_TRACK_FIELDS, player_rows)
    write_csv(resolve_path(config["outputs"]["player_track_gaps_csv"]), GAP_FIELDS, gap_rows)
    write_csv(resolve_path(config["outputs"]["identity_report_csv"]), ["metric", "value"], report_rows(player_rows, gap_rows, route_paths))
    print(f"player track rows: {len(player_rows)}")
    print(f"gap rows: {len(gap_rows)}")
    print(f"route images: {len(route_paths)}")
    print(f"player tracks csv: {resolve_path(config['outputs']['player_tracks_csv'])}")
    print(f"identity report: {resolve_path(config['outputs']['identity_report_csv'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
