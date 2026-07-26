from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.core.paths import project_path
from src.data_registry import display_path
from src.heatmap.stage_coordinates import (
    StageBox,
    roi_corner_control_points,
    stage_box_from_config,
)


DEFAULT_GRID_DIVISIONS = 10
DEFAULT_OUTPUT_ROOT = "outputs/stage_reference"

FILL_GUIDE = """# 控制点填写说明

这个目录是 stage 归一化 Goal 2 的参考帧包。目标：把 `control_points_draft.json`
里的 ROI 角点种子换成真实地标，然后提升为正式控制点资产。

## 步骤

1. 打开 `frames/` 里的网格参考帧。绿色外框是地图 ROI，网格标注的是
   ROI 线性归一化坐标（0.0 到 1.0）。
2. 在帧里找至少 4 个能在标准场地图上说出确切位置的地标：
   出生点垫、中心塔、固定平台角等。避免会移动的东西。
3. 对每个地标，把它的源视频像素坐标填进 `source`，把它在标准场地图上的
   0..1 坐标填进 `target`，并给 `name` 起一个能复查的名字。
4. 控制点不要共线：至少 4 个点要围出一个有面积的四边形。
5. 填完后把 `template` 改成 `false`。
6. 校验：

```bash
.venv/bin/python scripts/build_stage_control_points.py \\
  --config {config_path} \\
  --control-points {draft_path} \\
  --validate --strict
```

7. 校验通过后复制为正式资产 `config/stage_control_points/{stage_id}.json`，
   再用 `scripts/report_stage_coordinates.py --control-points ...` 导出
   homography 归一化坐标。

## 注意

- 草稿里的 4 个 ROI 角点只是种子：它们复现的是当前的线性映射，
  没有任何透视校正价值。真实地标没填完之前不要把 `template` 改成 `false`。
- 网格刻度是线性归一化坐标，只用来帮助读出像素位置，
  不是 homography 之后的目标坐标。
"""


def grid_lines(source_box: StageBox, divisions: int = DEFAULT_GRID_DIVISIONS) -> dict[str, list[dict[str, float]]]:
    """Grid line positions inside the ROI, labeled with linear normalized coordinates."""
    if divisions < 1:
        raise ValueError("divisions must be at least 1")
    vertical: list[dict[str, float]] = []
    horizontal: list[dict[str, float]] = []
    for index in range(divisions + 1):
        fraction = index / divisions
        vertical.append({"x": source_box.x1 + fraction * source_box.width, "label": round(fraction, 3)})
        horizontal.append({"y": source_box.y1 + fraction * source_box.height, "label": round(fraction, 3)})
    return {"vertical": vertical, "horizontal": horizontal}


def reference_times(config: Mapping[str, Any], extra_times: Sequence[float] | None = None) -> list[float]:
    """Default reference frame time from the config, plus optional extra probes."""
    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    sampling = config.get("sampling", {}) if isinstance(config, Mapping) else {}
    base = float(
        map_view.get("reference_time_seconds", sampling.get("start_seconds", 0.0))
        if isinstance(map_view, Mapping)
        else 0.0
    )
    times = [base]
    for value in extra_times or []:
        time_value = float(value)
        if time_value not in times:
            times.append(time_value)
    return sorted(times)


def frame_filename(time_seconds: float) -> str:
    return f"reference_{time_seconds:09.3f}s.jpg"


def build_draft_asset(stage_id: str, source_box: StageBox) -> dict[str, Any]:
    """ROI-corner seed marked template=true so it cannot enable homography until edited."""
    return {
        "schema_version": 1,
        "template": True,
        "stage_id": stage_id,
        "coordinate_space": "video_pixels",
        "target_coordinate_space": "stage_normalized_0_1",
        "control_points": roi_corner_control_points(source_box),
        "notes": [
            "Draft seeded from the map ROI corners; replace every point with a real stage landmark.",
            "Set template to false only after all points are replaced, then validate with scripts/build_stage_control_points.py.",
        ],
    }


def draw_reference_overlay(
    frame: Any,
    source_box: StageBox,
    *,
    divisions: int = DEFAULT_GRID_DIVISIONS,
    exclude_regions: Sequence[Mapping[str, Any]] | None = None,
) -> Any:
    """Draw the ROI border, normalized grid, and excluded regions onto a frame copy."""
    import cv2

    output = frame.copy()
    lines = grid_lines(source_box, divisions)
    y1, y2 = int(source_box.y1), int(source_box.y2)
    x1, x2 = int(source_box.x1), int(source_box.x2)

    for region in exclude_regions or []:
        cv2.rectangle(
            output,
            (int(region["x1"]), int(region["y1"])),
            (int(region["x2"]), int(region["y2"])),
            (40, 40, 40),
            2,
        )
    for line in lines["vertical"]:
        x = int(line["x"])
        cv2.line(output, (x, y1), (x, y2), (200, 200, 200), 1)
        cv2.putText(output, f"{line['label']:.1f}", (x + 2, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for line in lines["horizontal"]:
        y = int(line["y"])
        cv2.line(output, (x1, y), (x2, y), (200, 200, 200), 1)
        cv2.putText(output, f"{line['label']:.1f}", (x1 + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
    return output


def export_reference_frames(
    video_path: Path,
    times: Sequence[float],
    frames_dir: Path,
    source_box: StageBox,
    *,
    divisions: int = DEFAULT_GRID_DIVISIONS,
    exclude_regions: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frames_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    try:
        for time_seconds in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_seconds)) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                exported.append({"time": float(time_seconds), "status": "unreadable", "path": ""})
                continue
            overlay = draw_reference_overlay(
                frame, source_box, divisions=divisions, exclude_regions=exclude_regions
            )
            target = frames_dir / frame_filename(float(time_seconds))
            if not cv2.imwrite(str(target), overlay):
                exported.append({"time": float(time_seconds), "status": "write_failed", "path": ""})
                continue
            exported.append({"time": float(time_seconds), "status": "exported", "path": display_path(target)})
    finally:
        cap.release()
    return exported


def build_reference_package(
    config: Mapping[str, Any],
    *,
    config_path: str,
    stage_id: str,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    extra_times: Sequence[float] | None = None,
    divisions: int = DEFAULT_GRID_DIVISIONS,
) -> dict[str, Any]:
    source_box = stage_box_from_config(config)
    match = config.get("match", {}) if isinstance(config, Mapping) else {}
    video_path = project_path(str(match.get("input_video", "")))
    if not video_path.is_file():
        raise RuntimeError(f"input video not found: {video_path}")

    map_view = config.get("map_view", {}) if isinstance(config, Mapping) else {}
    exclude_regions = map_view.get("exclude_regions", []) if isinstance(map_view, Mapping) else []
    times = reference_times(config, extra_times)

    package_dir = project_path(output_root) / stage_id
    frames = export_reference_frames(
        video_path,
        times,
        package_dir / "frames",
        source_box,
        divisions=divisions,
        exclude_regions=exclude_regions,
    )

    draft_path = package_dir / "control_points_draft.json"
    draft = build_draft_asset(stage_id, source_box)
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    guide_path = package_dir / "README.md"
    guide_path.write_text(
        FILL_GUIDE.format(
            config_path=config_path,
            draft_path=display_path(draft_path),
            stage_id=stage_id,
        ),
        encoding="utf-8",
    )

    exported = [item for item in frames if item["status"] == "exported"]
    manifest = {
        "schema_version": 1,
        "status": "ready" if exported else "no_frames",
        "stage_id": stage_id,
        "config": config_path,
        "video": display_path(video_path),
        "source_roi": source_box.as_dict(),
        "grid_divisions": divisions,
        "frames": frames,
        "exported_frames": len(exported),
        "draft_asset": display_path(draft_path),
        "guide": display_path(guide_path),
        "next_step": (
            "Fill real landmarks in the draft, set template to false, then validate with "
            "scripts/build_stage_control_points.py --validate --strict"
        ),
    }
    manifest_path = package_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["manifest_path"] = display_path(manifest_path)
    return manifest
