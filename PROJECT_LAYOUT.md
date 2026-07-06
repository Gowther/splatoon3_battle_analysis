# Project Layout

This repository is now split into active development surfaces and legacy
references. The cleanup is intentionally behavior-preserving: supported
commands, model paths, CSV schemas, and heatmap outputs stay the same.

## Active Surfaces

| Path | Status | Notes |
| --- | --- | --- |
| `src/run_analysis.py` | Supported | Main offline image/video YOLO/OCR/weapon pipeline. |
| `src/detection.py` | Supported | YOLOv5 loading and UI element detection helpers. |
| `src/ocr.py` | Supported | Count OCR and message OCR helpers. |
| `src/weapons.py` | Supported | Weapon crop/classification helpers. |
| `src/weapon_training.py` | Supported | Weapon dataset validation, label sync, synthetic generation, and training helpers. |
| `src/media.py` | Supported | Image/video frame iteration. |
| `src/protocol.py` | Supported | 33-column CSV and GameState protocol contract. |
| `src/heatmap/` | Supported | Heatmap calibration, player marker detection, trajectory quality, annotation, and anomaly helpers. |
| `src/heatmap/comparison_report.py` | Supported | Cross-match heatmap quality and anomaly comparison reporting. |
| `src/data_registry.py` | Supported | Data registry path resolution and validation support. |
| `src/match_intake.py` | Supported | New match registry/evaluation intake planning and safe JSON updates. |
| `src/model_quality.py` | Supported | Aggregated registry/evaluation/weapon/assets quality overview. |
| `src/model_error_report.py` | Supported | CSV-level model/OCR risk signal reporting. |
| `src/model_experiments.py` | Supported | Model replacement experiment planning and prioritization. |
| `scripts/` | Supported | Health checks, summaries, evaluation, match intake, quality overview, model error reports, experiment planning, registry validation, heatmap reports, annotation exports, weapon training CLIs, and training planning. |
| `config/` | Supported | Evaluation, registry, and annotation sample configuration. |
| `models/` | Runtime assets | Canonical `.pt` and `.pth` weights used by supported commands. |
| `main_weapon_list.txt` | Runtime asset | Weapon classifier output-index to label mapping; must match the classifier output count. |
| `main_icons/` | Training asset | Active weapon icon source set used by synthetic dataset generation. |
| `sample/` | Runtime fixture | Small image fixture used by health checks. |
| `tests/` | Supported | Fast stdlib unit tests for refactored pure logic. |
| `yolov5/` | Vendor/runtime dependency | Upstream YOLOv5 code used by the current detector and raw smoke check. |

## Local Data

These paths are expected to be local-machine assets and are ignored by git:

- `footages/`: local match videos.
- `data/`: local heatmap output and intermediate data.
- `main_training_dataset/`: local weapon classifier training dataset.
- `outputs/`: generated reports and experiment outputs.
- `.cache/`: runtime caches for PyTorch, pip, matplotlib, and pycache.

## Legacy References

Legacy material is preserved under `legacy/` and `notebooks/legacy/`. It is not
compiled or exercised by `scripts/check_project.py` unless a file is promoted
back into a supported CLI/test.

See `legacy/README.md` and `notebooks/README.md` before using archived files.

## Development Rule

For new work, prefer adding a supported CLI under `scripts/` or a small module
under `src/`. Avoid restoring date-stamped root scripts or ad-hoc scripts under
`yolov5/`; use `legacy/` only for references that should not participate in the
daily health check.
