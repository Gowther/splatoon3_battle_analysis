# Legacy Archive

This folder keeps historical project material out of the active development
path without deleting it. Files here are references, not supported entrypoints.
They may depend on old paths, local devices, old data layouts, or packages that
are not covered by `scripts/check_project.py`.

## Contents

| Path | Contents | Use |
| --- | --- | --- |
| `root_scripts/` | Old date-stamped root scripts for weapon inference, weapon training, dataset generation, and icon resizing. | Reference when rebuilding formal training/data CLIs. |
| `realtime/src/` | Old realtime receiver and display experiments, including camera/weapon helpers. | Reference for a future supported realtime mode. |
| `yolov5_analysis/` | Historical custom YOLOv5 analysis scripts, CoreML experiments, realtime scripts, and conversion helpers. | Reference when comparing older analysis behavior or rebuilding export flows. |
| `yolov5_training/` | Old YOLOv5 training metadata and dataset notes. | Reference when rebuilding detector training workflows. |
| `artifacts/root_csv/` | Old root-level generated CSV outputs. | Historical output samples only. |
| `artifacts/weights/` | Duplicate root-level `.pth` weights that also exist under `models/`. | Historical backup; supported commands use `models/`. |
| `artifacts/yolov5/` | Old YOLOv5 output CSVs. | Historical output samples only. |
| `artifacts/model_exports/` | Generated TorchScript/CoreML model exports. | Keep as generated artifacts until an active entrypoint consumes them. |
| `artifacts/icons/` | Icon assets that are not part of the current 116-class weapon dataset. | Historical backup; active generation uses `main_icons/`. |
| `artifacts/labels/` | Old label lists that are not used by the supported classifier pipeline. | Historical backup; active inference uses `main_weapon_list.txt`. |

## Promotion Rule

If a legacy file becomes useful again, do not call it directly from its archive
location. First move the reusable logic into `src/`, add a small CLI under
`scripts/`, and include it in `scripts/check_project.py` or the fixed evaluation
suite when practical.
