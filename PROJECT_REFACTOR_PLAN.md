# Refactor Plan

This document freezes the refactor target for the restored Splatoon 3 battle
analysis project. The current goal is to keep the working pipeline
reproducible while separating active code from historical references.

## Supported Entrypoints

Use these entrypoints for active development:

| Entrypoint | Purpose |
| --- | --- |
| `python -m src.run_analysis` | Main offline image/video analysis pipeline. |
| `python -m src.heatmap.run_pipeline` | Overhead-map heatmap pipeline. |
| `python scripts/check_project.py` | Local health check for the restored baseline. |
| `python scripts/summarize_csv.py` | CSV sanity summary. |
| `python scripts/report_csv.py` | Lightweight Markdown report for an analysis CSV. |
| `python scripts/evaluate_matches.py` | Fixed match evaluation with count smoothing and heatmap quality checks. |
| `python scripts/intake_match.py` | Register new local match videos and generate registry/evaluation entries. |
| `python scripts/report_model_quality.py` | Aggregate registry, evaluation, weapon training, and asset status. |
| `python scripts/report_model_errors.py` | Report likely YOLO/OCR/weapon risk signals from analysis CSVs. |
| `python scripts/plan_model_experiments.py` | Prioritize model replacement experiments from quality signals. |
| `python scripts/validate_data_registry.py` | Validate local videos, analysis windows, and heatmap artifacts. |
| `python scripts/report_heatmap_quality.py` | Generate heatmap trajectory quality reports from the data registry. |
| `python scripts/report_heatmap_comparison.py` | Compare registered heatmap matches and trajectory anomalies. |
| `python scripts/report_heatmap_quality_loop.py` | Export/evaluate heatmap annotation loops for manual point-label quality. |
| `python scripts/prepare_heatmap_annotation_round.py` | Prepare named heatmap annotation rounds and summarize label progress. |
| `python scripts/build_heatmap_annotation_ui.py` | Build a static browser helper for filling manual heatmap point labels. |
| `python scripts/suggest_heatmap_tuning.py` | Suggest heatmap tracker tuning actions from manual labels and anomaly reports. |
| `python scripts/run_heatmap_parameter_experiments.py` | Generate candidate heatmap parameter configs and experiment commands. |
| `python scripts/report_stage_coordinates.py` | Report and optionally export normalized heatmap stage coordinates. |
| `python scripts/build_stage_control_points.py` | Build or validate a stage control-point asset and report homography reprojection error. |
| `python scripts/export_heatmap_annotation_package.py` | Export frames and CSV templates for manual heatmap point labels. |
| `python scripts/evaluate_heatmap_annotations.py` | Evaluate heatmap predictions against manual player point labels. |
| `python scripts/export_heatmap_anomalies.py` | Export low-quality heatmap frames for manual review. |
| `python scripts/inventory_project.py` | Local model/footage/dataset inventory. |
| `python scripts/report_dataset_governance.py` | Check weapon dataset, labels, and registry metadata governance. |
| `python scripts/plan_model_training.py` | Plan detector/OCR training targets and required data assets. |
| `python scripts/validate_model_training_datasets.py` | Dry-run detector/OCR YOLO dataset layout, class names, and split counts before training. |
| `python scripts/plan_weapon_training.py` | Weapon dataset and training-plan inspection. |
| `python scripts/train_weapon_classifier.py` | Supported weapon classifier training CLI. |
| `python scripts/generate_weapon_dataset.py` | Optional synthetic weapon dataset generation from icon assets. |
| `python scripts/benchmark_model_experiments.py` | Build a repeatable benchmark matrix for planned model experiments. |
| `python scripts/report_model_data_readiness.py` | Check whether labels, validation, runtime, and governance are ready for the next data/model phase. |
| `python scripts/report_model_benchmark_baseline.py` | Snapshot current validation reports into `outputs/model_benchmarks/`. |
| `python scripts/run_model_experiment_baseline.py` | Generate a fixed model experiment baseline package with registry, dataset dry-run, readiness, and manifest artifacts. |
| `python scripts/run_with_runtime_report.py` | Time a command and write JSON/Markdown runtime reports. |
| `python scripts/report_runtime_benchmarks.py` | Aggregate runtime reports into a benchmark summary. |
| `python scripts/write_experiment_manifest.py` | Write source/artifact hashes and verification into an experiment manifest. |
| `python scripts/report_heatmap_productization.py` | Report heatmap productization readiness and milestones. |
| `python scripts/report_change_package.py` | Summarize the current worktree into review/commit batches. |

Files outside this list are not routine development entrypoints. Historical
scripts, old date-stamped root scripts, notebooks, realtime experiments, old
CSV outputs, duplicate root weights, and model export artifacts live under
`legacy/` or `notebooks/legacy/` until they are intentionally migrated behind a
supported CLI or test.

## Phase 1: Freeze The Baseline

- Keep `.venv`, `.cache`, footage, datasets, and generated outputs outside
  source control.
- Keep the canonical runtime weights in `models/*.pt` and `models/*.pth`.
  Treat TorchScript/CoreML exports as generated artifacts unless a supported
  entrypoint starts consuming them.
- Use `scripts/check_project.py` as the repeatable local health check.
- Treat `src.run_analysis` and `src.heatmap.run_pipeline` as the only supported
  analysis entrypoints.
- Record known model and field limits instead of hiding them:
  `stage` is empty, message OCR is conservative, and historical realtime/CoreML
  scripts are not yet automated.

## Phase 2: Consolidate Core Modules

Move shared logic out of large scripts without changing behavior:

- `src/core/paths.py`: project paths, cache environment, model paths.
- `src/detection.py`: YOLOv5 model loading, class mapping, detection helpers.
- `src/ocr.py`: number OCR and message OCR helpers.
- `src/weapons.py`: weapon crop, classification, and warmup voting.
- `src/media.py`: image/video frame iteration.
- `src/protocol.py`: keep the 33-column CSV/GameState contract as the single
  protocol layer.
- `src/heatmap/`: keep the new heatmap pipeline separate but reuse shared CSV
  and state helpers where practical.

## Phase 3: Make Evaluation Reproducible

- Add small fixture-driven checks for model loading, sample image analysis,
  protocol conversion, and heatmap color calibration.
- Add optional longer checks for `match_1` MPS and selected heatmap matches.
- Extend `scripts/summarize_csv.py` with warning-oriented metrics such as count
  jumps, empty weapon spans, player-state gaps, and message noise.
- Use `scripts/report_csv.py` when a saved Markdown report is useful.
- Current bridge: `config/data_registry.json` records local videos and heatmap
  artifacts, while `scripts/evaluate_matches.py` writes per-match raw/smoothed
  reports plus heatmap trajectory quality reports.
- Current bridge: `config/annotation_samples.json` drives small manual heatmap
  annotation packages and anomaly exports, allowing real point-label accuracy
  checks once manual labels are filled in.

## Phase 4: Rebuild Training Workflows

- Convert weapon classifier training scripts into parameterized CLIs.
- Separate dataset generation from model training.
- Add a metrics file for each training run.
- Keep old notebooks in `notebooks/legacy/` as references, but do not make them
  required for routine development.
- Current bridge: `src/weapon_training.py` owns the reusable dataset summary,
  label sync, synthetic generation, and ResNet18 training helpers.
- Current bridge: `scripts/plan_weapon_training.py` validates dataset/label/model
  class-count consistency; `scripts/train_weapon_classifier.py` can dry-run or
  train and writes metrics; `scripts/generate_weapon_dataset.py` can generate
  icon-based synthetic samples into an output directory.

## Phase 5: Improve Models And New Data

- Add new labeled videos and images through a documented data registry.
- Evaluate the current YOLOv5/OCR/weapon models before replacing them.
- Consider YOLOv8/YOLO11, RT-DETR, PaddleOCR, or newer classifiers only after
  baseline metrics show where the current stack is weak.
- Current bridge: `scripts/inventory_project.py` records local assets and can
  include hashes for snapshots.
- Current bridge: `scripts/intake_match.py` creates dry-run reports and safe
  registry/evaluation updates for new videos such as `match_12` or `match_13`.
- Current bridge: `scripts/report_model_quality.py` gives one Markdown/JSON
  overview for deciding whether to add data, retrain, or evaluate model swaps.
- Current bridge: `scripts/report_model_errors.py` turns analysis CSVs into a
  model/OCR risk report before any model replacement work starts.
- Current bridge: `config/model_experiments.json` and
  `scripts/plan_model_experiments.py` keep YOLO/OCR/classifier replacement
  ideas as measured experiments instead of immediate runtime swaps.

## Phase 6: Productize Analysis Outputs

- Normalize heatmap coordinates to stage maps.
- Join events, state rows, and player tracks through documented schemas.
- Add report generation that compares matches consistently.
- Keep realtime display as a later supported entrypoint after the offline
  pipeline remains stable.
- Current bridge: `scripts/report_csv.py` provides a small report path for
  CSV-first analysis outputs while heatmap reporting evolves separately.
- Current bridge: `scripts/report_heatmap_comparison.py` compares registered
  heatmap matches and ranks trajectory anomalies before coordinate
  normalization/event joins are available.
- Current bridge: `scripts/report_heatmap_quality_loop.py` turns heatmap
  predictions into manual annotation packages and evaluates filled labels with
  recall/error gates.
- Current bridge: `scripts/prepare_heatmap_annotation_round.py` creates named
  annotation batches such as `first_manual_loop`; `scripts/suggest_heatmap_tuning.py`
  converts filled labels into concrete tracker-tuning suggestions.
- Current bridge: `src.heatmap.run_pipeline --clean-output` and
  `outputs/<heatmap>/run_manifest.json` keep generated heatmap artifacts
  deterministic.
- Current bridge: `scripts/report_stage_coordinates.py` maps heatmap video-pixel
  ROI coordinates into `stage_x/stage_y` 0-1 coordinates, ready for later
  homography once stage-map control points are labeled.

## Local Check Commands

Quick baseline:

```bash
source scripts/use_local_env.sh
python -m unittest discover -s tests -q
python scripts/check_project.py
python scripts/check_project.py --tooling
```

Include a short video smoke test:

```bash
source scripts/use_local_env.sh
python scripts/check_project.py --video-smoke --device mps
```

Include the longer MPS baseline:

```bash
source scripts/use_local_env.sh
python scripts/check_project.py --long-mps
```

Inventory and reporting:

```bash
source scripts/use_local_env.sh
python scripts/inventory_project.py --output outputs/project_inventory.json
python scripts/intake_match.py --match-id match_12 --video footages/match_12.mp4 --start-seconds 10 --stop-seconds 150 --dry-run --strict
python scripts/report_model_quality.py --output outputs/model_quality.md --json-output outputs/model_quality.json
python scripts/report_model_errors.py --evaluation-results outputs/evaluation/evaluation_results.json --output outputs/model_errors.md --json-output outputs/model_errors.json
python scripts/plan_model_experiments.py --model-errors outputs/model_errors.json --heatmap-comparison outputs/heatmap_comparison.json --output outputs/model_experiment_plan.md --json-output outputs/model_experiment_plan.json
python scripts/validate_data_registry.py --output outputs/data_registry.json --report outputs/data_registry.md --strict
python scripts/report_csv.py outputs/match_1.csv
python scripts/report_heatmap_quality.py --output-dir outputs/heatmap_quality --strict
python scripts/report_heatmap_comparison.py --output outputs/heatmap_comparison.md --json-output outputs/heatmap_comparison.json --strict
python scripts/report_heatmap_quality_loop.py --export-package --package-dir outputs/heatmap_quality_loop
python scripts/prepare_heatmap_annotation_round.py --round-id first_manual_loop --package-dir outputs/heatmap_annotation_round1
python scripts/build_heatmap_annotation_ui.py --annotation-csv outputs/heatmap_annotation_round1/annotation_template.csv --output outputs/heatmap_annotation_round1/annotation_ui.html
python scripts/suggest_heatmap_tuning.py --annotation-csv outputs/heatmap_annotation_round1/annotation_template.csv --heatmap-comparison outputs/heatmap_comparison.json
python scripts/run_heatmap_parameter_experiments.py --annotation-csv outputs/heatmap_annotation_round1/annotation_template.csv --write-configs
python scripts/report_stage_coordinates.py --config src/heatmap/config_match9.yaml --normalized-output outputs/heatmap_match9/player_tracks_stage.csv
python scripts/build_stage_control_points.py --config src/heatmap/config_match9.yaml --stage-id match9_seed --report outputs/stage_control_points.md --strict
python scripts/export_heatmap_annotation_package.py --output-dir outputs/annotation_samples
python scripts/evaluate_heatmap_annotations.py outputs/annotation_samples/annotation_template.csv
python scripts/export_heatmap_anomalies.py --output-dir outputs/heatmap_anomalies
python scripts/report_dataset_governance.py --output outputs/dataset_governance.md --json-output outputs/dataset_governance.json
python scripts/benchmark_model_experiments.py --experiment-plan outputs/model_experiment_plan.json --output outputs/model_benchmark_plan.md --json-output outputs/model_benchmark_plan.json
python scripts/report_model_benchmark_baseline.py --output outputs/model_benchmarks/baseline_snapshot.md --json-output outputs/model_benchmarks/baseline_snapshot.json
python scripts/report_model_data_readiness.py --output outputs/model_data_readiness.md --json-output outputs/model_data_readiness.json
python scripts/run_with_runtime_report.py --name validation_suite --output outputs/runtime/validation_suite.json -- python scripts/run_validation_suite.py
python scripts/report_runtime_benchmarks.py --output outputs/runtime/runtime_benchmarks.md --json-output outputs/runtime/runtime_benchmarks.json
python scripts/write_experiment_manifest.py --experiment-id local_refactor_baseline --artifact baseline=outputs/model_benchmarks/baseline_snapshot.json
python scripts/report_heatmap_productization.py --output outputs/heatmap_productization.md --json-output outputs/heatmap_productization.json
python scripts/report_change_package.py --verification "python -m unittest discover -s tests -q"
python scripts/plan_weapon_training.py
python scripts/train_weapon_classifier.py --dry-run --max-samples-per-class 1 --epochs 1
python scripts/generate_weapon_dataset.py --dry-run --images-per-class 1
```

Project structure:

```bash
cat PROJECT_LAYOUT.md
cat legacy/README.md
cat notebooks/README.md
```
