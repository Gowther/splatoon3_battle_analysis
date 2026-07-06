# Data And Training

This project should grow by adding reproducible data and evaluation paths
before swapping models.

## Asset Inventory

Use the inventory script before and after adding new local assets:

```bash
source scripts/use_local_env.sh
python scripts/inventory_project.py --output outputs/project_inventory.json
```

Add `--hash` when you need stable hashes for model or dataset snapshots.

## Weapon Classifier Planning

The old weapon classifier training scripts are historical references. The
supported flow starts by validating that the dataset folder order, label file,
and current model output class count agree:

```bash
source scripts/use_local_env.sh
python scripts/plan_weapon_training.py --dataset main_training_dataset --labels main_weapon_list.txt --strict
```

If the class list has intentionally changed, regenerate the label file from the
dataset folder order after reviewing the warning:

```bash
source scripts/use_local_env.sh
python scripts/plan_weapon_training.py --write-labels --strict
```

## Weapon Classifier Training

Run a dry-run first. It builds deterministic train/validation/test splits and
checks the class order without writing a model:

```bash
source scripts/use_local_env.sh
python scripts/train_weapon_classifier.py --dry-run --max-samples-per-class 1 --epochs 1
```

Run real training only after the dry-run looks right:

```bash
source scripts/use_local_env.sh
python scripts/train_weapon_classifier.py \
  --dataset main_training_dataset \
  --labels main_weapon_list.txt \
  --output models/main_weapons_classification_weight.pth \
  --metrics outputs/weapon_training_metrics.json \
  --epochs 25 \
  --batch-size 32 \
  --device auto \
  --write-labels
```

The training CLI keeps the existing runtime contract: it writes a full
TorchVision ResNet18 model object to `models/main_weapons_classification_weight.pth`
and writes the class order to `main_weapon_list.txt`.

## Synthetic Weapon Dataset Generation

The old random image generator is now optional and writes into a chosen output
directory instead of touching the active dataset by default:

```bash
source scripts/use_local_env.sh
python scripts/generate_weapon_dataset.py --dry-run --images-per-class 1
python scripts/generate_weapon_dataset.py \
  --icons main_icons \
  --backgrounds sample \
  --output-dir outputs/generated_weapon_dataset \
  --images-per-class 50 \
  --write-labels outputs/generated_weapon_labels.txt
```

## CSV Reports

For a quick Markdown summary of a CSV:

```bash
source scripts/use_local_env.sh
python scripts/report_csv.py outputs/match_1.csv
```

The report uses the same warning logic as `scripts/summarize_csv.py`.

## Quality Overview

Before adding data, retraining, or evaluating model swaps, generate the current
project quality overview:

```bash
source scripts/use_local_env.sh
python scripts/report_model_quality.py \
  --evaluation-results outputs/evaluation/evaluation_results.json \
  --output outputs/model_quality.md \
  --json-output outputs/model_quality.json
```

If your latest fixed evaluation lives outside `outputs/evaluation/`, pass that
`evaluation_results.json` path with `--evaluation-results`.

## Model Error Reports

Use the model error report before deciding whether a problem belongs to YOLO,
OCR, weapon classification, thresholds, or missing data:

```bash
source scripts/use_local_env.sh
python scripts/report_model_errors.py \
  --evaluation-results outputs/evaluation/evaluation_results.json \
  --output outputs/model_errors.md \
  --json-output outputs/model_errors.json
```

The report flags CSV-level risk signals such as missing player states, unstable
weapon rows, sparse count OCR, count jumps, and message OCR activity. These are
triage hints, not a replacement for manual labels.

## Heatmap Comparison

Once heatmap outputs are registered, compare matches and anomaly priorities:

```bash
source scripts/use_local_env.sh
python scripts/report_heatmap_comparison.py \
  --output outputs/heatmap_comparison.md \
  --json-output outputs/heatmap_comparison.json \
  --strict
```

Use the anomaly samples to pick frames for manual point labels before investing
in coordinate normalization or event joins.

## Model Experiment Planning

After generating quality and error reports, turn candidate model swaps into a
measured experiment plan:

```bash
source scripts/use_local_env.sh
python scripts/plan_model_experiments.py \
  --model-errors outputs/model_errors.json \
  --heatmap-comparison outputs/heatmap_comparison.json \
  --output outputs/model_experiment_plan.md \
  --json-output outputs/model_experiment_plan.json
```

The plan keeps YOLO11/YOLOv8, PaddleOCR, newer weapon classifiers, and heatmap
detector ideas as ranked experiments with baseline commands and pass criteria.

## Adding New Data

Start with the intake helper so registry and evaluation config stay in sync:

```bash
source scripts/use_local_env.sh
python scripts/intake_match.py \
  --match-id match_12 \
  --video footages/match_12.mp4 \
  --start-seconds 10 \
  --stop-seconds 150 \
  --sample-fps 5 \
  --device mps \
  --purpose analysis_candidate \
  --notes "source/mode/stage/colors notes" \
  --dry-run \
  --strict \
  --report outputs/match_12_intake.md
```

Review the report, then write the registry/evaluation entries:

```bash
python scripts/intake_match.py \
  --match-id match_12 \
  --video footages/match_12.mp4 \
  --start-seconds 10 \
  --stop-seconds 150 \
  --sample-fps 5 \
  --device mps \
  --purpose analysis_candidate \
  --notes "source/mode/stage/colors notes" \
  --write \
  --strict
```

Then validate and run the new analysis window:

```bash
python scripts/validate_data_registry.py --strict
python scripts/evaluate_matches.py --only match_12_10_150 --run-analysis --strict
```

Keep notes about video source, mode, stage, team colors, and visible OCR or
weapon mistakes. Only promote data into training/evaluation baselines once
labels or expected outputs are documented.

Current new-data smoke sample:

```bash
python scripts/evaluate_matches.py --only match_11_20_40 --run-analysis --strict
python scripts/report_model_quality.py \
  --evaluation-results outputs/evaluation/evaluation_results.json \
  --output outputs/model_quality.md \
  --json-output outputs/model_quality.json
```

`match_11_20_40` is intentionally a short analysis-candidate window. Use it to
verify the intake/evaluation path before promoting longer windows with fixed
expected metrics.
