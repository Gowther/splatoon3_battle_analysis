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

## Adding New Data

1. Put raw videos under `footages/`.
2. Run a short smoke analysis with `src.run_analysis`.
3. Run `scripts/summarize_csv.py` and `scripts/report_csv.py`.
4. Keep notes about video source, mode, stage, colors, and any visible OCR or
   weapon mistakes.
5. Only promote data into training/evaluation once labels or expected outputs
   are documented.
