# Evaluation

The fixed evaluation set lives in `config/evaluation_matches.json`.
The reusable local data registry lives in `config/data_registry.json`.

## Data Registry

Use this before adding a match to evaluation or training:

```bash
.venv/bin/python scripts/validate_data_registry.py --output outputs/data_registry.json --report outputs/data_registry.md --strict
```

The registry records local videos, analysis windows, heatmap output directories,
team colors, key artifacts, and conservative trajectory quality gates.

## Manual Heatmap Annotations

Use this to create a small manual labeling package from existing heatmap frames
and predictions:

```bash
.venv/bin/python scripts/export_heatmap_annotation_package.py --output-dir outputs/annotation_samples
```

This writes:

- `outputs/annotation_samples/annotation_template.csv`
- `outputs/annotation_samples/prediction_reference.csv`
- `outputs/annotation_samples/frames/`
- `outputs/annotation_samples/previews/`
- `outputs/annotation_samples/manifest.json`

Fill `x` and `y` in `annotation_template.csv` with manual player center points.
Prediction columns are reference hints only. Set `frame_complete` to `true`
only when every visible player for that team/frame has been labeled.

Evaluate completed labels with:

```bash
.venv/bin/python scripts/evaluate_heatmap_annotations.py outputs/annotation_samples/annotation_template.csv --report outputs/annotation_samples/evaluation.md
```

The evaluator reports matched labels, missed labels, recall, precision on
complete frame/team groups, and pixel error metrics.

## Heatmap Anomaly Export

Use this to export frames worth reviewing before labeling or model work:

```bash
.venv/bin/python scripts/export_heatmap_anomalies.py --output-dir outputs/heatmap_anomalies
```

The anomaly package includes jump resets, large movement steps, low-confidence
points, and track gaps. It writes copied frames, preview images, `anomalies.csv`,
and `summary.json`.

## Report Only

Use this when you want to check existing outputs without rerunning analysis:

```bash
.venv/bin/python scripts/evaluate_matches.py
```

This writes:

- `outputs/evaluation/evaluation_results.json`
- `outputs/evaluation/evaluation_report.md`
- `outputs/evaluation/<heatmap_id>/trajectory_quality.json`
- `outputs/evaluation/<heatmap_id>/trajectory_quality.md`

## Full Analysis Evaluation

Use this to rerun the current match baseline and regenerate per-match reports:

```bash
.venv/bin/python scripts/evaluate_matches.py --run-analysis --strict
```

For `match_1_10_150`, the script writes:

- `outputs/evaluation/match_1_10_150/raw.csv`
- `outputs/evaluation/match_1_10_150/smoothed.csv`
- `outputs/evaluation/match_1_10_150/count_smoothing.json`
- `outputs/evaluation/match_1_10_150/report.md`

The raw baseline is compared with the expected row counts in
`config/evaluation_matches.json`. The smoothed CSV is checked by a quality gate
that expects zero remaining count jump warnings.

## Count Smoothing Only

Use this when you already have a CSV and only want to remove short OCR count
jitter:

```bash
.venv/bin/python scripts/smooth_counts.py input.csv --output smoothed.csv --report count_smoothing.json
```

The smoothing step is conservative: it only rewrites short count or penalty
noise runs when they are bracketed by nearby stable values.

## Heatmap Trajectory Quality Only

Use this when you want to compare player-track quality without rerunning the
heatmap pipeline:

```bash
.venv/bin/python scripts/report_heatmap_quality.py --output-dir outputs/heatmap_quality --strict
```

The quality report checks player-track rows, gap ratio, jump-reset ratio, team
coverage, and player route image count for the heatmap matches in the registry.

## Project Health Check

The fixed evaluation suite is also available through the project check script:

```bash
.venv/bin/python scripts/check_project.py --evaluation --skip-yolov5-detect
```

That command writes evaluation artifacts under the check script's temporary
work directory. It validates the data registry before running the fixed match
evaluation.
