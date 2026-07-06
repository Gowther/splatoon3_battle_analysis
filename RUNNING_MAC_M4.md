# Running On Mac M4

This project is configured to keep the virtual environment and package caches inside the external-drive project directory.

See `PROJECT_STATE.md` for the full phase goal, asset hashes, verified MPS run, and current field limits.

## Setup

```bash
mkdir -p .cache/uv .cache/pip .cache/torch .cache/matplotlib outputs
UV_CACHE_DIR=.cache/uv PIP_CACHE_DIR=.cache/pip TORCH_HOME=.cache/torch MPLCONFIGDIR=.cache/matplotlib uv venv .venv --python /usr/bin/python3
UV_CACHE_DIR=.cache/uv PIP_CACHE_DIR=.cache/pip TORCH_HOME=.cache/torch MPLCONFIGDIR=.cache/matplotlib uv pip install --python .venv/bin/python -r requirements-mac-m4.txt
```

## Use

```bash
source scripts/use_local_env.sh
python -m src.run_analysis --input sample/battle.png --output outputs/sample.csv --max-frames 1
python -m src.run_analysis --input footages/YOUR_VIDEO.mp4 --output outputs/YOUR_VIDEO.csv --device mps --start-seconds 10 --sample-fps 5 --preview
```

Press `q` in the preview window to stop early.

For a baseline health check after changing code:

```bash
source scripts/use_local_env.sh
python scripts/check_project.py
python scripts/check_project.py --tooling
python scripts/check_project.py --video-smoke --device mps
```

For a non-interactive smoke test:

```bash
python -m src.run_analysis --input footages/match_1.mp4 --output outputs/match_1_smoke.csv --device mps --start-seconds 10 --sample-fps 5 --max-frames 40
python scripts/summarize_csv.py outputs/match_1_smoke.csv
```

For a longer verified range from the bundled footage:

```bash
python -m src.run_analysis --input footages/match_1.mp4 --output outputs/match_1_analysis_10_150.csv --device mps --start-seconds 10 --stop-seconds 150 --sample-fps 5
python scripts/summarize_csv.py outputs/match_1_analysis_10_150.csv
```

Start near the opening/team display when you want stable weapon names. Weapon
classification is voted from the first valid 8-player frames; starting in the
middle of a match can produce weaker votes.

To save annotated frames for debugging without opening a window:

```bash
python -m src.run_analysis --input footages/match_1.mp4 --output outputs/probe.csv --device mps --start-seconds 22.4 --sample-fps 5 --max-frames 10 --save-preview-dir outputs/previews_probe
```

## Output

The CSV keeps the existing 33-column protocol:

- `elapsed_time`
- `player_state_1` through `player_state_8`
- `count_left`, `count_right`, `penalty_left`, `penalty_right`
- `weapon_1` through `weapon_8`
- objective counts, message, player detection, timestamp, and reserved fields

## Notes

- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is required for these older YOLOv5 `.pt` model files on PyTorch 2.6+.
- The script uses MPS only when `torch.backends.mps.is_available()` reports true; otherwise it falls back to CPU. Codex sandbox may report MPS unavailable, while a normal Terminal session can still use MPS.
- The active detection model reports 24 classes from `models/the_model.pt`; `yolov5/data.yaml` is not treated as authoritative for runtime analysis.
- For quick verification, run `--max-frames 40` before analyzing a full match.
- Count OCR is filtered by confidence by default: `--count-box-conf 0.5` and `--digit-conf 0.5`. Lower these only when you are inspecting debug previews.
- Message OCR is still experimental. The default `--message-char-conf 0.55` intentionally suppresses low-confidence partial messages rather than writing noisy text.
