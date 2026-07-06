# Project State

## Goal

Restore this Splatoon 3 video analysis project so it can run reliably on a
Mac mini M4 16GB and produce trustworthy CSV data, with all runtime caches and
the Python environment kept inside this external-drive project directory.

For the full project flow and technical architecture notes, see
`PROJECT_FLOW_TECHNICAL_NOTES.md`.

For the overhead-map heatmap roadmap based on `footages/match_9.mp4`, see
`HEATMAP_GOALS.md`.

## Phases

1. Freeze and inventory the local project without deleting existing models,
   footage, notebooks, or historical scripts.
2. Rebuild a local Python environment and all package/runtime caches under
   `.venv` and `.cache`.
3. Establish one primary run path for video analysis.
4. Fix runtime issues around paths, device selection, PyTorch model loading,
   YOLOv5 class mapping, preview color handling, weapon warmup, and CSV output.
5. Validate on the sample image and a longer segment of bundled footage.
6. Document commands, output format, verified results, and remaining limits.
7. Extend confidence later by testing more footage and checking fields that are
   still model-limited.

## Primary Run Path

Use this entry point for normal analysis:

```bash
source scripts/use_local_env.sh
python -m src.run_analysis --input footages/match_1.mp4 --output outputs/match_1.csv --device mps --start-seconds 10 --sample-fps 5
```

Historical scripts under `src/` and `yolov5/` are kept as project history, but
they are not the current supported path.

## Match 9 Heatmap MVP

The overhead-map heatmap pipeline for `footages/match_9.mp4` is now runnable
through a single command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.run_pipeline --config src/heatmap/config_match9.yaml
```

Final report:

- `outputs/heatmap_match9/report.md`

Verified fresh-run summary:

- valid map frames: 140
- invalid map frames: 1
- raw marker candidates: 1107
- cleaned team points: 1099
- UI-state rows joined from `src.run_analysis`: 141
- enriched point rows: 1099
- event rows: 0, because no real external kill/death CSV has been supplied
- experimental slot/player rows: 1099
- player route images: 8

Important heatmap outputs:

- `outputs/heatmap_match9/rendered/heatmap_yellow.png`
- `outputs/heatmap_match9/rendered/heatmap_blue.png`
- `outputs/heatmap_match9/rendered/heatmap_combined.png`
- `outputs/heatmap_match9/rendered/team_routes.png`
- `outputs/heatmap_match9/player_routes/`

Known heatmap limits:

- Coordinates are still source-video pixels, not a normalized stage-map
  homography.
- Marker detection is an MVP and can confuse nearby ink patches with player
  markers.
- `player_tracks.csv` is experimental slot-level tracking, not verified player
  identity.
- Event joins are ready, but require an external event CSV.

## Local Storage

- `.venv`: local Python environment, about 825M.
- `.cache`: uv, pip, torch, matplotlib, and Python bytecode caches, about 917M.
- `outputs`: generated CSVs and preview images.
- `.gitignore` ignores `.cache/` and `outputs/`.

## Required Assets

| Path | Purpose | SHA-256 |
| --- | --- | --- |
| `models/the_model.pt` | main object detector | `6efcd8f127cf305bc0c3a6f6f7a91ebfc8326de4d4f0f66abeb1afef419fe898` |
| `models/ocr_model.pt` | number OCR | `09fc73a39b7b0cac283b92e7456799b031454c97194599aa6a1ebbc4c3aa1c46` |
| `models/message_ocr_model.pt` | message OCR | `f5567bc1ec4632927637d9778a8645f366d2f8b92ac3fc26e163ce51b4c12519` |
| `models/main_weapons_classification_weight.pth` | weapon classifier | `cd2896d6d9a39a799039c96af8cd6e0871fe150b5e07b7d04ba2155a9c29be3e` |
| `main_weapon_list.txt` | weapon labels | `7284d44b7f446dbb5c5e46d53b32ee8a4bdd2380fcd4ff8e73a59636c7cba62b` |

## Verified Result

Final MPS verification was run outside the Codex sandbox because the sandbox
does not expose Apple's MPS backend:

```bash
python -m src.run_analysis --input footages/match_1.mp4 --output outputs/match_1_analysis_10_150_mps_final.csv --device mps --start-seconds 10 --stop-seconds 150 --sample-fps 5
python scripts/summarize_csv.py outputs/match_1_analysis_10_150_mps_final.csv
```

Summary:

- Device: `mps`
- Rows: 701
- Elapsed range: 10.0 to 150.0 seconds
- 8-player state rows: 531
- Weapon rows: 677
- Count rows: 551
- Objective rows: 587
- Player-detected rows: 96
- Message rows: 0
- First stable count samples: right count `99` from 27.6 seconds onward
- Warmup weapon vote:
  `Splash-o-matic`, `Splattershot-Jr`, `Order_Blaster_Replica`,
  `Nautilus_79`, `Dualie-Squelchers`, `Splattershot-Jr`,
  `Annaki_Splattershot_Nova`, `Custom-Splattershot-Jr`

Additional checks:

- `py_compile` passed for `src/run_analysis.py` and `scripts/summarize_csv.py`
  when `PYTHONPYCACHEPREFIX` points to `.cache/pycache`.
- `sample/battle.png` runs successfully on CPU.
- The 72.0-73.2 second message probe now produces `message rows: 0`, avoiding
  the previous low-confidence single-character noise.

## Current Limits

- Message OCR is conservative by default. Lower `--message-char-conf` only when
  inspecting debug previews.
- Stage is not populated yet.
- Weapon classification depends on warmup voting from valid 8-player frames.
  Start near the opening/team display for the most stable names.
- The main validation target so far is `footages/match_1.mp4` from 10 to 150
  seconds. More videos and modes should be tested before treating every CSV
  field as broadly validated.
