# Heatmap Goals

This document breaks the overhead-map heatmap idea into executable goals. The
first sample target is `footages/match_9.mp4`.

The heatmap pipeline must stay independent from the verified analysis entry
point in `src/run_analysis.py`. It should write generated data under
`outputs/heatmap_match9/` and should not delete existing models, footage,
notebooks, or historical scripts.

## Current Sample Baseline

- Input: `footages/match_9.mp4`
- Resolution: 1920x1080
- FPS: 60
- Duration: about 308.07 seconds
- Estimated overhead-map interval: 20s to 160s
- Non-map ranges seen in probes:
  - 0s to 10s: opening/team scenes
  - about 170s onward: win/result scenes
- Useful probe outputs already generated:
  - `outputs/match9_overhead_contact.jpg`
  - `outputs/match9_overhead_contact_extra.jpg`
  - `outputs/match9_overhead_probe/`
  - `outputs/match9_model_probe_60/`

## Goal 1: Roadmap And Match9 Baseline Config

Create the planning and configuration base for the heatmap work.

Deliverables:

- `HEATMAP_GOALS.md`
- `src/heatmap/config_match9.yaml`
- Link from existing project notes, when useful.

Acceptance:

- All later goals are listed with deliverables.
- `match_9` has a configured input path, output directory, sample interval,
  initial map ROI, and first-pass detection settings.
- No existing runtime entry point is changed.

## Goal 2: Frame Extraction And Map-View Filtering

Build a small extractor for overhead-map frames.

Deliverables:

- `src/heatmap/extract_frames.py`
- `outputs/heatmap_match9/frames/`
- `outputs/heatmap_match9/valid_frames.csv`
- `outputs/heatmap_match9/invalid_frames.csv`
- `outputs/heatmap_match9/probes/contact.jpg`

Acceptance:

- The script can sample `match_9` between 20s and 160s.
- Opening, result, and clearly non-map frames are excluded or marked invalid.
- Sampling rate is configurable, with 1 FPS as the initial default.

Command:

```bash
python -m src.heatmap.extract_frames --config src/heatmap/config_match9.yaml --contact-limit 24
```

Current `match_9` baseline:

- sampled frames: 141
- valid frames: 140
- invalid frames: 1
- invalid sample: 156.0s, `low_team_color_ratio`

## Goal 3: Map ROI, Mask, And Pixel Coordinate System

Define a stable map coordinate frame.

Deliverables:

- ROI and mask generation in the heatmap pipeline.
- `outputs/heatmap_match9/map_roi_debug.jpg`
- `outputs/heatmap_match9/map_mask.png`

Acceptance:

- The first version uses source-video pixel coordinates.
- The configured ROI includes the overhead map and excludes most operation UI.
- The coordinate convention is documented as `x, y` in video pixels unless a
  later homography stage is added.

Command:

```bash
python -m src.heatmap.build_map_mask --config src/heatmap/config_match9.yaml
```

Current `match_9` geometry:

- coordinate space: `video_pixels`
- ROI: `x=0..1760`, `y=180..980`
- mask output: `outputs/heatmap_match9/map_mask.png`
- debug output: `outputs/heatmap_match9/map_roi_debug.jpg`
- usable mask ratio: about 0.66 of the 1920x1080 frame

## Goal 4: Team Marker Detection MVP

Detect yellow/blue team marker candidates on the overhead map.

Deliverables:

- `src/heatmap/detect_markers.py`
- `outputs/heatmap_match9/team_points_raw.csv`
- `outputs/heatmap_match9/debug_markers/`

Acceptance:

- The detector produces candidate points with `time`, `frame_index`, `team`,
  `x`, `y`, `confidence`, and `source`.
- First pass may use color thresholding, connected components, and geometric
  filtering.
- Player identity is optional and can remain empty.

Status: complete.

Implemented a label-guided color component detector for `match_9`.

Command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.detect_markers --config src/heatmap/config_match9.yaml
```

Current result:

- raw candidate output: `outputs/heatmap_match9/team_points_raw.csv`
- debug output: `outputs/heatmap_match9/debug_markers/`
- raw candidate rows: 1107
- team counts: yellow 560, blue 547
- per-frame candidate count: 131 frames with 8 points, 6 frames with 7 points,
  2 frames with 6 points, 1 frame with 5 points
- debug images: 36 files, from `20.000s` through `55.000s`

Known limitation:

- This is still an MVP candidate detector. It deliberately favors recall and
  stable per-frame point counts, so some points can still land on nearby ink
  patches instead of exact player icons. Goal 5 will clean these raw points.

## Goal 5: Point Cleaning And Team-Level Tracks

Turn raw marker candidates into cleaner team-level point data.

Deliverables:

- `outputs/heatmap_match9/team_points.csv`
- `outputs/heatmap_match9/team_tracks.csv`
- Debug plots for removed outliers.

Acceptance:

- Low-confidence and obviously invalid points are removed.
- Large frame-to-frame jumps are filtered or marked.
- Yellow and blue team activity paths are roughly coherent.

Status: complete.

Implemented `src/heatmap/clean_points.py`.

Command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.clean_points --config src/heatmap/config_match9.yaml
```

Current result:

- cleaned points: `outputs/heatmap_match9/team_points.csv`
- temporary team tracks: `outputs/heatmap_match9/team_tracks.csv`
- rejected points: `outputs/heatmap_match9/team_points_rejected.csv`
- cleaning report: `outputs/heatmap_match9/point_cleaning_report.csv`
- cleaning debug image:
  `outputs/heatmap_match9/cleaning_debug/cleaned_points_overview.jpg`
- raw points: 1107
- clean points: 1099
- rejected points: 8, all `low_confidence`
- clean team counts: yellow 560, blue 539
- track rows: 1099
- track status: `matched` 949, `jump_reset` 142, `new` 8
- per-frame cleaned count: 128 frames with 8 points, 5 frames with 7 points,
  5 frames with 6 points, 2 frames with 5 points

Known limitation:

- The four `track_slot` values per team are temporary continuity slots, not
  verified player identities. They are good enough for team-level path and
  heatmap rendering, but player-level identity should be a later goal.

## Goal 6: Team Heatmaps And Route Rendering

Render visual outputs from cleaned team points.

Deliverables:

- `src/heatmap/render_heatmaps.py`
- `outputs/heatmap_match9/rendered/heatmap_yellow.png`
- `outputs/heatmap_match9/rendered/heatmap_blue.png`
- `outputs/heatmap_match9/rendered/heatmap_combined.png`
- `outputs/heatmap_match9/rendered/team_routes.png`
- `outputs/heatmap_match9/render_report.csv`

Acceptance:

- Hot and cold areas are visible.
- Yellow and blue overlays are visually separable.
- The report records input video, sampled time range, point counts, and known
  failure cases.

Status: complete.

Implemented `src/heatmap/render_heatmaps.py`.

Command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.render_heatmaps --config src/heatmap/config_match9.yaml
```

Current result:

- yellow heatmap:
  `outputs/heatmap_match9/rendered/heatmap_yellow.png`
- blue heatmap:
  `outputs/heatmap_match9/rendered/heatmap_blue.png`
- combined heatmap:
  `outputs/heatmap_match9/rendered/heatmap_combined.png`
- team-level route/local movement map:
  `outputs/heatmap_match9/rendered/team_routes.png`
- render report: `outputs/heatmap_match9/render_report.csv`
- rendered from clean points: 1099
- team counts: yellow 560, blue 539
- route rendering uses `team_tracks.csv`, but only draws short matched segments
  up to `route_max_draw_step_px: 120`

Known limitation:

- `team_routes.png` is a team-level local movement visualization, not a true
  player identity route map. It intentionally avoids drawing long jumps because
  Goal 5 tracks are only temporary slots.

## Goal 7: Join With UI State CSV

Enrich heatmap points with match-state data from `src.run_analysis` CSV output.

Deliverables:

- `src/heatmap/join_state.py`
- `outputs/heatmap_match9/ui_state.csv`
- `outputs/heatmap_match9/team_points_enriched.csv`
- `outputs/heatmap_match9/state_join_report.csv`

Acceptance:

- Each heatmap point can be joined to nearest `elapsed_time` in the UI-state
  CSV.
- Enriched rows include score, penalty, player-state counts, weapons if known,
  and objective state.

Status: complete.

Generated a match-state CSV from `src.run_analysis` for the same `20s..160s`
map segment, then joined it to cleaned heatmap points.

UI-state command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.run_analysis --input footages/match_9.mp4 --output outputs/heatmap_match9/ui_state.csv --start-seconds 20 --stop-seconds 160 --sample-fps 1 --device auto --warmup-frames 10
```

Join command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.join_state --config src/heatmap/config_match9.yaml
```

Current result:

- UI-state rows: 141
- cleaned point rows: 1099
- enriched point rows: 1099
- matched rows: 1099
- unmatched rows: 0
- max join time delta: 0.0 seconds
- enriched output: `outputs/heatmap_match9/team_points_enriched.csv`
- join report: `outputs/heatmap_match9/state_join_report.csv`

Enriched fields include:

- `ui_count_left`, `ui_count_right`
- `ui_penalty_left`, `ui_penalty_right`
- `ui_asari_count`, `ui_hoko_count`, `ui_area_count`, `ui_yagura_count`
- `ui_message`, `ui_player_detected`
- `ui_player_state_1` through `ui_player_state_8`
- `ui_weapon_1` through `ui_weapon_8`

## Goal 8: Join With Kill/Death Event Data

Connect your existing event timeline to map points.

Expected input schema:

```csv
time,event,team,player,killer,victim,clip_path,segment_id,notes
```

Deliverables:

- `src/heatmap/join_events.py`
- `outputs/heatmap_match9/events_template.csv`
- `outputs/heatmap_match9/team_points_events.csv`
- `outputs/heatmap_match9/events_near_points.csv`
- `outputs/heatmap_match9/event_segments.csv`
- `outputs/heatmap_match9/event_join_report.csv`

Acceptance:

- Death and kill events are assigned to nearest plausible map points.
- Rapid consecutive events can be grouped into teamfight segments.
- Event rows include useful clip metadata for review.

Status: complete as an adapter layer.

No real kill/death event CSV exists in the current project, so this goal does
not invent events. It creates the input template and a join pipeline that can be
run when your external event timeline is available.

Default command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.join_events --config src/heatmap/config_match9.yaml
```

Command with your own event CSV:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.join_events --config src/heatmap/config_match9.yaml --events path/to/your_events.csv --window-seconds 2.0
```

Current result with empty template:

- event template: `outputs/heatmap_match9/events_template.csv`
- points with event columns: `outputs/heatmap_match9/team_points_events.csv`
- events-to-nearest-points output: `outputs/heatmap_match9/events_near_points.csv`
- grouped event segments: `outputs/heatmap_match9/event_segments.csv`
- event join report: `outputs/heatmap_match9/event_join_report.csv`
- point rows: 1099
- event rows: 0
- points with nearby events: 0
- events with nearby points: 0
- segment count: 0

The join logic supports optional `team` filtering. If an event row has a team,
only points from that team are considered. If the team is blank, all nearby map
points are considered.

## Goal 9: Player-Level Identity Tracking

Upgrade from team-level activity to player-level routes.

Potential inputs:

- map marker motion
- player names visible on the map
- top-bar player order
- weapon identity from UI CSV
- death/respawn periods from event data

Deliverables:

- `src/heatmap/infer_player_tracks.py`
- `outputs/heatmap_match9/player_tracks.csv`
- `outputs/heatmap_match9/player_track_gaps.csv`
- `outputs/heatmap_match9/identity_report.csv`
- per-player route images

Acceptance:

- At least a subset of players can be tracked consistently across meaningful
  intervals.
- Tracks include confidence and gaps instead of pretending uncertain identity is
  certain.

Status: complete as experimental slot-level tracking.

Implemented `src/heatmap/infer_player_tracks.py`.

Command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.infer_player_tracks --config src/heatmap/config_match9.yaml
```

Current result:

- experimental player/slot tracks:
  `outputs/heatmap_match9/player_tracks.csv`
- gap/reset rows: `outputs/heatmap_match9/player_track_gaps.csv`
- identity report: `outputs/heatmap_match9/identity_report.csv`
- route image directory: `outputs/heatmap_match9/player_routes/`
- player track rows: 1099
- gap rows: 150
- route images: 8
- route labels: `yellow_slot_1..4`, `blue_slot_1..4`
- method: `team_slot_weapon_hint`

Important limitation:

- These are not verified player identities. They are temporary track slots with
  weapon hints from the UI CSV. The output includes `identity_confidence`,
  `identity_method`, and `identity_note` so downstream analysis does not treat
  them as certain.

## Goal 10: Stable CLI And Review Report

Package the heatmap pipeline so it is easy to rerun.

Deliverables:

- A single documented command for the MVP pipeline.
- `outputs/heatmap_match9/report.md` or optional HTML report.
- Updated project documentation.

Acceptance:

- A fresh run can regenerate points, heatmaps, and reports for `match_9`.
- Existing video analysis remains runnable through `src.run_analysis`.

Status: complete.

Implemented `src/heatmap/run_pipeline.py`.

One-command MVP pipeline:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.run_pipeline --config src/heatmap/config_match9.yaml
```

Report-only command:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache .venv/bin/python -m src.heatmap.run_pipeline --config src/heatmap/config_match9.yaml --only-report
```

Fresh-run verification:

- full pipeline completed successfully
- valid map frames: 140
- invalid map frames: 1
- raw points: 1107
- clean points: 1099
- UI-state rows: 141
- enriched points: 1099
- points with event columns: 1099
- experimental slot/player rows: 1099
- player route images: 8
- final report: `outputs/heatmap_match9/report.md`

`src.run_analysis` remains the original standalone UI-analysis entry point; the
pipeline calls it as a subprocess for `outputs/heatmap_match9/ui_state.csv`.

## Original First MVP Boundary

The first implementation pass originally stopped at team-level heatmaps:

```text
match_9.mp4
  -> sample 20s-160s map frames
  -> detect yellow/blue marker candidates
  -> clean team points
  -> render yellow/blue/all heatmaps
  -> write a short report
```

Later goals added event-join and experimental slot-tracking adapters after the
team-level heatmap became visually useful. New model training and homography to
a standard map are still outside this first roadmap.
