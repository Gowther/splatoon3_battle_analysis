# Death/Kill Event Layer

这个文档记录接下来“死亡时间、死因、击杀者、复盘剪辑”的事件层建设。当前完成的是 Goal 1：从 `player_state_1..8` 的时间线中抽取死亡事件骨架。

## 1-8 Goal Roadmap

1. 死亡事件数据模型与 `player_state` 死亡时间抽取。
2. 根据死亡时间导出死亡窗口帧/短片段，供 OCR 和人工复核使用。
3. 建立 death screen / kill log OCR 候选队列。
4. 建立 killer / victim / weapon 归因规则引擎。
5. 在 Web 工作台加入死亡事件标注入口。
6. 将人工修正后的 death event 标注入库，并输出评估报告。
7. 根据事件生成自动复盘片段和精彩击杀候选剪辑。
8. 对齐多视角事件，导出整场复盘时间线。

## Goal 1 Outputs

新增命令：

```bash
python scripts/extract_death_events.py --analysis-csv outputs/example_analysis.csv
```

默认输出：

- `outputs/death_events/<analysis_stem>_death_events.csv`
- `outputs/death_events/<analysis_stem>_death_events.json`

CSV 兼容现有热力图事件联表，包含 `time/event/team/player/killer/victim/clip_path/segment_id/notes`，同时额外包含：

- `victim_slot`: 顶部状态栏的绝对槽位，范围 1-8。
- `victim_weapon`: 该槽位当时已识别到的武器。
- `cause_text` / `cause_weapon`: 预留给后续 death screen OCR 和武器归因。
- `confidence`: 当前只代表 `player_state` 转换证据强度。
- `clip_start` / `clip_end`: 后续剪辑导出的建议时间窗口。
- `source`: 当前为 `player_state_transition`。
- `evidence`: 例如 `player_state_1:0->1`。

## State Id Defaults

当前分析 CSV 中 `player_state_1..8` 写入的是 YOLO 类别 id。根据 `legacy/yolov5_training/data.yaml`：

- `alive` = `0`
- `dead` = `1`
- `special` = `14`

因此命令默认：

```bash
--dead-state-ids 1 --alive-state-ids 0,14
```

如果以后检测模型类别顺序变化，需要显式传入新的 id。

## Conservative Behavior

默认只在同一槽位出现“已知非死亡状态 -> dead”时生成事件。片段第一帧如果已经是 dead，不会直接生成死亡事件，因为真实死亡可能发生在片段开始之前。

如确实要把初始 dead 也当作事件候选，可以使用：

```bash
python scripts/extract_death_events.py \
  --analysis-csv outputs/example_analysis.csv \
  --include-initial-dead
```

如果状态检测抖动较大，可以要求连续多帧 dead：

```bash
python scripts/extract_death_events.py \
  --analysis-csv outputs/example_analysis.csv \
  --min-dead-frames 2
```

## Current Limits

Goal 1 只回答“哪个状态栏槽位在什么时间死亡”。它不会假装知道：

- 谁杀了这个玩家。
- 死因武器是什么。
- 画面上的死亡提示文字是否可靠。
- `team_1/team_2` 对应真实黄队、蓝队还是玩家视角队伍。

这些会在 Goal 2-6 通过 OCR、规则归因、Web 标注和评估逐步补上。

## Goal 2: 导出死亡窗口素材

Goal 2 新增命令：

```bash
python scripts/export_death_event_windows.py \
  --events-csv outputs/death_events/example_death_events.csv \
  --video footages/example.mp4 \
  --updated-events-csv outputs/death_events/example_death_events_with_assets.csv
```

默认输出：

- `outputs/death_events/assets/death_event_assets.csv`
- `outputs/death_events/assets/death_event_assets.json`
- 每个事件一个独立素材目录，目录中包含 `frames/*.jpg`

默认只导出复核帧，不强制导出 MP4 片段。需要短片段时加：

```bash
python scripts/export_death_event_windows.py \
  --events-csv outputs/death_events/example_death_events.csv \
  --video footages/example.mp4 \
  --write-clips
```

可以调整复核帧相对死亡时间的采样点：

```bash
python scripts/export_death_event_windows.py \
  --events-csv outputs/death_events/example_death_events.csv \
  --video footages/example.mp4 \
  --frame-offsets -4,-2,0,1.5,3
```

导出的 manifest 是后续 death screen OCR、Web 标注、归因和自动剪辑的共同素材入口。

## Goal 3: 生成死亡 OCR 候选队列

Goal 3 新增命令：

```bash
python scripts/build_death_ocr_candidates.py \
  --asset-manifest outputs/death_events/assets/death_event_assets.csv
```

默认输出：

- `outputs/death_events/ocr_candidates/death_ocr_candidates.csv`
- `outputs/death_events/ocr_candidates/death_ocr_candidates.json`
- `outputs/death_events/ocr_candidates/crops/<region>/*.jpg`

默认裁三个候选区域：

- `kill_log_right`: 画面右侧击杀日志候选区。
- `death_message_center`: 中央死亡提示候选区。
- `full_death_screen`: 全画面兜底候选区。

候选 CSV 保留现有 Web 工作台可识别的通用字段：

- `candidate_id`
- `target=death_event_ocr`
- `reason`
- `match_id`
- `elapsed_time`
- `frame_path`
- `details`

其中 `frame_path` 指向已经裁好的 OCR crop，`source_frame_path` 保留原始复核帧路径。
