# Web 工作台使用手册

这份手册面向日常操作：新增素材、生成候选样本、人工标注、写入训练集、训练预演、执行训练和模型提升。底层设计和 API 说明见 [ACTIVE_LEARNING_WORKBENCH.md](ACTIVE_LEARNING_WORKBENCH.md)。

## 1. 启动和打开

在项目根目录启动本地 Web 工作台：

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765
```

如果端口被占用，换一个端口：

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --port 8766
```

界面默认是中文，右上角可以切换到 English。语言选择会保存在浏览器本地。

## 2. 推荐使用顺序

1. 把新视频放到 `footages/`。
2. 在「素材收件箱」确认新视频出现。
3. 点「使用」，再点「接入视频」登记素材。
4. 点「运行验证」，必要时勾选「同时跑分析」。
5. 点「刷新候选样本」生成失败样本队列。
6. 在「标注队列」逐条选择样本，画框或点坐标。
7. 标注完成后点「保存」，先进入暂存区。
8. 点「预演」检查暂存标注能否写入训练集。
9. 预演无明显问题后点「应用」正式写入训练集。
10. 点「验证训练集」和「训练预演」。
11. 确认需要训练时再点「执行训练」。
12. 新模型验证通过后，用「提升计划」和「应用提升」替换正式模型。

如果想少点手动按钮，可以先点「自动预演」查看工作台准备自动执行哪些安全步骤；确认后点「自动推进」。自动推进只会执行已标记为安全的步骤，遇到人工标注、正式训练、正式入库、模型提升这类门禁会停下。

## 3. 页面区域说明

### 顶部栏

- 「主动学习工作台」旁边的状态显示当前整体状态。
- 「刷新」会重新读取报告、候选队列、素材收件箱和暂存状态。
- 语言下拉框用于切换中文/英文。
- 右侧时间是当前状态数据的更新时间。

### 报告卡片

页面上方的报告卡片汇总当前项目状态，包括验证套件、训练候选样本、热力图标注、训练数据集、模型/数据就绪、运行时基准、模型提升计划等。

常见状态含义：

| 状态 | 含义 |
| --- | --- |
| 就绪 / 通过 / 完成 | 当前报告没有发现阻塞问题 |
| 缺数据 / 需标注 / 需要关注 | 需要补数据、补标签或处理失败项 |
| 缺失 | 对应报告文件还没有生成 |
| 失败 / 阻塞 / 超时 | 命令失败、检查阻塞或执行超时 |

### 素材收件箱

「素材收件箱」扫描 `footages/` 目录。

- `新素材`：视频在 `footages/` 中，但还没有登记到 `config/data_registry.json`。
- `已登记`：视频已经在数据登记表中。
- 「使用」会把视频路径和建议的对战 ID 填入「操作」区的表单。

### 标注队列

左侧「标注队列」展示从失败样本和待标注样本生成的候选项。可以按目标和状态筛选。

队列会按优先级排序，并自动聚合相近重复帧。候选详情里的 `duplicates` 表示当前代表项背后还有多少同组样本，`priority` 表示工作台认为它的处理优先级。

常见目标：

| 目标 | 用途 |
| --- | --- |
| UI 检测 YOLO | 标注玩家状态、UI 元素等检测框 |
| 数字 OCR YOLO | 标注计数、数字区域等检测框 |
| 消息 OCR YOLO | 标注消息文本区域等检测框 |
| 热力图轨迹标注 | 标注角色位置点坐标 |

常见状态：

- `待处理`：还没有人工处理。
- `草稿`：已经保存，但暂时不准备入库。
- `已完成`：可以被「应用暂存」写入训练集。
- `已跳过`：本条样本不使用。

### 标注区

选择队列中的候选样本后，中间会显示对应图片。

如果候选 CSV 里已经带有机器可读的框或点坐标，工作台会自动把它载入为预标注草稿。你可以直接调整后保存，不需要从零开始画。

标注 YOLO/OCR 框：

1. 设置「类别 ID」。
2. 可选填写「类别名称」。
3. 在图片上按住鼠标拖动，画出矩形框。
4. 如需多个框，重复拖动。
5. 选择「训练」或「验证」划分。
6. 状态选择「已完成」或「草稿」。
7. 点「保存」。

标注热力图点：

1. 选择热力图候选样本。
2. 在图片中的角色位置点击一次。
3. 点坐标会按原图像素保存为 `x/y`。
4. 状态选择「已完成」后点「保存」。

辅助字段：

- 「文本」适合记录 OCR 文本或人工读数。
- 「备注」适合记录遮挡、低置信、看不清、需要复核等信息。
- 「清空」只清空当前页面上的框/点，保存前不会写入。
- 「跳过」会把当前候选写成 `skipped` 状态。

注意：点击「保存」只写入暂存文件，不会直接改正式训练集。

如果保存的是「已完成」，工作台会自动跑一次暂存入库预演，并把结果显示在「操作」输出区。这样能尽早发现图片缺失、框坐标非法、热力图点缺失等问题。

## 4. 操作区按钮说明

| 按钮 | 做什么 | 备注 |
| --- | --- | --- |
| 自动预演 | 生成并展示可自动执行的安全步骤 | 不执行命令 |
| 自动推进 | 执行安全步骤，遇到人工门禁停止 | 默认不跑长任务，不执行训练/提升 |
| 规则审阅 | 用本地规则给候选写入 LLM review 格式建议 | 不替代人工确认 |
| 预填热力图 | 把带有 `x/y` 的热力图候选预填为草稿 | 默认不标为已完成 |
| 刷新候选样本 | 重新导出失败样本候选队列 | 对应 `scripts/export_training_sample_candidates.py` |
| 验证训练集 | 检查模型训练数据集配置和文件 | 对应 `scripts/validate_model_training_datasets.py` |
| 刷新就绪状态 | 重新生成模型/数据就绪报告 | 对应 `scripts/report_model_data_readiness.py` |
| 生成 LLM 审阅包 | 把一批候选样本打包给 LLM 审阅 | 输出到 `outputs/active_learning_workbench/llm_review_pack.json` |
| 接入视频 | 把新视频登记为可分析素材 | 视频路径通常来自「素材收件箱」的「使用」 |
| 运行验证 | 跑验证套件 | 勾选「同时跑分析」会更耗时 |
| 训练预演 | 生成训练启动计划 | 不真正训练 |
| 执行训练 | 真正运行训练命令 | 会弹确认，可能耗时较久 |
| 提升计划 | 生成候选模型替换正式模型的计划 | 不直接替换 |
| 应用提升 | 把候选模型复制到登记的正式模型路径 | 会弹确认，旧模型由脚本备份 |

「执行训练」和「应用提升」是高影响操作。建议先完成验证、预演，并确认候选模型路径正确。

长任务会以后台 job 形式启动，页面不会一直阻塞。状态会写入 `outputs/active_learning_workbench/jobs.json`，最近任务也会出现在状态 API 中。

## 5. 暂存和入库

所有人工操作先写到：

```text
outputs/active_learning_workbench/staging_annotations.json
```

「应用暂存」区有两个按钮：

- 「预演」：检查哪些 `done` 标注可以写入训练集，只生成报告，不复制文件。
- 「应用」：把 `done` 标注正式写入训练集。

暂存应用报告写到：

```text
outputs/active_learning_workbench/apply_report.json
```

只有状态为 `已完成` 的标注会进入应用流程。`草稿` 和 `已跳过` 不会写入正式训练集。

## 6. 写入位置

`ui_detector_yolo` 写入：

```text
yolov5/train/images
yolov5/train/labels
yolov5/valid/images
yolov5/valid/labels
```

`count_ocr_yolo` 写入：

```text
outputs/model_training/count_ocr_dataset/images/train
outputs/model_training/count_ocr_dataset/images/val
outputs/model_training/count_ocr_dataset/labels/train
outputs/model_training/count_ocr_dataset/labels/val
```

`message_ocr_yolo` 写入：

```text
outputs/model_training/message_ocr_dataset/images/train
outputs/model_training/message_ocr_dataset/images/val
outputs/model_training/message_ocr_dataset/labels/train
outputs/model_training/message_ocr_dataset/labels/val
```

`heatmap_tracker_labels` 写入：

```text
outputs/active_learning_workbench/heatmap_staging_labels.csv
```

YOLO 标签文件只包含 `class_id x_center y_center width height`。类别名称、文本、备注和来源候选会进入旁路 metadata JSON，方便之后追踪。

## 7. LLM 审阅建议怎么用

点击「生成 LLM 审阅包」后，工作台会生成：

```text
outputs/active_learning_workbench/llm_review_pack.json
```

这个文件适合交给 LLM 判断：

- 样本是否值得标注。
- 框/点是否明显错误。
- 文本或类别是否需要人工复核。
- 是否建议跳过。

LLM 的建议只是辅助，不会自动写入训练集。最终仍需要人确认标注状态为 `已完成`，再通过「预演」和「应用」入库。

当前「规则审阅」是本地启发式审阅：它会识别缺图样本、可预标注样本和必须人工看的样本，并写成和 LLM review 一样的结构。后续接入真实视觉 LLM 时，可以复用同一份 review JSON。

## 8. 数据核验和证据核验页面

除了主页的标注流程，工作台还有两个独立的核验页面，用顶部导航进入：

| 页面 | 地址 | 用途 |
| --- | --- | --- |
| 数据核验 | `/data-review` | 把视频时间轴和分析 CSV 对齐，确认某个时间点的数据行是否和画面一致。 |
| 证据核验 | `/evidence-review` | 对武器识别和死亡事件导出画面证据，逐条判断对错，并修正武器标签。 |
| 场地标注 | `/stage-labeling` | 在网格参考帧上点选场地控制点，校验 homography 后提升为正式资产。 |

### 数据核验

页面会扫描 `outputs/` 下的分析 CSV 和 `footages/`、`sample/` 下的视频，按 `match` 编号自动配对。

1. 选择一个视频，页面列出匹配到的数据源。
2. 拖动视频到要检查的时间点。
3. 页面取该时间附近的数据行，显示时间差最小的若干行。
4. 判断数据和画面是否一致，写结论后保存。

核验结论追加写入：

```text
outputs/data_review_workbench/reviews.jsonl
```

### 证据核验

页面对选定视频导出两类证据：

- 武器证据：从有武器数据的行取一帧，并按 8 个槽位裁出武器图标。
- 死亡证据：从 `player_state_1..8` 的死亡行取一帧。

1. 选择视频，点生成证据。
2. 逐条查看导出的帧，判断识别是否正确。
3. 武器识别错误时，直接从武器列表里选正确名称提交修正。
4. 修正会连同槽位裁图一起写入修正数据集。

输出位置：

```text
outputs/evidence_review/reviews.jsonl
outputs/weapon_correction_dataset/corrections.jsonl
```

武器修正裁图按标签分目录存放，可以直接作为武器分类器的补充训练数据。

### 场地标注

前置条件：先用 `scripts/export_stage_reference.py` 为一场对战导出参考帧包
（见 `STAGE_NORMALIZATION_GOALS.md` Goal 2）。

1. 左侧选择参考帧包，可切换多个时间点的网格帧。
2. 在图上点击地标位置，源像素坐标自动写入表格。
3. 在表格里补齐每个点的 `stage_x/stage_y`（标准场地图 0..1 坐标）和名称。
4. 点「保存草稿并校验」：立即返回控制点校验和重投影误差。
   少于 4 个点时草稿保持模板状态，不会启用 homography。
5. 校验通过后点「提升为正式资产」，写入
   `config/stage_control_points/<stage_id>.json`。

输出位置：

```text
outputs/stage_reference/<stage_id>/control_points_draft.json
config/stage_control_points/<stage_id>.json
```

### 相关 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/data-review/state` | 视频列表、数据源列表和核验统计。 |
| GET | `/api/data-review/snapshot` | 指定数据源和时间点的邻近数据行。 |
| POST | `/api/data-review/review` | 追加一条数据核验结论。 |
| GET | `/api/evidence-review/state` | 可核验视频列表和已有结论统计。 |
| GET | `/api/evidence-review/video` | 为指定视频导出武器/死亡证据。 |
| POST | `/api/evidence-review/review` | 追加一条证据核验结论。 |
| POST | `/api/evidence-review/weapon-correction` | 写入武器标签修正和槽位裁图。 |
| GET | `/api/stage-labeling/state` | 参考帧包列表和草稿状态。 |
| POST | `/api/stage-labeling/save` | 保存场地控制点草稿并校验。 |
| POST | `/api/stage-labeling/promote` | 校验通过后提升为正式控制点资产。 |

这三个页面只读取视频和已生成的 CSV/参考帧，不会改动正式训练集，也不会替换模型。场地标注的「提升」只写 `config/stage_control_points/`。

## 9. 常见问题

### 页面打不开

确认服务进程还在运行，并确认端口和浏览器地址一致：

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --host 127.0.0.1 --port 8765
```

### 素材收件箱没有新视频

检查视频是否放在 `footages/`，并确认扩展名是项目支持的视频格式。放入后点页面顶部「刷新」。

### 队列为空

先点「运行验证」，再点「刷新候选样本」。如果仍为空，说明当前报告没有导出可标注候选，或者相关分析结果还没生成。

### 图片显示为空

候选样本需要有 `frame_path` 或 `preview_path`，并且文件必须存在于项目目录内。可以先刷新候选样本，或重新跑验证/分析。

### 预演出现 skipped

常见原因包括：

- 标注状态不是 `已完成`。
- 候选图片不存在。
- YOLO 框为空或坐标非法。
- 热力图点没有 `x/y`。
- 当前目标暂不支持直接写入训练集。

先修正对应样本，再重新点「预演」。

### 点了保存但训练集没变化

这是正常行为。保存只写暂存文件。要进入训练集，需要先「预演」，再「应用」。

### 训练很久没有结束

训练是真实命令，耗时取决于数据量、模型和硬件。可以查看「操作」区输出，或在终端查看启动服务的进程日志。

## 10. 安全边界

- Web 工作台默认只绑定本机地址 `127.0.0.1`，不要暴露到公网。
- 页面没有登录系统，适合本机开发使用。
- 自动化负责扫描、生成候选、校验、搬运和记录。
- 人负责最终确认高风险动作：正式入库、执行训练、应用模型提升。
- 每次正式写入训练集前，先跑一次「预演」。
- 每次模型提升前，先确认验证报告和候选模型路径。
- 自动推进不会绕过人工门禁；它只处理不需要视觉判断或高风险确认的步骤。

## 11. 快速检查清单

正式训练前确认：

- 新视频已经登记。
- 候选样本已经刷新。
- 需要的样本已标为 `已完成`。
- 「应用暂存」预演没有意外 skipped。
- 训练集验证通过。
- 训练预演输出的目标和命令正确。

模型提升前确认：

- 候选模型路径存在。
- 模型 ID 正确。
- 验证结果优于或至少不弱于当前基线。
- 已生成提升计划。
- 确认要替换正式模型后再点「应用提升」。
