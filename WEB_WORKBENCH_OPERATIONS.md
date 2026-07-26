# Web 工作台操作手册

覆盖本地 Web 工作台**全部四个页面**：每个区域是干什么的、操作步骤、
以及当前操作逻辑存在的问题和优化建议。

底层设计见 [ACTIVE_LEARNING_WORKBENCH.md](ACTIVE_LEARNING_WORKBENCH.md)，
场地归一化路线见 [STAGE_NORMALIZATION_GOALS.md](STAGE_NORMALIZATION_GOALS.md)。

---

## 0. 启动

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --host 127.0.0.1 --port 8765
```

打开 `http://127.0.0.1:8765`。端口占用就换 `--port 8766`。
默认只绑 `127.0.0.1`，无登录系统，**不要暴露公网**。

| 页面 | 地址 | 一句话 |
| --- | --- | --- |
| 主动学习 | `/` | 主控台：验证、生成候选、标注、入库、训练、模型提升 |
| 数据核验 | `/data-review` | 视频时间轴 ↔ 分析 CSV 对齐，确认某一时刻数据和画面一致 |
| 证据核验 | `/evidence-review` | 对武器/死亡识别逐条判对错，并修正武器标签产出训练数据 |
| 场地标注 | `/stage-labeling` | 在参考帧上点选场地控制点，校验后提升为正式资产 |

---

## 1. 主动学习页 `/`

唯一有中英文切换的页面（右上角，选择存浏览器本地）。

### 1.1 区域

| 区域 | 内容 | 数据来源 |
| --- | --- | --- |
| 顶部栏 | 整体状态、刷新、导航、语言、更新时间 | `GET /api/state` |
| 标注队列（左侧全高） | 目标筛选 + 状态筛选 + 最多 180 条候选 | `GET /api/candidates` |
| 报告卡片 | 14 张卡：标题 + 状态 + 路径（或"缺失"） | 每个 spec 取第一个存在的候选路径 |
| 候选面板 | 候选 id、目标/原因/优先级/重复数、图片 | 队列选中项 |
| 标注面板 | 类别 ID/名称、划分、状态、文本、备注 | 已有暂存 → 否则预标注 |
| 操作面板 | 所有动作按钮 | — |
| 应用暂存 | 预演/应用 + **全页唯一的输出控制台** | `POST /api/apply-staging` |
| 素材收件箱（底部） | `footages/` 每个视频一张卡，`new`/`registered` | 对照 `config/data_registry.json` |

整体状态判定：任一报告 blocker、任一新素材、或任一候选存在 → `needs_attention`，否则 `ready`。

### 1.2 推荐顺序

1. 视频放到 `footages/`
2. 收件箱确认出现 → 「使用」→「接入视频」登记
3. 「运行验证」（可勾「同时跑分析」）
4. 「刷新候选样本」
5. 队列逐条标注
6. 「保存」（只进暂存）
7. 「预演」检查能否入库
8. 无问题后「应用」
9. 「验证训练集」+「训练预演」
10. 「执行训练」
11. 验证通过后「提升计划」+「应用提升」

想少点手动：先「自动预演」看会执行哪些安全步骤，再「自动推进」。

### 1.3 标注

**画框（`ui_detector_yolo` / `count_ocr_yolo` / `message_ocr_yolo` / `death_event_ocr`）：**
设类别 ID → 拖动画框（可多个）→ 选训练/验证 → 状态选已完成/草稿 → 保存

**标点（`heatmap_tracker_labels`）：**
选候选 → 在角色位置点一次 → 坐标按**原图像素**保存 → 状态选已完成 → 保存

候选 CSV 已有机器可读框/点时会自动载入为预标注草稿。
保存「已完成」会自动跑一次入库预演。

**状态值实际是 `todo` / `draft` / `done` / `skipped`**（界面显示"待处理"对应 `todo`）。

队列排序：按 target+match+原因+2 秒时间桶分组去重，再按优先级排
（todo +100、draft +80、缺原因 +25、跳变 +20、预标注就绪 +15、**无图片 −50**）。

### 1.4 按钮速查

| 按钮 | 实际执行 | 同步/后台 | 有确认框 |
| --- | --- | --- | --- |
| 自动预演 | 只生成计划，无副作用 | 同步 | 否 |
| **自动推进** | 执行安全步骤 | 同步（**整条流水线阻塞请求**） | **否** |
| 规则审阅 | 本地启发式，不调 LLM | 同步 | 否 |
| 预填热力图 | 把带 `x/y` 候选预填为草稿 | 同步 | 否 |
| 刷新候选样本 | `export_training_sample_candidates.py` | 同步 | 否 |
| 验证训练集 | `validate_model_training_datasets.py` | 同步 | 否 |
| 刷新就绪状态 | `report_model_data_readiness.py` | 同步 | 否 |
| 生成 LLM 审阅包 | 写 `llm_review_pack.json` | 同步 | 否 |
| 接入视频 | `intake_samples.py --write` | 同步 | 否 |
| 运行验证 | `run_validation_suite.py` | **后台 job** | 否 |
| 训练预演 | `run_model_training_target.py` | 同步 | 否 |
| **执行训练** | 同上 `--execute` | **后台 job** | **是** |
| 提升计划 | `promote_model_candidate.py` | 同步 | 否 |
| **应用提升** | 同上 `--apply` | **同步（阻塞，600s 超时）** | **是** |
| 预演 / 应用（暂存） | `apply_staging_annotations` | 同步 | 应用有 |

所有动作都是 `subprocess.run(timeout=600)`，非零退出**不抛异常**，作为数据返回，
记录追加到 `outputs/active_learning_workbench/action_runs.json`。

### 1.5 自动推进的实际行为

计划是**从当前状态实时推导的**，条件不满足就不出现该步骤。

可自动执行：接入新视频 → 跑验证套件 → 刷新候选 → 暂存预演 → 验证训练集 → 刷新就绪状态

人工门禁（永远跳过）：`annotate_candidates`（有 todo 候选时）、`heatmap_labels`（需要标注时）

**关键限制**：两个自动化按钮都硬编码 `include_long:false`，
所以「跑验证套件」这一步**永远被跳过**——只能靠「运行验证」按钮单独触发。

执行是 fail-fast：某个动作步骤结果不是 `passed`/`ready` 就中断后续。
暂存预演步骤**永远是 dry-run**，自动推进不会真正入库。

### 1.6 写入位置

```
暂存      outputs/active_learning_workbench/staging_annotations.json
应用报告  outputs/active_learning_workbench/apply_report.json
动作记录  outputs/active_learning_workbench/action_runs.json
后台任务  outputs/active_learning_workbench/jobs.json（只留最近 100 条）
```

正式训练集：

| 目标 | 路径 |
| --- | --- |
| `ui_detector_yolo` | `yolov5/train\|valid/{images,labels}` |
| `count_ocr_yolo` | `outputs/model_training/count_ocr_dataset/…` |
| `message_ocr_yolo` | `outputs/model_training/message_ocr_dataset/…` |
| `death_event_ocr` | `outputs/active_learning_workbench/death_event_ocr_labels.csv` |
| `heatmap_tracker_labels` | `outputs/active_learning_workbench/heatmap_staging_labels.csv` |
| `weapon_classifier_resnet18` | **不支持入库，会被跳过** |

只有 `done` 的项进入应用流程。**dry-run 也会写 `apply_report.json`**，但不碰数据集文件。
应用按 id 幂等（同 stem 覆盖），且不会从暂存里移除项——重复应用会重复写入。

---

## 2. 数据核验页 `/data-review`

**用途**：确认某时间点的分析数据和画面是否一致。发现"数据看着对但其实错位"。

### 2.1 布局

左侧 340px：视频单选 + 数据源多选 + 使用推荐/全选/清空 + 核验统计
右上：视频播放器（支持 range 请求，可拖动）
右中（吸顶）：**可读摘要卡** — 比分和目标、玩家状态（8 个 chip，`1`=死亡红、`0`/`14`=存活绿）、武器、其它数据；下面每个数据源一张可折叠原始详情卡
底部整宽：判断（准确/不准确/需要复查/跳过）+ 有问题的字段勾选 + 备注 + 保存 + 前后退 2 秒

### 2.2 操作步骤

1. 选视频 → 自动选中推荐数据源
2. 拖动视频到要检查的时间点（**暂停时操作**，见下方问题）
3. 对照摘要卡和画面
4. 选判断 → 勾选有问题的字段 → 写备注
5. 「保存当前判断」

结论追加写入 `outputs/data_review_workbench/reviews.jsonl`。
记录含 `decision`、`incorrect_fields[]`、`note`，以及**完整快照**（每个源最多 80 行）。

### 2.3 前置条件

- 视频在 `footages/` 或 `sample/`（前 200 个）
- CSV 在 `outputs/` 下（前 600 个），且**必须含时间列**之一：
  `elapsed_time` / `time` / `event_time` / `nearest_point_time` / `clip_start`，否则静默丢弃
- 自动配对靠路径里的 `match_N` / `n_match_N` / `f_match_N` 标记，或前 25 行的 `match_id` 列
- **可读摘要卡额外要求**：数据源要有 `elapsed_time` **且**至少一个 `player_state_*` 列，
  否则只能看原始转储

时间窗口语义按类型不同：`death_events`/`heatmap_tracks` 等返回 ±0.35 秒内所有行；
`analysis_csv` 只返回最接近的**一行**。

---

## 3. 证据核验页 `/evidence-review`

**用途**：对武器/死亡识别逐条判对错，**并能直接修正武器标签产出训练数据**。

### 3.1 操作步骤

1. 选视频 → 「生成证据」
2. **武器识别**（左栏）：看整帧截图，8 个槽位显示识别结果
   - 判断后点「武器都准确 / 有武器错误 / 看不清+待复查」→ **立即保存，无确认**
3. **纠错某个槽位**：
   - 点该槽位的「选择」
   - **在截图上拖框**圈出真实武器图标
   - 选「真实武器」→「截取并加入待验证训练集」
4. **死亡时间点**（右栏）：每个事件一张卡，四个判断按钮 + 备注

### 3.2 一次纠错产出 5 张训练图

裁框会被**正方形化**（`max(宽,高) × 1.12`）、居中、贴到黑底、LANCZOS 缩放到 **64×64**：

```
outputs/weapon_correction_dataset/<正确武器名>/
  ..._orig.jpg              原图              质量 95
  ..._aug_brightness.jpg    亮度 ×1.16        质量 92
  ..._aug_contrast.jpg      对比度 ×1.18      质量 92
  ..._aug_soft.jpg          高斯模糊 r=0.45   质量 92
  ..._aug_shift.jpg         位移 (+2,−2)      质量 92
```

按标签分目录（ImageFolder 结构），可直接作为武器分类器补充训练数据。
与 `main_training_dataset/` 刻意分开，属于"待验证"数据。

### 3.3 前置条件

- 视频 + **token 匹配且含 `player_state_*` 列的分析 CSV**，否则整页降级为"没有找到匹配的分析 CSV"
- 数据源优先级（分数越低越优先）：`heatmap_*/ui_state.csv` 最优 → `evaluation/*/smoothed.csv`
  → `raw.csv` → `analysis_window_scan/` → `validation_suite/` 最差
- 至少一行有 `weapon_1..8` 才有武器证据；有 alive→dead 转换才有死亡证据
- 需要 **opencv-python**（导帧，缺了静默返回空路径）和 **Pillow**（裁图，缺了报明确错误）

---

## 4. 场地标注页 `/stage-labeling`

**前置**：必须先用命令行导出参考帧包（**页面不能创建包**）：

```bash
.venv/bin/python scripts/export_stage_reference.py \
  --config src/heatmap/config_match9.yaml --stage-id match9_stage --times 30,60,90
```

### 操作步骤

1. 左侧选包，可切换多个时间点（挑遮挡最少的）
2. **在图上点击地标** → `source_x/source_y` 自动填入（已换算回原图像素）
3. 表格补 `stage_x/stage_y`（标准场地图 0..1 坐标）和名称
4. 「保存草稿并校验」→ 返回重投影误差等校验结果
5. 通过后「提升为正式资产」

**机制**：少于 4 点保存按钮禁用、草稿保持 `template: true`，不可能提前启用 homography。
「删除」删的是**单个控制点行**，不是删包。提升前会再校验一次。

当前状态：`match9_stage` 有 3 帧、4 个 ROI 角点种子、`template=True`（等真实地标）。

---

## 5. 当前操作逻辑的问题

以下均经实际验证，按严重程度排序。

### 5.1 严重：提升场地资产时不跑几何质量门禁

Goal 5 的 CLI 有四道门禁，但网页「提升为正式资产」**只跑 Goal 1 基础校验**。

实测同一份"4 点挤在画面一角"的资产：

```
网页提升路径（Goal 1）:  ready         ← 会被接受
Goal 5 命令行门禁      :  needs_review  失败项 [coverage, corners]
```

它重投影误差是完美的 2e-16，但会把 ROI 右上角映射到 stage `(17.4, -1.5)`。
**网页会放行一个命令行明确拒绝的资产。**

规避：提升后手动跑
`report_stage_control_point_quality.py --config … --strict`

### 5.2 严重：后台任务跑完了，页面上看不到

「执行训练」和「运行验证」是后台 job，但：

- 页面**只在点击瞬间**把 job JSON 打进那个共享的输出框
- 输出框会被下一个按钮的结果**直接覆盖**
- `GET /api/jobs` 存在，`recent_jobs` 也在 state 里，但**页面从不渲染**

想知道训练跑完没有，只能手动开 `/api/jobs` 或读 `jobs.json`。
更糟的是：job 状态纯粹存文件，**服务重启后进行中的任务会永远卡在 `running`**。

### 5.3 严重：「自动推进」是写操作却没有确认框

它会真的执行接入视频、刷新候选、写报告等一串动作，
但和「自动预演」一样是一键触发，**没有任何二次确认**。
旁边的「应用」「执行训练」「应用提升」都有确认框，唯独它没有。

### 5.4 数据核验：勾选的字段会被自动清空

「有问题的字段」勾选框在**每次快照刷新时被整体重建**。
而快照会在 `timeupdate` / `seeked` 时自动重新拉取——
也就是说**视频只要在播放，勾选就会不断被清掉**。

只有暂停时操作才安全，但界面没有任何提示。

另外保存时记录的 `time` 取自 `video.currentTime`，
和快照里那个（防抖 180ms 后的）`snapshot.time` 可能差几帧。

### 5.5 证据核验：重新生成证据不会刷新旧截图

导帧函数发现目标文件已存在就**直接返回**。
所以对同一个视频再点「生成证据」，拿到的还是旧图。
要换一帧必须先手动删文件。

而且证据帧的时刻是**第一行有武器数据的行**决定的，不是用户能选的。

### 5.6 证据核验：状态永远是 ready，且看不到已核验数量

- 后端 `build_evidence_review_state` **无条件返回 `status: "ready"`**，
  哪怕一个视频都没有——这个状态徽章不是健康信号
- 没选视频就点「生成证据」→ 空路径解析成仓库根 → 500 错误直接甩在页面上
- `review_log_path` 后端返回了但**前端从不渲染**，
  页面上看不到"已核验 N 条"，只有纠错数据集路径

### 5.7 场地归一化链路大部分没有 Web 入口

Goal 2-8 的 6 个 CLI，只有标注这一步有页面：

| 命令 | Web 入口 |
| --- | --- |
| `export_stage_reference.py`（造参考帧包） | 无（页面只有一句文字说明） |
| `report_stage_control_point_quality.py` | 无 |
| `report_stage_registry.py` | 无 |
| `render_stage_heatmaps.py`（标准场地渲染） | 无 |
| `report_stage_aggregate.py`（跨场次对比） | 无 |
| `report_outputs_retention.py`（清理 outputs） | 无 |

用户标完点、提升成功后**页面上什么也看不到**，必须切回终端才能看到成果图。

### 5.8 target 坐标只能凭空手填

点选自动填 `source`，但 `stage_x/stage_y` 要人自己想——
**页面上没有任何标准场地图作对照**。这也放大了 5.1 的危害：
最容易填错的环节，恰好没有门禁拦。

### 5.9 导航命名不一致

`/data-review` 在证据核验页叫**「时间同步核验」**，在其余两页叫**「数据核验」**。

### 5.10 只有主页有中英文切换

主动学习页有 49 处 i18n 钩子，其余三页**都是 0**（纯中文硬编码）。
主页切成 English 后点进其他页仍是中文。

### 5.11 其它

- `needs_confirmation` 只是**元数据，服务端从不强制**。真正的拦截只有浏览器
  `confirm()` 和两个命令构造函数里的字符串检查。任何客户端可直接 POST 绕过
- `run_model_baseline` 动作**没有任何按钮**，只能通过 API 调用
- 证据核验的记录路径**没走 `safe_project_file` 校验**（虽然从不打开，但日志可含未校验文本）
- `ThreadingHTTPServer` + 无锁 JSON 读改写：两个任务同时结束可能**丢记录**
- 主页有一批状态字段计算了但从不渲染：`queue_summary`、`staging_summary`、
  `recent_actions`、`recent_jobs`、`automation_plan`

---

## 6. 优化建议

### P0：把质量门禁接进提升路径

`src/stage_labeling_workbench.py:171` 的 `promote_stage_labels` 同时调用
`build_control_point_quality_report`，覆盖度/角点不过就拒绝。
**这是唯一会让错误数据静默进入正式资产的问题。**

### P0：加后台任务状态区

`/api/jobs` 已经有了，前端补一个常驻面板显示最近任务和状态即可。
再给每个动作独立的输出区，别让所有按钮共用一个会被覆盖的框。
顺带修服务重启后 `running` 僵尸任务（启动时把孤儿标成 `interrupted`）。

### P1：「自动推进」加确认框

和「应用」「执行训练」保持一致。一行改动。

### P1：数据核验的勾选框改成增量更新

不要每次快照都 `innerHTML` 重建；或者至少在视频播放时暂停自动刷新快照，
并在界面上提示"暂停后再标记字段"。

### P1：补上标注后的反馈闭环

场地标注页加三个按钮：质量检查 / 渲染标准场地图 / 跨场次对比，
让用户在页面上看到自己标注的成果。

### P2：加标准场地图底图对照

点选区旁并排显示标准场地图，支持"在右图点一下"来填 target。
能同时解决猜数字和大部分标注错误来源。

### P2：证据核验补强

- 「生成证据」加 `--force` 语义或先删旧图
- 支持用户选择证据时刻，而不是固定取第一行有武器的行
- 渲染已核验统计
- 没选视频时前端拦截，别让 500 甩到页面上

### P3：统一命名 + 补 i18n + 参考帧包生成上页面 + outputs 清理入口

- 「时间同步核验」统一成「数据核验」
- 三个新页面抽 i18n
- 场地标注页加「新建参考帧包」
- `report_outputs_retention.py` 接成按钮（清理前 1.6G，其中 1.3G 可回收）

---

## 7. 安全边界

- 只绑 `127.0.0.1`，无登录，不要暴露公网
- 路径有项目根约束，实测 `/etc`、`/etc/passwd` 均被拒绝
- 自动化负责扫描、生成候选、校验、搬运、记录
- **人负责确认**：正式入库、执行训练、应用模型提升、提升场地资产
- 每次正式写入训练集前先「预演」
- 自动推进不会绕过人工门禁，也不会真正入库

---

## 8. 快速检查清单

**正式训练前：** 视频已登记 · 候选已刷新 · 样本标为已完成 ·
预演无意外 skipped · 训练集验证通过 · 训练预演命令正确

**模型提升前：** 候选模型路径存在 · 模型 ID 正确 · 验证不弱于基线 · 已生成提升计划

**场地资产提升前：** 至少 4 个真实地标（非 ROI 角点种子）· 控制点铺开不挤在一角 ·
**手动跑 `report_stage_control_point_quality.py --strict`**（页面暂不查这个）
