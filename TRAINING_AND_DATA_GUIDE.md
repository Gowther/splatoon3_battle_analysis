# Training And Data Guide

这份文档回答三个问题：

- 如果要重新训练或补新数据，应该先准备什么资产，放在哪里。
- 当前项目里哪些模型已经有支持的训练/数据流程，哪些还只是实验计划。
- 现在目录结构和代码质量是否足够清楚，以及下一步最值得重构什么。

## 当前结论

当前项目结构已经足够支撑日常分析、数据接入、武器分类器训练、热力图标注调参和验证回归。主线已经从历史脚本堆收束到 `src/`、`scripts/`、`config/`、`tests/`、`models/`、`outputs/` 这几块。

但要注意：现在“正式可训练”的只有武器分类器。YOLO UI 检测、数字 OCR、消息 OCR 目前是运行时模型和实验候选，还没有被封装成项目级训练流水线。热力图当前也不是深度模型训练，而是通过人工点标注、参数实验和质量报告来调 tracker。

## 模型和资产现状

| 目标 | 当前资产 | 状态 | 训练入口 |
| --- | --- | --- | --- |
| UI/玩家状态检测 | `models/the_model.pt` | 运行中 | 暂无支持的项目级训练 CLI，`yolov5/` 是 vendor/runtime 依赖 |
| 数字 OCR | `models/ocr_model.pt` | 运行中 | 暂无支持的项目级训练 CLI |
| 消息 OCR | `models/message_ocr_model.pt` | 运行中，默认偏保守 | 暂无支持的项目级训练 CLI |
| 武器分类器 | `models/main_weapons_classification_weight.pth` | 可训练 | `scripts/train_weapon_classifier.py` |
| 武器标签表 | `main_weapon_list.txt` | 必须和分类器输出顺序一致 | `scripts/plan_weapon_training.py --write-labels` |
| 热力图 tracker | `src/heatmap/config_*.yaml` + `outputs/heatmap_*` | 可调参 | 标注和参数实验脚本，不是深度模型训练 |

## 资产应该放在哪里

| 资产 | 推荐位置 | 是否进 git | 说明 |
| --- | --- | --- | --- |
| 原始对战视频 | `footages/` | 不进 git | 本地大文件，注册到 `config/data_registry.json`。 |
| 普通分析输出 CSV/报告 | `outputs/` | 不进 git | 由脚本生成，必要时用 manifest 记录。 |
| 武器分类器真实训练图 | `main_training_dataset/<weapon_label>/...` | 不进 git | ImageFolder 结构；每个子目录名就是类别名。 |
| 武器图标源 | `main_icons/` | 可进 git | 当前用于合成数据生成，一个 PNG 对应一个武器类。 |
| 合成武器训练数据 | `outputs/generated_weapon_dataset/` | 不进 git | 先生成到 outputs，确认后再决定是否合并进真实训练集。 |
| 运行时模型权重 | `models/` | 小心处理 | 当前 canonical 权重在这里；替换前先备份旧权重和 metrics。 |
| YOLO 检测数据集 | `yolov5/train/`、`yolov5/valid/`、`yolov5/test/` | 不进 git | `.gitignore` 已忽略；目前没有项目级包装流程。 |
| 热力图人工标注包 | `outputs/heatmap_annotation_round1/` | 不进 git | 包含帧图、预览图、`annotation_template.csv`、HTML 标注页。 |
| 数据/实验说明 | `outputs/experiment_manifest.*` 或文档 | 看情况 | 用来记录本地大资产来源、命令、指标和时间。 |

## 每次动数据或模型前先跑

先进入项目环境：

```bash
source scripts/use_local_env.sh
```

确认项目状态：

```bash
git status --short
python scripts/inventory_project.py --output outputs/project_inventory.json
python scripts/report_project_hygiene.py \
  --output outputs/project_hygiene.md \
  --json-output outputs/project_hygiene.json
```

确认数据和基线还健康：

```bash
python scripts/validate_data_registry.py --strict
python scripts/report_dataset_governance.py \
  --output outputs/dataset_governance.md \
  --json-output outputs/dataset_governance.json \
  --strict
python scripts/report_model_registry.py \
  --output outputs/model_registry.md \
  --json-output outputs/model_registry.json \
  --strict
python scripts/run_validation_suite.py
```

如果准备进入模型/数据实验阶段，再跑 readiness：

```bash
python scripts/report_model_data_readiness.py \
  --output outputs/model_data_readiness.md \
  --json-output outputs/model_data_readiness.json
```

当前 readiness 的硬门槛之一是至少 30 条真实热力图标注。没有这些标签时，不建议急着换 YOLO/OCR/热力图 detector，因为指标还不够稳。

## 武器分类器：新增数据后重新训练

### 1. 准备真实训练图

目录必须是 ImageFolder 风格：

```text
main_training_dataset/
  Splash-o-matic/
    clip_001_frame_0001.png
    clip_001_frame_0002.png
  Splattershot-Jr/
    clip_002_frame_0001.png
```

要求：

- 子目录名必须是武器标签名。
- 现有武器就放进现有目录。
- 新武器需要新建目录，然后重新生成并审查 `main_weapon_list.txt`。
- 文件名不影响类别，但建议带来源、帧号或批次，方便追溯。
- 不要直接把未审查的合成数据混进真实训练集；先放 `outputs/generated_weapon_dataset/`。

### 2. 先检查标签、类别和当前模型是否一致

```bash
python scripts/plan_weapon_training.py \
  --dataset main_training_dataset \
  --labels main_weapon_list.txt \
  --model models/main_weapons_classification_weight.pth \
  --strict
```

如果只是给已有类别加图片，这一步应该通过。

如果新增了类别，`--strict` 很可能失败，这是正常的。先人工确认目录顺序和类别名，再写标签：

```bash
python scripts/plan_weapon_training.py \
  --dataset main_training_dataset \
  --labels main_weapon_list.txt \
  --write-labels \
  --strict
```

### 3. 干跑训练计划

```bash
python scripts/train_weapon_classifier.py \
  --dataset main_training_dataset \
  --labels main_weapon_list.txt \
  --output outputs/weapon_classifier_candidate/main_weapons_classification_weight.pth \
  --metrics outputs/weapon_classifier_candidate/metrics.json \
  --dry-run \
  --max-samples-per-class 1 \
  --epochs 1
```

### 4. 正式训练候选模型

建议先训练到 `outputs/weapon_classifier_candidate/`，不要一开始覆盖 `models/main_weapons_classification_weight.pth`：

```bash
python scripts/train_weapon_classifier.py \
  --dataset main_training_dataset \
  --labels main_weapon_list.txt \
  --output outputs/weapon_classifier_candidate/main_weapons_classification_weight.pth \
  --metrics outputs/weapon_classifier_candidate/metrics.json \
  --epochs 25 \
  --batch-size 32 \
  --device auto \
  --initial-model models/main_weapons_classification_weight.pth
```

如果要从 ImageNet 初始化而不是当前项目模型初始化，改用 `--pretrained`。

注意：`--initial-model` 是权重初始化/fine-tune 入口，不是完整 checkpoint resume。它会加载已有 `.pth` 模型权重，并要求输出类别数和当前数据集类别数一致；它不会恢复 optimizer 或 scheduler 状态。

### 5. 评估候选模型

训练后至少跑：

```bash
python scripts/report_dataset_governance.py \
  --output outputs/dataset_governance.md \
  --json-output outputs/dataset_governance.json \
  --strict
python scripts/run_validation_suite.py
```

如果要把候选模型和现有模型对比，先保存 baseline：

```bash
python scripts/report_model_benchmark_baseline.py \
  --output outputs/model_benchmarks/baseline_snapshot.md \
  --json-output outputs/model_benchmarks/baseline_snapshot.json
python scripts/write_experiment_manifest.py \
  --experiment-id weapon_classifier_candidate_001 \
  --artifact metrics=outputs/weapon_classifier_candidate/metrics.json \
  --artifact baseline=outputs/model_benchmarks/baseline_snapshot.json \
  --output outputs/weapon_classifier_candidate/manifest.md \
  --json-output outputs/weapon_classifier_candidate/manifest.json
```

只有当 metrics 和固定验证集都满意时，再考虑备份旧模型并替换：

```bash
cp models/main_weapons_classification_weight.pth outputs/model_benchmarks/main_weapons_classification_weight.previous.pth
cp outputs/weapon_classifier_candidate/main_weapons_classification_weight.pth models/main_weapons_classification_weight.pth
python scripts/run_validation_suite.py
```

## 武器分类器：生成合成数据

合成数据入口用于扩充或 smoke test，不应该未经审查直接覆盖真实训练集。

先 dry-run：

```bash
python scripts/generate_weapon_dataset.py \
  --icons main_icons \
  --backgrounds sample \
  --output-dir outputs/generated_weapon_dataset \
  --images-per-class 1 \
  --dry-run
```

正式生成：

```bash
python scripts/generate_weapon_dataset.py \
  --icons main_icons \
  --backgrounds sample \
  --output-dir outputs/generated_weapon_dataset \
  --images-per-class 50 \
  --write-labels outputs/generated_weapon_labels.txt
```

然后单独检查 `outputs/generated_weapon_dataset/` 的质量。确认图像像真实 UI crop 后，再决定是否复制到 `main_training_dataset/` 对应类别下。

## 新增普通视频数据

把视频放进：

```text
footages/match_12.mp4
```

先生成 intake 计划：

```bash
python scripts/intake_match.py \
  --match-id match_12 \
  --video footages/match_12.mp4 \
  --start-seconds 10 \
  --stop-seconds 150 \
  --sample-fps 5 \
  --device mps \
  --purpose analysis_candidate \
  --notes "mode/stage/team colors/source notes" \
  --dry-run \
  --strict \
  --report outputs/match_12_intake.md
```

确认报告后写入 registry 和 evaluation config：

```bash
python scripts/intake_match.py \
  --match-id match_12 \
  --video footages/match_12.mp4 \
  --start-seconds 10 \
  --stop-seconds 150 \
  --sample-fps 5 \
  --device mps \
  --purpose analysis_candidate \
  --notes "mode/stage/team colors/source notes" \
  --write \
  --strict
```

然后验证并跑分析窗口：

```bash
python scripts/validate_data_registry.py --strict
python scripts/evaluate_matches.py --only match_12_10_150 --run-analysis --strict
```

新视频不要马上当 baseline。先记录来源、模式、地图、队伍颜色、明显 OCR/武器错误，再决定是否提升为验证样本。

## 热力图数据和标注

热力图当前的核心不是训练深度模型，而是：

1. 注册 heatmap 样本。
2. 生成候选帧和标注包。
3. 人工填写真实 `x/y`。
4. 跑参数实验和质量报告。

新增 heatmap 样本时，先从 registry 生成一份小型 override config：

```bash
python scripts/create_heatmap_config.py \
  --match-id f_match_6 \
  --stop-seconds 330 \
  --duration-seconds 368.6 \
  --output outputs/heatmap_config_templates/config_f_match_6.yaml
```

确认 config 能跑通后，再把它晋升到 `src/heatmap/config_f_match_6.yaml` 并更新 `config/data_registry.json`。

第一轮标注包：

```bash
python scripts/prepare_heatmap_annotation_round.py \
  --round-id first_manual_loop \
  --package-dir outputs/heatmap_annotation_round1 \
  --output outputs/heatmap_annotation_round1.md \
  --json-output outputs/heatmap_annotation_round1.json
```

生成 HTML 标注页，并把优先任务排在前面：

```bash
python scripts/build_heatmap_annotation_ui.py \
  --annotation-csv outputs/heatmap_annotation_round1/annotation_template.csv \
  --output outputs/heatmap_annotation_round1/annotation_ui.html \
  --priority-limit 24
```

打开 HTML 后点击帧图填写 `x/y`，下载填好的 CSV，再替换或另存为标注 CSV。填完至少 30 条后跑：

```bash
python scripts/run_heatmap_parameter_experiments.py \
  --annotation-csv outputs/heatmap_annotation_round1/annotation_template.csv \
  --output-root outputs/heatmap_parameter_experiments \
  --write-configs \
  --output outputs/heatmap_parameter_experiments.md \
  --json-output outputs/heatmap_parameter_experiments.json
python scripts/report_model_data_readiness.py \
  --output outputs/model_data_readiness.md \
  --json-output outputs/model_data_readiness.json
```

## YOLO/OCR 如果要重新训练

现在不建议直接把 YOLO/OCR 训练当作主线。项目已经有数据目录 dry-run 和候选命令规划，但还没有封装真正的项目级训练 CLI 或候选模型晋升命令。

如果确实要探索 YOLO 训练，建议作为独立实验处理：

先生成训练规划报告：

```bash
python scripts/plan_model_training.py \
  --output outputs/model_training_plan.md \
  --json-output outputs/model_training_plan.json
```

再跑数据集 dry-run：

```bash
python scripts/validate_model_training_datasets.py \
  --output outputs/model_training_datasets.md \
  --json-output outputs/model_training_datasets.json
```

然后按报告补齐数据目录。UI detector 的历史 YOLO 数据放在 `.gitignore` 已忽略的 `yolov5/train/`、`yolov5/valid/`、`yolov5/test/`；新的 count/message OCR 数据放在 `outputs/model_training/count_ocr_dataset/` 和 `outputs/model_training/message_ocr_dataset/`。

YOLO/OCR 目录规范现在记录在 `config/model_training_targets.json` 的 `dataset_spec` 里。每个目标都应包含：

- `data_yaml`：YOLO data.yaml。
- `splits.train.images` 和 `splits.train.labels`。
- `splits.val.images` 和 `splits.val.labels`。
- `class_names`：候选数据集的类别顺序。

规则：

1. 不要直接覆盖 `models/the_model.pt`、`models/ocr_model.pt` 或 `models/message_ocr_model.pt`。
2. 先跑 `report_model_benchmark_baseline.py`、`report_model_errors.py`、`run_validation_suite.py` 保存现状。
3. 训练命令和结果写入 `write_experiment_manifest.py`。
4. 只有当 `plan_model_training.py` 和 `validate_model_training_datasets.py` 对目标从 `needs_data` 变为 `ready`，才进入实际训练。

换模型或补数据前，推荐先生成一份固定 baseline 包：

```bash
python scripts/run_model_experiment_baseline.py \
  --output-dir outputs/model_experiment_baseline
```

如果要同时重新跑完整验证套件：

```bash
python scripts/run_model_experiment_baseline.py \
  --output-dir outputs/model_experiment_baseline \
  --run-validation-suite
```

当前 `yolov5/` 更像 vendor/runtime 边界，不适合继续塞项目业务逻辑。等 YOLO/OCR 训练变成明确目标后，应该新增 `scripts/train_ui_detector.py` 或类似包装，而不是让用户直接记 upstream YOLO 命令。

## 目录结构评价

现在的目录结构已经比较明确：

- `src/`：支持中的业务代码和可测试逻辑。
- `scripts/`：支持中的 CLI 工作流。
- `config/`：registry、evaluation、annotation、experiment 配置。
- `models/`：canonical 运行时权重。
- `tests/`：快速单元测试。
- `outputs/`：所有生成物和实验报告。
- `footages/`、`main_training_dataset/`：本地大资产。
- `legacy/`、`notebooks/legacy/`：历史参考。
- `yolov5/`：当前运行时依赖/vendor 代码，不应作为新业务代码入口。

还不够清楚的地方：

- `models/` 现在已有 `config/models.json` registry，但候选模型晋升时还需要把训练数据和 metrics 自动写回 manifest。
- `main_training_dataset/` 是本地忽略目录，缺少数据 manifest 后很难复现。
- YOLO/OCR 训练资产和武器分类器训练资产还没有统一实验目录规范。
- `src/heatmap/config_*.yaml` 数量变多后，需要模板化或 registry 驱动生成。
- `DATA_AND_TRAINING.md` 已经有很多命令，后续要避免继续变成流水账。

## 代码质量评价

整体质量已经从“历史脚本集合”推进到“可维护的本地分析项目”：

- 支持入口集中到了 `src/` 和 `scripts/`。
- 有 registry、validation suite、dataset governance、runtime benchmark、model readiness 这些门禁。
- 大量逻辑已经拆成可单测函数，验证套件能跑 100+ 个测试。
- legacy 和 notebooks 已经有边界，不再默认参与健康检查。

主要风险还在这些地方：

- `src/run_analysis.py` 仍然是重业务入口，YOLO/OCR/weapon/runtime 状态耦合较重。
- 训练 CLI 支持 `--initial-model` fine-tune，但还不支持恢复 optimizer/scheduler 的完整 checkpoint resume。
- YOLO/OCR 没有项目级训练流水线，容易回到手动脚本和不可复现实验。
- 模型晋升流程已有 `config/models.json` 起点，但还没有自动化的候选模型晋升/回滚命令。
- 热力图配置和输出越来越多，后续会需要更强的配置模板和注册工具。

## 推荐重构顺序

1. 给模型加 registry：记录每个 canonical 权重的路径、用途、来源、训练数据、metrics、替换日期。
2. 给 `train_weapon_classifier.py` 增加完整 checkpoint `--resume`，恢复 optimizer/scheduler 状态。
3. 固化候选模型目录：例如 `outputs/model_experiments/<experiment_id>/`，里面统一放 weights、metrics、manifest、baseline。
4. 给武器训练集生成 dataset manifest，记录每个类别样本数、来源批次、人工/合成标记。
5. 如果确定要训 YOLO/OCR，先写项目级 CLI 和数据规范，再碰 `models/the_model.pt` 等 canonical 权重。
6. 把 heatmap config 进一步模板化，避免每个新 match 都手写一份近似 YAML。
7. 继续缩小 `src/run_analysis.py`，把检测、OCR、武器识别、CSV 输出和运行时状态拆成更明确的服务边界。

## 最安全的下一步

如果目标是“马上提高模型表现”，推荐先不要碰 YOLO/OCR，而是按这个顺序做：

1. 给 `main_training_dataset/` 补真实武器 crop。
2. 跑 `plan_weapon_training.py --strict`。
3. 训练候选武器分类器到 `outputs/weapon_classifier_candidate/`。
4. 跑 validation suite 和 manifest。
5. 同时完成至少 30 条热力图真实 `x/y` 标注，让 readiness 从 `needs_data` 往前走。

这样项目会保持可回滚、可比较、可解释，不会因为一次模型替换把现有分析基线冲散。
