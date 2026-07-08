# Active Learning Workbench

这份文档说明本地 Web 工作台如何把“新增素材 -> 自动分析 -> 失败样本 -> 人/LLM 标注 -> 训练集 -> 训练/评估 -> 模型提升”串成闭环。

## 启动

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --host 127.0.0.1 --port 8765
```

打开：

```text
http://127.0.0.1:8765
```

如果端口被占用，换一个端口：

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --port 8766
```

## 8 个 goal 对应的页面能力

| Goal | Web 工作台能力 | 当前安全边界 |
| --- | --- | --- |
| 1. 总控台 | 读取 validation、model errors、training candidates、heatmap labels、dataset readiness、runtime、promotion 等报告 | 只读汇总 |
| 2. 新素材接入 | 扫描 `footages/`，识别未登记视频，表单触发 `scripts/intake_samples.py` | 只执行白名单脚本 |
| 3. 失败样本队列 | 读取 `outputs/training_sample_candidates/manifest.json`，展示 UI/OCR/heatmap 候选 | 队列来自现有导出器 |
| 4. 标注工作台 | Canvas 画 YOLO/OCR 框，heatmap 点坐标，保存到 staging | 不直接写正式训练集 |
| 5. LLM 辅助裁判 | 生成 `outputs/active_learning_workbench/llm_review_pack.json`，可记录 LLM 建议 | LLM 建议不自动入库 |
| 6. staging 入库 | 对 `done` 标注做 dry-run 或 apply，生成 YOLO label 和 sidecar metadata | apply 前可先 dry-run |
| 7. 训练/评估编排 | 触发 dataset validation、training dry-run、training execute、baseline | 真训练需要确认 |
| 8. 模型提升/回滚 | 触发 promotion plan/apply | apply 需要确认，现有脚本会备份旧模型 |

## 典型使用流程

1. 把新视频放到 `footages/`。
2. 打开工作台，在 Asset Inbox 里确认是否出现新视频。
3. 点击 Use，检查 match id，然后运行 Intake Video。
4. 运行 Run Validation，可按需勾选 run analysis。
5. 运行 Refresh Candidates，生成新的失败样本候选。
6. 在 Queue 里逐条标注或跳过。
7. 标注结果先进入：

```text
outputs/active_learning_workbench/staging_annotations.json
```

8. 运行 Apply Staging 的 Dry Run。
9. Dry Run 没有 blocker 后，再运行 Apply。
10. 运行 Validate Datasets 和 Training Dry Run。
11. 需要训练时再确认 Execute Training。
12. 训练结果通过验证后，用 Promotion Plan / Apply Promotion 替换正式模型。

## 生成 LLM review pack

Web 上点击 Build LLM Pack，或者直接调用 API：

```bash
curl -sS -X POST http://127.0.0.1:8765/api/llm-review-pack \
  -H 'content-type: application/json' \
  --data '{"limit":30}'
```

输出会写到：

```text
outputs/active_learning_workbench/llm_review_pack.json
```

这个 JSON 只要求 LLM 给建议、置信度和理由。最终是否标为 `done` 仍由人或规则确认。

## 训练集写入规则

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

`heatmap_tracker_labels` 当前写入工作台补充标签 CSV：

```text
outputs/active_learning_workbench/heatmap_staging_labels.csv
```

后续如果要让它直接回写 `outputs/heatmap_annotation_round1/annotation_template.csv`，建议先加一次 CSV merge 校验，避免覆盖已有人工标签。

## API smoke

```bash
curl -sS http://127.0.0.1:8765/api/state
curl -sS http://127.0.0.1:8765/api/candidates
curl -sS -X POST http://127.0.0.1:8765/api/apply-staging \
  -H 'content-type: application/json' \
  --data '{"dry_run":true}'
```

## 重要原则

- 自动化负责搬运、排序、校验和生成候选。
- LLM 负责建议，不负责最终入库。
- 人只处理低置信、高风险、最终入库、训练启动和模型提升。
- 正式训练集写入前先 dry-run。
- 模型提升前先验证，并保留备份。
