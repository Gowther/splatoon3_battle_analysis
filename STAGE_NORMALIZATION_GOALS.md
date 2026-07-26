# Stage Coordinate Normalization

这个文档记录“把热力图坐标从源视频像素变成标准场地坐标”的建设路线。当前完成的是 Goal 1-2：homography 已能真正生效，并且有了给人工填地标用的参考帧包。

## 为什么需要这一层

现在 `outputs/heatmap_match9/` 里的所有点位都是源视频像素坐标。这带来三个问题：

- 换一场对战、换一个录制分辨率，坐标就不可比。
- 同一张地图的不同镜头缩放会被当成不同位置。
- 无法把点位叠到标准场地图上做跨场次统计。

`src/heatmap/stage_coordinates.py` 已经实现了 homography 求解和归一化，但在 Goal 1 之前没有任何一场对战填过真实控制点，因此始终回退到 ROI 线性映射：

```text
homography_status: needs_control_points
control_point_count: 0
method: roi_linear_normalization
```

ROI 线性映射只是把矩形 ROI 拉成 0..1，它不校正俯视地图的透视形变，所以只能当占位方案。

## 1-8 Goal Roadmap

1. 为一场对战建立控制点资产，让 homography 真正启用，并报告重投影误差。
2. 导出带地图网格的参考帧和控制点标注模板，让人工可以按地标填点。
3. 在 Web 工作台加入控制点标注入口，支持点选和拖动修正。
4. 把 homography 接入 `src.heatmap.run_pipeline`，让主产物直接带 `stage_x/stage_y`。
5. 建立控制点质量评估：重投影误差门禁、跨帧稳定性、地标覆盖度。
6. 建立 stage 注册表，把同一张地图的控制点在多场对战之间复用。
7. 用归一化坐标渲染标准场地热力图，替换当前像素坐标渲染。
8. 跨场次聚合：同地图多场对战的占位热力图和路线对比。

## Goal 1 产出

新增命令：

```bash
.venv/bin/python scripts/build_stage_control_points.py \
  --config src/heatmap/config_match9.yaml \
  --stage-id scorch_gorge
```

默认输出：

- `config/stage_control_points/<stage_id>.json`

这个资产可以直接喂给已有的报告命令：

```bash
.venv/bin/python scripts/report_stage_coordinates.py \
  --config src/heatmap/config_match9.yaml \
  --control-points config/stage_control_points/scorch_gorge.json \
  --normalized-output outputs/heatmap_match9/player_tracks_stage.csv
```

启用后报告从 `roi_linear_normalization` 变成 `homography`，并且多出重投影误差字段。

## 控制点资产格式

沿用 `config/stage_control_points.template.json` 的结构：

```json
{
  "schema_version": 1,
  "template": false,
  "stage_id": "scorch_gorge",
  "coordinate_space": "video_pixels",
  "target_coordinate_space": "stage_normalized_0_1",
  "control_points": [
    {"name": "yellow_spawn_pad", "source": [412, 786], "target": [0.18, 0.82]}
  ]
}
```

约定：

- `source` 是源视频像素坐标，和 `map_view.roi` 同一个坐标系。
- `target` 是标准场地归一化坐标，`0..1`，左上为原点。
- `template` 必须是 `false`。模板资产会被显式拒绝启用 homography，避免示例点污染真实产物。
- 至少 4 个点，且不能有重复 `source`。

## 重投影误差

Goal 1 增加了 homography 的自检：把每个控制点的 `source` 代回矩阵，和它声明的 `target` 比较。

报告字段：

- `reprojection.max_error`: 最大误差，单位是归一化坐标。
- `reprojection.mean_error`: 平均误差。
- `reprojection.worst_point`: 误差最大的控制点名称。
- `reprojection.status`: 超过阈值时为 `high_error`。

默认阈值 `0.02`，约等于标准场地图短边的 2%。四点共线或接近共线时误差会明显变大，这是需要重新标注的信号。

## Current Limits

Goal 1 只解决“单场对战、单组人工控制点、单一 homography”。它不会假装知道：

- 控制点是否真的对准了地标，人工填错不会被发现，只有共线和重投影异常能被发现。
- 镜头在对战中缩放或平移时，单一矩阵是否仍然成立。
- `stage_id` 是否对应真实的官方地图名。
- 同一张地图在不同录制设置下能否直接复用同一组控制点。

这些会在 Goal 2-6 通过参考帧导出、Web 标注、跨帧稳定性评估和 stage 注册表逐步补上。

## Goal 2: 参考帧包

Goal 2 新增命令：

```bash
.venv/bin/python scripts/export_stage_reference.py \
  --config src/heatmap/config_match9.yaml \
  --stage-id match9_stage \
  --times 30,60,90
```

默认输出到 `outputs/stage_reference/<stage_id>/`：

- `frames/reference_*.jpg`: 带绿色 ROI 边框和归一化网格刻度的参考帧。
  默认取配置的 `reference_time_seconds`，`--times` 可以追加多个时间点，
  方便挑一帧遮挡最少的画面。
- `control_points_draft.json`: ROI 角点种子草稿，`template: true`。
  在把 4 个种子点换成真实地标之前，它无法通过校验，也无法启用 homography。
- `README.md`: 中文填写说明，含地标选择建议和校验命令。
- `manifest.json`: 导出清单和下一步提示。

网格刻度标注的是 ROI 线性归一化坐标，只用来帮助人读出像素位置；填进
`target` 的应该是地标在标准场地图上的位置，两者在有透视形变时不相等。

填完地标并把 `template` 改成 `false` 后：

```bash
.venv/bin/python scripts/build_stage_control_points.py \
  --config src/heatmap/config_match9.yaml \
  --control-points outputs/stage_reference/match9_stage/control_points_draft.json \
  --validate --strict
```

校验通过即可复制为 `config/stage_control_points/<stage_id>.json` 正式资产。

Goal 2 之后仍未解决：地标填错但不共线时只能靠人复查；这依赖 Goal 3 的
Web 标注入口（点选自动写坐标）和 Goal 5 的跨帧稳定性评估。

## 相关文件

| 路径 | 作用 |
| --- | --- |
| `src/heatmap/stage_coordinates.py` | ROI 归一化、控制点解析、homography 求解与重投影自检。 |
| `src/heatmap/stage_reference.py` | 网格参考帧导出、控制点草稿与填写说明生成。 |
| `scripts/build_stage_control_points.py` | 生成和校验控制点资产。 |
| `scripts/export_stage_reference.py` | 导出地标标注用的参考帧包。 |
| `scripts/report_stage_coordinates.py` | 报告归一化状态并导出 `stage_x/stage_y`。 |
| `config/stage_control_points.template.json` | 控制点模板。 |
| `config/stage_control_points/` | 真实控制点资产目录。 |
