# Stage Coordinate Normalization

这个文档记录“把热力图坐标从源视频像素变成标准场地坐标”的建设路线。当前完成的是 Goal 1-6：homography 已能真正生效，有参考帧包、Web 点选标注、管线自动产出、几何质量门禁，以及 stage 注册表跨场次复用与交叉验证。

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

## Goal 3: Web 点选标注

工作台新增「场地标注」页面：

```bash
.venv/bin/python scripts/serve_active_learning_workbench.py --port 8765
# 打开 http://127.0.0.1:8765/stage-labeling
```

流程：

1. 左侧列出 `outputs/stage_reference/` 下的所有参考帧包。
2. 选包后展示网格参考帧，可切换 `--times` 导出的多个时间点。
3. 在图上点击即写入 `source_x/source_y`（自动换算回原图像素），
   再在表格里填 `stage_x/stage_y` 目标坐标和地标名称。
4. 「保存草稿并校验」把点写回 `control_points_draft.json` 并立即返回
   Goal 1 的完整校验结果（含重投影误差）。少于 4 个点会保持
   `template: true`，不可能提前启用 homography。
5. 校验通过后「提升为正式资产」把草稿复制到
   `config/stage_control_points/<stage_id>.json`，并给出下一步
   `report_stage_coordinates.py` 命令。提升前会再校验一次，
   未通过的草稿会被拒绝。

相关 API：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/stage-labeling/state` | 参考帧包列表和草稿状态。 |
| POST | `/api/stage-labeling/save` | 保存点选结果并校验。 |
| POST | `/api/stage-labeling/promote` | 校验通过后提升为正式资产。 |

Goal 3 之后仍未解决：target 坐标还是手填数字，需要标准场地图底图对照
（Goal 7 的标准渲染可以反哺这里）；单帧标注无法发现镜头移动，
留给 Goal 5 的跨帧稳定性评估。

## Goal 4: 管线自动产出 stage 坐标

`src.heatmap.run_pipeline` 现在会在每次运行结束时自动寻找已提升的
控制点资产：

1. 优先用配置里显式声明的 `stage_coordinates.control_point_asset`。
2. 否则按 `config/stage_control_points/<match_id>.json` 和
   `<stage_id>.json` 查找。
3. 模板资产（`template: true`）会被忽略，不会静默启用。

找到资产时，管线把 `player_tracks.csv` 归一化为：

```text
outputs/<heatmap>/player_tracks_stage.csv
```

并且：

- `report.md` 增加 `Stage Normalization` 小节，展示方法、资产路径、
  行数和重投影最大误差；启用 homography 后，
  “坐标还是源视频像素”的已知限制行会替换成提醒复核地标的版本。
- `run_manifest.json` 增加 `stage_normalization` 字段和
  `stage_tracks` 产物项。
- `--clean-output` 会一并清掉 `player_tracks_stage.csv`。
- 没有资产时行为完全不变，只在输出里注明 `no_asset`。

也就是说：在 `/stage-labeling` 页面提升资产之后，重跑管线（或
`--only-report`）就能直接拿到 stage 坐标产物，不需要再手动跑
`report_stage_coordinates.py`。

Goal 4 之后仍未解决：stage 坐标只覆盖 `player_tracks.csv`，
enriched/team points 的归一化和标准场地渲染留给 Goal 7。

## Goal 5: 控制点质量门禁

Goal 1 的重投影自检有一个盲区：**它只检查控制点自己**。

如果 4 个地标全都挤在画面一角，拟合出的矩阵会把这 4 个点映射得完美无缺
（实测重投影误差 2e-16），但把 ROI 其余部分映射到场地之外。实测同一份
聚集控制点，ROI 右上角被映射到 stage `(17.4, -1.5)`。

Goal 5 增加三个重投影看不见的检查：

```bash
.venv/bin/python scripts/report_stage_control_point_quality.py \
  --config src/heatmap/config_match9.yaml \
  --strict
```

| 检查 | 含义 | 默认阈值 |
| --- | --- | --- |
| `reprojection` | 控制点自身的回代误差（Goal 1） | `<= 0.02` |
| `coverage` | 控制点凸包占地图 ROI 的面积比 | `>= 0.15` |
| `corners` | ROI 四角经 homography 后偏离 0..1 场地框的距离 | `<= 0.35` |
| `frame_drift` | 同名地标在多个参考帧之间的像素漂移 | `<= 12px` |

不传 `--control-points` 时，会自动复用 Goal 4 的资产发现逻辑。

`--labeled-frames` 接受一个 `{帧号: [控制点...]}` 的 JSON，用于跨帧稳定性：
同一个地标在不同时间点的标注如果漂移过大，说明镜头发生了缩放或平移，
单一 homography 不再成立。没有共享地标时该项为 `not_available`，不算失败。

实测对照：

- 铺满 ROI 的 4 个角点：全部通过，`coverage=1.000`，角点偏移 `<1e-6`。
- 挤在一角的 4 个点：重投影仍然 `ready`，但 `coverage=0.0096`、
  角点偏移 `16.43`，报告判为 `needs_review`，`--strict` 退出码 1。
- 镜头移动的跨帧标注：漂移 `52.02px`，判为 `unstable`。

Goal 5 之后仍未解决：这些都是几何自洽性检查，无法判断地标是否真的对准了
官方地图上的那个位置——填错但几何合理的点仍然只能靠人复查，
需要 Goal 6 的 stage 注册表跨场次交叉验证。

## Goal 6: stage 注册表与交叉验证

Goal 5 的四道门禁都是**单份资产的自洽性检查**。它们看不见一类错误：
地标点选得几何合理，但对错了位置。

实测：把某个角点的 `target` 从 `1.0` 改成 `0.75`，Goal 5 的
`reprojection`、`coverage`、`corners`、`frame_drift` **全部通过**，
状态是 `ready`。单份资产无论如何自查都发现不了这个错误——
因为它内部完全自洽。

唯一的外部参照是**同一张地图的另一次独立标注**。

```bash
# 把对战登记到 stage
.venv/bin/python scripts/report_stage_registry.py --register scorch_gorge match_9
.venv/bin/python scripts/report_stage_registry.py --register scorch_gorge match_10

# 交叉验证
.venv/bin/python scripts/report_stage_registry.py --strict
```

注册表 `config/stage_registry.json` 按 stage 而非 match 组织：

```json
{
  "schema_version": 1,
  "stages": [
    {
      "stage_id": "scorch_gorge",
      "matches": ["match_9", "match_10"],
      "control_point_asset": "config/stage_control_points/scorch_gorge.json"
    }
  ]
}
```

带来两件事：

1. **复用**：同一 stage 下的对战共享一份控制点资产，新对战登记后
   直接继承，不需要重新标注。一个 match 只能属于一个 stage，
   重复登记会自动从旧 stage 移出。
2. **交叉验证**：如果某场对战另有自己的 `<match_id>.json` 资产，
   会把 ROI 上 9×9 的采样点分别过两个 homography，比较它们落到的
   stage 坐标。同一张地图的两次标注应该把同一个像素送到同一个位置。

默认阈值 `0.05`（stage 宽度的 5%）。实测对照：

| 情况 | 最大分歧 | 判定 |
| --- | --- | --- |
| 同一 stage，手标有几像素抖动 | `0.0051` | `ready` |
| 一个地标对错位置（Goal 5 全绿） | `0.2500` | `disagrees` |

报告会直接点名出问题的对战，`--strict` 退出码 1。

Goal 6 之后仍未解决：交叉验证需要至少两次独立标注，只标了一场的 stage
仍然无人可对照；且两次都错到同一处时依然测不出来。此外目前没有任何
match 声明 stage，注册表是空的——真正的价值要等第二场同图对战标注后才体现。

## 相关文件

| 路径 | 作用 |
| --- | --- |
| `src/heatmap/stage_coordinates.py` | ROI 归一化、控制点解析、homography 求解与重投影自检。 |
| `src/heatmap/stage_reference.py` | 网格参考帧导出、控制点草稿与填写说明生成。 |
| `src/heatmap/stage_quality.py` | 控制点覆盖度、角点合理性与跨帧稳定性评估。 |
| `src/heatmap/stage_registry.py` | stage 注册表、跨场次资产复用与标注交叉验证。 |
| `src/stage_labeling_workbench.py` | Web 场地标注页面的包发现、草稿保存校验与资产提升。 |
| `scripts/build_stage_control_points.py` | 生成和校验控制点资产。 |
| `scripts/report_stage_control_point_quality.py` | 控制点几何质量门禁报告。 |
| `scripts/report_stage_registry.py` | stage 登记、复用状态与交叉验证报告。 |
| `scripts/export_stage_reference.py` | 导出地标标注用的参考帧包。 |
| `scripts/report_stage_coordinates.py` | 报告归一化状态并导出 `stage_x/stage_y`。 |
| `config/stage_control_points.template.json` | 控制点模板。 |
| `config/stage_control_points/` | 真实控制点资产目录。 |
| `config/stage_registry.json` | stage 到对战的映射与规范资产声明。 |
