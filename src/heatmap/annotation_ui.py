from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

from src.core.paths import relative_asset_path
from src.data_registry import display_path
from src.heatmap.annotation_samples import ANNOTATION_FIELDS
from src.heatmap.annotation_round import has_manual_position, is_visible_task, priority_score


def read_annotation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def priority_group(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("match_id", ""), row.get("time", ""), row.get("team", ""))


def priority_row_indices(rows: list[dict[str, str]], limit: int | None) -> list[int]:
    if limit is None or limit <= 0:
        return []

    candidates = [
        (index, row)
        for index, row in enumerate(rows)
        if is_visible_task(row) and not has_manual_position(row)
    ]
    candidates.sort(key=lambda item: (priority_score(item[1]), item[0]))

    selected_indices: list[int] = []
    seen_groups: set[tuple[str, str, str]] = set()
    for index, row in candidates:
        group = priority_group(row)
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected_indices.append(index)
        if len(selected_indices) >= limit:
            break
    return selected_indices


def order_annotation_rows(rows: list[dict[str, str]], priority_indices: list[int]) -> list[dict[str, str]]:
    if not priority_indices:
        return list(rows)

    selected = set(priority_indices)
    ordered_indices = priority_indices + [index for index in range(len(rows)) if index not in selected]
    ordered_rows: list[dict[str, str]] = []
    for index in ordered_indices:
        item = dict(rows[index])
        item["_row_index"] = str(index + 1)
        ordered_rows.append(item)
    return ordered_rows


def prioritize_annotation_rows(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    return order_annotation_rows(rows, priority_row_indices(rows, limit))


def prepare_rows(rows: list[dict[str, str]], output_html: Path) -> list[dict[str, str]]:
    prepared: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        item = {field: row.get(field, "") for field in ANNOTATION_FIELDS}
        item["_row_index"] = row.get("_row_index", str(index + 1))
        item["_frame_src"] = relative_asset_path(item.get("frame_path", ""), output_html)
        item["_preview_src"] = relative_asset_path(item.get("preview_path", ""), output_html)
        prepared.append(item)
    return prepared


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def render_annotation_html(rows: list[dict[str, str]], *, output_html: Path, title: str = "热力图人工标注") -> str:
    prepared = prepare_rows(rows, output_html)
    title_text = html.escape(title)
    fields_json = safe_json(ANNOTATION_FIELDS)
    rows_json = safe_json(prepared)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_text}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; min-height: 100vh; display: grid; grid-template-rows: auto minmax(0, 1fr); font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #f5f5f3; }}
    header {{ padding: 10px 16px; background: #1f2933; color: white; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    header h1 {{ font-size: 16px; margin: 0; font-weight: 650; }}
    .header-actions {{ display: flex; gap: 6px; }}
    #statusText {{ margin-left: auto; font-variant-numeric: tabular-nums; color: #e5e7eb; }}
    button, select, input {{ font: inherit; }}
    button {{ border: 1px solid #9aa4af; background: white; padding: 6px 10px; border-radius: 6px; cursor: pointer; }}
    button:disabled {{ cursor: not-allowed; opacity: 0.5; }}
    button.primary {{ background: #0f766e; color: white; border-color: #0f766e; }}
    button.warning {{ background: #fff7ed; color: #9a3412; border-color: #fdba74; white-space: nowrap; }}
    main {{ display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 12px; padding: 12px; min-height: 0; }}
    aside {{ display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; background: white; border: 1px solid #d7dce0; border-radius: 8px; overflow: hidden; }}
    .sidebar-heading {{ padding: 10px; border-bottom: 1px solid #e6e8ea; display: grid; gap: 2px; }}
    .sidebar-heading span {{ color: #6b7280; font-size: 12px; }}
    #rowList {{ overflow: auto; }}
    .row-button {{ width: 100%; text-align: left; border: 0; border-bottom: 1px solid #e6e8ea; border-radius: 0; padding: 8px 10px; }}
    .row-button.active {{ background: #dff4f0; }}
    .row-button.done {{ box-shadow: inset 4px 0 0 #0f766e; }}
    .row-button.skipped {{ box-shadow: inset 4px 0 0 #d97706; }}
    .workspace {{ display: grid; grid-template-rows: auto minmax(280px, 1fr) auto minmax(110px, 150px); gap: 12px; min-width: 0; min-height: 0; }}
    .guide {{ padding: 10px 12px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .guide-copy {{ min-width: 0; }}
    .guide-copy p {{ margin: 2px 0 0; color: #4b5563; font-size: 12px; }}
    #currentTaskText {{ font-weight: 650; color: #111827; }}
    #actionMessage {{ color: #9a3412; }}
    .images {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(240px, 360px); gap: 12px; min-height: 0; }}
    .panel {{ background: white; border: 1px solid #d7dce0; border-radius: 8px; overflow: hidden; min-width: 0; }}
    figure {{ margin: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); min-height: 0; }}
    figcaption {{ padding: 8px 10px; border-bottom: 1px solid #d7dce0; font-weight: 600; }}
    figcaption span {{ color: #6b7280; font-weight: 400; font-size: 12px; }}
    .image-wrap {{ position: relative; height: 100%; min-height: 0; display: grid; place-items: center; background: #111827; }}
    .image-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
    #frameImage {{ cursor: crosshair; }}
    .image-error {{ display: none; padding: 20px; color: #fecaca; text-align: center; }}
    .controls {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 8px; padding: 10px; align-items: start; }}
    label {{ display: grid; gap: 3px; color: #374151; font-size: 12px; min-width: 0; }}
    label small {{ color: #6b7280; min-height: 34px; }}
    input, select {{ border: 1px solid #cbd2d9; border-radius: 6px; padding: 6px 8px; background: white; color: #202124; min-width: 0; }}
    .data-preview {{ display: grid; grid-template-rows: auto minmax(0, 1fr); }}
    .data-heading {{ padding: 6px 10px; display: flex; gap: 10px; justify-content: space-between; border-bottom: 1px solid #d7dce0; font-size: 12px; }}
    .data-heading span {{ color: #9a3412; }}
    textarea {{ width: 100%; height: 100%; border: 0; padding: 8px 10px; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; resize: none; background: #f9fafb; }}
    @media (max-width: 980px) {{
      body {{ display: block; }}
      #statusText {{ width: 100%; margin-left: 0; }}
      main {{ grid-template-columns: 1fr; min-height: auto; }}
      aside {{ max-height: 240px; }}
      .workspace {{ grid-template-rows: auto auto auto 140px; }}
      .images {{ grid-template-columns: 1fr; }}
      figure {{ height: 420px; }}
      .controls {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
      .guide {{ align-items: stretch; flex-direction: column; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{title_text}</h1>
  <div class="header-actions">
    <button id="prevButton" title="返回上一条任务">← 上一条</button>
    <button id="nextButton" title="前往下一条任务">下一条 →</button>
  </div>
  <button id="downloadButton" class="primary" title="下载包含当前所有标注结果的新 CSV 文件">下载标注 CSV</button>
  <span id="statusText" role="status"></span>
</header>
<main>
  <aside>
    <div class="sidebar-heading">
      <strong>任务列表</strong>
      <span>绿线：已标坐标　黄线：已跳过</span>
    </div>
    <div id="rowList"></div>
  </aside>
  <section class="workspace">
    <div class="panel guide">
      <div class="guide-copy">
        <div id="currentTaskText">正在载入任务...</div>
        <p>只在左侧俯视地图的原始画面中，点击当前队伍、当前槽位玩家名字下方的小三角中心；右侧图片只是模型预测参考，不要直接照抄。</p>
        <p>若整张图不是俯视地图，请使用右侧跳过按钮。玩家暂时被遮挡时，在下方选择“被遮挡”；只有确认本队本帧全部可见玩家都已处理，才选择“是，已全部标完”。</p>
        <p id="actionMessage"></p>
      </div>
      <button id="skipNonOverheadButton" class="warning" title="将同一帧、同一队伍的任务全部标记为无法标注，不会计入模型误报">非俯视图：跳过本帧本队</button>
    </div>
    <div class="images">
      <figure class="panel">
        <figcaption>原始画面 <span>在这里点击玩家标记中心</span></figcaption>
        <div class="image-wrap">
          <img id="frameImage" alt="待标注的原始游戏画面">
          <div id="frameImageError" class="image-error"></div>
        </div>
      </figure>
      <figure class="panel">
        <figcaption>模型参考图 <span>仅用于对照预测位置</span></figcaption>
        <div class="image-wrap">
          <img id="previewImage" alt="带有模型预测标记的参考画面">
          <div id="previewImageError" class="image-error"></div>
        </div>
      </figure>
    </div>
    <div class="panel controls">
      <label>X 坐标
        <input id="xInput" type="number" min="0" step="0.1" inputmode="decimal" placeholder="点击原图自动填写">
        <small>玩家标记中心相对原图左边缘的位置，可手动微调。</small>
      </label>
      <label>Y 坐标
        <input id="yInput" type="number" min="0" step="0.1" inputmode="decimal" placeholder="点击原图自动填写">
        <small>玩家标记中心相对原图上边缘的位置，可手动微调。</small>
      </label>
      <label>玩家可见状态
        <select id="visibilityInput">
          <option value="visible">可见，正常标注</option>
          <option value="uncertain">位置不确定，仍保留标点</option>
          <option value="occluded">被遮挡，无法标点</option>
          <option value="absent">不在画面或跳过</option>
        </select>
        <small>“被遮挡”和“不在画面”会作为跳过项，不计入有效坐标。</small>
      </label>
      <label>本队本帧是否全部标完
        <select id="completeInput">
          <option value="false">否，仍可能有遗漏</option>
          <option value="true">是，已全部标完</option>
        </select>
        <small>仅在本队所有可见玩家都已处理后选“是”，用于评估模型误报。</small>
      </label>
      <label style="grid-column: span 2;">备注
        <input id="notesInput" placeholder="可填写遮挡、位置不确定等情况">
        <small>非俯视图按钮会自动写入标准跳过原因，无需重复填写。</small>
      </label>
    </div>
    <div class="panel data-preview">
      <div class="data-heading">
        <strong>CSV 实时预览（只读）</strong>
        <span>页面不会直接修改原 CSV；刷新或关闭前请下载标注 CSV。</span>
      </div>
      <textarea id="csvOutput" spellcheck="false" readonly aria-label="当前标注 CSV 实时预览"></textarea>
    </div>
  </section>
</main>
<script>
const fields = {fields_json};
const rows = {rows_json};
let selected = 0;
const list = document.getElementById("rowList");
const frameImage = document.getElementById("frameImage");
const previewImage = document.getElementById("previewImage");
const frameImageError = document.getElementById("frameImageError");
const previewImageError = document.getElementById("previewImageError");
const statusText = document.getElementById("statusText");
const currentTaskText = document.getElementById("currentTaskText");
const actionMessage = document.getElementById("actionMessage");
const xInput = document.getElementById("xInput");
const yInput = document.getElementById("yInput");
const visibilityInput = document.getElementById("visibilityInput");
const completeInput = document.getElementById("completeInput");
const notesInput = document.getElementById("notesInput");
const csvOutput = document.getElementById("csvOutput");

function csvCell(value) {{
  const text = String(value ?? "");
  return /[",\\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
}}
function csvText() {{
  return [fields.join(","), ...rows.map(row => fields.map(field => csvCell(row[field])).join(","))].join("\\n") + "\\n";
}}
function rowState(row) {{
  if (["occluded", "absent"].includes(row.visibility)) return "skipped";
  if (row.x && row.y) return "done";
  return "pending";
}}
function teamLabel(team) {{
  const labels = {{blue: "蓝队", cyan: "青队", green: "绿队", orange: "橙队", pink: "粉队", purple: "紫队", red: "红队", yellow: "黄队"}};
  return labels[team] ? `${{labels[team]}}（${{team}}）` : (team || "未指定队伍");
}}
function groupKey(row) {{
  return [row.match_id, row.time, row.frame_index, row.team].join("|");
}}
function updateProgress() {{
  const labeled = rows.filter(row => rowState(row) === "done").length;
  const skipped = rows.filter(row => rowState(row) === "skipped").length;
  const pending = rows.length - labeled - skipped;
  statusText.textContent = rows.length
    ? `第 ${{selected + 1}} / ${{rows.length}} 条　已标 ${{labeled}}　已跳过 ${{skipped}}　待处理 ${{pending}}`
    : "没有可标注任务";
}}
function showImage(image, errorBox, src, label) {{
  errorBox.style.display = "none";
  image.style.display = "block";
  image.alt = label;
  if (!src) {{
    image.removeAttribute("src");
    image.style.display = "none";
    errorBox.style.display = "block";
    errorBox.textContent = `${{label}}路径为空，无法显示。`;
    return;
  }}
  image.src = src;
}}
function saveControls() {{
  if (!rows.length) return;
  const row = rows[selected];
  row.x = xInput.value;
  row.y = yInput.value;
  row.visibility = visibilityInput.value;
  row.frame_complete = completeInput.value;
  row.notes = notesInput.value;
  csvOutput.value = csvText();
  renderList();
  updateProgress();
}}
function renderList() {{
  list.innerHTML = "";
  rows.forEach((row, index) => {{
    const button = document.createElement("button");
    const state = rowState(row);
    button.className = "row-button" + (index === selected ? " active" : "") + (state === "done" ? " done" : "") + (state === "skipped" ? " skipped" : "");
    button.textContent = `#${{row._row_index}}　${{row.match_id || "未命名比赛"}} · ${{row.time || "?"}} 秒 · ${{teamLabel(row.team)}} · 槽位 ${{row.slot_hint || "?"}}`;
    button.title = state === "done" ? "已标注坐标" : (state === "skipped" ? "已跳过" : "待处理");
    button.onclick = () => selectRow(index);
    list.appendChild(button);
  }});
}}
function selectRow(index) {{
  if (!rows.length) {{
    currentTaskText.textContent = "当前标注文件中没有任务。";
    csvOutput.value = csvText();
    renderList();
    updateProgress();
    ["prevButton", "nextButton", "downloadButton", "skipNonOverheadButton"].forEach(id => document.getElementById(id).disabled = true);
    return;
  }}
  selected = Math.max(0, Math.min(rows.length - 1, index));
  const row = rows[selected];
  showImage(frameImage, frameImageError, row._frame_src, "待标注的原始游戏画面");
  showImage(previewImage, previewImageError, row._preview_src, "带有模型预测标记的参考画面");
  xInput.value = row.x || "";
  yInput.value = row.y || "";
  visibilityInput.value = row.visibility || "visible";
  completeInput.value = row.frame_complete || "false";
  notesInput.value = row.notes || "";
  currentTaskText.textContent = `当前任务：比赛 ${{row.match_id || "未命名"}} · ${{row.time || "?"}} 秒 · ${{teamLabel(row.team)}} · 槽位 ${{row.slot_hint || "未指定"}}`;
  actionMessage.textContent = "";
  csvOutput.value = csvText();
  document.getElementById("prevButton").disabled = selected === 0;
  document.getElementById("nextButton").disabled = selected === rows.length - 1;
  renderList();
  updateProgress();
}}
frameImage.addEventListener("click", event => {{
  const rect = frameImage.getBoundingClientRect();
  if (!frameImage.naturalWidth || !frameImage.naturalHeight || !rect.width || !rect.height) return;
  const x = (event.clientX - rect.left) * frameImage.naturalWidth / rect.width;
  const y = (event.clientY - rect.top) * frameImage.naturalHeight / rect.height;
  xInput.value = x.toFixed(1);
  yInput.value = y.toFixed(1);
  if (["occluded", "absent"].includes(visibilityInput.value)) visibilityInput.value = "visible";
  saveControls();
}});
frameImage.addEventListener("load", () => {{ frameImage.style.display = "block"; frameImageError.style.display = "none"; }});
previewImage.addEventListener("load", () => {{ previewImage.style.display = "block"; previewImageError.style.display = "none"; }});
frameImage.addEventListener("error", () => {{ frameImage.style.display = "none"; frameImageError.style.display = "block"; frameImageError.textContent = "原始画面加载失败，请确认 frames 目录与此 HTML 保持在同一个标注包中。"; }});
previewImage.addEventListener("error", () => {{ previewImage.style.display = "none"; previewImageError.style.display = "block"; previewImageError.textContent = "模型参考图加载失败；仍可使用左侧原始画面标注。"; }});
[xInput, yInput, completeInput, notesInput].forEach(input => input.addEventListener("input", saveControls));
visibilityInput.addEventListener("change", () => {{
  if (["occluded", "absent"].includes(visibilityInput.value)) {{
    xInput.value = "";
    yInput.value = "";
  }}
  saveControls();
}});
document.getElementById("prevButton").onclick = () => selectRow(selected - 1);
document.getElementById("nextButton").onclick = () => selectRow(selected + 1);
document.getElementById("skipNonOverheadButton").onclick = () => {{
  if (!rows.length) return;
  saveControls();
  const key = groupKey(rows[selected]);
  const groupRows = rows.filter(row => groupKey(row) === key);
  const hasCoordinates = groupRows.some(row => row.x || row.y);
  if (hasCoordinates && !window.confirm("本帧本队已有坐标。继续会清空这些坐标并整组跳过，是否继续？")) return;
  const reason = "跳过：非俯视图，无法标注地图坐标（skip_reason=non_overhead_view）";
  groupRows.forEach(row => {{
    row.x = "";
    row.y = "";
    row.visibility = "absent";
    row.frame_complete = "false";
    if (!String(row.notes || "").includes("skip_reason=non_overhead_view")) {{
      row.notes = row.notes ? `${{row.notes}}；${{reason}}` : reason;
    }}
  }});
  selectRow(selected);
  actionMessage.textContent = `已将本帧本队的 ${{groupRows.length}} 条任务标记为“非俯视图并跳过”，不会计入有效坐标或模型误报。`;
}};
document.getElementById("downloadButton").onclick = () => {{
  saveControls();
  const blob = new Blob([csvOutput.value], {{type: "text/csv;charset=utf-8"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "annotation_template_filled.csv";
  link.click();
  URL.revokeObjectURL(link.href);
  actionMessage.textContent = "标注 CSV 已下载。后续请使用下载的 annotation_template_filled.csv 继续分析。";
}};
selectRow(0);
</script>
</body>
</html>
"""


def build_annotation_ui(
    annotation_csv: Path,
    output_html: Path,
    title: str = "热力图人工标注",
    priority_limit: int | None = None,
) -> dict[str, Any]:
    rows = read_annotation_rows(annotation_csv)
    priority_indices = priority_row_indices(rows, priority_limit)
    ordered_rows = order_annotation_rows(rows, priority_indices)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_annotation_html(ordered_rows, output_html=output_html, title=title), encoding="utf-8")
    return {
        "status": "ready" if rows else "empty",
        "annotation_csv": display_path(annotation_csv),
        "output_html": display_path(output_html),
        "rows": len(rows),
        "priority_limit": priority_limit,
        "priority_rows": len(priority_indices),
    }
