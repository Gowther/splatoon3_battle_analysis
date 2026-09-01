from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any

from src.data_registry import display_path, resolve_project_path
from src.heatmap.annotation_samples import ANNOTATION_FIELDS
from src.heatmap.annotation_round import has_manual_position, is_visible_task, priority_score


def read_annotation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relative_asset_path(value: str, output_html: Path) -> str:
    if not value:
        return ""
    target = (resolve_project_path(value) or Path(value).expanduser()).resolve()
    output_parent = output_html.expanduser().resolve().parent
    try:
        return Path(os.path.relpath(target, start=output_parent)).as_posix()
    except ValueError:
        return target.as_uri()


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


def render_annotation_html(rows: list[dict[str, str]], *, output_html: Path, title: str = "Heatmap Annotation") -> str:
    prepared = prepare_rows(rows, output_html)
    title_text = html.escape(title)
    fields_json = safe_json(ANNOTATION_FIELDS)
    rows_json = safe_json(prepared)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_text}</title>
  <style>
    body {{ margin: 0; font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #f5f5f3; }}
    header {{ padding: 12px 16px; background: #1f2933; color: white; display: flex; gap: 12px; align-items: center; }}
    header h1 {{ font-size: 16px; margin: 0; font-weight: 650; }}
    button, select, input {{ font: inherit; }}
    button {{ border: 1px solid #9aa4af; background: white; padding: 6px 10px; border-radius: 6px; cursor: pointer; }}
    button.primary {{ background: #0f766e; color: white; border-color: #0f766e; }}
    main {{ display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 12px; padding: 12px; height: calc(100vh - 49px); box-sizing: border-box; }}
    aside {{ overflow: auto; background: white; border: 1px solid #d7dce0; border-radius: 8px; }}
    .row-button {{ width: 100%; text-align: left; border: 0; border-bottom: 1px solid #e6e8ea; border-radius: 0; padding: 8px 10px; }}
    .row-button.active {{ background: #dff4f0; }}
    .row-button.done {{ box-shadow: inset 4px 0 0 #0f766e; }}
    .workspace {{ display: grid; grid-template-rows: minmax(0, 1fr) auto 180px; gap: 12px; min-width: 0; }}
    .images {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(240px, 360px); gap: 12px; min-height: 0; }}
    .panel {{ background: white; border: 1px solid #d7dce0; border-radius: 8px; overflow: hidden; min-width: 0; }}
    .image-wrap {{ height: 100%; display: grid; place-items: center; background: #111827; }}
    #frameImage {{ max-width: 100%; max-height: 100%; cursor: crosshair; }}
    #previewImage {{ max-width: 100%; max-height: 100%; }}
    .controls {{ display: grid; grid-template-columns: repeat(6, minmax(80px, 1fr)); gap: 8px; padding: 10px; align-items: end; }}
    label {{ display: grid; gap: 3px; color: #4b5563; font-size: 12px; }}
    input, select {{ border: 1px solid #cbd2d9; border-radius: 6px; padding: 6px 8px; background: white; color: #202124; min-width: 0; }}
    textarea {{ width: 100%; height: 100%; border: 0; border-top: 1px solid #d7dce0; box-sizing: border-box; padding: 10px; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; resize: none; }}
  </style>
</head>
<body>
<header>
  <h1>{title_text}</h1>
  <button id="prevButton">Prev</button>
  <button id="nextButton">Next</button>
  <button id="downloadButton" class="primary">Download CSV</button>
  <span id="statusText"></span>
</header>
<main>
  <aside id="rowList"></aside>
  <section class="workspace">
    <div class="images">
      <div class="panel image-wrap"><img id="frameImage" alt="frame"></div>
      <div class="panel image-wrap"><img id="previewImage" alt="preview"></div>
    </div>
    <div class="panel controls">
      <label>x<input id="xInput" inputmode="decimal"></label>
      <label>y<input id="yInput" inputmode="decimal"></label>
      <label>visibility<select id="visibilityInput"><option>visible</option><option>uncertain</option><option>occluded</option><option>absent</option></select></label>
      <label>complete<select id="completeInput"><option>false</option><option>true</option></select></label>
      <label style="grid-column: span 2;">notes<input id="notesInput"></label>
    </div>
    <div class="panel"><textarea id="csvOutput" spellcheck="false"></textarea></div>
  </section>
</main>
<script>
const fields = {fields_json};
const rows = {rows_json};
let selected = 0;
const list = document.getElementById("rowList");
const frameImage = document.getElementById("frameImage");
const previewImage = document.getElementById("previewImage");
const statusText = document.getElementById("statusText");
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
function rowDone(row) {{
  return row.x && row.y;
}}
function saveControls() {{
  const row = rows[selected];
  row.x = xInput.value;
  row.y = yInput.value;
  row.visibility = visibilityInput.value;
  row.frame_complete = completeInput.value;
  row.notes = notesInput.value;
  csvOutput.value = csvText();
  renderList();
}}
function renderList() {{
  list.innerHTML = "";
  rows.forEach((row, index) => {{
    const button = document.createElement("button");
    button.className = "row-button" + (index === selected ? " active" : "") + (rowDone(row) ? " done" : "");
    button.textContent = [row._row_index, row.match_id, row.time + "s", row.team, row.slot_hint || ""].join(" ");
    button.onclick = () => selectRow(index);
    list.appendChild(button);
  }});
}}
function selectRow(index) {{
  selected = Math.max(0, Math.min(rows.length - 1, index));
  const row = rows[selected];
  frameImage.src = row._frame_src || "";
  previewImage.src = row._preview_src || "";
  xInput.value = row.x || "";
  yInput.value = row.y || "";
  visibilityInput.value = row.visibility || "visible";
  completeInput.value = row.frame_complete || "false";
  notesInput.value = row.notes || "";
  statusText.textContent = `${{selected + 1}} / ${{rows.length}}`;
  csvOutput.value = csvText();
  renderList();
}}
frameImage.addEventListener("click", event => {{
  const rect = frameImage.getBoundingClientRect();
  const x = (event.clientX - rect.left) * frameImage.naturalWidth / rect.width;
  const y = (event.clientY - rect.top) * frameImage.naturalHeight / rect.height;
  xInput.value = x.toFixed(1);
  yInput.value = y.toFixed(1);
  saveControls();
}});
[xInput, yInput, visibilityInput, completeInput, notesInput].forEach(input => input.addEventListener("input", saveControls));
document.getElementById("prevButton").onclick = () => selectRow(selected - 1);
document.getElementById("nextButton").onclick = () => selectRow(selected + 1);
document.getElementById("downloadButton").onclick = () => {{
  saveControls();
  const blob = new Blob([csvOutput.value], {{type: "text/csv;charset=utf-8"}});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "annotation_template_filled.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}};
selectRow(0);
</script>
</body>
</html>
"""


def build_annotation_ui(
    annotation_csv: Path,
    output_html: Path,
    title: str = "Heatmap Annotation",
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
