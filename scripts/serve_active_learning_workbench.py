#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.active_learning_workbench import (
    DEFAULT_LLM_REVIEWS_PATH,
    DEFAULT_STAGING_PATH,
    apply_staging_annotations,
    build_llm_review_pack,
    build_workbench_state,
    load_candidate_queue,
    load_llm_reviews,
    load_staging,
    media_type_for_path,
    record_llm_review,
    run_workbench_action,
    safe_project_file,
    upsert_staging_annotation,
)


APP_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Active Learning Workbench</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f8;
      --surface: #ffffff;
      --ink: #172026;
      --muted: #65717b;
      --line: #d7dde2;
      --good: #13795b;
      --warn: #a16207;
      --bad: #b42318;
      --accent: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: var(--surface);
      position: sticky;
      top: 0;
      z-index: 4;
    }
    h1 { margin: 0; font-size: 17px; font-weight: 650; }
    h2 { margin: 0 0 8px; font-size: 14px; font-weight: 650; }
    button, input, select, textarea { font: inherit; }
    button {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
    }
    button.primary { border-color: var(--accent); background: var(--accent); color: white; }
    button.danger { border-color: var(--bad); color: var(--bad); }
    button.active { border-color: var(--accent); color: var(--accent); }
    main {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
      min-height: calc(100vh - 52px);
    }
    aside, section, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside { overflow: hidden; }
    .side-scroll { height: calc(100vh - 78px); overflow: auto; padding: 10px; }
    .stack { display: grid; gap: 12px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
    .report {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 86px;
    }
    .report strong { display: block; font-size: 13px; margin-bottom: 6px; }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef2f4;
      color: var(--muted);
      font-size: 12px;
    }
    .status.ready, .status.passed, .status.completed, .status.promoted { background: #e8f5ef; color: var(--good); }
    .status.needs_attention, .status.needs_review, .status.needs_data, .status.needs_labels, .status.needs_human, .status.has_drafts { background: #fff5d7; color: var(--warn); }
    .status.failed, .status.blocked, .status.timeout, .status.missing { background: #fdeceb; color: var(--bad); }
    .muted { color: var(--muted); font-size: 12px; }
    .tiny { font-size: 11px; color: var(--muted); overflow-wrap: anywhere; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
    label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
    input, select, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      color: var(--ink);
      background: white;
      min-width: 0;
    }
    textarea { min-height: 74px; resize: vertical; }
    .queue-item {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      margin-bottom: 8px;
    }
    .queue-item.active { border-color: var(--accent); background: #eefbf8; }
    .workspace { display: grid; grid-template-rows: auto minmax(360px, 1fr) auto; gap: 12px; min-width: 0; }
    .canvas-wrap {
      display: grid;
      place-items: center;
      min-height: 360px;
      background: #141b22;
      border-radius: 8px;
      overflow: hidden;
    }
    canvas { max-width: 100%; max-height: 70vh; background: #0f141a; cursor: crosshair; }
    .split { display: grid; grid-template-columns: minmax(0, 1fr) 310px; gap: 12px; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      padding: 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: #f7f9fa;
      max-height: 260px;
      overflow: auto;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    @media (max-width: 940px) {
      main, .split { grid-template-columns: 1fr; }
      .side-scroll { height: auto; max-height: 420px; }
    }
  </style>
</head>
<body>
<header>
  <h1>Active Learning Workbench</h1>
  <span id="appStatus" class="status">loading</span>
  <button id="refreshButton">Refresh</button>
  <span id="updatedAt" class="muted"></span>
</header>
<main>
  <aside>
    <div class="side-scroll">
      <h2>Queue</h2>
      <div class="toolbar">
        <select id="targetFilter"></select>
        <select id="statusFilter">
          <option value="">all status</option>
          <option value="todo">todo</option>
          <option value="draft">draft</option>
          <option value="done">done</option>
          <option value="skipped">skipped</option>
        </select>
      </div>
      <div id="queueList" style="margin-top:10px;"></div>
    </div>
  </aside>
  <div class="workspace">
    <section class="panel" style="padding:10px;">
      <div class="grid" id="reports"></div>
    </section>
    <div class="split">
      <section class="panel" style="padding:10px;">
        <div class="toolbar" style="justify-content:space-between;">
          <div>
            <h2 id="candidateTitle">No candidate selected</h2>
            <div id="candidateMeta" class="tiny"></div>
          </div>
          <span id="candidateStatus" class="status">idle</span>
        </div>
        <div class="canvas-wrap" style="margin-top:10px;">
          <canvas id="canvas" width="960" height="540"></canvas>
        </div>
      </section>
      <section class="stack">
        <div class="panel" style="padding:10px;">
          <h2>Annotation</h2>
          <div class="form-grid">
            <label>class id<input id="classId" type="number" min="0" value="0"></label>
            <label>class name<input id="className" placeholder="optional"></label>
            <label>split<select id="splitInput"><option>train</option><option>val</option></select></label>
            <label>status<select id="annotationStatus"><option>done</option><option>draft</option></select></label>
          </div>
          <label style="margin-top:8px;">text<input id="textInput" placeholder="OCR text or note"></label>
          <label style="margin-top:8px;">notes<textarea id="notesInput"></textarea></label>
          <div class="toolbar" style="margin-top:8px;">
            <button id="saveAnnotation" class="primary">Save</button>
            <button id="clearBoxes">Clear</button>
            <button id="skipCandidate">Skip</button>
          </div>
          <div id="annotationHint" class="tiny" style="margin-top:8px;"></div>
        </div>
        <div class="panel" style="padding:10px;">
          <h2>Actions</h2>
          <div class="toolbar">
            <button data-action="refresh_training_candidates">Refresh Candidates</button>
            <button data-action="validate_training_datasets">Validate Datasets</button>
            <button data-action="refresh_model_data_readiness">Refresh Readiness</button>
            <button id="llmPackButton">Build LLM Pack</button>
          </div>
          <div class="form-grid" style="margin-top:8px;">
            <label>video<input id="videoInput" placeholder="footages/n_match_6.mp4"></label>
            <label>match id<input id="matchIdInput" placeholder="n_match_6"></label>
          </div>
          <div class="toolbar" style="margin-top:8px;">
            <button id="intakeButton">Intake Video</button>
            <button id="validationButton">Run Validation</button>
            <label style="display:flex;align-items:center;gap:6px;"><input id="runAnalysis" type="checkbox"> run analysis</label>
          </div>
          <div class="form-grid" style="margin-top:8px;">
            <label>training target<select id="trainingTarget">
              <option>ui_detector_yolo</option>
              <option>count_ocr_yolo</option>
              <option>message_ocr_yolo</option>
            </select></label>
            <label>candidate model<input id="candidateModel" placeholder="outputs/model_training/.../weights/best.pt"></label>
            <label>model id<input id="modelId" placeholder="ui_detector_yolo"></label>
          </div>
          <div class="toolbar" style="margin-top:8px;">
            <button id="trainingDryRun">Training Dry Run</button>
            <button id="trainingExecute" class="danger">Execute Training</button>
            <button id="promotionPlan">Promotion Plan</button>
            <button id="promotionApply" class="danger">Apply Promotion</button>
          </div>
        </div>
        <div class="panel" style="padding:10px;">
          <h2>Apply Staging</h2>
          <div class="toolbar">
            <button id="applyDryRun">Dry Run</button>
            <button id="applyReal" class="danger">Apply</button>
          </div>
          <pre id="actionOutput"></pre>
        </div>
      </section>
    </div>
    <section class="panel" style="padding:10px;">
      <h2>Asset Inbox</h2>
      <div id="assetInbox" class="grid"></div>
    </section>
  </div>
</main>
<script>
const state = { app: null, candidates: [], selected: null, image: null, boxes: [], point: null, drag: null };
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function cls(status) { return "status " + String(status || "missing"); }
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch { payload = { text }; }
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
function postJson(path, body) {
  return api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}
function imageUrl(path) { return "/api/image?path=" + encodeURIComponent(path || ""); }

async function loadAll() {
  state.app = await api("/api/state");
  state.candidates = await api("/api/candidates");
  render();
}
function render() {
  document.getElementById("appStatus").className = cls(state.app.status);
  document.getElementById("appStatus").textContent = state.app.status;
  document.getElementById("updatedAt").textContent = state.app.updated_at || "";
  renderReports();
  renderInbox();
  renderQueueFilters();
  renderQueue();
  drawCanvas();
}
function renderReports() {
  const root = document.getElementById("reports");
  root.innerHTML = state.app.reports.map(report => `
    <div class="report">
      <strong>${escapeHtml(report.title)}</strong>
      <span class="${cls(report.status)}">${escapeHtml(report.status)}</span>
      <div class="tiny" style="margin-top:6px;">${escapeHtml(report.path || "missing")}</div>
    </div>`).join("");
}
function renderInbox() {
  const inbox = state.app.asset_inbox || { videos: [] };
  document.getElementById("assetInbox").innerHTML = inbox.videos.map(item => `
    <div class="report">
      <strong>${escapeHtml(item.suggested_match_id)}</strong>
      <span class="${cls(item.status)}">${escapeHtml(item.status)}</span>
      <div class="tiny" style="margin-top:6px;">${escapeHtml(item.path)}</div>
      ${item.status === "new" ? `<button style="margin-top:8px;" onclick="prefillIntake('${escapeHtml(item.path)}','${escapeHtml(item.suggested_match_id)}')">Use</button>` : ""}
    </div>`).join("");
}
function prefillIntake(path, matchId) {
  document.getElementById("videoInput").value = path;
  document.getElementById("matchIdInput").value = matchId;
}
function renderQueueFilters() {
  const select = document.getElementById("targetFilter");
  const targets = [...new Set(state.candidates.map(item => item.target))].sort();
  const current = select.value;
  select.innerHTML = `<option value="">all targets</option>` + targets.map(target => `<option>${escapeHtml(target)}</option>`).join("");
  select.value = current;
}
function renderQueue() {
  const target = document.getElementById("targetFilter").value;
  const status = document.getElementById("statusFilter").value;
  const items = state.candidates.filter(item => (!target || item.target === target) && (!status || item.status === status));
  document.getElementById("queueList").innerHTML = items.slice(0, 180).map(item => `
    <button class="queue-item ${state.selected && state.selected.id === item.id ? "active" : ""}" onclick="selectCandidate('${escapeHtml(item.id)}')">
      <strong>${escapeHtml(item.target)}</strong>
      <span class="${cls(item.status)}" style="float:right;">${escapeHtml(item.status)}</span>
      <div class="tiny">${escapeHtml(item.reason)} ${escapeHtml(item.match_id)} ${escapeHtml(item.elapsed_time)}</div>
    </button>`).join("");
}
function selectCandidate(id) {
  state.selected = state.candidates.find(item => item.id === id);
  state.boxes = [];
  state.point = null;
  const staged = state.selected.staging || {};
  const annotation = staged.annotation || {};
  if (Array.isArray(annotation.boxes)) state.boxes = annotation.boxes.map(item => ({ ...item }));
  if (annotation.point) state.point = { ...annotation.point };
  document.getElementById("textInput").value = annotation.text || "";
  document.getElementById("notesInput").value = annotation.notes || "";
  document.getElementById("splitInput").value = staged.split || "train";
  document.getElementById("annotationStatus").value = staged.status === "draft" ? "draft" : "done";
  const frame = state.selected.frame_path || state.selected.preview_path;
  if (frame) {
    const image = new Image();
    image.onload = () => { state.image = image; fitCanvas(image); drawCanvas(); };
    image.onerror = () => { state.image = null; drawCanvas(); };
    image.src = imageUrl(frame);
  } else {
    state.image = null;
  }
  renderCandidateHeader();
  renderQueue();
}
function renderCandidateHeader() {
  const item = state.selected;
  document.getElementById("candidateTitle").textContent = item ? item.id : "No candidate selected";
  document.getElementById("candidateMeta").textContent = item ? `${item.target} | ${item.reason} | ${item.frame_path || item.preview_path || ""}` : "";
  document.getElementById("candidateStatus").className = cls(item ? item.status : "idle");
  document.getElementById("candidateStatus").textContent = item ? item.status : "idle";
  const review = item && item.llm_review ? item.llm_review : {};
  document.getElementById("annotationHint").textContent = review.suggestion ? `LLM: ${review.suggestion} (${review.confidence ?? ""})` : "";
}
function fitCanvas(image) {
  const maxWidth = 980;
  const scale = Math.min(1, maxWidth / image.naturalWidth);
  canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
}
function drawCanvas(extraBox = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.image) {
    ctx.fillStyle = "#101820";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#d7dde2";
    ctx.fillText("Select a candidate with an image", 24, 32);
    return;
  }
  ctx.drawImage(state.image, 0, 0, canvas.width, canvas.height);
  const boxes = extraBox ? [...state.boxes, extraBox] : state.boxes;
  boxes.forEach((box, index) => {
    const x = (box.x_center - box.width / 2) * canvas.width;
    const y = (box.y_center - box.height / 2) * canvas.height;
    const w = box.width * canvas.width;
    const h = box.height * canvas.height;
    ctx.strokeStyle = "#00d6a3";
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(0, 214, 163, 0.88)";
    ctx.fillRect(x, Math.max(0, y - 18), 56, 18);
    ctx.fillStyle = "#06251d";
    ctx.fillText(String(box.class_name || box.class_id || index), x + 4, Math.max(12, y - 5));
  });
  if (state.point) {
    const x = Number(state.point.x) * canvas.width / state.image.naturalWidth;
    const y = Number(state.point.y) * canvas.height / state.image.naturalHeight;
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.stroke();
  }
}
function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}
canvas.addEventListener("mousedown", event => {
  if (!state.selected || !state.image) return;
  state.drag = canvasPoint(event);
});
canvas.addEventListener("mousemove", event => {
  if (!state.drag || !state.image || !state.selected) return;
  if (state.selected.annotation_type === "heatmap_point") return;
  const end = canvasPoint(event);
  const x1 = Math.min(state.drag.x, end.x);
  const y1 = Math.min(state.drag.y, end.y);
  const x2 = Math.max(state.drag.x, end.x);
  const y2 = Math.max(state.drag.y, end.y);
  drawCanvas({
    class_id: Number(document.getElementById("classId").value || 0),
    class_name: document.getElementById("className").value,
    x_center: ((x1 + x2) / 2) / canvas.width,
    y_center: ((y1 + y2) / 2) / canvas.height,
    width: (x2 - x1) / canvas.width,
    height: (y2 - y1) / canvas.height
  });
});
canvas.addEventListener("mouseup", event => {
  if (!state.drag || !state.image || !state.selected) return;
  const end = canvasPoint(event);
  if (state.selected.annotation_type === "heatmap_point") {
    state.point = {
      x: (end.x * state.image.naturalWidth / canvas.width).toFixed(1),
      y: (end.y * state.image.naturalHeight / canvas.height).toFixed(1),
      visibility: "visible"
    };
  } else {
    const x1 = Math.min(state.drag.x, end.x);
    const y1 = Math.min(state.drag.y, end.y);
    const x2 = Math.max(state.drag.x, end.x);
    const y2 = Math.max(state.drag.y, end.y);
    if (Math.abs(x2 - x1) > 4 && Math.abs(y2 - y1) > 4) {
      state.boxes.push({
        class_id: Number(document.getElementById("classId").value || 0),
        class_name: document.getElementById("className").value,
        x_center: ((x1 + x2) / 2) / canvas.width,
        y_center: ((y1 + y2) / 2) / canvas.height,
        width: (x2 - x1) / canvas.width,
        height: (y2 - y1) / canvas.height
      });
    }
  }
  state.drag = null;
  drawCanvas();
});
document.getElementById("clearBoxes").onclick = () => { state.boxes = []; state.point = null; drawCanvas(); };
document.getElementById("saveAnnotation").onclick = async () => {
  if (!state.selected) return;
  const annotation = {
    boxes: state.boxes,
    point: state.point,
    text: document.getElementById("textInput").value,
    notes: document.getElementById("notesInput").value
  };
  await postJson("/api/annotation", {
    id: state.selected.id,
    target: state.selected.target,
    annotation_type: state.selected.annotation_type,
    status: document.getElementById("annotationStatus").value,
    split: document.getElementById("splitInput").value,
    candidate: state.selected,
    annotation
  });
  await loadAll();
};
document.getElementById("skipCandidate").onclick = async () => {
  if (!state.selected) return;
  await postJson("/api/annotation", {
    id: state.selected.id,
    target: state.selected.target,
    annotation_type: state.selected.annotation_type,
    status: "skipped",
    candidate: state.selected,
    annotation: { notes: document.getElementById("notesInput").value }
  });
  await loadAll();
};
async function runAction(action_id, payload = {}) {
  document.getElementById("actionOutput").textContent = "running " + action_id;
  try {
    const result = await postJson("/api/action", { action_id, payload });
    document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
    await loadAll();
  } catch (error) {
    document.getElementById("actionOutput").textContent = String(error);
  }
}
document.querySelectorAll("[data-action]").forEach(button => {
  button.onclick = () => runAction(button.dataset.action);
});
document.getElementById("validationButton").onclick = () => runAction("run_validation_suite", { run_analysis: document.getElementById("runAnalysis").checked });
document.getElementById("intakeButton").onclick = () => runAction("intake_video", {
  video: document.getElementById("videoInput").value,
  match_id: document.getElementById("matchIdInput").value,
  scan_analysis_windows: true
});
document.getElementById("trainingDryRun").onclick = () => runAction("training_dry_run", { target: document.getElementById("trainingTarget").value });
document.getElementById("trainingExecute").onclick = () => {
  if (confirm("Start training for this target?")) runAction("training_execute", { target: document.getElementById("trainingTarget").value, confirm: "execute_training" });
};
document.getElementById("promotionPlan").onclick = () => runAction("promotion_plan", {
  model_id: document.getElementById("modelId").value,
  candidate: document.getElementById("candidateModel").value
});
document.getElementById("promotionApply").onclick = () => {
  if (confirm("Apply candidate model to the registered path?")) runAction("promotion_apply", {
    model_id: document.getElementById("modelId").value,
    candidate: document.getElementById("candidateModel").value,
    confirm: "apply_promotion"
  });
};
document.getElementById("llmPackButton").onclick = async () => {
  const result = await postJson("/api/llm-review-pack", { limit: 30 });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
};
document.getElementById("applyDryRun").onclick = async () => {
  const result = await postJson("/api/apply-staging", { dry_run: true });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
};
document.getElementById("applyReal").onclick = async () => {
  if (!confirm("Apply done staging annotations into training datasets?")) return;
  const result = await postJson("/api/apply-staging", { dry_run: false });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
  await loadAll();
};
document.getElementById("refreshButton").onclick = loadAll;
document.getElementById("targetFilter").onchange = renderQueue;
document.getElementById("statusFilter").onchange = renderQueue;
loadAll().catch(error => { document.getElementById("actionOutput").textContent = String(error); });
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local active-learning workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "SplatoonWorkbench/1.0"

    def send_bytes(self, body: bytes, status: HTTPStatus = HTTPStatus.OK, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"), status, "application/json; charset=utf-8")

    def send_error_json(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": message}, status)

    def read_json_body(self) -> dict:
        length = int(self.headers.get("content-length", "0") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_bytes(APP_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self.send_json(build_workbench_state())
            elif parsed.path == "/api/candidates":
                self.send_json(load_candidate_queue())
            elif parsed.path == "/api/staging":
                self.send_json(load_staging(DEFAULT_STAGING_PATH))
            elif parsed.path == "/api/llm-reviews":
                self.send_json(load_llm_reviews(DEFAULT_LLM_REVIEWS_PATH))
            elif parsed.path == "/api/image":
                query = parse_qs(parsed.query)
                image_path = safe_project_file(query.get("path", [""])[0])
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.send_error_json("unsupported image type", HTTPStatus.BAD_REQUEST)
                    return
                if not image_path.exists():
                    self.send_error_json("image not found", HTTPStatus.NOT_FOUND)
                    return
                self.send_bytes(image_path.read_bytes(), content_type=media_type_for_path(image_path))
            else:
                self.send_error_json("not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - local tool should return useful API errors.
            self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json_body()
            if parsed.path == "/api/annotation":
                self.send_json(upsert_staging_annotation(payload))
            elif parsed.path == "/api/action":
                self.send_json(run_workbench_action(str(payload.get("action_id", "")), payload.get("payload", {})))
            elif parsed.path == "/api/apply-staging":
                self.send_json(apply_staging_annotations(dry_run=bool(payload.get("dry_run", True))))
            elif parsed.path == "/api/llm-review-pack":
                self.send_json(build_llm_review_pack(limit=int(payload.get("limit", 30))))
            elif parsed.path == "/api/llm-review":
                self.send_json(record_llm_review(str(payload.get("id", "")), payload.get("review", {})))
            else:
                self.send_error_json("not found", HTTPStatus.NOT_FOUND)
        except Exception as exc:  # noqa: BLE001 - local tool should return useful API errors.
            self.send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> int:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), WorkbenchHandler)
    print(f"active learning workbench: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
