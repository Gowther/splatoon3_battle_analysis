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
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>主动学习工作台</title>
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
  <h1 data-i18n="app.title">主动学习工作台</h1>
  <span id="appStatus" class="status">加载中</span>
  <button id="refreshButton" data-i18n="button.refresh">刷新</button>
  <select id="languageSelect" aria-label="Language">
    <option value="zh-CN">中文</option>
    <option value="en">English</option>
  </select>
  <span id="updatedAt" class="muted"></span>
</header>
<main>
  <aside>
    <div class="side-scroll">
      <h2 data-i18n="section.queue">标注队列</h2>
      <div class="toolbar">
        <select id="targetFilter"></select>
        <select id="statusFilter">
          <option value="" data-i18n="filter.all_status">全部状态</option>
          <option value="todo" data-i18n="status.todo">待处理</option>
          <option value="draft" data-i18n="status.draft">草稿</option>
          <option value="done" data-i18n="status.done">已完成</option>
          <option value="skipped" data-i18n="status.skipped">已跳过</option>
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
            <h2 id="candidateTitle" data-i18n="candidate.empty">未选择候选样本</h2>
            <div id="candidateMeta" class="tiny"></div>
          </div>
          <span id="candidateStatus" class="status">空闲</span>
        </div>
        <div class="canvas-wrap" style="margin-top:10px;">
          <canvas id="canvas" width="960" height="540"></canvas>
        </div>
      </section>
      <section class="stack">
        <div class="panel" style="padding:10px;">
          <h2 data-i18n="section.annotation">标注</h2>
          <div class="form-grid">
            <label><span data-i18n="label.class_id">类别 ID</span><input id="classId" type="number" min="0" value="0"></label>
            <label><span data-i18n="label.class_name">类别名称</span><input id="className" data-i18n-placeholder="placeholder.optional" placeholder="可选"></label>
            <label><span data-i18n="label.split">数据集划分</span><select id="splitInput"><option value="train" data-i18n="split.train">训练</option><option value="val" data-i18n="split.val">验证</option></select></label>
            <label><span data-i18n="label.status">状态</span><select id="annotationStatus"><option value="done" data-i18n="status.done">已完成</option><option value="draft" data-i18n="status.draft">草稿</option></select></label>
          </div>
          <label style="margin-top:8px;"><span data-i18n="label.text">文本</span><input id="textInput" data-i18n-placeholder="placeholder.ocr_text" placeholder="OCR 文本或备注"></label>
          <label style="margin-top:8px;"><span data-i18n="label.notes">备注</span><textarea id="notesInput"></textarea></label>
          <div class="toolbar" style="margin-top:8px;">
            <button id="saveAnnotation" class="primary" data-i18n="button.save">保存</button>
            <button id="clearBoxes" data-i18n="button.clear">清空</button>
            <button id="skipCandidate" data-i18n="button.skip">跳过</button>
          </div>
          <div id="annotationHint" class="tiny" style="margin-top:8px;"></div>
        </div>
        <div class="panel" style="padding:10px;">
          <h2 data-i18n="section.actions">操作</h2>
          <div class="toolbar">
            <button data-action="refresh_training_candidates" data-i18n="action.refresh_training_candidates">刷新候选样本</button>
            <button data-action="validate_training_datasets" data-i18n="action.validate_training_datasets">验证训练集</button>
            <button data-action="refresh_model_data_readiness" data-i18n="action.refresh_model_data_readiness">刷新就绪状态</button>
            <button id="llmPackButton" data-i18n="button.llm_pack">生成 LLM 审阅包</button>
          </div>
          <div class="form-grid" style="margin-top:8px;">
            <label><span data-i18n="label.video">视频</span><input id="videoInput" placeholder="footages/n_match_6.mp4"></label>
            <label><span data-i18n="label.match_id">对战 ID</span><input id="matchIdInput" placeholder="n_match_6"></label>
          </div>
          <div class="toolbar" style="margin-top:8px;">
            <button id="intakeButton" data-i18n="action.intake_video">接入视频</button>
            <button id="validationButton" data-i18n="action.run_validation_suite">运行验证</button>
            <label style="display:flex;align-items:center;gap:6px;"><input id="runAnalysis" type="checkbox"> <span data-i18n="label.run_analysis">同时跑分析</span></label>
          </div>
          <div class="form-grid" style="margin-top:8px;">
            <label><span data-i18n="label.training_target">训练目标</span><select id="trainingTarget">
              <option>ui_detector_yolo</option>
              <option>count_ocr_yolo</option>
              <option>message_ocr_yolo</option>
            </select></label>
            <label><span data-i18n="label.candidate_model">候选模型</span><input id="candidateModel" placeholder="outputs/model_training/.../weights/best.pt"></label>
            <label><span data-i18n="label.model_id">模型 ID</span><input id="modelId" placeholder="ui_detector_yolo"></label>
          </div>
          <div class="toolbar" style="margin-top:8px;">
            <button id="trainingDryRun" data-i18n="action.training_dry_run">训练预演</button>
            <button id="trainingExecute" class="danger" data-i18n="action.training_execute">执行训练</button>
            <button id="promotionPlan" data-i18n="action.promotion_plan">提升计划</button>
            <button id="promotionApply" class="danger" data-i18n="action.promotion_apply">应用提升</button>
          </div>
        </div>
        <div class="panel" style="padding:10px;">
          <h2 data-i18n="section.apply_staging">应用暂存</h2>
          <div class="toolbar">
            <button id="applyDryRun" data-i18n="button.dry_run">预演</button>
            <button id="applyReal" class="danger" data-i18n="button.apply">应用</button>
          </div>
          <pre id="actionOutput"></pre>
        </div>
      </section>
    </div>
    <section class="panel" style="padding:10px;">
      <h2 data-i18n="section.asset_inbox">素材收件箱</h2>
      <div id="assetInbox" class="grid"></div>
    </section>
  </div>
</main>
<script>
const state = { app: null, candidates: [], selected: null, image: null, boxes: [], point: null, drag: null };
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const I18N = {
  "zh-CN": {
    "app.title": "主动学习工作台",
    "button.refresh": "刷新",
    "section.queue": "标注队列",
    "section.annotation": "标注",
    "section.actions": "操作",
    "section.apply_staging": "应用暂存",
    "section.asset_inbox": "素材收件箱",
    "filter.all_status": "全部状态",
    "filter.all_targets": "全部目标",
    "candidate.empty": "未选择候选样本",
    "canvas.empty": "选择一个带图片的候选样本",
    "label.class_id": "类别 ID",
    "label.class_name": "类别名称",
    "label.split": "数据集划分",
    "label.status": "状态",
    "label.text": "文本",
    "label.notes": "备注",
    "label.video": "视频",
    "label.match_id": "对战 ID",
    "label.run_analysis": "同时跑分析",
    "label.training_target": "训练目标",
    "label.candidate_model": "候选模型",
    "label.model_id": "模型 ID",
    "placeholder.optional": "可选",
    "placeholder.ocr_text": "OCR 文本或备注",
    "split.train": "训练",
    "split.val": "验证",
    "button.save": "保存",
    "button.clear": "清空",
    "button.skip": "跳过",
    "button.llm_pack": "生成 LLM 审阅包",
    "button.dry_run": "预演",
    "button.apply": "应用",
    "button.use": "使用",
    "action.refresh_training_candidates": "刷新候选样本",
    "action.validate_training_datasets": "验证训练集",
    "action.refresh_model_data_readiness": "刷新就绪状态",
    "action.intake_video": "接入视频",
    "action.run_validation_suite": "运行验证",
    "action.training_dry_run": "训练预演",
    "action.training_execute": "执行训练",
    "action.promotion_plan": "提升计划",
    "action.promotion_apply": "应用提升",
    "message.running": "正在运行",
    "message.llm_prefix": "LLM 建议",
    "confirm.training": "要开始训练这个目标吗？",
    "confirm.promotion": "要把候选模型应用到登记的正式模型路径吗？",
    "confirm.apply_staging": "要把已完成的暂存标注写入训练集吗？",
    "status.loading": "加载中",
    "status.idle": "空闲",
    "status.missing": "缺失",
    "status.ready": "就绪",
    "status.passed": "通过",
    "status.completed": "完成",
    "status.promoted": "已提升",
    "status.needs_attention": "需要关注",
    "status.needs_review": "需要复核",
    "status.needs_data": "缺数据",
    "status.needs_labels": "需标注",
    "status.needs_human": "需人工",
    "status.has_drafts": "有草稿",
    "status.failed": "失败",
    "status.blocked": "阻塞",
    "status.timeout": "超时",
    "status.todo": "待处理",
    "status.draft": "草稿",
    "status.done": "已完成",
    "status.skipped": "已跳过",
    "status.new": "新素材",
    "status.registered": "已登记",
    "target.ui_detector_yolo": "UI 检测 YOLO",
    "target.count_ocr_yolo": "数字 OCR YOLO",
    "target.message_ocr_yolo": "消息 OCR YOLO",
    "target.weapon_classifier_resnet18": "武器分类器",
    "target.heatmap_tracker_labels": "热力图轨迹标注"
  },
  en: {
    "app.title": "Active Learning Workbench",
    "button.refresh": "Refresh",
    "section.queue": "Queue",
    "section.annotation": "Annotation",
    "section.actions": "Actions",
    "section.apply_staging": "Apply Staging",
    "section.asset_inbox": "Asset Inbox",
    "filter.all_status": "all status",
    "filter.all_targets": "all targets",
    "candidate.empty": "No candidate selected",
    "canvas.empty": "Select a candidate with an image",
    "label.class_id": "class id",
    "label.class_name": "class name",
    "label.split": "split",
    "label.status": "status",
    "label.text": "text",
    "label.notes": "notes",
    "label.video": "video",
    "label.match_id": "match id",
    "label.run_analysis": "run analysis",
    "label.training_target": "training target",
    "label.candidate_model": "candidate model",
    "label.model_id": "model id",
    "placeholder.optional": "optional",
    "placeholder.ocr_text": "OCR text or note",
    "split.train": "train",
    "split.val": "val",
    "button.save": "Save",
    "button.clear": "Clear",
    "button.skip": "Skip",
    "button.llm_pack": "Build LLM Pack",
    "button.dry_run": "Dry Run",
    "button.apply": "Apply",
    "button.use": "Use",
    "action.refresh_training_candidates": "Refresh Candidates",
    "action.validate_training_datasets": "Validate Datasets",
    "action.refresh_model_data_readiness": "Refresh Readiness",
    "action.intake_video": "Intake Video",
    "action.run_validation_suite": "Run Validation",
    "action.training_dry_run": "Training Dry Run",
    "action.training_execute": "Execute Training",
    "action.promotion_plan": "Promotion Plan",
    "action.promotion_apply": "Apply Promotion",
    "message.running": "running",
    "message.llm_prefix": "LLM",
    "confirm.training": "Start training for this target?",
    "confirm.promotion": "Apply candidate model to the registered path?",
    "confirm.apply_staging": "Apply done staging annotations into training datasets?",
    "status.loading": "loading",
    "status.idle": "idle",
    "status.missing": "missing",
    "status.ready": "ready",
    "status.passed": "passed",
    "status.completed": "completed",
    "status.promoted": "promoted",
    "status.needs_attention": "needs attention",
    "status.needs_review": "needs review",
    "status.needs_data": "needs data",
    "status.needs_labels": "needs labels",
    "status.needs_human": "needs human",
    "status.has_drafts": "has drafts",
    "status.failed": "failed",
    "status.blocked": "blocked",
    "status.timeout": "timeout",
    "status.todo": "todo",
    "status.draft": "draft",
    "status.done": "done",
    "status.skipped": "skipped",
    "status.new": "new",
    "status.registered": "registered",
    "target.ui_detector_yolo": "UI detector YOLO",
    "target.count_ocr_yolo": "count OCR YOLO",
    "target.message_ocr_yolo": "message OCR YOLO",
    "target.weapon_classifier_resnet18": "weapon classifier",
    "target.heatmap_tracker_labels": "heatmap tracker labels"
  }
};
let language = localStorage.getItem("workbenchLanguage") || "zh-CN";
if (!I18N[language]) language = "zh-CN";

function cls(status) { return "status " + String(status || "missing"); }
function t(key) {
  return (I18N[language] && I18N[language][key]) || I18N.en[key] || key;
}
function statusLabel(status) {
  return t("status." + String(status || "missing"));
}
function targetLabel(target) {
  return t("target." + target) || target;
}
function reportTitle(report) {
  return language === "zh-CN" ? (report.title_zh || report.title) : report.title;
}
function translateStatic() {
  document.documentElement.lang = language;
  document.title = t("app.title");
  document.getElementById("languageSelect").value = language;
  document.querySelectorAll("[data-i18n]").forEach(node => { node.textContent = t(node.dataset.i18n); });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(node => { node.placeholder = t(node.dataset.i18nPlaceholder); });
}
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
  translateStatic();
  document.getElementById("appStatus").className = cls(state.app.status);
  document.getElementById("appStatus").textContent = statusLabel(state.app.status);
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
      <strong>${escapeHtml(reportTitle(report))}</strong>
      <span class="${cls(report.status)}">${escapeHtml(statusLabel(report.status))}</span>
      <div class="tiny" style="margin-top:6px;">${escapeHtml(report.path || t("status.missing"))}</div>
    </div>`).join("");
}
function renderInbox() {
  const inbox = state.app.asset_inbox || { videos: [] };
  document.getElementById("assetInbox").innerHTML = inbox.videos.map(item => `
    <div class="report">
      <strong>${escapeHtml(item.suggested_match_id)}</strong>
      <span class="${cls(item.status)}">${escapeHtml(statusLabel(item.status))}</span>
      <div class="tiny" style="margin-top:6px;">${escapeHtml(item.path)}</div>
      ${item.status === "new" ? `<button style="margin-top:8px;" onclick="prefillIntake('${escapeHtml(item.path)}','${escapeHtml(item.suggested_match_id)}')">${escapeHtml(t("button.use"))}</button>` : ""}
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
  select.innerHTML = `<option value="">${escapeHtml(t("filter.all_targets"))}</option>` + targets.map(target => `<option value="${escapeHtml(target)}">${escapeHtml(targetLabel(target))}</option>`).join("");
  select.value = current;
}
function renderQueue() {
  const target = document.getElementById("targetFilter").value;
  const status = document.getElementById("statusFilter").value;
  const items = state.candidates.filter(item => (!target || item.target === target) && (!status || item.status === status));
  document.getElementById("queueList").innerHTML = items.slice(0, 180).map(item => `
    <button class="queue-item ${state.selected && state.selected.id === item.id ? "active" : ""}" onclick="selectCandidate('${escapeHtml(item.id)}')">
      <strong>${escapeHtml(targetLabel(item.target))}</strong>
      <span class="${cls(item.status)}" style="float:right;">${escapeHtml(statusLabel(item.status))}</span>
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
  document.getElementById("candidateTitle").textContent = item ? item.id : t("candidate.empty");
  document.getElementById("candidateMeta").textContent = item ? `${item.target} | ${item.reason} | ${item.frame_path || item.preview_path || ""}` : "";
  document.getElementById("candidateStatus").className = cls(item ? item.status : "idle");
  document.getElementById("candidateStatus").textContent = statusLabel(item ? item.status : "idle");
  const review = item && item.llm_review ? item.llm_review : {};
  document.getElementById("annotationHint").textContent = review.suggestion ? `${t("message.llm_prefix")}: ${review.suggestion} (${review.confidence ?? ""})` : "";
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
    ctx.fillText(t("canvas.empty"), 24, 32);
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
  document.getElementById("actionOutput").textContent = `${t("message.running")} ${action_id}`;
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
  if (confirm(t("confirm.training"))) runAction("training_execute", { target: document.getElementById("trainingTarget").value, confirm: "execute_training" });
};
document.getElementById("promotionPlan").onclick = () => runAction("promotion_plan", {
  model_id: document.getElementById("modelId").value,
  candidate: document.getElementById("candidateModel").value
});
document.getElementById("promotionApply").onclick = () => {
  if (confirm(t("confirm.promotion"))) runAction("promotion_apply", {
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
  if (!confirm(t("confirm.apply_staging"))) return;
  const result = await postJson("/api/apply-staging", { dry_run: false });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
  await loadAll();
};
document.getElementById("languageSelect").onchange = event => {
  language = event.target.value;
  localStorage.setItem("workbenchLanguage", language);
  render();
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
