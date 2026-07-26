#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
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
    auto_record_llm_reviews,
    apply_staging_annotations,
    build_automation_plan,
    build_llm_review_pack,
    build_workbench_state,
    finish_job_record,
    load_candidate_queue,
    load_jobs,
    load_llm_reviews,
    load_staging,
    media_type_for_path,
    prefill_heatmap_staging,
    record_llm_review,
    run_automation_pipeline,
    run_workbench_action,
    safe_project_file,
    start_job_record,
    upsert_staging_annotation,
)
from src.data_review_workbench import (
    build_data_review_state,
    build_time_snapshot,
    is_video_path,
    record_data_review,
)
from src.evidence_review_workbench import (
    build_evidence_review_state,
    build_video_evidence,
    record_evidence_review,
    record_weapon_correction,
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
    a.nav-link {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      text-decoration: none;
    }
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
  <a class="nav-link" href="/data-review">数据核验</a>
  <a class="nav-link" href="/evidence-review">证据核验</a>
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
            <button id="automationDryRun" data-i18n="button.automation_dry_run">自动预演</button>
            <button id="automationRun" class="primary" data-i18n="button.automation_run">自动推进</button>
            <button id="autoReviewButton" data-i18n="button.auto_review">规则审阅</button>
            <button id="heatmapPrefillButton" data-i18n="button.heatmap_prefill">预填热力图</button>
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
    "button.automation_dry_run": "自动预演",
    "button.automation_run": "自动推进",
    "button.auto_review": "规则审阅",
    "button.heatmap_prefill": "预填热力图",
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
    "message.job_started": "后台任务已启动",
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
    "target.heatmap_tracker_labels": "热力图轨迹标注",
    "target.death_event_ocr": "死亡事件 OCR"
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
    "button.automation_dry_run": "Automation Dry Run",
    "button.automation_run": "Run Automation",
    "button.auto_review": "Rule Review",
    "button.heatmap_prefill": "Prefill Heatmap",
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
    "message.job_started": "background job started",
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
    "target.heatmap_tracker_labels": "heatmap tracker labels",
    "target.death_event_ocr": "death event OCR"
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
  const preannotation = state.selected.preannotation || {};
  const annotation = staged.annotation || preannotation.annotation || {};
  if (Array.isArray(annotation.boxes)) state.boxes = annotation.boxes.map(item => ({ ...item }));
  if (annotation.point) state.point = { ...annotation.point };
  document.getElementById("textInput").value = annotation.text || "";
  document.getElementById("notesInput").value = annotation.notes || "";
  document.getElementById("splitInput").value = staged.split || "train";
  document.getElementById("annotationStatus").value = staged.status === "done" ? "done" : "draft";
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
  document.getElementById("candidateMeta").textContent = item ? `${targetLabel(item.target)} | ${item.reason} | priority ${item.priority_score ?? ""} | duplicates ${item.duplicate_count ?? 1} | ${item.frame_path || item.preview_path || ""}` : "";
  document.getElementById("candidateStatus").className = cls(item ? item.status : "idle");
  document.getElementById("candidateStatus").textContent = statusLabel(item ? item.status : "idle");
  const review = item && item.llm_review ? item.llm_review : {};
  const pre = item && item.preannotation ? item.preannotation : {};
  const hints = [];
  if (pre.status === "ready") hints.push(`preannotation: ${pre.source} (${pre.confidence ?? ""})`);
  if (review.suggestion) hints.push(`${t("message.llm_prefix")}: ${review.suggestion} (${review.confidence ?? ""})`);
  document.getElementById("annotationHint").textContent = hints.join(" | ");
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
  const saved = await postJson("/api/annotation", {
    id: state.selected.id,
    target: state.selected.target,
    annotation_type: state.selected.annotation_type,
    status: document.getElementById("annotationStatus").value,
    split: document.getElementById("splitInput").value,
    candidate: state.selected,
    annotation
  });
  const output = { saved };
  if (saved.status === "done") {
    output.dry_run = await postJson("/api/apply-staging", { dry_run: true });
  }
  document.getElementById("actionOutput").textContent = JSON.stringify(output, null, 2);
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
    const definition = (state.app.actions || []).find(item => item.id === action_id) || {};
    if (definition.long_running) {
      const job = await postJson("/api/job", { action_id, payload });
      document.getElementById("actionOutput").textContent = `${t("message.job_started")}\\n` + JSON.stringify(job, null, 2);
      await loadAll();
      return;
    }
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
document.getElementById("automationDryRun").onclick = async () => {
  const result = await postJson("/api/automation-run", { dry_run: true, include_long: false, max_steps: 8 });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
  await loadAll();
};
document.getElementById("automationRun").onclick = async () => {
  const result = await postJson("/api/automation-run", { dry_run: false, include_long: false, max_steps: 8 });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
  await loadAll();
};
document.getElementById("autoReviewButton").onclick = async () => {
  const result = await postJson("/api/llm-review-auto", { limit: 50 });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
  await loadAll();
};
document.getElementById("heatmapPrefillButton").onclick = async () => {
  const result = await postJson("/api/heatmap-prefill", { limit: 30, status: "draft" });
  document.getElementById("actionOutput").textContent = JSON.stringify(result, null, 2);
  await loadAll();
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


DATA_REVIEW_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>数据核验工作台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --surface: #ffffff;
      --ink: #172026;
      --muted: #687782;
      --line: #d8e0e5;
      --accent: #0f766e;
      --accent-soft: #e8f6f3;
      --good: #13795b;
      --warn: #a16207;
      --bad: #b42318;
      --info: #315a9f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 54px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 16px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0; font-size: 17px; font-weight: 700; }
    h2 { margin: 0 0 8px; font-size: 14px; font-weight: 700; }
    h3 { margin: 0; font-size: 13px; font-weight: 700; }
    button, input, select, textarea { font: inherit; }
    button, a.nav-link {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      text-decoration: none;
      cursor: pointer;
    }
    button.primary, button.active {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    button.good { border-color: var(--good); color: var(--good); }
    button.warn { border-color: var(--warn); color: var(--warn); }
    button.bad { border-color: var(--bad); color: var(--bad); }
    button.good.active { background: var(--good); color: #fff; }
    button.warn.active { background: var(--warn); color: #fff; }
    button.bad.active { background: var(--bad); color: #fff; }
    main {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
      min-height: calc(100vh - 54px);
    }
    aside, section, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    aside { padding: 12px; align-self: start; position: sticky; top: 66px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fff;
      color: var(--ink);
      min-width: 0;
    }
    select[multiple] { min-height: 210px; }
    textarea { min-height: 76px; resize: vertical; }
    video {
      width: 100%;
      max-height: 58vh;
      background: #111820;
      display: block;
      border-radius: 8px;
    }
    .stack { display: grid; gap: 12px; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(360px, 1.35fr) minmax(320px, 0.95fr);
      align-items: start;
      gap: 12px;
      min-width: 0;
    }
    .workspace > * { min-width: 0; }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2f4;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .status.ready, .status.saved, .status.accurate { background: #e8f5ef; color: var(--good); }
    .status.needs_data, .status.needs_review { background: #fff5d7; color: var(--warn); }
    .status.incorrect { background: #fdeceb; color: var(--bad); }
    .muted { color: var(--muted); font-size: 12px; }
    .tiny { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .video-panel, .review-panel, .source-panel { padding: 12px; }
    .source-panel {
      position: sticky;
      top: 66px;
      max-height: calc(100vh - 78px);
      overflow: auto;
    }
    .review-panel { grid-column: 1 / -1; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      min-height: 58px;
      background: #fbfcfd;
    }
    .metric strong { display: block; font-size: 12px; margin-bottom: 4px; }
    .summary-grid {
      display: grid;
      gap: 8px;
    }
    .summary-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .summary-card h3 { margin-bottom: 8px; }
    .summary-line {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      min-height: 28px;
      border-bottom: 1px solid #e6ecef;
      padding: 4px 0;
    }
    .summary-line:last-child { border-bottom: 0; }
    .summary-value { font-weight: 700; text-align: right; overflow-wrap: anywhere; }
    .player-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(70px, 1fr));
      gap: 6px;
    }
    .player-chip {
      border: 1px solid #e2e8ec;
      border-radius: 6px;
      padding: 6px;
      background: #fff;
      min-height: 44px;
    }
    .player-chip strong { display: block; font-size: 12px; }
    .player-chip.dead { border-color: #f3b4ae; background: #fff3f2; color: var(--bad); }
    .player-chip.alive { border-color: #b7dfcf; background: #f0fbf6; color: var(--good); }
    .source-details summary { list-style: none; }
    .source-details summary::-webkit-details-marker { display: none; }
    .source-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }
    .source-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 10px;
      background: #f7fafb;
      border-bottom: 1px solid var(--line);
    }
    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 6px;
      padding: 10px;
    }
    .kv {
      border: 1px solid #e6ecef;
      border-radius: 6px;
      padding: 6px;
      min-height: 46px;
      background: #fff;
    }
    .kv span { display: block; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .kv strong { display: block; margin-top: 3px; font-size: 13px; overflow-wrap: anywhere; }
    .table-wrap { overflow: auto; max-height: 340px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid #e6ecef; padding: 6px 7px; text-align: left; white-space: nowrap; }
    th { position: sticky; top: 0; background: #f7fafb; z-index: 1; }
    .field-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 6px;
      max-height: 180px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fbfcfd;
    }
    .field-list label {
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--ink);
      font-size: 12px;
      min-width: 0;
    }
    pre {
      margin: 0;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f7f9fa;
      overflow: auto;
      max-height: 180px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; }
    }
    @media (max-width: 760px) {
      .workspace { grid-template-columns: 1fr; }
      .source-panel { position: static; max-height: none; }
    }
  </style>
</head>
<body>
<header>
  <h1>数据核验工作台</h1>
  <span id="pageStatus" class="status">加载中</span>
  <a class="nav-link" href="/">主动学习</a>
  <a class="nav-link" href="/evidence-review">证据核验</a>
  <button id="refreshButton">刷新</button>
  <span id="saveStatus" class="muted"></span>
</header>
<main>
  <aside class="stack">
    <section class="stack" style="border:0;">
      <label>视频
        <select id="videoSelect"></select>
      </label>
      <label>数据源
        <select id="sourceSelect" multiple></select>
      </label>
      <div class="toolbar">
        <button id="suggestedButton">使用推荐</button>
        <button id="selectAllButton">全选</button>
        <button id="clearSourcesButton">清空</button>
      </div>
      <div id="sourceHint" class="tiny"></div>
    </section>
    <section class="panel" style="padding:10px;">
      <h2>核验统计</h2>
      <div id="reviewSummary" class="tiny"></div>
    </section>
  </aside>
  <div class="workspace">
    <section class="video-panel">
      <div class="toolbar" style="justify-content:space-between; margin-bottom:10px;">
        <div>
          <h2 id="videoTitle">未选择视频</h2>
          <div id="videoMeta" class="tiny"></div>
        </div>
        <span id="timeBadge" class="status">0.000s</span>
      </div>
      <video id="reviewVideo" controls playsinline preload="metadata"></video>
    </section>
    <section class="source-panel stack">
      <div class="toolbar" style="justify-content:space-between;">
        <h2>当前时间点数据</h2>
        <span id="snapshotStatus" class="status">等待视频时间</span>
      </div>
      <div class="metric-grid" id="snapshotMetrics"></div>
      <div id="readableSummary" class="summary-grid"></div>
      <div class="stack" id="sourceSnapshots"></div>
    </section>
    <section class="review-panel stack">
      <div class="toolbar" style="justify-content:space-between;">
        <h2>判断</h2>
        <span id="decisionBadge" class="status accurate">准确</span>
      </div>
      <div class="toolbar">
        <button class="good active" data-decision="accurate">准确</button>
        <button class="bad" data-decision="incorrect">不准确</button>
        <button class="warn" data-decision="needs_review">需要复查</button>
        <button data-decision="skipped">跳过</button>
      </div>
      <label>有问题的字段
        <div id="fieldList" class="field-list"></div>
      </label>
      <label>备注
        <textarea id="reviewNote" placeholder="例如：比分左侧识别错了，player_state_3 实际已经死亡"></textarea>
      </label>
      <div class="toolbar">
        <button id="saveReviewButton" class="primary">保存当前判断</button>
        <button id="jumpBackButton">后退 2 秒</button>
        <button id="jumpForwardButton">前进 2 秒</button>
      </div>
      <pre id="actionOutput"></pre>
    </section>
  </div>
</main>
<script>
let state = null;
let snapshot = null;
let activeDecision = "accurate";
let snapshotTimer = null;

const pageStatus = document.getElementById("pageStatus");
const video = document.getElementById("reviewVideo");
const videoSelect = document.getElementById("videoSelect");
const sourceSelect = document.getElementById("sourceSelect");
const sourceSnapshots = document.getElementById("sourceSnapshots");
const readableSummary = document.getElementById("readableSummary");
const fieldList = document.getElementById("fieldList");
const actionOutput = document.getElementById("actionOutput");

function jsonFetch(url, options = {}) {
  return fetch(url, options).then(async response => {
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || response.statusText);
    return payload;
  });
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[character]));
}

function selectedVideo() {
  return state?.videos.find(item => item.path === videoSelect.value) || null;
}

function selectedSourcePaths() {
  return Array.from(sourceSelect.selectedOptions).map(option => option.value);
}

function setStatus(element, value) {
  element.className = "status " + String(value || "");
  element.textContent = value || "ready";
}

function renderState() {
  videoSelect.innerHTML = "";
  for (const item of state.videos) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.label;
    videoSelect.appendChild(option);
  }
  sourceSelect.innerHTML = "";
  for (const item of state.sources) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = `${item.kind} · ${item.path}`;
    sourceSelect.appendChild(option);
  }
  setStatus(pageStatus, state.status);
  renderReviewSummary();
  chooseSuggestedSources();
  applyVideoSource();
}

function renderReviewSummary() {
  const summary = state.review_summary || {};
  const decisions = summary.by_decision || {};
  document.getElementById("reviewSummary").innerHTML = `
    <div>已保存：${escapeHtml(summary.count || 0)}</div>
    <div>准确：${escapeHtml(decisions.accurate || 0)} · 不准确：${escapeHtml(decisions.incorrect || 0)} · 复查：${escapeHtml(decisions.needs_review || 0)}</div>
    <div>${escapeHtml(summary.path || "")}</div>
  `;
}

function chooseSuggestedSources() {
  const item = selectedVideo();
  const suggested = new Set(item?.suggested_sources || []);
  for (const option of sourceSelect.options) {
    option.selected = suggested.has(option.value);
  }
  document.getElementById("sourceHint").textContent = suggested.size
    ? `已选择 ${suggested.size} 个推荐数据源`
    : "没有匹配到推荐数据源，可以手动选择";
}

function applyVideoSource() {
  const item = selectedVideo();
  if (!item) return;
  video.src = `/api/video?path=${encodeURIComponent(item.path)}`;
  document.getElementById("videoTitle").textContent = item.label;
  document.getElementById("videoMeta").textContent = item.path;
  requestSnapshot();
}

function requestSnapshot() {
  if (snapshotTimer) clearTimeout(snapshotTimer);
  snapshotTimer = setTimeout(loadSnapshot, 180);
}

async function loadSnapshot() {
  const item = selectedVideo();
  if (!item) return;
  const params = new URLSearchParams();
  params.set("video", item.path);
  params.set("time", String(video.currentTime || 0));
  for (const source of selectedSourcePaths()) params.append("source", source);
  try {
    snapshot = await jsonFetch(`/api/data-review/snapshot?${params.toString()}`);
    renderSnapshot();
  } catch (error) {
    document.getElementById("snapshotStatus").textContent = String(error);
  }
}

function firstSourceByKind(kind) {
  return snapshot?.sources.find(source => source.kind === kind && (source.rows || []).length) || null;
}

function firstRowValues(source) {
  return source?.rows?.[0]?.values || {};
}

function displayValue(value) {
  const text = String(value ?? "").trim();
  return text || "无数据";
}

function playerStateInfo(value) {
  const text = String(value ?? "").trim();
  if (!text) return { label: "无数据", className: "" };
  if (text === "1") return { label: "死亡", className: "dead" };
  if (text === "0" || text === "14") return { label: "存活", className: "alive" };
  return { label: `状态 ${text}`, className: "" };
}

function renderPlayerSummary(row) {
  return `
    <div class="summary-card">
      <h3>玩家状态</h3>
      <div class="player-grid">
        ${Array.from({ length: 8 }, (_, index) => {
          const slot = index + 1;
          const state = playerStateInfo(row[`player_state_${slot}`]);
          return `<div class="player-chip ${state.className}"><strong>${slot} 号</strong>${escapeHtml(state.label)}</div>`;
        }).join("")}
      </div>
    </div>
  `;
}

function renderWeaponSummary(row) {
  const weapons = Array.from({ length: 8 }, (_, index) => row[`weapon_${index + 1}`]).filter(value => String(value || "").trim());
  if (!weapons.length) return "";
  return `
    <div class="summary-card">
      <h3>武器</h3>
      ${weapons.map((weapon, index) => `
        <div class="summary-line"><span>${index + 1} 号</span><span class="summary-value">${escapeHtml(weapon)}</span></div>
      `).join("")}
    </div>
  `;
}

function heatmapPointCount() {
  return (snapshot?.sources || [])
    .filter(source => source.kind === "heatmap_tracks")
    .reduce((count, source) => count + Number(source.display_row_count || 0), 0);
}

function renderReadableSummary() {
  const analysisSource = firstSourceByKind("analysis_csv");
  const analysis = firstRowValues(analysisSource);
  const heatmapCount = heatmapPointCount();
  const cards = [
    `
      <div class="summary-card">
        <h3>比分和目标</h3>
        <div class="summary-line"><span>比分</span><span class="summary-value">${escapeHtml(displayValue(analysis.count_left))} : ${escapeHtml(displayValue(analysis.count_right))}</span></div>
        <div class="summary-line"><span>罚分</span><span class="summary-value">${escapeHtml(displayValue(analysis.penalty_left))} : ${escapeHtml(displayValue(analysis.penalty_right))}</span></div>
        <div class="summary-line"><span>蛤蜊 / 鱼 / 区 / 塔</span><span class="summary-value">${escapeHtml(displayValue(analysis.asari_count))} / ${escapeHtml(displayValue(analysis.hoko_count))} / ${escapeHtml(displayValue(analysis.area_count))} / ${escapeHtml(displayValue(analysis.yagura_count))}</span></div>
      </div>
    `,
    renderPlayerSummary(analysis),
    renderWeaponSummary(analysis),
    `
      <div class="summary-card">
        <h3>其它数据</h3>
        <div class="summary-line"><span>消息文字</span><span class="summary-value">${escapeHtml(displayValue(analysis.message))}</span></div>
        <div class="summary-line"><span>检测到玩家</span><span class="summary-value">${escapeHtml(displayValue(analysis.player_detected))}</span></div>
        <div class="summary-line"><span>热力图点数</span><span class="summary-value">${escapeHtml(heatmapCount)}</span></div>
      </div>
    `
  ];
  if (!analysisSource) {
    cards.unshift('<div class="summary-card"><h3>分析 CSV</h3><div class="muted">当前推荐数据源里没有分析 CSV，只能查看下方原始详情。</div></div>');
  }
  return cards.filter(Boolean).join("");
}

function renderSnapshot() {
  if (!snapshot) return;
  setStatus(document.getElementById("snapshotStatus"), snapshot.status);
  document.getElementById("timeBadge").textContent = `${Number(snapshot.time || 0).toFixed(3)}s`;
  document.getElementById("snapshotMetrics").innerHTML = `
    <div class="metric"><strong>数据源</strong>${snapshot.sources.length}</div>
    <div class="metric"><strong>时间窗口</strong>${escapeHtml(snapshot.window)}s</div>
    <div class="metric"><strong>当前视频</strong>${escapeHtml(snapshot.video?.label || "")}</div>
  `;
  readableSummary.innerHTML = renderReadableSummary();
  sourceSnapshots.innerHTML = snapshot.sources.map(renderSource).join("");
  renderFieldList();
}

function renderSource(source) {
  const rows = source.rows || [];
  const delta = source.delta === null || source.delta === undefined ? "" : `Δ ${source.delta}s`;
  return `
    <details class="source-card source-details">
      <summary class="source-head">
        <div>
          <h3>${escapeHtml(source.label || source.path)}</h3>
          <div class="tiny">${escapeHtml(source.path)} · ${escapeHtml(source.time_field || "")} · ${escapeHtml(delta)}</div>
        </div>
        <span class="status ${escapeHtml(source.status)}">${escapeHtml(source.status)}</span>
      </summary>
      ${rows.length === 0 ? '<div class="kv-grid"><div class="muted">当前时间点没有数据</div></div>' : renderRows(source, rows)}
    </details>
  `;
}

function renderRows(source, rows) {
  const fields = source.fieldnames || Object.keys(rows[0]?.values || {});
  if (rows.length === 1) {
    return `<div class="kv-grid">${fields.map(field => `
      <div class="kv">
        <span>${escapeHtml(field)}</span>
        <strong>${escapeHtml(rows[0].values?.[field] ?? "")}</strong>
      </div>`).join("")}</div>`;
  }
  return `
    <div class="table-wrap">
      <table>
        <thead><tr>${fields.map(field => `<th>${escapeHtml(field)}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map(row => `<tr>${fields.map(field => `<td>${escapeHtml(row.values?.[field] ?? "")}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderFieldList() {
  if (!snapshot || !snapshot.sources.length) {
    fieldList.innerHTML = '<span class="muted">没有可标记字段</span>';
    return;
  }
  const html = [];
  for (const source of snapshot.sources) {
    for (const field of source.fieldnames || []) {
      const value = `${source.path}::${field}`;
      html.push(`<label><input type="checkbox" value="${escapeHtml(value)}"> ${escapeHtml(source.kind)} / ${escapeHtml(field)}</label>`);
    }
  }
  fieldList.innerHTML = html.join("");
}

function setDecision(decision) {
  activeDecision = decision;
  for (const button of document.querySelectorAll("[data-decision]")) {
    button.classList.toggle("active", button.dataset.decision === decision);
  }
  const labels = { accurate: "准确", incorrect: "不准确", needs_review: "需要复查", skipped: "跳过" };
  const badge = document.getElementById("decisionBadge");
  badge.className = `status ${decision}`;
  badge.textContent = labels[decision] || decision;
}

async function saveReview() {
  if (!snapshot) await loadSnapshot();
  const incorrectFields = Array.from(fieldList.querySelectorAll("input:checked")).map(input => input.value);
  const payload = {
    video_path: videoSelect.value,
    time: video.currentTime || 0,
    source_paths: selectedSourcePaths(),
    decision: activeDecision,
    incorrect_fields: incorrectFields,
    note: document.getElementById("reviewNote").value,
    snapshot
  };
  const result = await jsonFetch("/api/data-review/review", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
  state.review_summary = result.summary;
  renderReviewSummary();
  document.getElementById("saveStatus").textContent = `已保存 ${Number(payload.time).toFixed(3)}s`;
  actionOutput.textContent = JSON.stringify(result.review, null, 2);
}

document.getElementById("refreshButton").onclick = async () => {
  await loadState();
};
document.getElementById("suggestedButton").onclick = () => {
  chooseSuggestedSources();
  requestSnapshot();
};
document.getElementById("selectAllButton").onclick = () => {
  for (const option of sourceSelect.options) option.selected = true;
  requestSnapshot();
};
document.getElementById("clearSourcesButton").onclick = () => {
  for (const option of sourceSelect.options) option.selected = false;
  requestSnapshot();
};
document.getElementById("saveReviewButton").onclick = () => saveReview().catch(error => {
  actionOutput.textContent = String(error);
});
document.getElementById("jumpBackButton").onclick = () => {
  video.currentTime = Math.max(0, (video.currentTime || 0) - 2);
  requestSnapshot();
};
document.getElementById("jumpForwardButton").onclick = () => {
  video.currentTime = (video.currentTime || 0) + 2;
  requestSnapshot();
};
videoSelect.onchange = () => {
  chooseSuggestedSources();
  applyVideoSource();
};
sourceSelect.onchange = requestSnapshot;
video.ontimeupdate = requestSnapshot;
video.onseeked = requestSnapshot;
for (const button of document.querySelectorAll("[data-decision]")) {
  button.onclick = () => setDecision(button.dataset.decision);
}

async function loadState() {
  state = await jsonFetch("/api/data-review/state");
  renderState();
}

loadState().catch(error => {
  pageStatus.textContent = String(error);
});
</script>
</body>
</html>
"""


EVIDENCE_REVIEW_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>证据核验工作台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7f8;
      --surface: #ffffff;
      --ink: #172026;
      --muted: #65717b;
      --line: #d8e0e5;
      --accent: #0f766e;
      --good: #13795b;
      --warn: #a16207;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      height: 54px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 16px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 { margin: 0; font-size: 17px; font-weight: 700; }
    h2 { margin: 0; font-size: 16px; font-weight: 700; }
    h3 { margin: 0; font-size: 14px; font-weight: 700; }
    button, select, textarea { font: inherit; }
    button, a.nav-link {
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      text-decoration: none;
      cursor: pointer;
    }
    button.primary { border-color: var(--accent); background: var(--accent); color: #fff; }
    button.good { border-color: var(--good); color: var(--good); }
    button.warn { border-color: var(--warn); color: var(--warn); }
    button.bad { border-color: var(--bad); color: var(--bad); }
    main { padding: 12px; display: grid; gap: 12px; }
    section, .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    select, textarea {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fff;
      color: var(--ink);
      width: 100%;
    }
    textarea { min-height: 62px; resize: vertical; }
    img {
      width: 100%;
      border-radius: 8px;
      background: #111820;
      display: block;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto;
      gap: 10px;
      align-items: end;
    }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
    .status {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2f4;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .status.ready, .status.saved, .status.accurate { background: #e8f5ef; color: var(--good); }
    .status.needs_data, .status.needs_review, .status.empty { background: #fff5d7; color: var(--warn); }
    .status.incorrect, .status.not_death { background: #fdeceb; color: var(--bad); }
    .muted { color: var(--muted); font-size: 12px; }
    .tiny { color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .evidence-grid {
      display: grid;
      grid-template-columns: minmax(340px, 0.95fr) minmax(360px, 1.25fr);
      gap: 12px;
      align-items: start;
    }
    .weapon-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .weapon-slot, .death-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fbfcfd;
    }
    .weapon-slot strong { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .weapon-image-wrap {
      position: relative;
      margin-top: 10px;
      border-radius: 8px;
      overflow: hidden;
      background: #111820;
      user-select: none;
      touch-action: none;
    }
    .weapon-image-wrap.ready-to-draw { cursor: crosshair; }
    .weapon-image-wrap img { border-radius: 0; }
    .manual-crop-box {
      position: absolute;
      display: none;
      border: 2px solid #22c55e;
      background: rgba(34, 197, 94, 0.18);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.22);
      pointer-events: none;
    }
    .manual-crop-box::after {
      content: attr(data-label);
      position: absolute;
      left: 0;
      top: -24px;
      min-width: max-content;
      padding: 2px 6px;
      border-radius: 5px;
      background: #16a34a;
      color: #fff;
      font-size: 12px;
      font-weight: 700;
    }
    .slot-marker {
      position: absolute;
      display: grid;
      place-items: center;
      min-width: 28px;
      min-height: 28px;
      padding: 0;
      border: 2px solid #f97316;
      background: rgba(255, 247, 237, 0.16);
      color: #fff;
      text-shadow: 0 1px 2px #000;
      font-weight: 800;
      box-shadow: 0 0 0 2px rgba(0, 0, 0, 0.3);
    }
    .slot-marker.active {
      border-color: #22c55e;
      background: rgba(34, 197, 94, 0.28);
    }
    .slot-marker::before {
      content: "";
      position: absolute;
      top: -16px;
      left: 50%;
      transform: translateX(-50%);
      width: 0;
      height: 0;
      border-left: 7px solid transparent;
      border-right: 7px solid transparent;
      border-top: 12px solid #f97316;
    }
    .slot-marker.active::before { border-top-color: #22c55e; }
    .correction-panel {
      display: grid;
      gap: 8px;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfd;
    }
    .field-row {
      display: grid;
      grid-template-columns: minmax(160px, 1fr) minmax(220px, 1.4fr);
      gap: 8px;
      align-items: end;
    }
    .crop-result {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fff;
      min-height: 34px;
    }
    .death-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }
    .death-card img { aspect-ratio: 16 / 9; object-fit: cover; margin: 8px 0; }
    .decision-panel {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .decision-panel .toolbar button { flex: 1 1 auto; }
    .message {
      border: 1px dashed var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--muted);
      background: #fbfcfd;
    }
    @media (max-width: 900px) {
      .evidence-grid, .topbar { grid-template-columns: 1fr; }
      .field-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header>
  <h1>证据核验工作台</h1>
  <span id="pageStatus" class="status">加载中</span>
  <a class="nav-link" href="/">主动学习</a>
  <a class="nav-link" href="/data-review">时间同步核验</a>
  <span id="saveStatus" class="muted"></span>
</header>
<main>
  <section class="topbar">
    <label>先选视频
      <select id="videoSelect"></select>
    </label>
    <div class="toolbar">
      <button id="loadButton" class="primary">生成证据</button>
      <button id="refreshButton">刷新列表</button>
    </div>
  </section>
  <div id="evidenceRoot" class="message">请选择一个视频，然后点“生成证据”。</div>
</main>
<script>
let state = null;
let evidence = null;
let selectedWeaponSlot = null;
let manualWeaponCropBox = null;
let cropDragStart = null;
let cropPointerId = null;

const pageStatus = document.getElementById("pageStatus");
const videoSelect = document.getElementById("videoSelect");
const evidenceRoot = document.getElementById("evidenceRoot");
const saveStatus = document.getElementById("saveStatus");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[character]));
}

async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function setStatus(element, status) {
  element.className = "status " + String(status || "");
  element.textContent = status || "ready";
}

function imageTag(path, alt) {
  if (!path) return '<div class="message">没有可用截图</div>';
  return `<img src="/api/image?path=${encodeURIComponent(path)}" alt="${escapeHtml(alt)}">`;
}

function weaponBySlot(slot) {
  return (evidence?.weapon?.weapons || []).find(item => Number(item.slot) === Number(slot)) || null;
}

function weaponImageHtml(weapon) {
  if (!weapon.image_path) return '<div class="message">没有可用截图</div>';
  const cropStyle = manualWeaponCropBox
    ? `left:${manualWeaponCropBox.left_pct}%;top:${manualWeaponCropBox.top_pct}%;width:${manualWeaponCropBox.width_pct}%;height:${manualWeaponCropBox.height_pct}%;display:block;`
    : "";
  const cropLabel = manualWeaponCropBox
    ? `${manualWeaponCropBox.slot} 号 ${manualWeaponCropBox.width}x${manualWeaponCropBox.height}`
    : "";
  return `
    <div class="weapon-image-wrap ${selectedWeaponSlot ? "ready-to-draw" : ""}">
      ${imageTag(weapon.image_path, "武器证据截图")}
      <div id="manualCropBox" class="manual-crop-box" data-label="${escapeHtml(cropLabel)}" style="${escapeHtml(cropStyle)}"></div>
    </div>
  `;
}

function weaponOptionsHtml() {
  const labels = state?.weapon_labels || [];
  return [
    '<option value="">选择真实武器</option>',
    ...labels.map(label => `<option value="${escapeHtml(label)}">${escapeHtml(label)}</option>`)
  ].join("");
}

function selectedWeaponBox() {
  return manualWeaponCropBox || {};
}

function renderWeaponCorrectionPanel(weapon) {
  const selected = weaponBySlot(selectedWeaponSlot);
  const hasCrop = Boolean(manualWeaponCropBox && manualWeaponCropBox.width >= 4 && manualWeaponCropBox.height >= 4);
  const selectedText = selected
    ? `${selected.slot} 号 · 当前识别：${selected.weapon || "无数据"}`
    : "先在下方选择 1-8 号武器";
  const cropText = selected
    ? (hasCrop ? `已框选 ${manualWeaponCropBox.width}x${manualWeaponCropBox.height}` : "在截图上拖框圈出真实武器图标")
    : "选择槽位后再框选";
  return `
    <div class="correction-panel">
      <div class="toolbar" style="justify-content:space-between;">
        <div>
          <h3>纠错样本</h3>
          <div class="tiny">待验证训练集：${escapeHtml(state?.weapon_correction_dataset || "")}</div>
        </div>
        <span id="weaponCorrectionState" class="status">${escapeHtml(selectedText)} · ${escapeHtml(cropText)}</span>
      </div>
      <div class="field-row">
        <label>真实武器
          <select id="actualWeaponSelect" ${selected ? "" : "disabled"}>${weaponOptionsHtml()}</select>
        </label>
        <label>备注
          <textarea id="weaponCorrectionNote" ${selected ? "" : "disabled"} placeholder="可选"></textarea>
        </label>
      </div>
      <div class="toolbar">
        <button id="saveWeaponCorrectionButton" class="primary" disabled>截取并加入待验证训练集</button>
      </div>
      <div id="weaponCorrectionResult" class="crop-result tiny">先选择槽位，再在截图上拖框；新样本会保存到独立目录，不会进入正式训练集。</div>
    </div>
  `;
}

function zhReason(value) {
  const text = String(value || "");
  if (text === "no player-state death transitions found") {
    return "当前分析数据里没有识别到玩家从存活变成死亡的时间点。";
  }
  if (text.includes("没有找到匹配的分析 CSV")) return text;
  return text;
}

function renderState() {
  videoSelect.innerHTML = "";
  for (const video of state.videos || []) {
    const option = document.createElement("option");
    option.value = video.path;
    option.textContent = video.label;
    videoSelect.appendChild(option);
  }
  setStatus(pageStatus, state.status);
}

function weaponListHtml(weapon) {
  const items = weapon.weapons || [];
  if (!items.length) return '<div class="message">没有武器识别结果。</div>';
  return `<div class="weapon-grid">${items.map(item => `
    <div class="weapon-slot">
      <strong>${escapeHtml(item.slot)} 号</strong>
      <div>${escapeHtml(item.weapon || "无数据")}</div>
      <button data-slot-select="${escapeHtml(item.slot)}" style="margin-top:8px;">选择</button>
    </div>
  `).join("")}</div>`;
}

function decisionPanel(itemType, itemId, sourcePath, payload, decisions) {
  const noteId = `note_${itemType}_${String(itemId).replace(/[^a-zA-Z0-9_-]/g, "_")}`;
  return `
    <div class="decision-panel">
      <div class="toolbar">
        ${decisions.map(item => `<button class="${item.className || ""}" data-item-type="${escapeHtml(itemType)}" data-item-id="${escapeHtml(itemId)}" data-source-path="${escapeHtml(sourcePath || "")}" data-decision="${escapeHtml(item.decision)}" data-note-id="${escapeHtml(noteId)}">${escapeHtml(item.label)}</button>`).join("")}
      </div>
      <textarea id="${escapeHtml(noteId)}" placeholder="可选备注，比如第 3 号武器错了，或这个时间点不是死亡"></textarea>
    </div>
  `;
}

function renderWeaponSection(weapon) {
  const status = weapon.status || "missing";
  return `
    <section>
      <div class="toolbar" style="justify-content:space-between;">
        <div>
          <h2>武器识别</h2>
          <div class="tiny">证据时间：${escapeHtml(weapon.time ?? "无")}s · 数据源：${escapeHtml(weapon.source_path || "")}</div>
        </div>
        <span class="status ${escapeHtml(status)}">${escapeHtml(status)}</span>
      </div>
      ${weapon.blocking_reason ? `<div class="message">${escapeHtml(weapon.blocking_reason)}</div>` : ""}
      ${weaponImageHtml(weapon)}
      ${weaponListHtml(weapon)}
      ${renderWeaponCorrectionPanel(weapon)}
      ${decisionPanel("weapon", weapon.id || "weapon", weapon.source_path || "", weapon, [
        { decision: "accurate", label: "武器都准确", className: "good" },
        { decision: "incorrect", label: "有武器错误", className: "bad" },
        { decision: "needs_review", label: "看不清/待复查", className: "warn" }
      ])}
    </section>
  `;
}

function renderDeathCard(event, sourcePath) {
  return `
    <article class="death-card">
      <div class="toolbar" style="justify-content:space-between;">
        <h3>${escapeHtml(event.time)}s</h3>
        <span class="status">${escapeHtml(event.victim || event.player || "")}</span>
      </div>
      <div class="tiny">槽位：${escapeHtml(event.victim_slot || "")} · 武器：${escapeHtml(event.victim_weapon || "无数据")} · 证据：${escapeHtml(event.evidence || "")}</div>
      ${imageTag(event.image_path, `死亡事件 ${event.time}s`)}
      ${decisionPanel("death", event.review_id || event.event_id || "", sourcePath, event, [
        { decision: "accurate", label: "死亡时间准确", className: "good" },
        { decision: "incorrect", label: "时间不对", className: "bad" },
        { decision: "not_death", label: "不是死亡", className: "bad" },
        { decision: "needs_review", label: "看不清", className: "warn" }
      ])}
    </article>
  `;
}

function renderDeathSection(death) {
  const events = death.events || [];
  return `
    <section>
      <div class="toolbar" style="justify-content:space-between;">
        <div>
          <h2>死亡时间点</h2>
          <div class="tiny">数据源：${escapeHtml(death.source_path || "")}</div>
        </div>
        <span class="status ${escapeHtml(death.status || "")}">${escapeHtml(death.event_count || 0)} 个</span>
      </div>
      ${death.blocking_reason ? `<div class="message">${escapeHtml(zhReason(death.blocking_reason))}</div>` : ""}
      ${events.length ? `<div class="death-grid">${events.map(event => renderDeathCard(event, death.source_path || "")).join("")}</div>` : '<div class="message">没有死亡事件截图。这里为空代表当前系统还没有拿到死亡时间点数据，需要后续先把死亡检测/OCR 跑出来。</div>'}
    </section>
  `;
}

function renderEvidence() {
  if (!evidence) {
    evidenceRoot.className = "message";
    evidenceRoot.textContent = "请选择一个视频，然后点“生成证据”。";
    return;
  }
  evidenceRoot.className = "evidence-grid";
  evidenceRoot.innerHTML = `
    ${renderWeaponSection(evidence.weapon || {})}
    ${renderDeathSection(evidence.death || {})}
  `;
}

async function loadState() {
  state = await jsonFetch("/api/evidence-review/state");
  renderState();
}

async function loadEvidence() {
  evidenceRoot.className = "message";
  evidenceRoot.textContent = "正在生成证据截图...";
  selectedWeaponSlot = null;
  manualWeaponCropBox = null;
  cropDragStart = null;
  cropPointerId = null;
  const params = new URLSearchParams();
  params.set("video", videoSelect.value);
  evidence = await jsonFetch(`/api/evidence-review/video?${params.toString()}`);
  setStatus(pageStatus, evidence.status);
  renderEvidence();
}

async function saveDecision(button) {
  const note = document.getElementById(button.dataset.noteId)?.value || "";
  const itemType = button.dataset.itemType;
  const itemId = button.dataset.itemId;
  const payload = {
    item_type: itemType,
    item_id: itemId,
    video_path: videoSelect.value,
    source_path: button.dataset.sourcePath || "",
    decision: button.dataset.decision,
    note,
    payload: itemType === "weapon"
      ? evidence.weapon
      : (evidence.death?.events || []).find(event => (event.review_id || event.event_id) === itemId)
  };
  const result = await jsonFetch("/api/evidence-review/review", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
  saveStatus.textContent = `已保存：${button.textContent}`;
  button.classList.add("primary");
  setTimeout(() => button.classList.remove("primary"), 900);
  return result;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function imageMetrics(wrapper) {
  const image = wrapper.querySelector("img");
  const rect = image.getBoundingClientRect();
  return {
    rect,
    naturalWidth: image.naturalWidth || Math.round(rect.width),
    naturalHeight: image.naturalHeight || Math.round(rect.height)
  };
}

function pointFromEvent(event, wrapper) {
  const metrics = imageMetrics(wrapper);
  const displayX = clamp(event.clientX - metrics.rect.left, 0, metrics.rect.width);
  const displayY = clamp(event.clientY - metrics.rect.top, 0, metrics.rect.height);
  return {
    x: Math.round(displayX * metrics.naturalWidth / Math.max(1, metrics.rect.width)),
    y: Math.round(displayY * metrics.naturalHeight / Math.max(1, metrics.rect.height)),
    naturalWidth: metrics.naturalWidth,
    naturalHeight: metrics.naturalHeight
  };
}

function cropBoxFromPoints(start, end) {
  const left = Math.min(start.x, end.x);
  const top = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);
  const naturalWidth = end.naturalWidth || start.naturalWidth || 1;
  const naturalHeight = end.naturalHeight || start.naturalHeight || 1;
  return {
    slot: Number(selectedWeaponSlot),
    left,
    top,
    width,
    height,
    left_pct: +(left / naturalWidth * 100).toFixed(3),
    top_pct: +(top / naturalHeight * 100).toFixed(3),
    width_pct: +(width / naturalWidth * 100).toFixed(3),
    height_pct: +(height / naturalHeight * 100).toFixed(3)
  };
}

function syncCorrectionControls() {
  const selected = weaponBySlot(selectedWeaponSlot);
  const hasCrop = Boolean(manualWeaponCropBox && manualWeaponCropBox.width >= 4 && manualWeaponCropBox.height >= 4);
  const actualWeapon = document.getElementById("actualWeaponSelect")?.value || "";
  const select = document.getElementById("actualWeaponSelect");
  const note = document.getElementById("weaponCorrectionNote");
  const button = document.getElementById("saveWeaponCorrectionButton");
  if (select) select.disabled = !selected;
  if (note) note.disabled = !selected;
  if (button) button.disabled = !(selected && hasCrop && actualWeapon);
  const stateNode = document.getElementById("weaponCorrectionState");
  if (stateNode) {
    const selectedText = selected
      ? `${selected.slot} 号 · 当前识别：${selected.weapon || "无数据"}`
      : "先在下方选择 1-8 号武器";
    const cropText = selected
      ? (hasCrop ? `已框选 ${manualWeaponCropBox.width}x${manualWeaponCropBox.height}` : "在截图上拖框圈出真实武器图标")
      : "选择槽位后再框选";
    stateNode.textContent = `${selectedText} · ${cropText}`;
  }
  const resultBox = document.getElementById("weaponCorrectionResult");
  if (resultBox && selected && hasCrop) {
    resultBox.textContent = actualWeapon
      ? `已框选 ${manualWeaponCropBox.width}x${manualWeaponCropBox.height}，可以保存。`
      : `已框选 ${manualWeaponCropBox.width}x${manualWeaponCropBox.height}，选择真实武器后即可保存。`;
  }
}

function updateManualCropOverlay() {
  const box = document.getElementById("manualCropBox");
  if (!box) return;
  if (!manualWeaponCropBox) {
    box.style.display = "none";
    box.dataset.label = "";
    syncCorrectionControls();
    return;
  }
  box.style.display = "block";
  box.style.left = `${manualWeaponCropBox.left_pct}%`;
  box.style.top = `${manualWeaponCropBox.top_pct}%`;
  box.style.width = `${manualWeaponCropBox.width_pct}%`;
  box.style.height = `${manualWeaponCropBox.height_pct}%`;
  box.dataset.label = `${manualWeaponCropBox.slot} 号 ${manualWeaponCropBox.width}x${manualWeaponCropBox.height}`;
  syncCorrectionControls();
}

function startManualCrop(event) {
  const wrapper = event.target.closest(".weapon-image-wrap");
  if (!wrapper) return;
  if (!selectedWeaponSlot) {
    saveStatus.textContent = "请先在武器识别结果里选择 1-8 号";
    return;
  }
  cropDragStart = pointFromEvent(event, wrapper);
  cropPointerId = event.pointerId;
  manualWeaponCropBox = cropBoxFromPoints(cropDragStart, cropDragStart);
  wrapper.setPointerCapture?.(event.pointerId);
  event.preventDefault();
  updateManualCropOverlay();
}

function updateManualCrop(event) {
  if (cropPointerId === null || event.pointerId !== cropPointerId || !cropDragStart) return;
  const wrapper = document.querySelector(".weapon-image-wrap");
  if (!wrapper) return;
  manualWeaponCropBox = cropBoxFromPoints(cropDragStart, pointFromEvent(event, wrapper));
  event.preventDefault();
  updateManualCropOverlay();
}

function finishManualCrop(event) {
  if (cropPointerId === null || event.pointerId !== cropPointerId) return;
  if (manualWeaponCropBox && (manualWeaponCropBox.width < 4 || manualWeaponCropBox.height < 4)) {
    manualWeaponCropBox = null;
    saveStatus.textContent = "框太小了，请重新拖一个包含武器图标的区域";
  }
  cropDragStart = null;
  cropPointerId = null;
  updateManualCropOverlay();
}

async function saveWeaponCorrection() {
  const actualWeapon = document.getElementById("actualWeaponSelect")?.value || "";
  if (!selectedWeaponSlot) {
    saveStatus.textContent = "请先选择一个武器槽位";
    return null;
  }
  if (!manualWeaponCropBox || manualWeaponCropBox.width < 4 || manualWeaponCropBox.height < 4) {
    saveStatus.textContent = "请先在截图上拖框圈出真实武器图标";
    return null;
  }
  if (!actualWeapon) {
    saveStatus.textContent = "请选择真实武器";
    return null;
  }
  const selected = weaponBySlot(selectedWeaponSlot) || {};
  const payload = {
    video_path: videoSelect.value,
    source_path: evidence?.weapon?.source_path || "",
    evidence_image_path: evidence?.weapon?.image_path || "",
    time: evidence?.weapon?.time,
    slot: Number(selectedWeaponSlot),
    predicted_weapon: selected.weapon || "",
    actual_weapon: actualWeapon,
    crop_box: selectedWeaponBox(),
    note: document.getElementById("weaponCorrectionNote")?.value || ""
  };
  const result = await jsonFetch("/api/evidence-review/weapon-correction", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
  saveStatus.textContent = `已保存纠错样本：${actualWeapon}`;
  const resultBox = document.getElementById("weaponCorrectionResult");
  if (resultBox) {
    const record = result.record || {};
    resultBox.innerHTML = `
      已生成 ${escapeHtml(1 + (record.augmented_paths || []).length)} 张图。<br>
      原图：${escapeHtml(record.original_path || "")}<br>
      目录：${escapeHtml(result.dataset_root || "")}
    `;
  }
  return result;
}

document.getElementById("refreshButton").onclick = loadState;
document.getElementById("loadButton").onclick = () => loadEvidence().catch(error => {
  evidenceRoot.className = "message";
  evidenceRoot.textContent = String(error);
});
evidenceRoot.addEventListener("click", event => {
  const slotButton = event.target.closest("button[data-slot-select]");
  if (slotButton) {
    selectedWeaponSlot = Number(slotButton.dataset.slotSelect);
    manualWeaponCropBox = null;
    renderEvidence();
    saveStatus.textContent = "已选择槽位，请在截图上拖框圈出武器图标";
    return;
  }
  const correctionButton = event.target.closest("#saveWeaponCorrectionButton");
  if (correctionButton) {
    saveWeaponCorrection().catch(error => {
      saveStatus.textContent = String(error);
    });
    return;
  }
  const button = event.target.closest("button[data-decision]");
  if (!button) return;
  saveDecision(button).catch(error => {
    saveStatus.textContent = String(error);
  });
});
evidenceRoot.addEventListener("change", event => {
  if (event.target?.id === "actualWeaponSelect") {
    syncCorrectionControls();
  }
});
evidenceRoot.addEventListener("pointerdown", startManualCrop);
evidenceRoot.addEventListener("pointermove", updateManualCrop);
window.addEventListener("pointerup", finishManualCrop);
window.addEventListener("pointercancel", finishManualCrop);

loadState().catch(error => {
  pageStatus.textContent = String(error);
});
</script>
</body>
</html>
"""


def run_background_job(job: dict, action_id: str, payload: dict) -> None:
    try:
        result = run_workbench_action(action_id, payload)
    except Exception as exc:  # noqa: BLE001 - local tool should persist job failures.
        result = {"status": "failed", "error": str(exc)}
    finish_job_record(str(job["id"]), result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local active-learning workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "SplatoonWorkbench/1.0"

    def log_message(self, format: str, *args: object) -> None:
        path = self.path.split("?", 1)[0]
        if path in {"/api/video", "/api/data-review/snapshot"}:
            return
        super().log_message(format, *args)

    def send_bytes(self, body: bytes, status: HTTPStatus = HTTPStatus.OK, content_type: str = "application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"), status, "application/json; charset=utf-8")

    def send_error_json(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        self.send_json({"error": message}, status)

    def send_file_response(self, path: Path) -> None:
        size = path.stat().st_size
        start = 0
        end = size - 1
        status = HTTPStatus.OK
        range_header = self.headers.get("range", "")
        if range_header.startswith("bytes="):
            status = HTTPStatus.PARTIAL_CONTENT
            range_value = range_header.removeprefix("bytes=").split(",", 1)[0]
            start_text, _, end_text = range_value.partition("-")
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            elif end_text:
                suffix_length = int(end_text)
                start = max(0, size - suffix_length)
                end = size - 1
            if start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("content-range", f"bytes */{size}")
                self.end_headers()
                return
            end = min(end, size - 1)

        length = max(0, end - start + 1)
        self.send_response(status)
        self.send_header("content-type", media_type_for_path(path))
        self.send_header("accept-ranges", "bytes")
        self.send_header("content-length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("content-range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

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
            elif parsed.path == "/data-review":
                self.send_bytes(DATA_REVIEW_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            elif parsed.path == "/evidence-review":
                self.send_bytes(EVIDENCE_REVIEW_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            elif parsed.path == "/api/state":
                self.send_json(build_workbench_state())
            elif parsed.path == "/api/data-review/state":
                self.send_json(build_data_review_state())
            elif parsed.path == "/api/data-review/snapshot":
                query = parse_qs(parsed.query)
                source_paths = query.get("source", [])
                self.send_json(
                    build_time_snapshot(
                        query.get("video", [""])[0],
                        float(query.get("time", ["0"])[0] or 0),
                        source_paths=source_paths,
                    )
                )
            elif parsed.path == "/api/evidence-review/state":
                self.send_json(build_evidence_review_state())
            elif parsed.path == "/api/evidence-review/video":
                query = parse_qs(parsed.query)
                self.send_json(build_video_evidence(query.get("video", [""])[0]))
            elif parsed.path == "/api/candidates":
                self.send_json(load_candidate_queue())
            elif parsed.path == "/api/staging":
                self.send_json(load_staging(DEFAULT_STAGING_PATH))
            elif parsed.path == "/api/llm-reviews":
                self.send_json(load_llm_reviews(DEFAULT_LLM_REVIEWS_PATH))
            elif parsed.path == "/api/automation-plan":
                self.send_json(build_automation_plan())
            elif parsed.path == "/api/jobs":
                self.send_json(load_jobs())
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
            elif parsed.path == "/api/video":
                query = parse_qs(parsed.query)
                video_path = safe_project_file(query.get("path", [""])[0])
                if not is_video_path(video_path):
                    self.send_error_json("unsupported video type", HTTPStatus.BAD_REQUEST)
                    return
                if not video_path.exists():
                    self.send_error_json("video not found", HTTPStatus.NOT_FOUND)
                    return
                self.send_file_response(video_path)
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
            elif parsed.path == "/api/data-review/review":
                self.send_json(record_data_review(payload))
            elif parsed.path == "/api/evidence-review/review":
                self.send_json(record_evidence_review(payload))
            elif parsed.path == "/api/evidence-review/weapon-correction":
                self.send_json(record_weapon_correction(payload))
            elif parsed.path == "/api/action":
                self.send_json(run_workbench_action(str(payload.get("action_id", "")), payload.get("payload", {})))
            elif parsed.path == "/api/apply-staging":
                self.send_json(apply_staging_annotations(dry_run=bool(payload.get("dry_run", True))))
            elif parsed.path == "/api/llm-review-pack":
                self.send_json(build_llm_review_pack(limit=int(payload.get("limit", 30))))
            elif parsed.path == "/api/llm-review":
                self.send_json(record_llm_review(str(payload.get("id", "")), payload.get("review", {})))
            elif parsed.path == "/api/llm-review-auto":
                self.send_json(auto_record_llm_reviews(limit=int(payload.get("limit", 30))))
            elif parsed.path == "/api/heatmap-prefill":
                self.send_json(
                    prefill_heatmap_staging(
                        limit=int(payload.get("limit", 30)),
                        status=str(payload.get("status", "draft")),
                    )
                )
            elif parsed.path == "/api/automation-run":
                self.send_json(
                    run_automation_pipeline(
                        include_long=bool(payload.get("include_long", False)),
                        max_steps=int(payload.get("max_steps", 8)),
                        dry_run=bool(payload.get("dry_run", False)),
                    )
                )
            elif parsed.path == "/api/job":
                action_id = str(payload.get("action_id", ""))
                action_payload = payload.get("payload", {}) if isinstance(payload.get("payload", {}), dict) else {}
                job = start_job_record(action_id, action_payload)
                threading.Thread(target=run_background_job, args=(job, action_id, action_payload), daemon=True).start()
                self.send_json(job)
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
