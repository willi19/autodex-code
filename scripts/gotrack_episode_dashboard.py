#!/usr/bin/env python3
"""Browser dashboard for episode-level offline GoTrack scheduling."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.tracking.episode_queue import (  # noqa: E402
    DEFAULT_PCS,
    EpisodeScheduleStore,
    localize_shared_path,
    overlay_done,
    overlay_files,
    summarize_schedule,
)


STAGE_DEFINITIONS = [
    {
        "key": "pending",
        "label": "Pending",
        "description": "Episode is in the queue and no worker has claimed it yet.",
        "output": "task json",
    },
    {
        "key": "claimed",
        "label": "Claimed",
        "description": "A capture PC created the episode lock and is preparing the batch command.",
        "output": "claim lock",
    },
    {
        "key": "gotrack_or_overlay",
        "label": "Starting",
        "description": "The worker process is running; first stage-specific progress has not appeared yet.",
        "output": "worker log",
    },
    {
        "key": "gotrack",
        "label": "GoTrack",
        "description": "Multi-view object tracking over all camera videos for one episode.",
        "output": "world_pose_records.json",
    },
    {
        "key": "overlay",
        "label": "Overlay",
        "description": "Per-camera mesh overlay videos are rendered from the tracked poses.",
        "output": "overlay_*.mp4",
    },
    {
        "key": "complete",
        "label": "Complete",
        "description": "Required outputs already exist or were produced successfully; the task is not rerun.",
        "output": "done/skipped_done",
    },
    {
        "key": "failed",
        "label": "Failed",
        "description": "The command failed or required outputs were missing after completion.",
        "output": "reason + log",
    },
    {
        "key": "dry_run",
        "label": "Dry Run",
        "description": "The worker claimed the episode without running tracking or overlay.",
        "output": "dry_run_done",
    },
]


def _stage_summary(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    phase_counts: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    for task in tasks:
        phase = str(task.get("phase") or task.get("status") or "unknown")
        status = str(task.get("status") or "unknown")
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        status_counts[status] = status_counts.get(status, 0) + 1
    return {"phase_counts": phase_counts, "status_counts": status_counts}


def _load_events(path: Path, limit: int = 40) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append({"event": "corrupt_event", "raw": line})
    return out


def _augment_overlay_metadata(
    store: EpisodeScheduleStore,
    data: Dict[str, Any],
    include_files: bool = False,
) -> None:
    done_statuses = {"done", "skipped_done", "dry_run_done"}
    tasks: List[Dict[str, Any]] = []
    for task in data.get("tasks", []):
        if not isinstance(task, dict):
            continue
        out = dict(task)
        for key in ("episode_dir", "videos_dir"):
            if out.get(key):
                out[key] = str(localize_shared_path(out[key]))

        overlay_dir = store.overlay_dir_for(out)
        found: List[str] = []
        if include_files:
            for item in out.get("overlay_files", []) or []:
                try:
                    path = localize_shared_path(item).resolve()
                except OSError:
                    continue
                if path.name.startswith("overlay_") and path.suffix.lower() == ".mp4" and path.is_file():
                    found.append(str(path))

            if not found and str(out.get("status")) in done_statuses:
                found = overlay_files(overlay_dir)

        if out.get("serials"):
            out["overlay_done"] = overlay_done(overlay_dir, out.get("serials", []))
        out["overlay_output_dir"] = str(overlay_dir)
        out["overlay_file_count"] = (
            len(set(found))
            if include_files
            else int(len(out.get("overlay_files") or []) or (len(out.get("serials") or []) if out.get("overlay_done") else 0))
        )
        out["overlay_files"] = sorted(set(found)) if include_files else []
        tasks.append(out)

    data["tasks"] = tasks


def collect(schedule_dir: Path, include_overlay_files: bool = False) -> Dict[str, Any]:
    store = EpisodeScheduleStore.open(schedule_dir)
    data = summarize_schedule(store)
    _augment_overlay_metadata(store, data, include_files=include_overlay_files)
    data["tasks"] = [_compact_task_for_status(t) for t in data.get("tasks", []) if isinstance(t, dict)]
    data["events"] = _load_events(store.events_path)
    data["expected_pcs"] = list(DEFAULT_PCS)
    data["stage_definitions"] = STAGE_DEFINITIONS
    data["stage_summary"] = _stage_summary(data.get("tasks", []))
    data["overlay_file_count"] = sum(int(t.get("overlay_file_count") or 0) for t in data.get("tasks", []) if isinstance(t, dict))
    data["overlay_episode_count"] = sum(1 for t in data.get("tasks", []) if isinstance(t, dict) and int(t.get("overlay_file_count") or 0) > 0)
    return data


def _compact_task_for_status(task: Dict[str, Any]) -> Dict[str, Any]:
    drop_keys = {
        "command",
        "last_line",
        "serials",
        "videos_dir",
    }
    return {k: v for k, v in task.items() if k not in drop_keys}


def find_task(store: EpisodeScheduleStore, task_id: str) -> Dict[str, Any] | None:
    for task in store.tasks():
        if str(task.get("task_id") or "") == str(task_id):
            return task
    return None


def allowed_overlay_path(schedule_dir: Path, requested: Path) -> Path | None:
    store = EpisodeScheduleStore.open(schedule_dir)
    try:
        requested = requested.expanduser().resolve()
    except OSError:
        return None
    if not (requested.name.startswith("overlay_") and requested.suffix.lower() == ".mp4"):
        return None
    for task in store.tasks():
        try:
            overlay_dir = store.overlay_dir_for(task).expanduser().resolve()
            requested.relative_to(overlay_dir)
        except (OSError, ValueError):
            continue
        return requested if requested.is_file() else None
    return None


def overlay_payload_for_task(schedule_dir: Path, task_id: str) -> Dict[str, Any]:
    store = EpisodeScheduleStore.open(schedule_dir)
    task = find_task(store, task_id)
    if task is None:
        raise KeyError(task_id)
    overlay_dir = store.overlay_dir_for(task)
    files = overlay_files(overlay_dir)
    return {
        "task_id": task_id,
        "overlay_output_dir": str(overlay_dir),
        "overlay_files": files,
        "overlay_file_count": len(files),
    }


def _thumbnail_path(schedule_dir: Path, video: Path) -> Path:
    key = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:24]
    return schedule_dir / ".overlay_thumbs" / f"{key}.jpg"


def ensure_overlay_thumbnail(schedule_dir: Path, video: Path) -> Path:
    thumb = _thumbnail_path(schedule_dir, video)
    if thumb.exists() and thumb.stat().st_mtime >= video.stat().st_mtime:
        return thumb
    thumb.parent.mkdir(parents=True, exist_ok=True)
    tmp = thumb.with_suffix(".jpg.tmp")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", "1",
        "-i", str(video),
        "-frames:v", "1",
        "-vf", "scale=320:-1",
        "-q:v", "5",
        "-f", "image2",
        str(tmp),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0 or not tmp.exists():
        raise RuntimeError(result.stderr.strip() or "ffmpeg thumbnail generation failed")
    tmp.replace(thumb)
    return thumb


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GoTrack Episode Scheduler</title>
<style>
:root {
  color-scheme: dark;
  --bg: #111315;
  --panel: #1b1f22;
  --panel2: #22272b;
  --line: #31383e;
  --text: #e6edf3;
  --muted: #8b949e;
  --green: #3fb950;
  --yellow: #d29922;
  --red: #f85149;
  --blue: #58a6ff;
  --gray: #6e7681;
  --violet: #bc8cff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 13px/1.4 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  border-bottom: 1px solid var(--line);
  background: #15191c;
}
h1 { font-size: 17px; margin: 0; font-weight: 650; letter-spacing: 0; }
h2 { font-size: 14px; margin: 0; color: #c9d1d9; font-weight: 650; }
.sub { color: var(--muted); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; }
main { padding: 16px 18px 22px; max-width: 1580px; margin: 0 auto; }
.section-title { display: flex; align-items: flex-end; justify-content: space-between; gap: 14px; margin: 18px 0 8px; }
.section-title:first-child { margin-top: 0; }
.overview-grid { display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 10px; }
.risk-grid { display: grid; grid-template-columns: repeat(4, minmax(190px, 1fr)); gap: 10px; }
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  min-width: 0;
}
.metric-label { color: var(--muted); font-size: 12px; }
.metric-value { margin-top: 4px; font-size: 22px; font-weight: 700; letter-spacing: 0; }
.metric-small { color: var(--muted); margin-top: 4px; font-size: 12px; overflow-wrap: anywhere; }
.bar { width: 100%; height: 8px; background: #30363d; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; width: 0%; background: var(--green); transition: width .2s ease; }
.bar.slim { width: 84px; height: 6px; flex: none; }
.bar.warn > div { background: var(--yellow); }
.table-wrap {
  width: 100%;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
table { width: 100%; border-collapse: collapse; min-width: 1040px; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: middle; }
th { color: var(--muted); font-size: 12px; font-weight: 500; background: var(--panel2); }
tr:last-child td { border-bottom: 0; }
td.path { max-width: 420px; white-space: normal; overflow-wrap: anywhere; }
.status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 7px;
  border-radius: 999px;
  background: #2a3035;
  color: var(--muted);
  font-weight: 600;
}
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--gray); flex: none; }
.done .dot, .skipped_done .dot, .dry_run_done .dot, .complete .dot, .idle .dot, .stopped .dot { background: var(--green); }
.running .dot, .pending .dot, .claimed .dot, .gotrack .dot, .overlay .dot, .starting .dot { background: var(--yellow); }
.failed .dot, .stale .dot { background: var(--red); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.muted { color: var(--muted); }
.bad { color: var(--red); }
.ok { color: var(--green); }
.warn { color: var(--yellow); }
.progress-cell { display: flex; align-items: center; gap: 8px; min-width: 180px; }
.progress-cell .pct { width: 44px; }
.episode-cell { display: flex; flex-direction: column; gap: 2px; min-width: 150px; }
.episode-cell .task-id { max-width: 280px; overflow: hidden; text-overflow: ellipsis; }
.overlay-links { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; min-width: 150px; }
.overlay-button {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--blue);
  cursor: pointer;
  padding: 0;
  font: inherit;
}
.overlay-button:hover { text-decoration: underline; }
.stage-grid { display: grid; grid-template-columns: repeat(4, minmax(210px, 1fr)); gap: 10px; }
.stage-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 11px 12px;
  min-width: 0;
}
.stage-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
.stage-name { font-weight: 700; }
.stage-count { font: 700 18px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.stage-desc { margin-top: 7px; color: #c9d1d9; min-height: 38px; }
.stage-output { margin-top: 7px; color: var(--muted); overflow-wrap: anywhere; }
.stage-card.active { border-color: #506172; background: #20262a; }
.overlay-carousel {
  display: grid;
  grid-template-columns: 28px minmax(128px, 168px) 28px;
  align-items: center;
  gap: 6px;
  width: 236px;
}
.overlay-step,
.viewer-nav {
  appearance: none;
  border: 1px solid var(--line);
  background: #30363d;
  color: var(--text);
  cursor: pointer;
  border-radius: 6px;
  height: 32px;
  font: 700 14px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.overlay-step:disabled,
.viewer-nav:disabled { color: var(--gray); cursor: default; opacity: .55; }
.overlay-thumb-button {
  appearance: none;
  border: 1px solid var(--line);
  border-radius: 6px;
  overflow: hidden;
  background: #0d1117;
  color: var(--text);
  cursor: pointer;
  padding: 0;
  text-align: left;
  min-width: 0;
}
.overlay-thumb {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  object-fit: cover;
  background: #000;
}
.overlay-caption {
  display: block;
  padding: 4px 6px;
  color: var(--muted);
  font: 11px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.risk-title { display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 12px; }
.risk-value { margin-top: 6px; font-size: 18px; font-weight: 700; }
.risk-list { margin-top: 8px; display: flex; flex-direction: column; gap: 5px; }
.risk-line { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.events {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 4px 0;
  max-height: 260px;
  overflow: auto;
}
.event { display: grid; grid-template-columns: 82px 170px 1fr; gap: 10px; padding: 6px 10px; border-bottom: 1px solid var(--line); }
.event:last-child { border-bottom: 0; }
.viewer-backdrop {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(0, 0, 0, 0.72);
}
.viewer-backdrop.open { display: flex; }
.viewer {
  width: min(1120px, 96vw);
  max-height: 92vh;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.55);
}
.viewer-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  background: var(--panel2);
}
.viewer-body {
  display: grid;
  grid-template-columns: 42px minmax(0, 1fr) 42px;
  align-items: center;
  gap: 8px;
  padding: 8px;
  background: #0d1117;
}
.viewer-close {
  flex: none;
  width: 30px;
  height: 30px;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: #30363d;
  color: var(--text);
  cursor: pointer;
  font: inherit;
}
.viewer video { width: 100%; max-height: 74vh; display: block; background: #000; }
.viewer-nav { height: 54px; }
.viewer-meta { padding: 8px 12px 10px; color: var(--muted); border-top: 1px solid var(--line); overflow-wrap: anywhere; }
.controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
select {
  appearance: none;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel2);
  color: var(--text);
  padding: 5px 28px 5px 8px;
  font: inherit;
}
@media (max-width: 1200px) {
  .overview-grid { grid-template-columns: repeat(3, minmax(140px, 1fr)); }
  .risk-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
  .stage-grid { grid-template-columns: repeat(2, minmax(210px, 1fr)); }
}
@media (max-width: 760px) {
  header { align-items: flex-start; flex-direction: column; }
  .overview-grid, .risk-grid { grid-template-columns: 1fr; }
  .stage-grid { grid-template-columns: 1fr; }
  .section-title { align-items: flex-start; flex-direction: column; }
  table { font-size: 12px; }
  th, td { padding: 7px 8px; }
  .event { grid-template-columns: 74px 1fr; }
  .event .detail { grid-column: 1 / -1; }
  .overlay-carousel { width: 210px; grid-template-columns: 26px minmax(120px, 158px) 26px; }
  .viewer-body { grid-template-columns: 34px minmax(0, 1fr) 34px; }
}
</style>
</head>
<body>
<header>
  <div>
    <h1>GoTrack Episode Scheduler</h1>
    <div class="sub mono" id="subtitle">loading</div>
  </div>
  <div class="sub mono" id="clock">-</div>
</header>
<main>
  <div class="section-title">
    <div>
      <h2>Queue Overview</h2>
      <div class="sub" id="summaryMeta">schedule</div>
    </div>
  </div>
  <div class="overview-grid">
    <div class="panel"><div class="metric-label">Completion</div><div class="metric-value" id="completion">-</div><div class="bar" style="margin-top:8px"><div id="completionBar"></div></div></div>
    <div class="panel"><div class="metric-label">Episodes</div><div class="metric-value" id="episodes">-</div><div class="metric-small" id="episodeCounts">-</div></div>
    <div class="panel"><div class="metric-label">Active PCs</div><div class="metric-value" id="activePcs">-</div><div class="metric-small" id="pcCounts">-</div></div>
    <div class="panel"><div class="metric-label">Global ETA</div><div class="metric-value" id="globalEta">-</div><div class="metric-small" id="globalEtaDetail">-</div></div>
    <div class="panel"><div class="metric-label">Throughput</div><div class="metric-value" id="throughput">-</div><div class="metric-small" id="throughputDetail">episodes/hour</div></div>
    <div class="panel"><div class="metric-label">Overlay Videos</div><div class="metric-value" id="overlayVideos">-</div><div class="metric-small" id="overlayDetail">-</div></div>
  </div>

  <div class="section-title">
    <div>
      <h2>Completed Overlay Playback</h2>
      <div class="sub" id="completedOverlayMeta">completed episodes with overlay videos</div>
    </div>
    <span class="sub" id="completedOverlayHint">latest first</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Status</th><th>Object</th><th>Episode</th><th>Overlay Playback</th><th>Worker</th><th>Runtime</th><th>Output</th></tr></thead>
      <tbody id="completedOverlayRows"><tr><td colspan="7" class="muted">loading</td></tr></tbody>
    </table>
  </div>

  <div class="section-title">
    <div>
      <h2>Stage Flow</h2>
      <div class="sub">episode lifecycle and current distribution</div>
    </div>
    <span class="sub" id="stageMeta">stages</span>
  </div>
  <div class="stage-grid" id="stageFlow"></div>

  <div class="section-title">
    <div>
      <h2>PC ETA</h2>
      <div class="sub">capture worker state and current episode forecast</div>
    </div>
    <span class="sub" id="workerMeta">workers</span>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>PC</th><th>State</th><th>Current Episode</th><th>Stage</th><th>Progress</th><th>Current ETA</th><th>Done / Failed</th><th>Avg Runtime</th><th>Last Update</th></tr></thead>
      <tbody id="workerRows"><tr><td colspan="9" class="muted">loading</td></tr></tbody>
    </table>
  </div>

  <div class="section-title">
    <div>
      <h2>Bottlenecks</h2>
      <div class="sub">failed, stale, slow, and missing output signals</div>
    </div>
  </div>
  <div class="risk-grid" id="riskPanels"></div>

  <div class="section-title">
    <div>
      <h2>Episode Work Queue</h2>
      <div class="sub" id="queueMeta">tasks</div>
    </div>
    <div class="controls">
      <select id="queueFilter" aria-label="Task queue filter">
        <option value="active">Active + Pending</option>
        <option value="playback">Completed With Overlay</option>
        <option value="completed">All Completed</option>
        <option value="failed">Failed</option>
        <option value="all">All Tasks</option>
      </select>
    </div>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Status</th><th>Object</th><th>Episode</th><th>Stage</th><th>Progress</th><th>Worker</th><th>Task ETA</th><th>Overlay Playback</th><th>Runtime</th><th>Output</th></tr></thead>
      <tbody id="taskRows"><tr><td colspan="10" class="muted">loading</td></tr></tbody>
    </table>
  </div>

  <div class="section-title">
    <div>
      <h2>Scheduler Events</h2>
      <div class="sub">events.jsonl tail</div>
    </div>
  </div>
  <div class="events mono" id="events"></div>
</main>
<div class="viewer-backdrop" id="overlayViewer" aria-hidden="true">
  <div class="viewer" role="dialog" aria-modal="true" aria-labelledby="overlayViewerTitle">
    <div class="viewer-head">
      <div>
        <h2 id="overlayViewerTitle">Overlay Playback</h2>
        <div class="sub mono" id="overlayViewerSub">-</div>
      </div>
      <button class="viewer-close" type="button" id="overlayClose" aria-label="Close overlay video">x</button>
    </div>
    <div class="viewer-body">
      <button class="viewer-nav" type="button" id="overlayPrev" aria-label="Previous overlay video">&lt;</button>
      <video id="overlayVideo" controls playsinline></video>
      <button class="viewer-nav" type="button" id="overlayNext" aria-label="Next overlay video">&gt;</button>
    </div>
    <div class="viewer-meta mono" id="overlayMeta">-</div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const safe = v => (v === undefined || v === null || v === '') ? '-' : String(v);
const esc = s => safe(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const statusClass = s => safe(s).replace(/[^a-zA-Z0-9_]/g, '_');
const doneStatuses = new Set(['done', 'skipped_done', 'dry_run_done']);
const activeStatuses = new Set(['running', 'claimed']);
const failedStatuses = new Set(['failed']);
let lastData = null;
let overlayGroups = {};
let overlayIndexes = {};
let activeOverlayGroup = null;
let statusFetchInFlight = false;
let completedOverlaySignature = '';
const fmtTime = ts => ts ? new Date(Number(ts) * 1000).toLocaleTimeString() : '-';
const asNumber = value => {
  if (value === undefined || value === null || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
};
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));
const fmtDuration = seconds => {
  if (seconds === undefined || seconds === null || seconds === '') return '-';
  const n = Number(seconds);
  if (!Number.isFinite(n)) return '-';
  const s = Math.max(0, Math.round(n));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(r).padStart(2, '0')}s`;
  return `${r}s`;
};
const fmtAge = (ts, now) => {
  const n = asNumber(ts);
  return n === null ? '-' : `${fmtDuration(Math.max(0, now - n))} ago`;
};

function statusBadge(status) {
  const value = status || '-';
  return `<span class="status ${statusClass(value)}"><span class="dot"></span>${esc(value)}</span>`;
}

function overlayCell(task) {
  const count = Number(task.overlay_file_count || (task.overlay_files || []).length || 0);
  if (!count) return '<span class="muted">-</span>';
  const group = safe(task.task_id || `${safe(task.obj)}__${safe(task.episode)}`);
  if ((task.overlay_files || []).length) overlayGroups[group] = (task.overlay_files || []).filter(Boolean);
  const files = overlayGroups[group] || [];
  const rawIndex = Number.isInteger(overlayIndexes[group]) ? overlayIndexes[group] : 0;
  const index = files.length ? clamp(rawIndex, 0, files.length - 1) : 0;
  overlayIndexes[group] = index;
  const path = files[index] || task.overlay_output_dir || '';
  const serial = files.length ? serialFromPath(path) : 'click to load';
  const disabled = count < 2 ? ' disabled' : '';
  return `<div class="overlay-carousel">
    <button type="button" class="overlay-step" data-overlay-group="${esc(group)}" data-overlay-delta="-1" aria-label="Previous overlay video"${disabled}>&lt;</button>
    <button type="button" class="overlay-thumb-button" data-overlay-open="1" data-overlay-group="${esc(group)}" data-overlay-task="${esc(task.task_id || '')}" title="${esc(path)}">
      <span class="overlay-caption">${esc(index + 1)}/${esc(count)} ${esc(serial)}</span>
    </button>
    <button type="button" class="overlay-step" data-overlay-group="${esc(group)}" data-overlay-delta="1" aria-label="Next overlay video"${disabled}>&gt;</button>
  </div>`;
}

function progressRatio(task) {
  const direct = asNumber(task.progress_ratio);
  if (direct !== null) return clamp(direct, 0, 1);
  const done = asNumber(task.frame_done);
  const total = asNumber(task.frame_total);
  if (done !== null && total && total > 0) return clamp(done / total, 0, 1);
  if (doneStatuses.has(String(task.status || ''))) return 1;
  return null;
}

function progressCell(task) {
  const ratio = progressRatio(task);
  const pct = ratio === null ? 0 : ratio * 100;
  const done = asNumber(task.frame_done);
  const total = asNumber(task.frame_total);
  const pctText = ratio === null ? '-' : `${pct >= 99.5 ? 100 : pct.toFixed(pct > 0 && pct < 10 ? 1 : 0)}%`;
  const frameText = done !== null && total ? `${done}/${total}` : safe(task.phase);
  const warn = failedStatuses.has(String(task.status || '')) ? ' warn' : '';
  return `<div class="progress-cell"><span class="bar slim${warn}"><div style="width:${pct}%"></div></span><span class="mono pct">${esc(pctText)}</span><span class="muted mono">${esc(frameText)}</span></div>`;
}

function runtimeOf(task) {
  const runtime = asNumber(task.runtime_sec);
  if (runtime !== null) return runtime;
  const start = asNumber(task.started_at);
  const finish = asNumber(task.finished_at);
  if (start !== null && finish !== null && finish >= start) return finish - start;
  return null;
}

function taskStart(task) {
  const started = asNumber(task.started_at);
  if (started !== null) return started;
  return asNumber(task.claimed_at);
}

function taskElapsed(task, now) {
  const started = taskStart(task);
  return started === null ? null : Math.max(0, now - started);
}

function taskEta(task, globalAvg, workerAvg, now) {
  const status = String(task.status || '');
  if (doneStatuses.has(status)) return 0;
  if (!activeStatuses.has(status)) return null;
  const elapsed = taskElapsed(task, now);
  if (elapsed === null) return null;
  const ratio = progressRatio(task);
  if (ratio !== null && ratio > 0.02 && ratio < 0.995) {
    return Math.max(0, elapsed * (1 - ratio) / ratio);
  }
  const avg = workerAvg || globalAvg;
  return avg ? Math.max(0, avg - elapsed) : null;
}

function buildRuntimeStats(tasks) {
  const byWorker = new Map();
  let runtimeTotal = 0;
  let runtimeCount = 0;
  for (const task of tasks) {
    const worker = safe(task.worker_id);
    if (worker === '-') continue;
    if (!byWorker.has(worker)) byWorker.set(worker, {done: 0, failed: 0, runtimeTotal: 0, runtimeCount: 0});
    const entry = byWorker.get(worker);
    const status = String(task.status || '');
    if (doneStatuses.has(status)) entry.done += 1;
    if (failedStatuses.has(status)) entry.failed += 1;
    const runtime = runtimeOf(task);
    if (runtime !== null && doneStatuses.has(status)) {
      entry.runtimeTotal += runtime;
      entry.runtimeCount += 1;
      runtimeTotal += runtime;
      runtimeCount += 1;
    }
  }
  return {
    byWorker,
    globalAvg: runtimeCount ? runtimeTotal / runtimeCount : null,
    finishedRuntimeCount: runtimeCount
  };
}

function workerSummaries(data, tasks, stats, now) {
  const workerRecords = new Map((data.workers || []).filter(Boolean).map(w => [safe(w.worker_id), w]));
  const ids = new Set(data.expected_pcs || []);
  for (const id of workerRecords.keys()) if (id !== '-') ids.add(id);
  for (const task of tasks) {
    const worker = safe(task.worker_id);
    if (worker !== '-') ids.add(worker);
  }
  return Array.from(ids).sort().map(id => {
    const record = workerRecords.get(id) || {worker_id: id, status: 'not_started'};
    let current = null;
    if (record.task_id) current = tasks.find(t => safe(t.task_id) === safe(record.task_id)) || null;
    if (!current) current = tasks.find(t => safe(t.worker_id) === id && activeStatuses.has(String(t.status || ''))) || null;
    const stat = stats.byWorker.get(id) || {done: 0, failed: 0, runtimeTotal: 0, runtimeCount: 0};
    const workerAvg = stat.runtimeCount ? stat.runtimeTotal / stat.runtimeCount : null;
    const updatedAt = asNumber(record.updated_at);
    const stale = updatedAt !== null && (now - updatedAt > 120) && (String(record.status || '') === 'running' || Boolean(current));
    const state = stale ? 'stale' : (record.status || (current ? 'running' : 'not_started'));
    return {
      id,
      record,
      state,
      current,
      stat,
      workerAvg,
      currentEta: current ? taskEta(current, stats.globalAvg, workerAvg, now) : null,
      stale,
      updatedAt
    };
  });
}

function renderRiskList(items, renderItem, emptyText) {
  if (!items.length) return `<div class="risk-list"><div class="risk-line muted">${esc(emptyText)}</div></div>`;
  return `<div class="risk-list">${items.slice(0, 4).map(renderItem).join('')}</div>`;
}

function renderRisks(tasks, workers, stats, manifest, now) {
  const failed = tasks.filter(t => failedStatuses.has(String(t.status || '')));
  const staleWorkers = workers.filter(w => w.stale);
  const runningTasks = tasks.filter(t => activeStatuses.has(String(t.status || '')));
  const longest = runningTasks.slice().sort((a, b) => (taskElapsed(b, now) || 0) - (taskElapsed(a, now) || 0))[0] || null;
  const doneTasks = tasks.filter(t => doneStatuses.has(String(t.status || '')));
  const overlayRequired = String(manifest.stages || 'both') !== 'gotrack';
  const missingOverlay = overlayRequired ? doneTasks.filter(t => !t.overlay_done || !Number(t.overlay_file_count || 0)) : [];
  const overlayFiles = tasks.reduce((n, t) => n + Number(t.overlay_file_count || 0), 0);
  const activeWorkerCount = workers.filter(w => w.state === 'running').length;
  const panels = [
    `<div class="panel"><div class="risk-title"><span>Failed Episodes</span><span class="${failed.length ? 'bad' : 'ok'}">${failed.length}</span></div><div class="risk-value">${failed.length ? 'Needs retry' : 'Clear'}</div>${renderRiskList(failed, t => `<div class="risk-line mono" title="${esc(t.reason || '')}">${esc(t.obj)}/${esc(t.episode)} ${esc(t.reason || '')}</div>`, 'no failed tasks')}</div>`,
    `<div class="panel"><div class="risk-title"><span>Stale PCs</span><span class="${staleWorkers.length ? 'bad' : 'ok'}">${staleWorkers.length}</span></div><div class="risk-value">${activeWorkerCount}/${workers.length} active</div>${renderRiskList(staleWorkers, w => `<div class="risk-line mono">${esc(w.id)} updated ${esc(fmtAge(w.updatedAt, now))}</div>`, 'no stale workers')}</div>`,
    `<div class="panel"><div class="risk-title"><span>Longest Active</span><span>${runningTasks.length}</span></div><div class="risk-value">${longest ? esc(fmtDuration(taskElapsed(longest, now))) : '-'}</div>${longest ? `<div class="risk-list"><div class="risk-line mono">${esc(longest.obj)}/${esc(longest.episode)}</div><div class="risk-line muted">worker ${esc(longest.worker_id)} · ETA ${esc(fmtDuration(taskEta(longest, stats.globalAvg, null, now)))}</div></div>` : '<div class="risk-list"><div class="risk-line muted">no running tasks</div></div>'}</div>`,
    `<div class="panel"><div class="risk-title"><span>Overlay Readiness</span><span>${overlayFiles}</span></div><div class="risk-value">${missingOverlay.length ? `${missingOverlay.length} missing` : 'Ready'}</div>${renderRiskList(missingOverlay, t => `<div class="risk-line mono">${esc(t.obj)}/${esc(t.episode)}</div>`, overlayRequired ? 'all completed overlays found' : 'overlay stage disabled')}</div>`
  ];
  $('riskPanels').innerHTML = panels.join('');
}

function renderStageFlow(data, tasks) {
  const defs = data.stage_definitions || [];
  const phaseCounts = (data.stage_summary && data.stage_summary.phase_counts) || {};
  const statusCounts = (data.stage_summary && data.stage_summary.status_counts) || {};
  const activePhase = new Set(tasks.filter(t => activeStatuses.has(String(t.status || ''))).map(t => String(t.phase || '')));
  $('stageMeta').textContent = `mode ${safe((data.manifest || {}).stages || 'both')} · skipped ${statusCounts.skipped_done || 0}`;
  $('stageFlow').innerHTML = defs.map(def => {
    const key = String(def.key || '');
    const count = phaseCounts[key] || 0;
    const active = count > 0 || activePhase.has(key) ? ' active' : '';
    return `<div class="stage-card${active}">
      <div class="stage-head"><span class="stage-name">${esc(def.label || key)}</span><span class="stage-count">${esc(count)}</span></div>
      <div class="stage-desc">${esc(def.description || '')}</div>
      <div class="stage-output mono">${esc(def.output || '')}</div>
    </div>`;
  }).join('');
}

function taskRecency(task) {
  return asNumber(task.finished_at) || asNumber(task.updated_at) || asNumber(task.started_at) || asNumber(task.created_at) || 0;
}

function sortTasks(tasks) {
  const order = {running: 0, failed: 1, pending: 2, done: 3, skipped_done: 4, dry_run_done: 5};
  return tasks.slice().sort((a, b) => {
    const statusOrder = (order[a.status] ?? 9) - (order[b.status] ?? 9);
    if (statusOrder) return statusOrder;
    const recent = taskRecency(b) - taskRecency(a);
    if (recent) return recent;
    return safe(a.obj).localeCompare(safe(b.obj)) || safe(a.episode).localeCompare(safe(b.episode));
  });
}

function completedOverlayTasks(tasks) {
  return tasks
    .filter(t => doneStatuses.has(String(t.status || '')) && Number(t.overlay_file_count || 0) > 0)
    .slice()
    .sort((a, b) => taskRecency(b) - taskRecency(a) || safe(a.obj).localeCompare(safe(b.obj)) || safe(a.episode).localeCompare(safe(b.episode)));
}

function filterQueueTasks(tasks, mode) {
  if (mode === 'all') return tasks;
  if (mode === 'failed') return tasks.filter(t => failedStatuses.has(String(t.status || '')));
  if (mode === 'completed') return tasks.filter(t => doneStatuses.has(String(t.status || '')));
  if (mode === 'playback') return completedOverlayTasks(tasks);
  return tasks.filter(t => activeStatuses.has(String(t.status || '')) || String(t.status || '') === 'pending');
}

function renderCompletedOverlay(tasks) {
  const playback = completedOverlayTasks(tasks);
  $('completedOverlayMeta').textContent = `${playback.length} completed episodes have overlay playback`;
  $('completedOverlayHint').textContent = 'all completed episodes';
  const signature = playback.map(t => `${safe(t.task_id)}:${safe(t.status)}:${safe(t.overlay_file_count)}:${safe(t.updated_at)}`).join('|');
  if (signature === completedOverlaySignature) return;
  completedOverlaySignature = signature;
  $('completedOverlayRows').innerHTML = playback.length ? playback.map(t => {
    const output = t.overlay_output_dir || '-';
    return `<tr>
      <td>${statusBadge(t.status || '-')}</td>
      <td>${esc(t.obj)}</td>
      <td class="mono">${esc(t.episode)}</td>
      <td>${overlayCell(t)}</td>
      <td class="mono">${esc(t.worker_id)}</td>
      <td class="mono">${esc(t.runtime_sec ? fmtDuration(t.runtime_sec) : '-')}</td>
      <td class="mono path">${esc(output)}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="7" class="muted">no completed overlay videos found for this schedule</td></tr>';
}

function render(data) {
  lastData = data;
  const manifest = data.manifest || {};
  const counts = data.counts || {};
  const now = asNumber(data.now) || Date.now() / 1000;
  const tasks = sortTasks(data.tasks || []);
  const done = Number(data.done_like || 0);
  const total = Number(data.n_tasks || tasks.length || 0);
  const pct = total ? clamp(100 * done / total, 0, 100) : 0;
  const running = counts.running || 0;
  const pending = counts.pending || 0;
  const failed = counts.failed || 0;
  const skipped = counts.skipped_done || 0;
  const overlayFiles = Number(data.overlay_file_count || tasks.reduce((n, t) => n + Number(t.overlay_file_count || 0), 0));
  const overlayEpisodes = Number(data.overlay_episode_count || tasks.filter(t => Number(t.overlay_file_count || 0) > 0).length);
  const stats = buildRuntimeStats(tasks);
  const workers = workerSummaries(data, tasks, stats, now);
  const activeWorkers = workers.filter(w => w.state === 'running').length;
  const staleWorkers = workers.filter(w => w.stale).length;

  $('clock').textContent = new Date().toLocaleTimeString();
  $('subtitle').textContent = data.schedule_dir || '-';
  $('summaryMeta').textContent = `${manifest.schedule_id || '-'} · ${manifest.hand || '-'} · ${manifest.stages || 'both'}`;
  $('completion').textContent = pct.toFixed(1) + '%';
  $('completionBar').style.width = pct + '%';
  $('episodes').textContent = `${done}/${total}`;
  $('episodeCounts').textContent = `running=${running} pending=${pending} failed=${failed} skipped=${skipped}`;
  $('activePcs').textContent = `${activeWorkers}/${workers.length}`;
  $('pcCounts').textContent = staleWorkers ? `stale=${staleWorkers}` : 'no stale workers';
  $('globalEta').textContent = data.eta_sec === null || data.eta_sec === undefined ? '-' : fmtDuration(data.eta_sec);
  $('globalEtaDetail').textContent = `remaining ${data.remaining ?? '-'} · elapsed ${fmtDuration(data.elapsed_sec)}`;
  $('throughput').textContent = Number(data.throughput_eps_per_hour || 0).toFixed(2);
  $('throughputDetail').textContent = stats.globalAvg ? `avg ${fmtDuration(stats.globalAvg)} / episode` : 'waiting for first completion';
  $('overlayVideos').textContent = String(overlayFiles);
  $('overlayDetail').textContent = `${overlayEpisodes} episodes with playback`;

  renderCompletedOverlay(tasks);
  renderStageFlow(data, tasks);

  $('workerMeta').textContent = `${workers.length} PCs · current-task ETA`;
  $('workerRows').innerHTML = workers.length ? workers.map(w => {
    const current = w.current;
    const stage = current ? safe(current.phase) : safe(w.record.reason || w.record.last_status || '-');
    const currentLabel = current
      ? `<div class="episode-cell"><span>${esc(current.obj)}</span><span class="mono">${esc(current.episode)}</span><span class="task-id muted mono" title="${esc(current.task_id)}">${esc(current.task_id)}</span></div>`
      : '<span class="muted">-</span>';
    const eta = w.currentEta === null ? (current ? 'warming up' : '-') : fmtDuration(w.currentEta);
    const doneFailed = `${w.stat.done}${w.stat.failed ? ` / ${w.stat.failed} failed` : ''}`;
    const ageClass = w.stale ? 'bad' : 'muted';
    return `<tr>
      <td class="mono">${esc(w.id)}</td>
      <td>${statusBadge(w.state)}</td>
      <td>${currentLabel}</td>
      <td class="mono">${esc(stage)}</td>
      <td>${current ? progressCell(current) : '<span class="muted">-</span>'}</td>
      <td class="mono">${esc(eta)}</td>
      <td class="mono">${esc(doneFailed)}</td>
      <td class="mono">${esc(w.workerAvg ? fmtDuration(w.workerAvg) : '-')}</td>
      <td class="mono ${ageClass}">${esc(fmtAge(w.updatedAt, now))}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="9" class="muted">no workers yet</td></tr>';

  renderRisks(tasks, workers, stats, manifest, now);

  const queueMode = $('queueFilter') ? $('queueFilter').value : 'active';
  const queueTasks = filterQueueTasks(tasks, queueMode);
  $('queueMeta').textContent = `${queueTasks.length}/${tasks.length} matched · showing first 160`;
  $('taskRows').innerHTML = queueTasks.length ? queueTasks.slice(0, 160).map(t => {
    const status = t.status || '-';
    const output = t.overlay_output_dir || (t.episode_dir ? t.episode_dir + '/object_tracking/gotrack_output' : '-');
    const worker = safe(t.worker_id);
    const workerAvg = worker !== '-' && stats.byWorker.has(worker) && stats.byWorker.get(worker).runtimeCount
      ? stats.byWorker.get(worker).runtimeTotal / stats.byWorker.get(worker).runtimeCount
      : null;
    const eta = taskEta(t, stats.globalAvg, workerAvg, now);
    const etaText = eta === null ? (activeStatuses.has(String(status)) ? 'warming up' : '-') : fmtDuration(eta);
    return `<tr>
      <td>${statusBadge(status)}</td>
      <td>${esc(t.obj)}</td>
      <td class="mono">${esc(t.episode)}</td>
      <td class="mono">${esc(t.phase)}</td>
      <td>${progressCell(t)}</td>
      <td class="mono">${esc(t.worker_id)}</td>
      <td class="mono">${esc(etaText)}</td>
      <td>${overlayCell(t)}</td>
      <td class="mono">${esc(t.runtime_sec ? fmtDuration(t.runtime_sec) : '-')}</td>
      <td class="mono path">${esc(output)}</td>
    </tr>`;
  }).join('') : '<tr><td colspan="10" class="muted">no tasks</td></tr>';

  const events = data.events || [];
  $('events').innerHTML = events.length ? events.slice().reverse().map(e => {
    const detail = {...e};
    delete detail.ts;
    delete detail.event;
    return `<div class="event">
      <div class="muted">${esc(fmtTime(e.ts))}</div>
      <div>${esc(e.event)}</div>
      <div class="detail muted">${esc(JSON.stringify(detail))}</div>
    </div>`;
  }).join('') : '<div class="event"><div class="muted">-</div><div>no events</div><div></div></div>';
}

function serialFromPath(path) {
  const name = (path || '').split('/').pop() || 'overlay.mp4';
  return name.replace(/^overlay_/, '').replace(/\.mp4$/i, '');
}

async function ensureOverlayGroup(group, taskId) {
  if ((overlayGroups[group] || []).length) return overlayGroups[group];
  if (!taskId) return [];
  const res = await fetch('/api/overlay-files?task_id=' + encodeURIComponent(taskId), {cache: 'no-store'});
  if (!res.ok) throw new Error(await res.text());
  const payload = await res.json();
  overlayGroups[group] = (payload.overlay_files || []).filter(Boolean);
  return overlayGroups[group];
}

async function openOverlayVideo(group, index, taskId) {
  const loaded = await ensureOverlayGroup(group, taskId);
  if (loaded.length && lastData) render(lastData);
  const files = overlayGroups[group] || [];
  if (!files.length) return;
  const n = files.length;
  const nextIndex = ((Number(index) || 0) % n + n) % n;
  const path = files[nextIndex];
  const serial = serialFromPath(path);
  activeOverlayGroup = group;
  overlayIndexes[group] = nextIndex;
  $('overlayViewerTitle').textContent = `Overlay ${nextIndex + 1}/${n} ${serial}`;
  $('overlayViewerSub').textContent = path || '-';
  $('overlayMeta').textContent = path || '-';
  $('overlayPrev').disabled = n < 2;
  $('overlayNext').disabled = n < 2;
  const video = $('overlayVideo');
  video.pause();
  video.src = '/overlay?path=' + encodeURIComponent(path);
  video.load();
  $('overlayViewer').classList.add('open');
  $('overlayViewer').setAttribute('aria-hidden', 'false');
  const play = video.play();
  if (play && play.catch) play.catch(() => {});
}

function stepOverlayGroup(group, delta, openViewer) {
  const files = overlayGroups[group] || [];
  if (!files.length) return;
  const current = Number.isInteger(overlayIndexes[group]) ? overlayIndexes[group] : 0;
  const next = ((current + Number(delta || 0)) % files.length + files.length) % files.length;
  overlayIndexes[group] = next;
  if (openViewer) {
    openOverlayVideo(group, next).catch(err => {
      $('subtitle').textContent = 'overlay load error: ' + err;
    });
  } else if (lastData) {
    render(lastData);
  }
}

function stepActiveOverlay(delta) {
  if (!activeOverlayGroup) return;
  stepOverlayGroup(activeOverlayGroup, delta, true);
}

function closeOverlayVideo() {
  const video = $('overlayVideo');
  video.pause();
  video.removeAttribute('src');
  video.load();
  $('overlayViewer').classList.remove('open');
  $('overlayViewer').setAttribute('aria-hidden', 'true');
  activeOverlayGroup = null;
}

async function tick() {
  if (statusFetchInFlight) return;
  statusFetchInFlight = true;
  try {
    const res = await fetch('/api/status', {cache: 'no-store'});
    render(await res.json());
  } catch (err) {
    $('subtitle').textContent = 'fetch error: ' + err;
  } finally {
    statusFetchInFlight = false;
  }
}

document.addEventListener('click', ev => {
  const target = ev.target instanceof Element ? ev.target : null;
  const step = target ? target.closest('[data-overlay-delta]') : null;
  if (step) {
    ev.preventDefault();
    stepOverlayGroup(step.dataset.overlayGroup, Number(step.dataset.overlayDelta), false);
    return;
  }
  const open = target ? target.closest('[data-overlay-open]') : null;
  if (open) {
    ev.preventDefault();
    const group = open.dataset.overlayGroup;
    const taskId = open.dataset.overlayTask;
    const index = Number.isInteger(overlayIndexes[group]) ? overlayIndexes[group] : 0;
    openOverlayVideo(group, index, taskId).catch(err => {
      $('subtitle').textContent = 'overlay load error: ' + err;
    });
    return;
  }
  if (target === $('overlayPrev')) {
    ev.preventDefault();
    stepActiveOverlay(-1);
    return;
  }
  if (target === $('overlayNext')) {
    ev.preventDefault();
    stepActiveOverlay(1);
    return;
  }
  if (target === $('overlayViewer') || target === $('overlayClose')) closeOverlayVideo();
});

document.addEventListener('keydown', ev => {
  if (ev.key === 'Escape' && $('overlayViewer').classList.contains('open')) closeOverlayVideo();
  if (ev.key === 'ArrowLeft' && $('overlayViewer').classList.contains('open')) stepActiveOverlay(-1);
  if (ev.key === 'ArrowRight' && $('overlayViewer').classList.contains('open')) stepActiveOverlay(1);
});

$('queueFilter').addEventListener('change', () => {
  if (lastData) render(lastData);
});

tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


def make_handler(schedule_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/status":
                body = json.dumps(collect(schedule_dir), sort_keys=True, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/overlay-files":
                task_id = parse_qs(parsed.query).get("task_id", [""])[0]
                if not task_id:
                    self.send_error(400, "missing task_id")
                    return
                try:
                    payload = overlay_payload_for_task(schedule_dir, task_id)
                except KeyError:
                    self.send_error(404, "task not found")
                    return
                body = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/overlay":
                self._serve_overlay(parsed)
                return
            if parsed.path == "/overlay-thumb":
                self._serve_overlay_thumb(parsed)
                return
            if parsed.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_overlay(self, parsed) -> None:
            target = parse_qs(parsed.query).get("path", [""])[0]
            if not target:
                self.send_error(400, "missing path")
                return
            try:
                requested = Path(target).expanduser().resolve()
            except OSError:
                self.send_error(400, "invalid path")
                return
            video = allowed_overlay_path(schedule_dir, requested)
            if video is None:
                self.send_error(403, "overlay file is not in this schedule")
                return
            if not video.is_file():
                self.send_error(404, "overlay video not found")
                return
            size = video.stat().st_size
            start, end, status_code = 0, size - 1, 200
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    unit, spec = range_header.split("=", 1)
                    if unit.strip().lower() != "bytes":
                        raise ValueError("unsupported unit")
                    spec = spec.split(",", 1)[0].strip()
                    if spec.startswith("-"):
                        start = max(0, size - int(spec[1:]))
                    else:
                        left, _, right = spec.partition("-")
                        start = int(left)
                        if right:
                            end = int(right)
                    end = min(end, size - 1)
                    if start < 0 or start > end:
                        raise ValueError("invalid range")
                    status_code = 206
                except Exception:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            content_len = end - start + 1
            self.send_response(status_code)
            self.send_header("Content-Type", mimetypes.guess_type(video.name)[0] or "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(content_len))
            if status_code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with video.open("rb") as f:
                f.seek(start)
                remaining = content_len
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _serve_overlay_thumb(self, parsed) -> None:
            target = parse_qs(parsed.query).get("path", [""])[0]
            if not target:
                self.send_error(400, "missing path")
                return
            try:
                requested = Path(target).expanduser().resolve()
            except OSError:
                self.send_error(400, "invalid path")
                return
            video = allowed_overlay_path(schedule_dir, requested)
            if video is None:
                self.send_error(403, "overlay file is not in this schedule")
                return
            if not video.is_file():
                self.send_error(404, "overlay video not found")
                return
            try:
                thumb = ensure_overlay_thumbnail(schedule_dir, video)
            except Exception as exc:
                self.send_error(500, f"thumbnail generation failed: {exc}")
                return
            body = thumb.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    return Handler


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--schedule-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--open-browser", action="store_true")
    args = p.parse_args()

    schedule_dir = Path(args.schedule_dir).expanduser()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(schedule_dir))
    url = f"http://{args.host}:{args.port}/"
    print(url, flush=True)
    print(f"reading {schedule_dir}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
