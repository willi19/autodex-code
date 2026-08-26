#!/usr/bin/env python3
"""Serve a live dashboard for interactive 3D batch export logs."""
from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


DEFAULT_LOG_ROOT = Path.home() / "shared_data" / "AutoDex" / "interactive_3d" / "_batch_logs"


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoDex Interactive 3D Export Monitor</title>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
      }
    }
  </script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f4;
      --surface: #ffffff;
      --surface-2: #f0f2ed;
      --line: #d9ded3;
      --line-strong: #bdc6b8;
      --text: #222621;
      --muted: #687067;
      --ok: #2f7d55;
      --ok-bg: #e6f2eb;
      --fail: #b2463f;
      --fail-bg: #f7e7e4;
      --accent: #315f6a;
      --warn: #946620;
      --shadow: 0 12px 28px rgba(30, 35, 27, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
      padding: 18px 22px 14px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, .88);
      backdrop-filter: blur(8px);
      position: sticky;
      top: 0;
      z-index: 20;
    }
    h1 {
      margin: 0;
      font-size: clamp(20px, 2.4vw, 32px);
      line-height: 1.05;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 5px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 680;
    }
    .header-right {
      display: grid;
      gap: 5px;
      justify-items: end;
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
    }
    .progress-wrap {
      padding: 14px 22px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }
    .progress-meta {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 780;
    }
    .progress {
      height: 14px;
      overflow: hidden;
      border-radius: 999px;
      background: var(--surface-2);
      border: 1px solid var(--line);
    }
    .bar {
      width: 0%;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--ok));
      transition: width .25s ease;
    }
    main {
      display: grid;
      grid-template-columns: 330px minmax(0, 1fr);
      gap: 14px;
      padding: 14px 22px 22px;
      min-height: 0;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .side {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 10px;
    }
    .stat {
      padding: 11px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fbfcfa;
    }
    .stat .label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 850;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .stat .value {
      margin-top: 4px;
      font-size: 23px;
      line-height: 1;
      font-weight: 850;
      font-variant-numeric: tabular-nums;
    }
    .stat.ok .value { color: var(--ok); }
    .stat.fail .value { color: var(--fail); }
    .tools {
      display: grid;
      gap: 8px;
      padding: 10px;
      border-top: 1px solid var(--line);
    }
    .search {
      width: 100%;
      height: 36px;
      padding: 0 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--text);
      background: #fff;
      outline: none;
      font: inherit;
      font-size: 13px;
    }
    .filters {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;
    }
    .filters button {
      min-height: 32px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 850;
      cursor: pointer;
    }
    .filters button.active {
      border-color: var(--accent);
      color: var(--text);
      background: #edf4f5;
    }
    .reasons {
      padding: 10px;
    }
    .reasons h2,
    .list-head h2 {
      margin: 0;
      font-size: 13px;
      line-height: 1.2;
    }
    .reason-list {
      display: grid;
      gap: 7px;
      margin-top: 10px;
    }
    .reason {
      display: grid;
      gap: 3px;
      padding: 9px;
      border-radius: 7px;
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--muted);
      font-size: 12px;
    }
    .reason strong {
      color: var(--fail);
      font-size: 14px;
      font-variant-numeric: tabular-nums;
    }
    .content {
      display: grid;
      gap: 14px;
      min-width: 0;
      min-height: 0;
    }
    .preview {
      overflow: hidden;
    }
    .preview-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
    }
    .preview-head h2 {
      margin: 0;
      font-size: 13px;
      line-height: 1.2;
    }
    .preview-head .path {
      color: var(--muted);
      font-size: 12px;
      font-weight: 780;
      overflow-wrap: anywhere;
      text-align: right;
    }
    .viewer {
      position: relative;
      height: min(42vh, 420px);
      min-height: 280px;
      background: #f7f8f4;
    }
    .viewer canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    .viewer-empty {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      padding: 24px;
      color: var(--muted);
      text-align: center;
      font-size: 13px;
      font-weight: 780;
    }
    .viewer-status {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 2;
      max-width: calc(100% - 20px);
      padding: 7px 9px;
      border: 1px solid rgba(216, 221, 210, .9);
      border-radius: 6px;
      background: rgba(255, 255, 255, .84);
      color: var(--muted);
      font-size: 12px;
      font-weight: 850;
      backdrop-filter: blur(8px);
    }
    .viewer-controls {
      position: absolute;
      left: 10px;
      right: 10px;
      bottom: 10px;
      z-index: 2;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto auto;
      gap: 9px;
      align-items: center;
      padding: 8px;
      border: 1px solid rgba(216, 221, 210, .9);
      border-radius: 7px;
      background: rgba(255, 255, 255, .86);
      backdrop-filter: blur(8px);
    }
    .viewer-controls button,
    .viewer-controls select {
      height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font-size: 12px;
      font-weight: 850;
    }
    .viewer-controls input {
      width: 100%;
      accent-color: var(--accent);
    }
    .viewer-time {
      min-width: 74px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 850;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .list {
      min-height: 0;
      overflow: hidden;
    }
    .list-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .list-head .count {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    .rows {
      height: calc(100vh - 240px);
      overflow: auto;
      padding: 8px;
    }
    .row {
      display: grid;
      grid-template-columns: 78px minmax(0, 1fr) 86px 92px;
      gap: 10px;
      align-items: start;
      padding: 9px 10px;
      border: 1px solid transparent;
      border-radius: 7px;
      color: var(--muted);
      font-size: 12px;
    }
    .row:nth-child(odd) {
      background: #fafbf8;
    }
    .row:hover {
      border-color: var(--line-strong);
      background: #fff;
    }
    .row.clickable {
      cursor: pointer;
    }
    .row.selected {
      border-color: var(--accent);
      background: #eef5f6;
    }
    .badge {
      display: inline-flex;
      min-width: 58px;
      height: 24px;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .04em;
    }
    .badge.ok {
      color: var(--ok);
      background: var(--ok-bg);
    }
    .badge.failed {
      color: var(--fail);
      background: var(--fail-bg);
    }
    .path {
      color: var(--text);
      font-weight: 780;
      overflow-wrap: anywhere;
    }
    .detail {
      margin-top: 3px;
      overflow-wrap: anywhere;
    }
    .mono {
      font-variant-numeric: tabular-nums;
      font-feature-settings: "tnum";
    }
    .empty {
      padding: 30px;
      color: var(--muted);
      text-align: center;
      font-weight: 750;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      .rows { height: auto; max-height: 65vh; }
      .viewer { height: 360px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div>
        <h1>Interactive 3D Export Monitor</h1>
        <div class="sub" id="log-dir">Loading...</div>
      </div>
      <div class="header-right">
        <div id="updated">-</div>
        <div id="eta">ETA -</div>
      </div>
    </header>
    <section class="progress-wrap">
      <div class="progress-meta">
        <span id="progress-text">0 / 0</span>
        <span id="rate">-</span>
      </div>
      <div class="progress"><div class="bar" id="bar"></div></div>
    </section>
    <main>
      <aside class="side">
        <section class="panel">
          <div class="stats">
            <div class="stat"><div class="label">Total</div><div class="value mono" id="total">0</div></div>
            <div class="stat"><div class="label">Done</div><div class="value mono" id="done">0</div></div>
            <div class="stat ok"><div class="label">Success</div><div class="value mono" id="success">0</div></div>
            <div class="stat fail"><div class="label">Failed</div><div class="value mono" id="failed">0</div></div>
          </div>
          <div class="tools">
            <input class="search" id="search" placeholder="Filter path or error">
            <div class="filters">
              <button class="active" data-filter="all">All</button>
              <button data-filter="ok">Success</button>
              <button data-filter="failed">Failed</button>
            </div>
          </div>
        </section>
        <section class="panel reasons">
          <h2>Failure Reasons</h2>
          <div class="reason-list" id="reasons"></div>
        </section>
      </aside>
      <section class="content">
        <section class="panel preview">
          <div class="preview-head">
            <h2>Animated GLB Preview</h2>
            <div class="path" id="preview-path">Select a successful episode</div>
          </div>
          <div class="viewer" id="viewer">
            <div class="viewer-empty" id="viewer-empty">Click a successful row to load and play its animated.glb.</div>
            <div class="viewer-status" id="viewer-status">No asset selected</div>
            <div class="viewer-controls">
              <button id="viewer-play" type="button">Play</button>
              <input id="viewer-scrub" type="range" min="0" max="1000" value="0">
              <div class="viewer-time" id="viewer-time">0.00s</div>
              <select id="viewer-speed">
                <option value="0.5">0.5x</option>
                <option value="1" selected>1x</option>
                <option value="2">2x</option>
                <option value="4">4x</option>
              </select>
            </div>
          </div>
        </section>
        <section class="panel list">
          <div class="list-head">
            <h2>Completed Episodes</h2>
            <div class="count" id="visible-count">0 rows</div>
          </div>
          <div class="rows" id="rows"></div>
        </section>
      </section>
    </main>
  </div>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

    const els = Object.fromEntries([
      'log-dir', 'updated', 'eta', 'progress-text', 'rate', 'bar', 'total', 'done',
      'success', 'failed', 'search', 'reasons', 'rows', 'visible-count', 'preview-path',
      'viewer', 'viewer-empty', 'viewer-status', 'viewer-play', 'viewer-scrub', 'viewer-time',
      'viewer-speed'
    ].map(id => [id, document.getElementById(id)]));
    const state = { data: null, filter: 'all', search: '', selectedAsset: '' };
    const preview = {
      renderer: null,
      scene: null,
      camera: null,
      controls: null,
      mixer: null,
      clock: new THREE.Clock(),
      model: null,
      duration: 0,
      playing: false,
      currentAsset: ''
    };

    document.querySelectorAll('.filters button').forEach(button => {
      button.addEventListener('click', () => {
        state.filter = button.dataset.filter;
        document.querySelectorAll('.filters button').forEach(b => b.classList.toggle('active', b === button));
        renderRows();
      });
    });
    els.search.addEventListener('input', () => {
      state.search = els.search.value.trim().toLowerCase();
      renderRows();
    });
    els['viewer-play'].addEventListener('click', () => {
      preview.playing = !preview.playing;
      preview.clock.getDelta();
      updatePreviewPlayButton();
    });
    els['viewer-scrub'].addEventListener('input', () => {
      setPreviewTime((Number(els['viewer-scrub'].value) || 0) / 1000);
    });
    els['viewer-speed'].addEventListener('change', () => preview.clock.getDelta());

    function fmt(n) {
      return Number(n || 0).toLocaleString();
    }
    function fmtSec(s) {
      if (!Number.isFinite(s) || s < 0) return '-';
      if (s < 60) return `${s.toFixed(0)}s`;
      const m = Math.floor(s / 60);
      const sec = Math.round(s % 60);
      if (m < 60) return `${m}m ${sec}s`;
      const h = Math.floor(m / 60);
      return `${h}h ${m % 60}m`;
    }
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[c]));
    }

    async function load() {
      const response = await fetch('/api/state', { cache: 'no-store' });
      state.data = await response.json();
      render();
    }
    function render() {
      const data = state.data;
      if (!data) return;
      const summary = data.summary || {};
      const total = summary.tasks_total || data.total || 0;
      const done = summary.completed || data.completed || 0;
      const failed = summary.failed || data.failed || 0;
      const success = Math.max(0, done - failed);
      const pct = total ? (done / total) * 100 : 0;
      els['log-dir'].textContent = data.log_dir || '-';
      els.updated.textContent = `Updated ${data.updated_local || '-'}`;
      els.eta.textContent = `ETA ${data.eta_sec == null ? '-' : fmtSec(data.eta_sec)}`;
      els['progress-text'].textContent = `${fmt(done)} / ${fmt(total)} completed (${pct.toFixed(1)}%)`;
      els.rate.textContent = data.rate_per_min ? `${data.rate_per_min.toFixed(1)} episodes/min` : '-';
      els.bar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
      els.total.textContent = fmt(total);
      els.done.textContent = fmt(done);
      els.success.textContent = fmt(success);
      els.failed.textContent = fmt(failed);
      renderReasons();
      renderRows();
      maybeAutoloadPreview();
    }
    function renderReasons() {
      const items = state.data.failure_reasons || [];
      els.reasons.innerHTML = items.length ? items.map(item => `
        <div class="reason">
          <strong>${fmt(item.count)}</strong>
          <span>${escapeHtml(item.reason)}</span>
        </div>
      `).join('') : '<div class="empty">No failures yet</div>';
    }
    function renderRows() {
      const data = state.data;
      if (!data) return;
      const rows = (data.records || []).filter(row => {
        if (state.filter !== 'all' && row.status !== state.filter) return false;
        if (!state.search) return true;
        return [row.relative_path, row.error, row.error_type, row.animated_glb]
          .filter(Boolean)
          .some(value => String(value).toLowerCase().includes(state.search));
      }).reverse();
      els['visible-count'].textContent = `${fmt(rows.length)} rows`;
      if (!rows.length) {
        els.rows.innerHTML = '<div class="empty">No rows match the current filter</div>';
        return;
      }
      els.rows.innerHTML = rows.map(row => {
        const status = row.status || 'unknown';
        const detail = status === 'ok'
          ? `${fmt(row.frames)} frames | ${escapeHtml(row.animated_glb || '')}`
          : `${escapeHtml(row.error_type || 'Error')}: ${escapeHtml(row.error || '')}`;
        const asset = row.animated_glb ? `/asset?path=${encodeURIComponent(row.animated_glb)}` : '';
        return `
          <div class="row ${status === 'ok' ? 'clickable' : ''} ${asset && asset === state.selectedAsset ? 'selected' : ''}" data-asset="${escapeHtml(asset)}" data-path="${escapeHtml(row.relative_path || '')}">
            <div><span class="badge ${escapeHtml(status)}">${escapeHtml(status)}</span></div>
            <div>
              <div class="path">${escapeHtml(row.relative_path || row.episode_root || '-')}</div>
              <div class="detail">${detail}</div>
            </div>
            <div class="mono">${row.elapsed_sec == null ? '-' : `${Number(row.elapsed_sec).toFixed(1)}s`}</div>
            <div class="mono">${row.frames ? `${fmt(row.frames)} fr` : ''}</div>
          </div>
        `;
      }).join('');
      els.rows.querySelectorAll('.row.clickable').forEach(row => {
        row.addEventListener('click', () => {
          const asset = row.dataset.asset;
          if (!asset) return;
          loadPreview(asset, row.dataset.path || '');
        });
      });
    }

    function maybeAutoloadPreview() {
      if (state.selectedAsset || preview.currentAsset) return;
      const records = state.data?.records || [];
      const firstOk = [...records].reverse().find(row => row.status === 'ok' && row.animated_glb);
      if (!firstOk) return;
      const asset = `/asset?path=${encodeURIComponent(firstOk.animated_glb)}`;
      loadPreview(asset, firstOk.relative_path || '').catch(err => {
        els['viewer-status'].textContent = `Preview failed: ${err.message || err}`;
      });
    }

    function initPreview() {
      if (preview.renderer) return;
      preview.scene = new THREE.Scene();
      preview.scene.background = new THREE.Color(0xf7f8f4);
      preview.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 20);
      preview.camera.up.set(0, 0, 1);
      preview.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      preview.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      preview.renderer.outputColorSpace = THREE.SRGBColorSpace;
      preview.renderer.toneMapping = THREE.ACESFilmicToneMapping;
      preview.renderer.toneMappingExposure = 0.95;
      els.viewer.appendChild(preview.renderer.domElement);
      preview.controls = new OrbitControls(preview.camera, preview.renderer.domElement);
      preview.controls.enableDamping = true;
      preview.controls.screenSpacePanning = true;
      preview.scene.add(new THREE.HemisphereLight(0xffffff, 0x9fa69e, 1.8));
      const key = new THREE.DirectionalLight(0xffffff, 2.2);
      key.position.set(1.1, -1.2, 1.8);
      preview.scene.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.8);
      fill.position.set(-0.8, 0.9, 1.0);
      preview.scene.add(fill);
      addGrid();
      window.addEventListener('resize', resizePreview);
      resizePreview();
      requestAnimationFrame(animatePreview);
    }

    function addGrid(size = 1.2, divisions = 24) {
      const group = new THREE.Group();
      const material = new THREE.LineBasicMaterial({ color: 0xc7cdc3, transparent: true, opacity: 0.58 });
      const half = size / 2;
      const step = size / divisions;
      for (let i = 0; i <= divisions; i += 1) {
        const v = -half + i * step;
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-half, v, 0), new THREE.Vector3(half, v, 0)
        ]), material));
        group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(v, -half, 0), new THREE.Vector3(v, half, 0)
        ]), material));
      }
      preview.scene.add(group);
      preview.scene.add(new THREE.AxesHelper(0.18));
    }

    function resizePreview() {
      if (!preview.renderer) return;
      const rect = els.viewer.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      preview.camera.aspect = rect.width / rect.height;
      preview.camera.updateProjectionMatrix();
      preview.renderer.setSize(rect.width, rect.height, false);
    }

    async function loadPreview(asset, relativePath) {
      initPreview();
      state.selectedAsset = asset;
      renderRows();
      clearPreviewModel();
      els['viewer-empty'].style.display = 'none';
      els['preview-path'].textContent = relativePath || asset;
      els['viewer-status'].textContent = 'Loading animated.glb...';
      preview.currentAsset = asset;
      const gltf = await new Promise((resolve, reject) => {
        new GLTFLoader().load(asset, resolve, event => {
          if (event.lengthComputable && event.total > 0) {
            const pct = Math.max(0, Math.min(100, event.loaded / event.total * 100));
            els['viewer-status'].textContent = `Loading animated.glb... ${pct.toFixed(0)}%`;
          }
        }, reject);
      });
      if (preview.currentAsset !== asset) return;
      preview.model = gltf.scene;
      preview.scene.add(preview.model);
      preview.mixer = new THREE.AnimationMixer(preview.model);
      preview.duration = Math.max(0, ...((gltf.animations || []).map(clip => clip.duration || 0)));
      for (const clip of gltf.animations || []) {
        const action = preview.mixer.clipAction(clip);
        action.reset();
        action.setLoop(THREE.LoopRepeat, Infinity);
        action.play();
      }
      fitCamera(preview.model);
      els['viewer-scrub'].max = String(Math.max(1, Math.round(preview.duration * 1000)));
      preview.playing = true;
      preview.clock.getDelta();
      updatePreviewPlayButton();
      setPreviewTime(0);
      els['viewer-status'].textContent = preview.duration
        ? `Playing | ${gltf.animations.length} clip | ${preview.duration.toFixed(2)}s`
        : 'Loaded GLB has no animation clip';
    }

    function clearPreviewModel() {
      if (!preview.model) return;
      preview.scene.remove(preview.model);
      preview.model.traverse(node => {
        node.geometry?.dispose?.();
        const materials = Array.isArray(node.material) ? node.material : [node.material].filter(Boolean);
        materials.forEach(material => material.dispose?.());
      });
      preview.model = null;
      preview.mixer = null;
      preview.duration = 0;
      preview.playing = false;
      updatePreviewPlayButton();
    }

    function fitCamera(root) {
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.25);
      preview.camera.position.set(center.x + radius * 1.0, center.y - radius * 1.35, center.z + radius * 0.75);
      preview.controls.target.copy(center);
      preview.controls.update();
    }

    function setPreviewTime(timeSec) {
      if (!preview.mixer) return;
      const t = Math.max(0, Math.min(timeSec, preview.duration || timeSec));
      preview.mixer.setTime(t);
      preview.model?.updateMatrixWorld(true);
      els['viewer-scrub'].value = String(Math.round(t * 1000));
      els['viewer-time'].textContent = `${t.toFixed(2)}s`;
      preview.renderer?.render(preview.scene, preview.camera);
    }

    function updatePreviewPlayButton() {
      els['viewer-play'].textContent = preview.playing ? 'Pause' : 'Play';
    }

    function animatePreview() {
      requestAnimationFrame(animatePreview);
      const delta = preview.clock.getDelta();
      if (preview.mixer && preview.playing) {
        const speed = Number(els['viewer-speed'].value) || 1;
        preview.mixer.update(delta * speed);
        if (preview.duration) {
          const t = preview.mixer.time % preview.duration;
          els['viewer-scrub'].value = String(Math.round(t * 1000));
          els['viewer-time'].textContent = `${t.toFixed(2)}s`;
        }
      }
      preview.controls?.update();
      preview.renderer?.render(preview.scene, preview.camera);
    }
    load().catch(err => {
      els.rows.innerHTML = `<div class="empty">${escapeHtml(err.message || err)}</div>`;
    });
    setInterval(() => load().catch(console.error), 2000);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    log_root: Path
    log_dir: Path | None

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"", "/"}:
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(read_state(resolve_log_dir(self.log_root, self.log_dir)))
            return
        if path == "/asset":
            query = parse_qs(parsed.query)
            asset_path = Path(query.get("path", [""])[0]).expanduser().resolve()
            self._serve_asset(asset_path)
            return
        self.send_error(404, "Not found")

    def _send_json(self, payload: dict) -> None:
        self._send_bytes(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), "application/json")

    def _send_bytes(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_asset(self, path: Path) -> None:
        output_root = Path.home() / "shared_data" / "AutoDex" / "interactive_3d"
        try:
            path.relative_to(output_root)
        except ValueError:
            self.send_error(403, "Forbidden")
            return
        if not path.is_file():
            self.send_error(404, "Asset not found")
            return
        mime = "model/gltf-binary" if path.suffix == ".glb" else mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), mime)


def resolve_log_dir(log_root: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit.expanduser().resolve()
    candidates = sorted(log_root.expanduser().glob("interactive_3d_export_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No interactive_3d_export_* logs under {log_root}")
    return candidates[0].resolve()


def read_json(path: Path, default: dict) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_records(events_path: Path) -> list[dict]:
    records: list[dict] = []
    if not events_path.is_file():
        return records
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("status") in {"ok", "failed"}:
                records.append(item)
    return records


def read_state(log_dir: Path) -> dict:
    summary = read_json(log_dir / "summary.json", {})
    records = read_records(log_dir / "events.jsonl")
    failed = [record for record in records if record.get("status") == "failed"]
    reasons = Counter(f"{record.get('error_type', 'Error')}: {record.get('error', '')}" for record in failed)
    now = time.time()
    started = float(summary.get("started_at") or now)
    completed = int(summary.get("completed") or len(records))
    total = int(summary.get("tasks_total") or completed)
    elapsed = max(0.001, now - started)
    rate_per_sec = completed / elapsed if completed else 0.0
    remaining = max(0, total - completed)
    eta_sec = remaining / rate_per_sec if rate_per_sec > 0 else None
    updated = float(summary.get("updated_at") or now)
    return {
        "log_dir": str(log_dir),
        "summary": summary,
        "records": records,
        "completed": completed,
        "failed": len(failed),
        "total": total,
        "failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(8)
        ],
        "rate_per_min": rate_per_sec * 60.0 if rate_per_sec else None,
        "eta_sec": eta_sec,
        "updated_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DashboardHandler.log_root = Path(args.log_root)
    DashboardHandler.log_dir = Path(args.log_dir) if args.log_dir else None
    log_dir = resolve_log_dir(DashboardHandler.log_root, DashboardHandler.log_dir)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[dashboard] log_dir={log_dir}")
    print(f"[dashboard] url=http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[dashboard] stopped")


if __name__ == "__main__":
    main()
