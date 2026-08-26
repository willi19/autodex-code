#!/usr/bin/env python3
"""Serve a local Three.js viewer for exported AutoDex interactive 3D assets."""
from __future__ import annotations

import argparse
import json
import mimetypes
import posixpath
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_ASSET_DIR = (
    Path.home()
    / "shared_data"
    / "AutoDex"
    / "interactive_3d"
    / "allegro"
    / "wood_organizer"
    / "20260326_121427"
)


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoDex Interactive 3D Preview</title>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.160.0/build/three.module.js"
      }
    }
  </script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8f4;
      --panel: #ffffff;
      --ink: #222421;
      --muted: #686d67;
      --line: #d8ddd2;
      --accent: #326f6d;
      --warn: #a5541c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-rows: auto 1fr auto;
      height: 100vh;
      min-height: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
    }
    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 680;
      letter-spacing: 0;
    }
    .meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      justify-content: flex-end;
    }
    .viewport {
      position: relative;
      min-height: 0;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
    }
    .loading {
      position: absolute;
      inset: 0;
      display: grid;
      place-items: center;
      background: rgba(247, 248, 244, 0.86);
      color: var(--muted);
      font-size: 14px;
      pointer-events: none;
    }
    .loading.hidden { display: none; }
    .hud {
      position: absolute;
      left: 14px;
      bottom: 14px;
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      border: 1px solid rgba(216, 221, 210, 0.9);
      background: rgba(255, 255, 255, 0.82);
      backdrop-filter: blur(8px);
      font-size: 12px;
      color: var(--muted);
      max-width: min(420px, calc(100vw - 28px));
    }
    .hud strong { color: var(--ink); font-weight: 650; }
    .warning {
      color: var(--warn);
      display: none;
    }
    .warning.visible { display: block; }
    footer {
      display: grid;
      grid-template-columns: auto auto auto auto 1fr auto auto;
      align-items: center;
      gap: 12px;
      padding: 12px 18px;
      border-top: 1px solid var(--line);
      background: #ffffff;
    }
    button {
      width: 38px;
      height: 34px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--ink);
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 650;
    }
    button:hover { border-color: var(--accent); color: var(--accent); }
    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .readout {
      min-width: 168px;
      text-align: right;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }
    select {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #ffffff;
      color: var(--ink);
      padding: 0 8px;
    }
    @media (max-width: 720px) {
      header {
        display: grid;
        align-items: start;
      }
      .meta { justify-content: flex-start; }
      footer {
        grid-template-columns: auto 1fr;
      }
      .readout {
        grid-column: 1 / -1;
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <h1>AutoDex Interactive 3D Preview</h1>
      <div class="meta">
        <span id="episode">episode</span>
        <span id="frame-count">0 frames</span>
        <span id="coord">frame</span>
      </div>
    </header>
    <main class="viewport" id="viewport">
      <div class="loading" id="loading">Loading animated GLB...</div>
      <div class="hud">
        <div><strong id="status">status</strong> <span id="stage"></span></div>
        <div id="quality">tracking metrics</div>
        <div id="motion">mesh playback</div>
        <div class="warning" id="missing"></div>
      </div>
    </main>
    <footer>
      <button id="first" title="Jump to first frame">|&lt;</button>
      <button id="play" title="Play or pause">Play</button>
      <button id="middle" title="Jump to middle frame">Mid</button>
      <button id="last" title="Jump to last frame">&gt;|</button>
      <input id="scrub" type="range" min="0" max="0" value="0">
      <div class="readout" id="readout">frame 0 / 0</div>
      <select id="speed" title="Playback speed">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1">1x</option>
        <option value="2">2x</option>
        <option value="4" selected>4x</option>
      </select>
    </footer>
  </div>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

    const els = {
      viewport: document.getElementById('viewport'),
      loading: document.getElementById('loading'),
      episode: document.getElementById('episode'),
      frameCount: document.getElementById('frame-count'),
      coord: document.getElementById('coord'),
      status: document.getElementById('status'),
      stage: document.getElementById('stage'),
      quality: document.getElementById('quality'),
      motion: document.getElementById('motion'),
      missing: document.getElementById('missing'),
      first: document.getElementById('first'),
      play: document.getElementById('play'),
      middle: document.getElementById('middle'),
      last: document.getElementById('last'),
      scrub: document.getElementById('scrub'),
      readout: document.getElementById('readout'),
      speed: document.getElementById('speed'),
    };

    let renderer;
    let camera;
    let controls;
    let scene;
    let model;
    let mixer;
    let animationDuration = 0;
    let frames = [];
    let frameTimes = [];
    let currentIndex = 0;
    let playing = false;
    let playStartWall = 0;
    let playStartTime = 0;
    let nodeByName = new Map();
    let objectNodeName = 'object::mesh';
    let lastTelemetryAt = 0;
    const query = new URLSearchParams(window.location.search);
    const shouldAutoplay = query.get('autoplay') !== '0';
    const showDebugPaths = query.get('paths') === '1';

    function setLoading(message, hidden = false) {
      els.loading.textContent = message;
      els.loading.classList.toggle('hidden', hidden);
    }

    function normalizeName(name) {
      return String(name || '').replace(/[^A-Za-z0-9_.-]+/g, '_');
    }

    function looseName(name) {
      return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
    }

    function rememberNode(obj) {
      if (!obj.name) return;
      nodeByName.set(obj.name, obj);
      nodeByName.set(normalizeName(obj.name), obj);
      nodeByName.set(looseName(obj.name), obj);
    }

    function findNode(name) {
      return nodeByName.get(name) || nodeByName.get(normalizeName(name)) || nodeByName.get(looseName(name));
    }

    function matrixFromRows(rows) {
      const m = new THREE.Matrix4();
      m.set(
        rows[0][0], rows[0][1], rows[0][2], rows[0][3],
        rows[1][0], rows[1][1], rows[1][2], rows[1][3],
        rows[2][0], rows[2][1], rows[2][2], rows[2][3],
        rows[3][0], rows[3][1], rows[3][2], rows[3][3],
      );
      return m;
    }

    function applyMatrix(obj, rows) {
      if (!obj || !rows) return;
      const worldMatrix = matrixFromRows(rows);
      const localMatrix = worldMatrix.clone();
      if (obj.parent) {
        obj.parent.updateMatrixWorld(true);
        localMatrix.premultiply(obj.parent.matrixWorld.clone().invert());
      }
      localMatrix.decompose(obj.position, obj.quaternion, obj.scale);
      obj.matrixAutoUpdate = true;
      obj.updateMatrix();
      obj.updateMatrixWorld(true);
    }

    function formatVec(v) {
      if (!v) return 'n/a';
      return `${v.x.toFixed(3)}, ${v.y.toFixed(3)}, ${v.z.toFixed(3)}`;
    }

    function translationFromRows(rows) {
      if (!rows) return null;
      return new THREE.Vector3(Number(rows[0][3]), Number(rows[1][3]), Number(rows[2][3]));
    }

    function addXYGrid(size = 1.2, divisions = 24) {
      const group = new THREE.Group();
      group.name = 'xy-grid';
      const material = new THREE.LineBasicMaterial({ color: 0xc7cdc3, transparent: true, opacity: 0.58 });
      const half = size / 2;
      const step = size / divisions;
      for (let i = 0; i <= divisions; i += 1) {
        const v = -half + i * step;
        const a = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-half, v, 0),
          new THREE.Vector3(half, v, 0),
        ]);
        const b = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(v, -half, 0),
          new THREE.Vector3(v, half, 0),
        ]);
        group.add(new THREE.Line(a, material));
        group.add(new THREE.Line(b, material));
      }
      scene.add(group);
    }

    function makePathLine(name, points, color, opacity) {
      if (points.length < 2) return null;
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthTest: true,
      });
      const line = new THREE.Line(geometry, material);
      line.name = name;
      return line;
    }

    function addTrajectoryPaths() {
      if (!showDebugPaths) return;
      const objectPoints = frames
        .map(f => translationFromRows(f.object_pose_world))
        .filter(Boolean);
      const objectPath = makePathLine('object-trajectory-path', objectPoints, 0xd95b2b, 0.95);
      if (objectPath) scene.add(objectPath);

      const candidateNodes = ['robot::base_link.obj', 'robot::link6.obj'];
      let wristKey = null;
      for (const key of candidateNodes) {
        if (frames.some(f => f.robot_geometry_poses_world?.[key])) {
          wristKey = key;
          break;
        }
      }
      if (wristKey) {
        const wristPoints = frames
          .map(f => translationFromRows(f.robot_geometry_poses_world?.[wristKey]))
          .filter(Boolean);
        const wristPath = makePathLine('wrist-trajectory-path', wristPoints, 0x326f6d, 0.85);
        if (wristPath) scene.add(wristPath);
      }
    }

    function setupRenderer() {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf7f8f4);

      camera = new THREE.PerspectiveCamera(45, 1, 0.001, 20);
      camera.up.set(0, 0, 1);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 0.95;
      els.viewport.prepend(renderer.domElement);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.screenSpacePanning = true;

      scene.add(new THREE.HemisphereLight(0xffffff, 0x9fa69e, 1.8));
      const key = new THREE.DirectionalLight(0xffffff, 2.2);
      key.position.set(1.1, -1.2, 1.8);
      scene.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.8);
      fill.position.set(-0.8, 0.9, 1.0);
      scene.add(fill);
      addXYGrid();
      scene.add(new THREE.AxesHelper(0.18));
      resize();
      window.addEventListener('resize', resize);
    }

    function resize() {
      const rect = els.viewport.getBoundingClientRect();
      if (!rect.width || !rect.height || !renderer) return;
      camera.aspect = rect.width / rect.height;
      camera.updateProjectionMatrix();
      renderer.setSize(rect.width, rect.height, false);
    }

    function fitCamera(root) {
      const box = new THREE.Box3().setFromObject(root);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 0.25);
      camera.position.set(center.x + radius * 1.0, center.y - radius * 1.35, center.z + radius * 0.75);
      controls.target.copy(center);
      controls.update();
    }

    function makeFrameTimes() {
      const raw = frames.map((f, i) => Number.isFinite(f.time_sec) ? Number(f.time_sec) : i / 30);
      const t0 = raw[0] || 0;
      frameTimes = raw.map((t, i) => {
        const v = t - t0;
        return Number.isFinite(v) && v >= 0 ? v : i / 30;
      });
      for (let i = 1; i < frameTimes.length; i += 1) {
        if (frameTimes[i] <= frameTimes[i - 1]) frameTimes[i] = frameTimes[i - 1] + 1 / 30;
      }
    }

    function rangeMeters(points) {
      if (!points.length) return 0;
      const min = points[0].clone();
      const max = points[0].clone();
      for (const p of points) {
        min.min(p);
        max.max(p);
      }
      return max.sub(min).length();
    }

    function updateMotionSummary() {
      const objectPoints = frames
        .map(f => translationFromRows(f.object_pose_world))
        .filter(Boolean);
      const objectRangeMm = rangeMeters(objectPoints) * 1000;
      const wristKey = ['robot::base_link.obj', 'robot::link6.obj'].find(
        key => frames.some(f => f.robot_geometry_poses_world?.[key])
      );
      const wristPoints = wristKey
        ? frames.map(f => translationFromRows(f.robot_geometry_poses_world?.[wristKey])).filter(Boolean)
        : [];
      const wristRangeMm = rangeMeters(wristPoints) * 1000;
      els.motion.textContent = `mesh playback | object range ${objectRangeMm.toFixed(1)} mm | hand range ${wristRangeMm.toFixed(1)} mm`;
    }

    function frameIndexAtTime(t) {
      if (!frameTimes.length) return 0;
      if (t <= frameTimes[0]) return 0;
      const last = frameTimes[frameTimes.length - 1];
      if (t >= last) return frameTimes.length - 1;
      let lo = 0;
      let hi = frameTimes.length - 1;
      while (lo + 1 < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (frameTimes[mid] <= t) lo = mid;
        else hi = mid;
      }
      return lo;
    }

    function applyFrame(index, userInitiated = false) {
      if (!frames.length) return;
      currentIndex = Math.max(0, Math.min(frames.length - 1, index));
      const frame = frames[currentIndex];
      window.__AUTODEX_FRAME_INDEX = currentIndex;
      const t = frameTimes[currentIndex] ?? 0;

      if (mixer) {
        mixer.setTime(Math.max(0, Math.min(t, animationDuration || t)));
        model?.updateMatrixWorld(true);
      } else {
        const objectNode = findNode(objectNodeName);
        if (objectNode) applyMatrix(objectNode, frame.object_pose_world);
        const poses = frame.robot_geometry_poses_world || {};
        for (const [name, matrix] of Object.entries(poses)) {
          const node = findNode(name);
          if (node) applyMatrix(node, matrix);
        }
      }

      els.scrub.value = String(currentIndex);
      els.readout.textContent = `frame ${frame.frame_index} (${currentIndex + 1}/${frames.length})  ${t.toFixed(2)}s`;
      els.status.textContent = frame.status || 'unknown';
      els.stage.textContent = frame.stage ? ` / ${frame.stage}` : '';
      const tr = frame.tracking || {};
      els.quality.textContent = [
        tr.num_inlier_anchors != null ? `inliers ${tr.num_inlier_anchors}` : null,
        tr.num_triangulated_anchors != null ? `triangulated ${tr.num_triangulated_anchors}` : null,
        tr.mean_anchor_fit_residual_mm != null ? `fit ${Number(tr.mean_anchor_fit_residual_mm).toFixed(2)} mm` : null,
      ].filter(Boolean).join(' | ') || 'no tracking metrics';
      const poses = frame.robot_geometry_poses_world || {};
      const objT = translationFromRows(frame.object_pose_world);
      const handPose = poses['robot::base_link.obj'] || poses['robot::link6.obj'];
      const handT = translationFromRows(handPose);
      const objectNode = findNode(objectNodeName);
      const handNode = findNode('robot::base_link.obj') || findNode('robot::link6.obj');
      const actualObj = objectNode ? objectNode.getWorldPosition(new THREE.Vector3()) : null;
      const actualHand = handNode ? handNode.getWorldPosition(new THREE.Vector3()) : null;
      els.motion.textContent = `${mixer ? 'animated.glb playback' : 'trajectory playback'} | object xyz ${formatVec(actualObj || objT)} | hand xyz ${formatVec(actualHand || handT)}`;
      const now = performance.now();
      if (now - lastTelemetryAt > 1000) {
        lastTelemetryAt = now;
        const params = new URLSearchParams({
          frame: String(currentIndex),
          source_frame: String(frame.frame_index),
          object: objT ? `${objT.x.toFixed(3)},${objT.y.toFixed(3)},${objT.z.toFixed(3)}` : 'n/a',
          hand: handT ? `${handT.x.toFixed(3)},${handT.y.toFixed(3)},${handT.z.toFixed(3)}` : 'n/a',
          actual_object: actualObj ? `${actualObj.x.toFixed(3)},${actualObj.y.toFixed(3)},${actualObj.z.toFixed(3)}` : 'n/a',
          actual_hand: actualHand ? `${actualHand.x.toFixed(3)},${actualHand.y.toFixed(3)},${actualHand.z.toFixed(3)}` : 'n/a',
        });
        fetch(`/telemetry?${params}`, { cache: 'no-store', keepalive: true }).catch(() => {});
      }

      els.missing.classList.remove('visible');

      if (userInitiated && playing) {
        playStartWall = performance.now();
        playStartTime = frameTimes[currentIndex] || 0;
      }
    }

    function setPlaying(next) {
      playing = next;
      els.play.textContent = playing ? 'Pause' : 'Play';
      if (playing) {
        playStartWall = performance.now();
        playStartTime = frameTimes[currentIndex] || 0;
      }
    }

    function loadGltfWithProgress(url, label) {
      const loader = new GLTFLoader();
      return new Promise((resolve, reject) => {
        loader.load(
          url,
          resolve,
          event => {
            if (event.lengthComputable && event.total > 0) {
              const pct = Math.max(0, Math.min(100, (event.loaded / event.total) * 100));
              setLoading(`Loading ${label}... ${pct.toFixed(0)}%`);
            } else {
              setLoading(`Loading ${label}...`);
            }
          },
          reject,
        );
      });
    }

    async function loadAssets() {
      setLoading('Initializing WebGL...');
      setupRenderer();
      setLoading('Loading manifest...');
      const manifest = await fetch('/asset/manifest.json').then(r => r.json());
      setLoading('Loading trajectory...');
      const trajectory = await fetch('/asset/trajectory.json').then(r => r.json());
      objectNodeName = trajectory.object?.node_name || 'object::mesh';
      frames = trajectory.frames || [];
      if (!frames.length) throw new Error('trajectory.json has no frames');
      makeFrameTimes();

      els.episode.textContent = manifest.relative_episode_path || trajectory.relative_episode_path || 'episode';
      els.frameCount.textContent = `${frames.length} frames`;
      els.coord.textContent = trajectory.coordinate_frame || manifest.coordinate_frame || 'frame';
      els.scrub.max = String(frames.length - 1);

      const animatedAsset = manifest.outputs?.animated_glb || 'animated.glb';
      const gltf = await loadGltfWithProgress(`/asset/${animatedAsset}`, animatedAsset);
      setLoading('Preparing playback...');
      scene.add(gltf.scene);
      model = gltf.scene;
      gltf.scene.traverse(rememberNode);
      if (!gltf.animations?.length) {
        throw new Error(`${animatedAsset} has no animation clips`);
      }
      mixer = new THREE.AnimationMixer(gltf.scene);
      animationDuration = Math.max(...gltf.animations.map(clip => clip.duration || 0));
      for (const clip of gltf.animations) {
        const action = mixer.clipAction(clip);
        action.reset();
        action.setLoop(THREE.LoopRepeat, Infinity);
        action.play();
      }
      window.__AUTODEX_ANIMATION_COUNT = gltf.animations.length;
      window.__AUTODEX_ANIMATION_DURATION = animationDuration;
      updateMotionSummary();
      addTrajectoryPaths();
      applyFrame(0);
      fitCamera(gltf.scene);
      window.__AUTODEX_VIEWER_READY = true;
      setLoading('', true);
      setPlaying(shouldAutoplay);
    }

    els.play.addEventListener('click', () => setPlaying(!playing));
    els.first.addEventListener('click', () => applyFrame(0, true));
    els.middle.addEventListener('click', () => applyFrame(Math.floor((frames.length - 1) / 2), true));
    els.last.addEventListener('click', () => applyFrame(frames.length - 1, true));
    els.scrub.addEventListener('input', () => applyFrame(Number(els.scrub.value), true));
    els.speed.addEventListener('change', () => {
      if (playing) {
        playStartWall = performance.now();
        playStartTime = frameTimes[currentIndex] || 0;
      }
    });
    window.addEventListener('keydown', event => {
      if (event.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
      if (event.code === 'Space') {
        event.preventDefault();
        setPlaying(!playing);
      } else if (event.code === 'ArrowRight') {
        event.preventDefault();
        applyFrame(currentIndex + 1, true);
      } else if (event.code === 'ArrowLeft') {
        event.preventDefault();
        applyFrame(currentIndex - 1, true);
      }
    });

    function animate(now) {
      requestAnimationFrame(animate);
      if (playing && frames.length) {
        const speed = Number(els.speed.value) || 1;
        const duration = Math.min(animationDuration || Infinity, frameTimes[frameTimes.length - 1] || frames.length / 30);
        let t = playStartTime + ((now - playStartWall) / 1000) * speed;
        if (t > duration) {
          playStartWall = now;
          playStartTime = 0;
          t = 0;
        }
        const idx = frameIndexAtTime(t);
        if (idx !== currentIndex) applyFrame(idx);
      }
      controls?.update();
      renderer?.render(scene, camera);
    }

    loadAssets().catch(err => {
      console.error(err);
      els.loading.textContent = `Failed to load viewer: ${err.message}`;
      els.loading.classList.remove('hidden');
    });
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    asset_dir: Path

    def log_message(self, fmt: str, *args) -> None:
        print(f"[viewer] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self._handle_request(send_body=False)

    def _handle_request(self, *, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path in {"", "/", "/gallery.html"}:
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8", send_body=send_body)
            return
        if path.startswith("/asset/"):
            rel = posixpath.normpath(path[len("/asset/") :]).lstrip("/")
            self._serve_asset(rel, send_body=send_body)
            return
        if path == "/telemetry":
            query = parsed.query
            print(f"[viewer:telemetry] {query}")
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404, "Not found")

    def _serve_asset(self, rel: str, *, send_body: bool) -> None:
        root = self.asset_dir.resolve()
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            self.send_error(403, "Forbidden")
            return
        if not target.is_file():
            self.send_error(404, f"Asset not found: {rel}")
            return
        if target.suffix == ".glb":
            mime = "model/gltf-binary"
        elif target.suffix == ".json":
            mime = "application/json"
        else:
            mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), mime, send_body=send_body)

    def _send_bytes(self, payload: bytes, content_type: str, *, send_body: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                return


def validate_asset_dir(asset_dir: Path) -> Path:
    asset_dir = asset_dir.expanduser().resolve()
    required = ["manifest.json", "trajectory.json", "scene.glb", "animated.glb"]
    missing = [name for name in required if not (asset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} under {asset_dir}")
    return asset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_dir = validate_asset_dir(args.asset_dir)
    ViewerHandler.asset_dir = asset_dir
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}/"
    manifest = json.loads((asset_dir / "manifest.json").read_text(encoding="utf-8"))
    print(f"[viewer] asset_dir={asset_dir}")
    print(f"[viewer] episode={manifest.get('relative_episode_path', asset_dir.name)}")
    print(f"[viewer] url={url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[viewer] stopped")


if __name__ == "__main__":
    main()
