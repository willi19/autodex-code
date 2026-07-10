#!/usr/bin/env python3
"""Self-contained v8 grasp-generation dashboard (HTML).

Scans local filesystem + running procs and writes ~/AutoDex/v8_status.html.
Run on a loop; the page meta-refreshes so an open tab stays live.

Shows: per-hand progress, the object currently being processed, and a
per-object drill-down (per scene_type: #scenes, #candidates, gap distribution
= the live adaptive state), plus the watcher log tail.
"""
import os, re, glob, json, shutil, subprocess, html
from datetime import datetime, timezone, timedelta

HOME = os.path.expanduser("~")
OBJ_LIST = f"{HOME}/AutoDex/src/grasp_generation/obj_list_v8.txt"
LOGDIR = f"{HOME}/AutoDex/logging/adaptive"
SCENE_BASE = f"{HOME}/shared_data/AutoDex/scene"
OUT = f"{HOME}/AutoDex/v8_status.html"
ALL_HANDS = ["allegro", "inspire", "inspire_left"]
SCENE_TYPES = ["box", "shelf", "wall"]
KST = timezone(timedelta(hours=9))


def _ps():
    try:
        return subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def objs():
    if not os.path.isfile(OBJ_LIST):
        return []
    return [l.strip() for l in open(OBJ_LIST) if l.strip() and not l.startswith("#")]


def done_dir(h):  return f"{HOME}/shared_data/AutoDex/bodex_outputs/{h}/v8/.done"
def cand_base(h): return f"{HOME}/AutoDex/candidates/{h}/v8"


def done_names(h):
    d = done_dir(h)
    return set(os.listdir(d)) if os.path.isdir(d) else set()


def active(ps):
    m = re.search(r"adaptive_orchestrator\.py .*?--hand\s+(\S+).*?--obj\s+(\S+)", ps)
    return (m.group(1), m.group(2)) if m else (None, None)


def phases(ps):
    out = []
    if "run_sim_filter.py" in ps:
        out.append("sim_filter")
    m = re.search(r"generate\.py -c sim_\w+/paradex_(\w+)\.yml", ps)
    if m:
        out.append(f"BODex:{m.group(1)}")
    return out or ["(scheduling…)"]


def watcher_alive(ps):
    return "run_v8_watch.sh" in ps


def cand_count(hand, obj, st=None):
    """# passing candidates. Derives from cand_by_scene so it inherits the
    _pre_bonus fold (scenes mid-0.0-bonus keep their succeeded count)."""
    sts = [st] if st else SCENE_TYPES
    return sum(sum(cand_by_scene(hand, obj, s).values()) for s in sts)


SUCCESS_THRESHOLD = 5


def cand_by_scene(hand, obj, st):
    """{scene_id: #candidate seed dirs} for one scene_type.

    During the 0.0 bonus phase the orchestrator MOVES a succeeded scene's
    candidates to '{sid}_pre_bonus' while it tests gap 0.0, leaving the base
    dir momentarily empty. Fold '_pre_bonus' back into its base id (max) so a
    scene that already succeeded stays counted as done, not flipped to gray."""
    root = os.path.join(cand_base(hand), obj, st)
    out = {}
    if not os.path.isdir(root):
        return out
    try:
        for sid in os.scandir(root):
            if not sid.is_dir():
                continue
            name = sid.name
            base = name[:-len("_pre_bonus")] if name.endswith("_pre_bonus") else name
            cnt = sum(1 for s in os.scandir(sid.path) if s.is_dir())
            out[base] = max(out.get(base, 0), cnt)
    except OSError:
        pass
    return out


def scene_ids(hand, obj, st):
    """All scene ids for a scene_type (from the scene JSONs on disk)."""
    d = os.path.join(SCENE_BASE, hand, obj, st)
    if not os.path.isdir(d):
        return []
    ids = [os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".json")]
    return sorted(ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))


def scene_progress(hand, obj):
    """Per scene_type -> (done, total, [(sid, cand, status, gap)], gmap)."""
    res = {}
    for st in SCENE_TYPES:
        cands = cand_by_scene(hand, obj, st)
        gmap = scene_gap_map(hand, obj, st)
        ids = scene_ids(hand, obj, st) or sorted(cands.keys())
        rows, done = [], 0
        for sid in ids:
            c = cands.get(sid, 0)
            status = "done" if c >= SUCCESS_THRESHOLD else ("cur" if c > 0 else "pend")
            if status == "done":
                done += 1
            rows.append((sid, c, status, gmap.get(sid)))
        res[st] = (done, len(ids), rows, gmap)
    return res


def scene_gap_map(hand, obj, st):
    """{scene_id: current gap/height} from the scene JSONs on disk."""
    d = os.path.join(SCENE_BASE, hand, obj, st)
    m = {}
    if not os.path.isdir(d):
        return m
    for f in glob.glob(os.path.join(d, "*.json")):
        sid = os.path.splitext(os.path.basename(f))[0]
        try:
            meta = json.load(open(f)).get("meta", {}).get("param", {})
        except Exception:
            continue
        m[sid] = meta.get("gap", meta.get("height_offset"))
    return m


def gap_hist_from(gmap):
    hist = {}
    for g in gmap.values():
        key = f"{g:g}" if isinstance(g, (int, float)) else "?"
        hist[key] = hist.get(key, 0) + 1
    return hist


def active_round(st):
    """(gap, N) the orchestrator is CURRENTLY testing for a scene_type, parsed
    from the newest watcher log's last '[st] gap=X N=Y: Z active' line."""
    p = newest_watch_log()
    if not p:
        return None
    try:
        lines = open(p, errors="replace").read().splitlines()
    except Exception:
        return None
    found = None
    tag = f"[{st}]"
    for l in lines:
        if tag not in l:
            continue
        m = re.search(r"gap=(\S+)\s+N=(\S+)", l)
        if m:
            found = (m.group(1), m.group(2).rstrip(":"))
        elif "bonus phase" in l:
            bm = re.search(r"gap=(\S+)", l)
            if bm:
                found = (bm.group(1), "bonus")
    return found


def disk():
    t = shutil.disk_usage("/")
    return f"{t.free/1e12:.2f}T free", 100*t.used/t.total


def newest_watch_log():
    logs = glob.glob(f"{LOGDIR}/run_v8_watch_*.log")
    return max(logs, key=os.path.getmtime) if logs else None


def log_tail(n=22):
    p = newest_watch_log()
    if not p:
        return "(no watcher log yet)"
    try:
        lines = [l for l in open(p, errors="replace").read().splitlines() if l.strip()]
    except Exception:
        return "(log read error)"
    return "\n".join(lines[-n:]) or "(empty)"


def obj_log_lines(obj, n=14):
    """Recent orchestrator progress lines mentioning this object's scene-type steps."""
    p = newest_watch_log()
    if not p:
        return ""
    try:
        lines = open(p, errors="replace").read().splitlines()
    except Exception:
        return ""
    pat = re.compile(r"\[(box|shelf|wall)\]|bonus|--- \w+ / " + re.escape(obj))
    hit = [l for l in lines if pat.search(l)]
    return "\n".join(hit[-n:])


def build():
    ps = _ps()
    O = objs()
    total = len(O)
    hand, obj = active(ps)
    disp_hand = hand or "allegro"
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    disk_s, disk_pct = disk()
    alive = watcher_alive(ps)

    # per-hand progress
    hand_cards = []
    for h in ALL_HANDS:
        dn = done_names(h)
        nd = len([o for o in O if o in dn])
        pct = 100*nd/total if total else 0
        badge = " ◀" if h == hand else ""
        hand_cards.append(f"""<div class="card">
          <div class="k">{h}{badge}</div>
          <div class="v">{nd}<span class="d">/{total}</span></div>
          <div class="bar"><i style="width:{pct:.1f}%"></i></div></div>""")

    # current object detail
    if obj:
        ph = ", ".join(html.escape(p) for p in phases(ps))
        tot_c = cand_count(hand, obj)
        sp = scene_progress(hand, obj)              # a scene is "done" at >=5 candidates
        sd_tot = sum(v[0] for v in sp.values())
        sn_tot = sum(v[1] for v in sp.values())
        spct = 100*sd_tot/sn_tot if sn_tot else 0
        blocks, rows = [], []
        for st in SCENE_TYPES:
            d, n, rows_s, gmap = sp[st]
            def gtxt(g):
                return f"{g:g}" if isinstance(g, (int, float)) else "?"
            chips_s = "".join(
                f'<span class="schip {stt}" title="{sid}: {c} cand @ gap={gtxt(g)}">{html.escape(sid)}</span>'
                for sid, c, stt, g in rows_s)
            ar = active_round(st)
            testing = (f'<span class="testing">▶ testing gap={html.escape(ar[0])} N={html.escape(ar[1])}</span>'
                       if ar else "")
            blocks.append(f"""<div class="stblock">
              <div class="sthdr">{st} &middot; <b>{d}/{n}</b> scenes done {testing}</div>
              <div class="schips">{chips_s or '<span class="muted">no scenes</span>'}</div></div>""")
            hist = gap_hist_from(gmap)
            gaps = " ".join(f'<span class="g">{k}:{v}</span>' for k, v in sorted(hist.items()))
            rows.append(f"<tr><td>{st}</td><td>{n}</td><td><b>{cand_count(hand,obj,st)}</b></td><td>{gaps or '—'}</td></tr>")
        detail = f"""
        <div class="k lbl">Now processing</div>
        <div class="objhdr"><b>{html.escape(hand)} / {html.escape(obj)}</b>
          &middot; {tot_c} candidates &middot; <span class="ph">{ph}</span></div>
        <div class="lbl" style="margin-top:6px">scenes complete (&ge;{SUCCESS_THRESHOLD} candidates): {sd_tot}/{sn_tot} &middot; {spct:.0f}%</div>
        <div class="bar" style="margin-bottom:14px"><i style="width:{spct:.1f}%"></i></div>
        {''.join(blocks)}
        <table style="margin-top:12px"><thead><tr><th>scene_type</th><th>scenes</th><th>candidates</th>
          <th>gap distribution (gap:#scenes)</th></tr></thead>
          <tbody>{''.join(rows)}</tbody></table>
        <div class="k lbl" style="margin-top:14px">This object — orchestrator steps</div>
        <pre class="small">{html.escape(obj_log_lines(obj)) or '(waiting…)'}</pre>"""
    else:
        detail = '<div class="objhdr">— no object actively processing —</div>'

    dn0 = done_names(disp_hand)
    chips = []
    for o in O:
        if o in dn0:
            chips.append(f'<span class="chip done" title="{cand_count(disp_hand,o)} cand">{html.escape(o)}</span>')
        elif o == obj:
            chips.append(f'<span class="chip cur">{html.escape(o)} ●</span>')
        else:
            chips.append(f'<span class="chip pend">{html.escape(o)}</span>')

    badge = '<span class="ok">RUNNING</span>' if alive else '<span class="bad">WATCHER DOWN</span>'
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="20">
<title>v8 progress {sum(1 for h in ALL_HANDS for o in O if o in done_names(h))}</title>
<style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}
body{{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 2px}}.sub{{color:#8b949e;font-size:12px;margin-bottom:18px}}
.ok{{color:#3fb950;font-weight:700}}.bad{{color:#f85149;font-weight:700}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:13px}}
.card .k{{color:#8b949e;font-size:12px;font-weight:600}}.card .v{{font-size:26px;font-weight:700;margin-top:2px}}
.card .d{{font-size:13px;color:#8b949e}}
.bar{{height:8px;background:#21262d;border-radius:6px;overflow:hidden;margin-top:8px}}
.bar>i{{display:block;height:100%;background:linear-gradient(90deg,#238636,#3fb950)}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:18px}}
.lbl{{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
.objhdr{{font-size:16px;margin-bottom:10px}}.ph{{color:#d29922}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:6px 10px;border-bottom:1px solid #21262d}}
th{{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase}}
.g{{display:inline-block;background:#21262d;border-radius:5px;padding:1px 6px;margin:1px;font-size:11px;color:#79c0ff}}
.chips{{display:flex;flex-wrap:wrap;gap:5px;margin:6px 0 18px}}
.chip{{font-size:11px;padding:3px 7px;border-radius:6px;border:1px solid #30363d}}
.chip.done{{background:#132e1a;border-color:#238636;color:#5fd377}}
.chip.cur{{background:#3a2d05;border-color:#d29922;color:#f0c14b;animation:pulse 1.5s infinite}}
.chip.pend{{background:#161b22;color:#6e7681}}@keyframes pulse{{50%{{opacity:.5}}}}
.stblock{{margin-bottom:10px}}.sthdr{{font-size:12px;color:#8b949e;margin-bottom:4px}}
.schips{{display:flex;flex-wrap:wrap;gap:3px}}
.schip{{font-size:10px;padding:2px 5px;border-radius:4px;min-width:20px;text-align:center}}
.schip.done{{background:#132e1a;color:#5fd377;border:1px solid #238636}}
.schip.cur{{background:#3a2d05;color:#f0c14b;border:1px solid #d29922}}
.schip.pend{{background:#1a1f26;color:#6e7681;border:1px solid #30363d}}
.muted{{color:#6e7681;font-size:11px}}
.testing{{color:#f0c14b;font-weight:600;margin-left:8px}}
pre{{background:#010409;border:1px solid #30363d;border-radius:10px;padding:12px;overflow:auto;font-size:12px;color:#c9d1d9;max-height:300px}}
pre.small{{max-height:200px;margin:0}}
</style></head><body><div class="wrap">
<h1>v8 grasp generation</h1>
<div class="sub">updated {now} &middot; auto-refresh 20s &middot; {badge} &middot; disk {html.escape(disk_s)} ({disk_pct:.0f}% used)</div>
<div class="cards">{''.join(hand_cards)}</div>
<div class="panel">{detail}</div>
<div class="lbl">objects — {html.escape(disp_hand)} (hover done for candidate count)</div>
<div class="chips">{''.join(chips)}</div>
<div class="lbl">watcher log (tail)</div>
<pre>{html.escape(log_tail())}</pre>
</div></body></html>"""
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write(doc)
    os.replace(tmp, OUT)


if __name__ == "__main__":
    build()
    print(f"wrote {OUT}")
