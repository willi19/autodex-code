"""Count sim_filter passes per (obj, scene) for a given version."""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--version", default="reset_0",
                        help="BODex reset experiment; suffix is release height in cm")
    parser.add_argument("--bodex_root", default=str(Path.home() / "AutoDex" / "bodex_outputs"))
    parser.add_argument("--per_scene", action="store_true",
                        help="show per-(obj, scene_id) breakdown")
    parser.add_argument("--zero_only", action="store_true",
                        help="only print scenes with 0 passes")
    parser.add_argument("--failed_pairs", action="store_true",
                        help="print only 'obj i j' lines for scenes with 0 passes (for next stage)")
    args = parser.parse_args()
    if not args.version.startswith("reset_"):
        parser.error("--version must be a reset_<height_mm> experiment")

    root = Path(args.bodex_root) / args.hand / args.version
    if not root.is_dir():
        print(f"not found: {root}")
        return

    per_scene = defaultdict(lambda: [0, 0, 0])  # [pass, total, evaluated]
    per_obj = defaultdict(lambda: [0, 0, 0])

    for obj in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for st in sorted(p.name for p in (root / obj).iterdir() if p.is_dir()):
            for sid in sorted(p.name for p in (root / obj / st).iterdir() if p.is_dir()):
                for seed in (root / obj / st / sid).iterdir():
                    if not seed.is_dir():
                        continue
                    grasp_exists = (seed / "grasp_pose.npy").exists()
                    eval_path = seed / "sim_eval.json"
                    per_scene[(obj, st, sid)][1] += int(grasp_exists)
                    per_obj[obj][1] += int(grasp_exists)
                    if eval_path.exists():
                        try:
                            d = json.load(open(eval_path))
                            succ = bool(d.get("success", False))
                            per_scene[(obj, st, sid)][0] += int(succ)
                            per_scene[(obj, st, sid)][2] += 1
                            per_obj[obj][0] += int(succ)
                            per_obj[obj][2] += 1
                        except Exception:
                            pass

    if args.failed_pairs:
        for (obj, st, sid), (p, t, e) in sorted(per_scene.items()):
            if p == 0:
                i, j = sid.split("_", 1)
                print(f"{obj} {i} {j}")
        return

    if args.per_scene or args.zero_only:
        print(f"{'OBJ':<22} {'SCENE_TYPE':<14} {'SCENE_ID':<14} {'PASS':>5} {'EVAL':>6} {'TOTAL':>7}")
        for (obj, st, sid), (p, t, e) in sorted(per_scene.items()):
            if args.zero_only and p > 0:
                continue
            print(f"{obj:<22} {st:<14} {sid:<14} {p:>5} {e:>6} {t:>7}")
        print()

    print(f"{'OBJ':<22} {'PASS':>6} {'EVAL':>7} {'TOTAL':>7}  pass%")
    grand = [0, 0, 0]
    for obj, (p, t, e) in sorted(per_obj.items()):
        grand[0] += p; grand[1] += t; grand[2] += e
        pct = 100.0 * p / e if e else 0.0
        print(f"{obj:<22} {p:>6} {e:>7} {t:>7}  {pct:5.1f}%")
    pct = 100.0 * grand[0] / grand[2] if grand[2] else 0.0
    print(f"{'TOTAL':<22} {grand[0]:>6} {grand[2]:>7} {grand[1]:>7}  {pct:5.1f}%")


if __name__ == "__main__":
    main()
