#!/usr/bin/env python3
"""Standalone object-surface-coverage vs #cameras figure (right panel of the old
cost_combined figure, relabeled). Best-subset coverage = fraction of the object
surface visible from >=1 camera in the best k-camera subset, with vs without the
robot occluding views. Source: rebuttal/coverage_results.json (compute_coverage.py).
"""
import json
import statistics as st
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent / "coverage_results.json"
OUTS = [
    Path.home() / "69d3239d2f336c3826888fa9" / "figures" / "from_rebuttal" / "surface_coverage.pdf",
]


def aggregate(records, key):
    by_k = {}
    for r in records:
        for e in r.get(key) or []:
            by_k.setdefault(e["n_cams"], []).append(e["max_coverage"])
    ks = sorted(by_k)
    return ks, [100.0 * st.mean(by_k[k]) for k in ks]


def main():
    d = json.load(open(SRC))
    kw, vw = aggregate(d, "coverage_with_robot")
    kn, vn = aggregate(d, "coverage_no_robot")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot(kn, vn, "-s", color="#3b7eb8", ms=3.5, lw=1.3, label="No robot")
    ax.plot(kw, vw, "-o", color="#d1503c", ms=3.5, lw=1.3, label="With robot")
    ax.set_xlabel("# Cameras")
    ax.set_ylabel("Object surface coverage (%)")
    ax.set_xlim(0.5, 24.5)
    ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    ax.grid(True, lw=0.4, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4)

    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
