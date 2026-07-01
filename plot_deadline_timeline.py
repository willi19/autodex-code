"""Submission-deadline timeline: today (2026-06-17) -> RSS 2027 (~Jan 2027).
Robotics-centric, with key vision/HCI venues. Confirmed vs estimated marked."""
import locale
try:
    locale.setlocale(locale.LC_TIME, "C")
except locale.Error:
    pass
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import date

# (label, date, field, confirmed?)
# field: robo / vision / hci
events = [
    ("Humanoids 2026",  date(2026, 7, 24), "robo",   True),
    ("3DV 2027",        date(2026, 8, 18), "vision", False),
    ("CHI 2027",        date(2026, 9, 11), "hci",    False),
    ("ICRA 2027",       date(2026, 9, 15), "robo",   False),
    ("WACV 2027",       date(2026, 9, 19), "vision", False),
    ("CVPR 2027",       date(2026, 11, 13),"vision", False),
    ("RSS 2027",        date(2027, 1, 30), "robo",   False),
]
today = date(2026, 6, 17)

MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
def fmt(d):   # "Jul 24, 2026"
    return f"{MON[d.month]} {d.day}, {d.year}"

color = {"robo": "#1f5fa8", "vision": "#2e8b57", "hci": "#d2691e"}
fname = {"robo": "Robotics", "vision": "Vision", "hci": "HCI"}

fig, ax = plt.subplots(figsize=(14, 6.2))

# baseline
ax.axhline(0, color="#444", lw=1.4, zorder=1)

# alternate stem heights to avoid label collision
levels = [1.0, -1.0, 1.7, -1.7, 1.0, -1.0, 1.0]
for (label, d, field, confirmed), lvl in zip(events, levels):
    c = color[field]
    ax.plot([d, d], [0, lvl], color=c, lw=1.6, zorder=2)
    ax.plot(d, lvl, "o", ms=13, color=c if confirmed else "white",
            markeredgecolor=c, markeredgewidth=2.2, zorder=3)
    tag = "" if confirmed else "  (est.)"
    va = "bottom" if lvl > 0 else "top"
    off = 0.18 if lvl > 0 else -0.18
    ax.text(d, lvl + off, f"{label}\n{fmt(d)}{tag}",
            ha="center", va=va, fontsize=10.5, fontweight="bold", color=c,
            linespacing=1.25, zorder=4)

# today marker
ax.axvline(today, color="#c62828", lw=1.8, ls="--", zorder=2)
ax.text(today, 2.55, f"TODAY\n{MON[today.month]} {today.day}", ha="center", va="bottom",
        fontsize=10, fontweight="bold", color="#c62828")

# axis cosmetics
ax.set_xlim(date(2026, 6, 1), date(2027, 2, 20))
ax.set_ylim(-2.6, 2.9)
ax.yaxis.set_visible(False)
for s in ["left", "right", "top"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_position(("data", -2.6))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda x, _pos: (lambda dt: f"{MON[dt.month]}\n{dt.year}")(mdates.num2date(x))))
ax.tick_params(axis="x", labelsize=10)

# legend
handles = [plt.Line2D([0],[0], marker="o", ls="", ms=11, color=color[k],
           label=fname[k]) for k in ["robo","vision","hci"]]
handles.append(plt.Line2D([0],[0], marker="o", ls="", ms=11, color="white",
               markeredgecolor="#555", markeredgewidth=2, label="estimated (hollow)"))
ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=False, ncol=4)

ax.set_title("Paper Submission Deadlines  ·  Jun 2026 → Jan 2027  (robotics-centric)",
             fontsize=15, fontweight="bold", pad=14)

plt.tight_layout()
plt.savefig("/home/mingi/AutoDex/deadline_timeline_2026.png", dpi=160, bbox_inches="tight")
print("saved deadline_timeline_2026.png")
