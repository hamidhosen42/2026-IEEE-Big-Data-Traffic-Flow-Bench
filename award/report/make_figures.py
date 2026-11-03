"""Generate the report figures from numbers stored in the repository (no competition data is read).

Inputs
  ../../all_submitted_notebooks/README.md   submission table (number, date, kernel, public LB, description)
  figdata/transfer.csv                      offline gain vs public-LB delta, each row with its TRACKER source
Outputs (figures/)
  lb_progression.pdf   public LB of every scored submission + running best
  lb_by_task.pdf       running-best increments summed by the task the change touched
  transfer.pdf         offline estimate vs realised public-LB delta (Task 1 lab J; Task 2 FM-PM)
  lb_steps.csv         the running-best chain with its category (traceability of lb_by_task.pdf)

Run: .venv/bin/python award/report/make_figures.py
"""
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# reference categorical palette (dataviz skill, light mode), fixed order
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "pdf.fonttype": 42,
})

# ------------------------------------------------------------------ submissions
rows = []
with open(os.path.join(ROOT, "all_submitted_notebooks", "README.md")) as f:
    for line in f:
        m = re.match(r"\|\s*(\d+)\s*\|\s*(\S+)\s*\|[^|]*\|\s*`([^`]*)`\s*\|\s*([0-9.]+|error)\s*\|(.*)\|\s*$", line)
        if m and m.group(4) != "error":
            rows.append(dict(n=int(m.group(1)), date=m.group(2), kernel=m.group(3), lb=float(m.group(4)),
                             desc=m.group(5).strip()))
assert rows and rows[-1]["lb"] == 0.88281, "submission table not parsed as expected"

# Task touched by every submission that set a new running best (from the README descriptions / TRACKER).
CAT = {
    6: "T2", 7: "T1", 8: "T1", 14: "T1", 11: "T2 ongoing", 13: "T2 onset", 15: "T2", 18: "mixed T1+T2", 20: "T1", 21: "T2 ongoing",
    23: "T1", 24: "T2 ongoing", 25: "T2 ongoing", 26: "T1", 27: "T2 ongoing", 28: "T1", 29: "T1", 30: "T2 ongoing",
    32: "T1", 33: "T1", 34: "T3 (recon/TV)", 35: "T2 ongoing", 37: "T3 (recon/TV)", 38: "T1", 39: "T2 ongoing",
    40: "T2 ongoing", 41: "T2 ongoing", 42: "T2 onset", 43: "T1", 47: "T2 ongoing", 50: "T2 ongoing", 53: "T1",
    57: "T1", 59: "T1", 61: "T1", 63: "T1", 66: "T1", 68: "T1", 70: "T1", 72: "T1", 76: "T1",
}
GROUP = {"T2": "Task 2 (model / both)", "T2 ongoing": "Task 2 ongoing", "T2 onset": "Task 2 onset",
         "T1": "Task 1 reconstruction", "T3 (recon/TV)": "Task 3 post-proc. (recon, TV)", "mixed T1+T2": "Combined T1+T2"}

start = next(r for r in rows if r["n"] == 5)            # soukeaizenz public notebook, 0.80939
best, chain = start["lb"], []
for r in rows:
    if r["n"] <= 5:
        continue
    if r["lb"] > best + 1e-12:
        chain.append(dict(n=r["n"], kernel=r["kernel"], lb=r["lb"], inc=r["lb"] - best, cat=CAT[r["n"]]))
        best = r["lb"]
with open(os.path.join(OUT, "lb_steps.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["n", "kernel", "lb", "inc", "cat"])
    w.writeheader()
    for c in chain:
        w.writerow({**c, "inc": f"{c['inc']:.5f}"})

# ------------------------------------------------------------------ Fig: LB progression
fig, ax = plt.subplots(figsize=(3.45, 2.4))
xs = [r["n"] for r in rows if r["n"] >= 5]
ys = [r["lb"] for r in rows if r["n"] >= 5]
ax.scatter(xs, ys, s=9, color=INK2, alpha=0.55, linewidths=0, label="submission (public LB)", zorder=2)
bx, by, b = [5], [start["lb"]], start["lb"]
for r in rows:
    if r["n"] > 5 and r["lb"] > b:
        b = r["lb"]
    if r["n"] > 5:
        bx.append(r["n"]); by.append(b)
ax.step(bx, by, where="post", color=C1, lw=1.6, label="running best", zorder=3)
ax.set_ylim(0.80, 0.888)
ax.set_xlabel("submission number (chronological)")
ax.set_ylabel("public LB (March)")
for n, txt, dy in [(5, "public base\n0.809", 0.004), (6, "T2 classifier", -0.004), (34, "FD recon", -0.006),
                   (40, "F03 ongoing", -0.007), (76, "final 0.8828", -0.009)]:
    y = next(r["lb"] for r in rows if r["n"] == n)
    ax.annotate(txt, (n, y), xytext=(n + 1.5, y + dy), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
ax.text(9.5, 0.8015, "zero-condition probes (0.737) below axis", fontsize=6, color=INK2)
ax.legend(loc="center right", frameon=False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "lb_progression.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ Fig: increments by task
tot = {}
for c in chain:
    tot[GROUP[c["cat"]]] = tot.get(GROUP[c["cat"]], 0.0) + c["inc"]
order = sorted(tot, key=lambda k: tot[k])
colors = {"Task 2 (model / both)": C1, "Task 2 ongoing": C1, "Task 2 onset": C1, "Task 1 reconstruction": C2,
          "Task 3 post-proc. (recon, TV)": C3, "Combined T1+T2": C4}
fig, ax = plt.subplots(figsize=(3.45, 1.75))
ax.barh(order, [tot[k] for k in order], color=[colors[k] for k in order], height=0.6, zorder=2)
for i, k in enumerate(order):
    ax.text(tot[k] + 0.0004, i, f"+{tot[k]:.4f}", va="center", fontsize=6.5, color=INK)
ax.set_xlabel("sum of running-best LB increments")
ax.set_xlim(0, max(tot.values()) * 1.25)
ax.grid(axis="y", visible=False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "lb_by_task.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ Fig: offline -> LB transfer
tr = list(csv.DictReader(open(os.path.join(HERE, "figdata", "transfer.csv"))))
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85))
for ax, panel, col, lab, unit in [(axs[0], "T1", C2, "lab $\\Delta J$", 1e3),
                                  (axs[1], "T2", C1, "FM-PM", 1e3)]:
    sub = [r for r in tr if r["panel"] == panel]
    x = [float(r["offline_gain"]) * unit for r in sub]
    y = [float(r["lb_delta"]) * unit for r in sub]
    lo, hi = min(x + y + [0]) * 1.1, max(x + y) * 1.15
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--", zorder=1)
    if panel == "T1":
        ax.plot([0, hi], [0, 0.5 * hi], color=INK2, lw=0.6, ls=":", zorder=1)
        ax.text(hi * 0.62, hi * 0.24, "0.5x", fontsize=6, color=INK2)
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.scatter(x, y, s=16, color=col, edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(lab + " ($10^{-3}$)", fontsize=7)
    ax.set_title("Task 1 members / blackout" if panel == "T1" else "Task 2 ongoing probes", fontsize=7)
axs[0].set_ylabel("public-LB $\\Delta$ ($10^{-3}$)", fontsize=7)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(os.path.join(OUT, "transfer.pdf"))
plt.close(fig)

for c in chain:
    print(f"#{c['n']:>2} {c['kernel']:<14} {c['lb']:.5f} +{c['inc']:.5f} {c['cat']}")
print({k: round(v, 5) for k, v in tot.items()}, "sum", round(sum(tot.values()), 5))
