#!/usr/bin/env python
"""Followup 002 figures for the chat canvas."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FU = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update"
agg = json.load(open(f"{FU}/results/aggregate.json"))

OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})
TRAITS = ["evil", "sycophantic", "hallucinating"]
TCOL = {"evil": OKABE_ITO[3], "sycophantic": OKABE_ITO[0], "hallucinating": OKABE_ITO[2]}
ladder = agg["ladder"]
slugs = [d["model"] for d in ladder]
eras = [d["era"] for d in ladder]
# short x labels
SHORT = {"Mistral-7B-Instruct-v0.2": "Mistral-7B\n2023-12",
         "Llama-3.1-8B-Instruct": "Llama-3.1-8B\n2024-07",
         "Qwen2.5-7B-Instruct": "Qwen2.5-7B*\n2024-09",
         "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1\nDistill-8B\n2025-01",
         "Qwen3-8B-Nonthinking": "Qwen3-8B\n2025-04"}
xlabels = [SHORT[s] for s in slugs]

def marker_for(slug):
    return "D" if agg["models"][slug]["_meta"]["kind"] == "reused" else "o"

# ---------------- Figure 1: over-time arc, 3 panels ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
panels = [
    ("A. Steering effect size", "max-coef trait score − baseline", "C1_effect", "delta_over_baseline"),
    ("B. Projection-monitoring", "Pearson r  (proj vs trait)", "C2_pearson", "value"),
    ("C. Sample separability", "AUC (misaligned vs normal)", "C13_auc", "value"),
]
for ax, (title, ylab, claim, field) in zip(axes, panels):
    for t in TRAITS:
        xs, ys = [], []
        for d in ladder:
            m = agg["models"][d["model"]]
            entry = m[claim].get(t)
            if entry is None:
                continue
            val = entry[field]
            if val is None:
                continue
            xs.append(d["era"]); ys.append(val)
        ax.plot(xs, ys, "-", color=TCOL[t], alpha=0.55, zorder=1)
        for d in ladder:
            m = agg["models"][d["model"]]
            entry = m[claim].get(t)
            if entry is None or entry[field] is None:
                continue
            ax.scatter(d["era"], entry[field], color=TCOL[t], marker=marker_for(d["model"]),
                       s=90, edgecolor="black", linewidth=0.6, zorder=3,
                       label=t if d["era"] == 1 else None)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xticks(eras); ax.set_xticklabels(xlabels, fontsize=8.5)
    if claim == "C2_pearson":
        ax.set_ylim(0, 1.0)
    if claim == "C13_auc":
        ax.set_ylim(0.85, 1.005)
axes[0].legend(title="trait", fontsize=9, loc="lower right")
fig.suptitle("Persona-vector mechanisms across the 2023→2025 chat-model era ladder\n"
             "(paper-exact gpt-4.1-mini judge; * = Qwen2.5-7B reused from replication, diamond marker)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{FU}/results/fig1_over_time.png", facecolor="white")
print("wrote fig1_over_time.png")

# ---------------- Figure 2: steering dose-response curves ----------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, t in zip(axes, TRAITS):
    for i, d in enumerate(ladder):
        m = agg["models"][d["model"]]
        curve = m["C1_curve"][t]["curve"]
        cs = sorted(float(k) for k in curve)
        ys = [curve[k] if k in curve else curve[str(k)] for k in
              (cs if all(k in curve for k in cs) else [str(c) for c in cs])]
        # curve keys may be str after json round-trip
        cs2 = sorted(float(k) for k in curve.keys())
        ys = [curve[str(k)] if str(k) in curve else curve[k] for k in cs2]
        ax.plot(cs2, ys, marker=marker_for(d["model"]), color=OKABE_ITO[i % len(OKABE_ITO)],
                label=f"{SHORT[d['model']].splitlines()[0]} (L{m['C1_curve'][t]['layer']})")
    ax.set_title(f"{t}")
    ax.set_xlabel("steering coefficient")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("trait score (gpt-4.1-mini, 0–100)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Steering dose-response at each model's selected layer (C1): trait score rises monotonically with coefficient",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f"{FU}/results/fig2_steering_curves.png", facecolor="white")
print("wrote fig2_steering_curves.png")
