import json, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/003-theory-update/results"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
})
df = pd.read_csv(os.path.join(RES, "theory_table.csv"))
tr = json.load(open(os.path.join(RES, "transfer_results.json")))
TRAITS = ["evil", "sycophantic", "hallucinating"]
SHORT = {"Mistral-7B-Instruct-v0.2": "Mistral\n(e1)", "Llama-3.1-8B-Instruct": "Llama3.1\n(e2)",
         "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1\n(e4)", "Qwen3-8B-Nonthinking": "Qwen3\n(e5)"}

# ---------- Figure 1: cross-model cosine heatmaps ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
models = tr["matrices"]["evil"]["models"]
labs = [SHORT[m] for m in models]
for ax, trait in zip(axes, TRAITS):
    M = np.array(tr["matrices"][trait]["cosine"])
    im = ax.imshow(M, vmin=-0.1, vmax=1.0, cmap="cividis")
    ax.set_xticks(range(len(models))); ax.set_yticks(range(len(models)))
    ax.set_xticklabels(labs, fontsize=8); ax.set_yticklabels(labs, fontsize=8)
    ax.set_title(f"{trait}")
    ax.grid(False)
    for i in range(len(models)):
        for j in range(len(models)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.55 else "black", fontsize=8)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="cosine similarity")
fig.suptitle("Cross-model persona-direction cosine at each model's selected layer (hidden dim = 4096)\n"
             "only Llama↔DeepSeek (same architecture family) align; independently-trained families ≈ 0",
             fontsize=11)
fig.savefig(os.path.join(RES, "fig1_transfer_heatmap.png"), facecolor="white")
plt.close(fig)
print("wrote fig1_transfer_heatmap.png")

# ---------- Figure 2: partial/marginal dependence ----------
ALIGN_COLOR = {"base-RLHF": OKABE_ITO[0], "newer-RLHF": OKABE_ITO[2], "reasoning-distill": OKABE_ITO[3]}
fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
panels = [("era", "steer_delta", "Steering effect size (Δ over baseline)", "era (2023→2025)"),
          ("depth_frac", "steer_delta", "Steering effect size (Δ over baseline)", "selected-layer depth fraction"),
          ("era", "monitor_r", "Monitoring Pearson r (C2)", "era (2023→2025)"),
          ("depth_frac", "monitor_r", "Monitoring Pearson r (C2)", "selected-layer depth fraction")]
for ax, (xcol, ycol, ylab, xlab) in zip(axes.ravel(), panels):
    for al, c in ALIGN_COLOR.items():
        sub = df[df.alignment == al]
        ax.scatter(sub[xcol], sub[ycol], color=c, s=55, label=al, edgecolor="k", linewidth=0.4, zorder=3)
    lr = stats.linregress(df[xcol], df[ycol])
    xs = np.linspace(df[xcol].min(), df[xcol].max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, color="0.35", ls="--", lw=1.4,
            label=f"OLS slope={lr.slope:.3f}, p={lr.pvalue:.2f}, R²={lr.rvalue**2:.2f}")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.legend(fontsize=7.5, loc="best")
axes[0, 1].axvspan(0.5, 0.7, color=OKABE_ITO[6], alpha=0.18, zorder=0)
axes[1, 1].axvspan(0.5, 0.7, color=OKABE_ITO[6], alpha=0.18, zorder=0)
fig.suptitle("Marginal dependence of steering effect & projection-monitoring r on era and layer-depth "
             "(15 model×trait points; low power)\nshaded band = hypothesised 0.5–0.7 late-layer locus",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(RES, "fig2_partial_dependence.png"), facecolor="white")
plt.close(fig)
print("wrote fig2_partial_dependence.png")

# ---------- Figure 3: depth-fraction era-invariance + DeepSeek deviation ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
for al, c in ALIGN_COLOR.items():
    sub = df[df.alignment == al]
    ax1.scatter(sub.era, sub.depth_frac, color=c, s=55, label=al, edgecolor="k", linewidth=0.4, zorder=3)
ax1.axhspan(0.5, 0.7, color=OKABE_ITO[6], alpha=0.2, label="0.5–0.7 depth band")
lr = stats.linregress(df.era, df.depth_frac)
xs = np.linspace(1, 5, 20)
ax1.plot(xs, lr.intercept + lr.slope * xs, color="0.35", ls="--", lw=1.4,
         label=f"slope={lr.slope:.3f}, p={lr.pvalue:.2f}")
ax1.set_xlabel("era (2023→2025)"); ax1.set_ylabel("selected-layer depth fraction")
ax1.set_title("Effective steering layer is era-invariant (~0.55 depth)")
ax1.set_xticks([1, 2, 3, 4, 5]); ax1.legend(fontsize=7.5)

# DeepSeek deviation: per-model mean of each normalized axis
metrics = {"steer_delta": "steer Δ", "monitor_r": "monitor r", "depth_frac": "depth frac", "sep_auc": "sep AUC"}
order = ["Mistral-7B-Instruct-v0.2", "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct",
         "DeepSeek-R1-Distill-Llama-8B", "Qwen3-8B-Nonthinking"]
short2 = {"Mistral-7B-Instruct-v0.2": "Mistral", "Llama-3.1-8B-Instruct": "Llama3.1",
          "Qwen2.5-7B-Instruct": "Qwen2.5", "DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1",
          "Qwen3-8B-Nonthinking": "Qwen3"}
mm = df.groupby("model")[list(metrics)].mean()
x = np.arange(len(order)); w = 0.2
for k, (col, lab) in enumerate(metrics.items()):
    vals = [mm.loc[m, col] for m in order]
    if col == "steer_delta":
        vals = [v / 100 for v in vals]; lab += " /100"
    ax2.bar(x + (k - 1.5) * w, vals, w, label=lab, color=OKABE_ITO[k])
ax2.set_xticks(x); ax2.set_xticklabels([short2[m] for m in order], rotation=20, fontsize=8)
ax2.set_ylabel("model-mean value (0–1 scale)")
ax2.set_title("DeepSeek-R1-Distill: monitor r & steering Δ collapse")
ax2.legend(fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig3_depth_invariance_deepseek.png"), facecolor="white")
plt.close(fig)
print("wrote fig3_depth_invariance_deepseek.png")
