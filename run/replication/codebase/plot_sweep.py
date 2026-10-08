"""Step 4 analysis: Figure-3-style layer/coef sweep plots + argmax-layer selection."""
import os, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUM = "eval_persona_eval/Qwen2.5-7B-Instruct/steer_sweep_summary.csv"
OUTDIR = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/analysis"
PAPER_SEL = {"evil": 20, "sycophantic": 20, "hallucinating": 16}

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(SUM)
    traits = list(df.trait.unique())
    fig, axes = plt.subplots(1, len(traits), figsize=(6*len(traits), 4.5))
    if len(traits) == 1: axes = [axes]
    selection = {}
    for ax, trait in zip(axes, traits):
        sub = df[df.trait == trait]
        for layer in sorted(sub.layer.unique()):
            s = sub[sub.layer == layer].sort_values("coef")
            ax.plot(s.coef, s.trait_mean, marker="o", label=f"L{layer}")
        ax.set_title(trait); ax.set_xlabel("steering coef"); ax.set_ylabel("trait score")
        ax.legend(fontsize=7, ncol=2)
        # argmax layer: layer with highest trait score at each coef; report per-coef and overall
        per_coef = {}
        for coef in sorted(sub.coef.unique()):
            sc = sub[sub.coef == coef]
            best = sc.loc[sc.trait_mean.idxmax()]
            per_coef[float(coef)] = dict(layer=int(best.layer), trait=float(best.trait_mean))
        # "most informative layer": at the largest coef where coherence still reasonable,
        # take the layer maximizing trait at a representative coef (paper uses fixed coef).
        # Report the modal argmax layer across coefs 1.0-2.0 (coherence not collapsed).
        mid = sub[sub.coef.isin([1.0, 1.5, 2.0])]
        arg = mid.loc[mid.groupby("coef").trait_mean.idxmax()]
        modal = arg.layer.mode().iloc[0]
        selection[trait] = dict(per_coef_argmax=per_coef, modal_argmax_layer_c1_2=int(modal),
                                paper_selected_layer=PAPER_SEL.get(trait))
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/step4_layer_sweep.png", dpi=120)
    json.dump(selection, open(f"{OUTDIR}/step4_layer_selection.json", "w"), indent=2)
    print(json.dumps(selection, indent=2))
    print("saved", OUTDIR)

if __name__ == "__main__":
    main()
