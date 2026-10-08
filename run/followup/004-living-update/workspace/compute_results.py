#!/usr/bin/env python
"""Followup 002 aggregation: compute C5/C8/C1/C2/C13 for the 5-model era ladder.

- Qwen2.5-7B-Instruct is REUSED verbatim from the replication artifacts on disk
  (never recomputed here beyond reading the paper-judge trait columns already in them).
- The 4 other models (Mistral, Llama, DeepSeek, Qwen3) are computed from THIS study's
  paper-exact gpt-4.1-mini rejudge sidecars + monitor projection columns + step8 summaries.

All trait scores are the paper-exact gpt-4.1-mini judge (same instrument for old and new).
Writes results/aggregate.json and results/tidy_results_long.csv.
"""
import os, json, glob
import numpy as np, pandas as pd
from scipy.stats import pearsonr

FU = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update"
RUN = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run"
CB = f"{FU}/workspace/codebase"                 # this study's generation outputs
REJ = f"{FU}/results/rejudged"                  # this study's paper-judge sidecars
STEP8 = f"{FU}/results/step8"
LOGS = f"{FU}/results/logs"
# reused replication artifacts (read-only history)
REPL_CB = f"{RUN}/replication/codebase/eval_persona_eval/Qwen2.5-7B-Instruct"
REPL_STEP8 = f"{RUN}/replication/analysis/step8_separability_summary.csv"

TRAITS = ["evil", "sycophantic", "hallucinating"]
COEFS_CONFIRM = [0.5, 1.0, 1.5, 2.0, 2.5]

# era ladder (release dates per 001 spec model table)
LADDER = [
    ("Mistral-7B-Instruct-v0.2", 1, "2023-12", 32, "new"),
    ("Llama-3.1-8B-Instruct",    2, "2024-07", 32, "new"),
    ("Qwen2.5-7B-Instruct",      3, "2024-09", 28, "reused"),
    ("DeepSeek-R1-Distill-Llama-8B", 4, "2025-01", 32, "new"),
    ("Qwen3-8B-Nonthinking",     5, "2025-04", 36, "new"),
]

def relp(p):
    return os.path.relpath(p, RUN)

# ---------- reused Qwen2.5 (from replication artifacts) ----------
def qwen25():
    out = {"selected_layer": {"evil": 20, "sycophantic": 20, "hallucinating": 16}}
    # C5 baseline
    out["C5_baseline"] = {}
    for t in TRAITS:
        f = f"{REPL_CB}/{t}_baseline.csv"
        out["C5_baseline"][t] = {"value": round(float(pd.read_csv(f)[t].mean()), 3),
                                 "n": int(len(pd.read_csv(f))), "source": relp(f)}
    # C8 selected layer = argmax trait_mean over layer (any coef) in full sweep summary
    ss = pd.read_csv(f"{REPL_CB}/steer_sweep_summary.csv")
    out["C8_selected_layer"] = {}
    out["C1_curve"] = {}
    out["C1_effect"] = {}
    for t in TRAITS:
        sub = ss[ss.trait == t]
        # selected layer = layer achieving the max trait_mean anywhere in the sweep
        best = sub.loc[sub.trait_mean.idxmax()]
        sel = int(best.layer)
        out["C8_selected_layer"][t] = {"value": sel, "source": relp(f"{REPL_CB}/steer_sweep_summary.csv")}
        # C1 curve at the selected layer across available coefs
        curve = sub[sub.layer == sel].sort_values("coef")[["coef", "trait_mean"]]
        cvals = {float(r.coef): round(float(r.trait_mean), 3) for r in curve.itertuples()}
        out["C1_curve"][t] = {"layer": sel, "curve": cvals,
                              "source": relp(f"{REPL_CB}/steer_sweep_summary.csv")}
        maxcoef_mean = float(curve.trait_mean.iloc[-1])
        base = out["C5_baseline"][t]["value"]
        out["C1_effect"][t] = {"max_coef_trait_mean": round(maxcoef_mean, 3),
                               "delta_over_baseline": round(maxcoef_mean - base, 3),
                               "layer": sel}
    # C2 pearson r from monitor pos/neg (trait col + proj col in same file)
    out["C2_pearson"] = {}
    for t in TRAITS:
        sel = out["C8_selected_layer"][t]["value"]
        projcol = f"Qwen2.5-7B-Instruct_{t}_response_avg_diff_prompt_last_proj_layer{sel}"
        dfs = []
        for pn in ["pos", "neg"]:
            f = f"{REPL_CB}/{t}_monitor_{pn}.csv"
            d = pd.read_csv(f)
            dfs.append(d[[t, projcol]].rename(columns={t: "score", projcol: "proj"}))
        d = pd.concat(dfs).dropna()
        r, p = pearsonr(d.proj, d.score)
        out["C2_pearson"][t] = {"value": round(float(r), 3), "n": int(len(d)), "layer": sel,
                                "source": relp(f"{REPL_CB}/{t}_monitor_{{pos,neg}}.csv")}
    # C13 AUC
    s8 = pd.read_csv(REPL_STEP8)
    out["C13_auc"] = {}
    for _, row in s8.iterrows():
        key = row["trait"] if row["dataset"] != "mistake_opinions" else "EM_like"
        out["C13_auc"][key] = {"value": round(float(row.auc_separability), 4),
                               "dataset": row["dataset"], "source": relp(REPL_STEP8)}
    return out

# ---------- new models (from this study) ----------
def new_model(slug):
    out = {}
    sel = json.load(open(f"{LOGS}/{slug}/sel_layer.json"))
    out["selected_layer"] = sel
    rej = f"{REJ}/{slug}"
    # C5 baseline (mean gpt_<trait>)
    out["C5_baseline"] = {}
    for t in TRAITS:
        f = f"{rej}/{t}_baseline.csv"
        s = pd.read_csv(f)[f"gpt_{t}"]
        out["C5_baseline"][t] = {"value": round(float(s.mean()), 3), "n": int(s.notna().sum()),
                                 "source": relp(f)}
    # C8 selected layer (argmax over coarse grid, already computed by pick_layers)
    out["C8_selected_layer"] = {t: {"value": int(sel[t]), "source": relp(f"{LOGS}/{slug}/sel_layer.json")}
                                for t in TRAITS}
    # C1 curve at selected layer from confirm sidecars
    out["C1_curve"] = {}
    out["C1_effect"] = {}
    for t in TRAITS:
        L = sel[t]
        cvals = {}
        for c in COEFS_CONFIRM:
            f = f"{rej}/confirm/{t}_steer_layer{L}_coef{c}.csv"
            if os.path.exists(f):
                cvals[c] = round(float(pd.read_csv(f)[f"gpt_{t}"].mean()), 3)
        out["C1_curve"][t] = {"layer": L, "curve": cvals, "source": relp(f"{rej}/confirm/")}
        maxcoef_mean = cvals[max(cvals)] if cvals else None
        base = out["C5_baseline"][t]["value"]
        out["C1_effect"][t] = {"max_coef_trait_mean": maxcoef_mean,
                               "delta_over_baseline": round(maxcoef_mean - base, 3) if maxcoef_mean is not None else None,
                               "layer": L}
    # C2 pearson r: proj col from CB monitor csv, gpt_<trait> from rejudge sidecar (row aligned)
    out["C2_pearson"] = {}
    for t in TRAITS:
        L = sel[t]
        projcol = f"{slug}_{t}_response_avg_diff_prompt_last_proj_layer{L}"
        parts = []
        for pn in ["pos", "neg"]:
            mon = pd.read_csv(f"{CB}/eval_persona_eval/{slug}/{t}_monitor_{pn}.csv")
            side = pd.read_csv(f"{rej}/{t}_monitor_{pn}.csv")
            assert len(mon) == len(side), f"{slug} {t} {pn} len mismatch"
            parts.append(pd.DataFrame({"proj": mon[projcol].values, "score": side[f"gpt_{t}"].values}))
        d = pd.concat(parts).dropna()
        r, p = pearsonr(d.proj, d.score)
        out["C2_pearson"][t] = {"value": round(float(r), 3), "n": int(len(d)), "layer": L,
                                "source": f"{relp(CB)}/eval_persona_eval/{slug}/{t}_monitor_{{pos,neg}}.csv + {relp(rej)}/{t}_monitor_{{pos,neg}}.csv"}
    # C13 AUC
    s8 = pd.read_csv(f"{STEP8}/{slug}/step8_separability_summary.csv")
    out["C13_auc"] = {}
    for _, row in s8.iterrows():
        key = row["trait"] if row["dataset"] != "mistake_opinions" else "EM_like"
        out["C13_auc"][key] = {"value": round(float(row.auc_separability), 4),
                               "dataset": row["dataset"], "source": relp(f"{STEP8}/{slug}/step8_separability_summary.csv")}
    return out

def main():
    agg = {"ladder": [], "models": {}}
    for slug, era, rel, nlayers, kind in LADDER:
        agg["ladder"].append({"model": slug, "era": era, "release": rel, "n_layers": nlayers, "kind": kind})
        agg["models"][slug] = qwen25() if kind == "reused" else new_model(slug)
        agg["models"][slug]["_meta"] = {"era": era, "release": rel, "n_layers": nlayers, "kind": kind}
    json.dump(agg, open(f"{FU}/results/aggregate.json", "w"), indent=2)

    # tidy long format
    rows = []
    for slug, era, rel, nlayers, kind in LADDER:
        m = agg["models"][slug]
        for t in TRAITS:
            rows.append(dict(model=slug, era=era, release=rel, trait=t, claim="C5_baseline",
                             value=m["C5_baseline"][t]["value"], layer="", kind=kind,
                             source=m["C5_baseline"][t]["source"]))
            rows.append(dict(model=slug, era=era, release=rel, trait=t, claim="C8_selected_layer",
                             value=m["C8_selected_layer"][t]["value"], layer=m["C8_selected_layer"][t]["value"],
                             kind=kind, source=m["C8_selected_layer"][t]["source"]))
            eff = m["C1_effect"][t]
            rows.append(dict(model=slug, era=era, release=rel, trait=t, claim="C1_steer_maxcoef",
                             value=eff["max_coef_trait_mean"], layer=eff["layer"], kind=kind,
                             source=m["C1_curve"][t]["source"]))
            rows.append(dict(model=slug, era=era, release=rel, trait=t, claim="C1_steer_delta",
                             value=eff["delta_over_baseline"], layer=eff["layer"], kind=kind,
                             source=m["C1_curve"][t]["source"]))
            rows.append(dict(model=slug, era=era, release=rel, trait=t, claim="C2_pearson_r",
                             value=m["C2_pearson"][t]["value"], layer=m["C2_pearson"][t]["layer"], kind=kind,
                             source=m["C2_pearson"][t]["source"]))
            if t in m["C13_auc"]:
                rows.append(dict(model=slug, era=era, release=rel, trait=t, claim="C13_auc",
                                 value=m["C13_auc"][t]["value"], layer="", kind=kind,
                                 source=m["C13_auc"][t]["source"]))
    pd.DataFrame(rows).to_csv(f"{FU}/results/tidy_results_long.csv", index=False)
    print("wrote aggregate.json and tidy_results_long.csv")
    # quick console summary
    for slug, era, rel, nlayers, kind in LADDER:
        m = agg["models"][slug]
        print(f"\n=== {slug} (era {era}, {rel}, {kind}) sel={m['selected_layer']} ===")
        for t in TRAITS:
            print(f"  {t:14s} base={m['C5_baseline'][t]['value']:6.2f}  steer_max={m['C1_effect'][t]['max_coef_trait_mean']}  "
                  f"r={m['C2_pearson'][t]['value']:.3f}  auc={m['C13_auc'].get(t,{}).get('value')}")

if __name__ == "__main__":
    main()
