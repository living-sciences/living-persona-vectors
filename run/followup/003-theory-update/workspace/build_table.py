"""Build the tidy per-(model,trait) analysis table from 002's aggregate.json."""
import json, os, numpy as np, pandas as pd
from model_meta import MODEL_META, TRAITS

FU002 = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update"
OUT = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/003-theory-update/results"

agg = json.load(open(os.path.join(FU002, "results/aggregate.json")))
rows = []
for model, m in MODEL_META.items():
    md = agg["models"][model]
    for trait in TRAITS:
        sel = md["C8_selected_layer"][trait]["value"]
        eff = md["C1_effect"][trait]
        rows.append(dict(
            model=model, family=m["family"], era=m["era"], release=m["release"],
            params_b=m["params_b"], n_layers=m["n_layers"], hidden=m["hidden"],
            alignment=m["alignment"], trait=trait,
            selected_layer=sel,
            depth_frac=sel / m["n_layers"],
            baseline_C5=md["C5_baseline"][trait]["value"],
            steer_maxcoef=eff["max_coef_trait_mean"],
            steer_delta=eff["delta_over_baseline"],
            monitor_r=md["C2_pearson"][trait]["value"],
            sep_auc=md["C13_auc"][trait]["value"],
            log10_params=np.log10(m["params_b"] * 1e9),
        ))
df = pd.DataFrame(rows)
os.makedirs(OUT, exist_ok=True)
df.to_csv(os.path.join(OUT, "theory_table.csv"), index=False)
print(df.to_string(index=False))
print("\nwrote", os.path.join(OUT, "theory_table.csv"))
