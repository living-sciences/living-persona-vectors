#!/usr/bin/env python
"""Pick per-trait selected steering layer = argmax (over the coarse layer x coef grid)
of the paper-exact gpt-judge trait_mean. Reads rejudge sidecars.
Usage: pick_layers.py <SLUG> <COARSE_LAYERS_CSV> <ACTIVE_TRAITS_CSV>"""
import sys, os, json, glob
import numpy as np, pandas as pd

SLUG, COARSE, ACTIVE = sys.argv[1], sys.argv[2], sys.argv[3]
FU = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update"
REJ = f"{FU}/results/rejudged/{SLUG}"
layers = [int(x) for x in COARSE.split(",")]
traits = [t for t in ACTIVE.split(",") if t]
COEFS = [1.0, 1.5, 2.0]

sel = {}
grid = {}
for t in traits:
    best_mean, best_layer, best_coef = -1e9, None, None
    rows = []
    for L in layers:
        for c in COEFS:
            p = f"{REJ}/{t}_steer_layer{L}_coef{c}.csv"
            if not os.path.exists(p):
                rows.append((L, c, None)); continue
            s = pd.read_csv(p)[f"gpt_{t}"]
            m = float(s.mean())
            rows.append((L, c, m))
            if m > best_mean:
                best_mean, best_layer, best_coef = m, L, c
    sel[t] = best_layer
    grid[t] = rows
    print(f"{t}: selected layer={best_layer} (coef {best_coef}, gpt_trait_mean={best_mean:.2f})")
    for L, c, m in rows:
        print(f"    L{L} c{c}: {m}")

os.makedirs(f"{FU}/results/logs/{SLUG}", exist_ok=True)
json.dump(sel, open(f"{FU}/results/logs/{SLUG}/sel_layer.json", "w"))
json.dump(grid, open(f"{FU}/results/logs/{SLUG}/coarse_grid.json", "w"), indent=1)
print("wrote sel_layer.json:", sel)
