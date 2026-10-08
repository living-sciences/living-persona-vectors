"""Analysis 2: cross-family persona-direction cosine at each model's selected layer.
Vectors: persona_vectors/<slug>/<trait>_response_avg_diff.pt, shape [n_layers+1, hidden];
row index = 1-indexed layer (row 0 = embedding). Only same-hidden-dim pairs are comparable.
"""
import json, os, itertools, numpy as np, pandas as pd, torch
from model_meta import MODEL_META, TRAITS

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/003-theory-update/results"
PV = {
    "Mistral-7B-Instruct-v0.2": "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update/workspace/codebase/persona_vectors",
    "Llama-3.1-8B-Instruct": "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update/workspace/codebase/persona_vectors",
    "DeepSeek-R1-Distill-Llama-8B": "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update/workspace/codebase/persona_vectors",
    "Qwen3-8B-Nonthinking": "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update/workspace/codebase/persona_vectors",
    "Qwen2.5-7B-Instruct": "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/codebase/persona_vectors",
}
df = pd.read_csv(os.path.join(RES, "theory_table.csv"))


def sel_layer(model, trait):
    return int(df[(df.model == model) & (df.trait == trait)].selected_layer.iloc[0])


def load_vec(model, trait, layer):
    t = torch.load(os.path.join(PV[model], model, f"{trait}_response_avg_diff.pt"),
                   map_location="cpu").float().numpy()
    return t[layer]  # row index = 1-indexed layer


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


DIM4096 = [m for m in MODEL_META if MODEL_META[m]["hidden"] == 4096]  # Mistral,Llama,DeepSeek,Qwen3
SAME_FAMILY_PAIR = ("Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B")

out = {"hidden_dims": {m: MODEL_META[m]["hidden"] for m in MODEL_META},
       "comparable_set_4096": DIM4096,
       "excluded_dim_mismatch": {"Qwen2.5-7B-Instruct": 3584},
       "per_trait": {}, "matrices": {}}
sf_cos, cf_cos = [], []
for trait in TRAITS:
    vecs = {m: load_vec(m, trait, sel_layer(m, trait)) for m in DIM4096}
    M = np.zeros((len(DIM4096), len(DIM4096)))
    for i, a in enumerate(DIM4096):
        for j, b in enumerate(DIM4096):
            M[i, j] = cos(vecs[a], vecs[b])
    out["matrices"][trait] = {"models": DIM4096, "cosine": np.round(M, 4).tolist()}
    pair_c = {}
    for a, b in itertools.combinations(DIM4096, 2):
        c = cos(vecs[a], vecs[b])
        same = (set((a, b)) == set(SAME_FAMILY_PAIR))
        pair_c[f"{a} vs {b}"] = dict(cosine=round(c, 4),
                                     pair="same-family(Llama<->DeepSeek)" if same else "cross-family")
        (sf_cos if same else cf_cos).append(c)
    out["per_trait"][trait] = pair_c

out["summary"] = dict(
    same_family_pairs_n=len(sf_cos), cross_family_pairs_n=len(cf_cos),
    mean_cosine_same_family=round(float(np.mean(sf_cos)), 4),
    mean_cosine_cross_family=round(float(np.mean(cf_cos)), 4),
    min_max_same=[round(min(sf_cos), 4), round(max(sf_cos), 4)],
    min_max_cross=[round(min(cf_cos), 4), round(max(cf_cos), 4)],
    interpretation=("Llama<->DeepSeek share residual basis (DeepSeek is a reasoning-distill of "
                    "Llama-3.1-8B) so directions align strongly; independently-trained families "
                    "have unaligned bases so cross-family raw cosine is ~0 (not a claim about "
                    "conceptual dissimilarity, only about basis non-alignment)."))

# robustness: Llama<->DeepSeek at a MATCHED layer (both at DeepSeek's selected layer)
matched = {}
for trait in TRAITS:
    L = sel_layer("DeepSeek-R1-Distill-Llama-8B", trait)
    a = load_vec("Llama-3.1-8B-Instruct", trait, L)
    b = load_vec("DeepSeek-R1-Distill-Llama-8B", trait, L)
    matched[trait] = dict(layer=L, cosine=round(cos(a, b), 4))
out["same_family_matched_layer"] = matched

json.dump(out, open(os.path.join(RES, "transfer_results.json"), "w"), indent=2)
# also write flat CSV of pair cosines
rows = []
for trait, pc in out["per_trait"].items():
    for pair, d in pc.items():
        rows.append(dict(trait=trait, pair=pair, kind=d["pair"], cosine=d["cosine"]))
pd.DataFrame(rows).to_csv(os.path.join(RES, "cosine_pairs.csv"), index=False)
print(json.dumps(out, indent=2))
