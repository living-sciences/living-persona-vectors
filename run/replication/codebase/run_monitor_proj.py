"""Step 5 (projection + correlation): prompt-induced monitoring.

For each trait, load the monitor CSVs (pos + neg system prompts), compute the
last-prompt-token projection onto the persona vector (response_avg_diff) at the
selected layer, then compute Pearson r between the per-response trait score and
the projection (pooled pos+neg), and also at the per-system-prompt-mean level.
Loads the HF model once for all traits. Saves a results JSON + scatter figure.
"""
import os, json, torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVALDIR = "eval_persona_eval/Qwen2.5-7B-Instruct"
VECDIR = "persona_vectors/Qwen2.5-7B-Instruct"
OUTDIR = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/analysis"
SEL_LAYER = {"evil": 20, "sycophantic": 20, "hallucinating": 16}

def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)

def add_projection(model, tokenizer, df, vector, layer):
    projs = []
    for _, d in df.iterrows():
        prompt, answer = d["prompt"], d["answer"]
        inputs = tokenizer(prompt + str(answer), return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last_prompt = out.hidden_states[layer][:, prompt_len - 1, :].detach().cpu().float()
        projs.append(a_proj_b(last_prompt, vector).item())
    return projs

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Loading HF model once...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, trait in zip(axes, ["evil", "sycophantic", "hallucinating"]):
        layer = SEL_LAYER[trait]
        vecs = torch.load(f"{VECDIR}/{trait}_response_avg_diff.pt", weights_only=False)
        vector = vecs[layer].float()
        col = f"Qwen2.5-7B-Instruct_{trait}_response_avg_diff_prompt_last_proj_layer{layer}"
        frames = []
        for ptype in ["pos", "neg"]:
            path = f"{EVALDIR}/{trait}_monitor_{ptype}.csv"
            df = pd.read_csv(path)
            df[col] = add_projection(model, tokenizer, df, vector, layer)
            df["ptype"] = ptype
            df.to_csv(path, index=False)
            frames.append(df)
            print(f"{trait} {ptype}: projected {len(df)} rows", flush=True)
        alldf = pd.concat(frames, ignore_index=True)
        # per-response correlation (drop rows with missing trait score)
        sub = alldf.dropna(subset=[trait, col])
        r_resp, p_resp = pearsonr(sub[col], sub[trait])
        # per-system-prompt-mean correlation (group by question_id prefix instruction index)
        # system prompt identity encoded in question_id suffix _k; use ptype+k grouping
        alldf["sysid"] = alldf["question_id"].astype(str).str.extract(r"_(pos|neg)_(\d+)$").agg("_".join, axis=1)
        grp = alldf.dropna(subset=[trait, col]).groupby("sysid").agg(trait_mean=(trait, "mean"), proj_mean=(col, "mean")).reset_index()
        if len(grp) >= 3:
            r_sys, p_sys = pearsonr(grp["proj_mean"], grp["trait_mean"])
        else:
            r_sys, p_sys = float("nan"), float("nan")
        results[trait] = dict(layer=layer, n_rows=int(len(sub)),
                              pearson_r_per_response=float(r_resp), p_per_response=float(p_resp),
                              n_sysprompts=int(len(grp)),
                              pearson_r_per_sysprompt=float(r_sys), p_per_sysprompt=float(p_sys))
        print(f"=== {trait} (layer {layer}): per-response r={r_resp:.3f} (p={p_resp:.1e}), "
              f"per-sysprompt r={r_sys:.3f} over {len(grp)} prompts ===", flush=True)
        ax.scatter(sub[col], sub[trait], s=8, alpha=0.3)
        ax.set_xlabel("last-prompt-token projection")
        ax.set_ylabel(f"{trait} score")
        ax.set_title(f"{trait} (L{layer}) r={r_resp:.2f}")
    plt.tight_layout()
    fig.savefig(f"{OUTDIR}/step5_monitoring_scatter.png", dpi=120)
    json.dump(results, open(f"{OUTDIR}/step5_monitoring_correlations.json", "w"), indent=2)
    print("saved analysis to", OUTDIR, flush=True)

if __name__ == "__main__":
    main()
