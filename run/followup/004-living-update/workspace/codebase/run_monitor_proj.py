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

import json as _json
MODEL = os.environ.get("PV_MODEL", "Qwen/Qwen2.5-7B-Instruct")
SLUG = os.environ.get("PV_SLUG", "Qwen2.5-7B-Instruct")
EVALDIR = os.environ.get("PV_OUTDIR", "eval_persona_eval/Qwen2.5-7B-Instruct")
VECDIR = os.environ.get("PV_VECDIR", "persona_vectors/Qwen2.5-7B-Instruct")
OUTDIR = os.environ.get("PV_ANALYSIS", "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/001-living-update/results/monitor_proj")
SEL_LAYER = _json.loads(os.environ.get("PV_SEL_LAYER", '{"evil": 20, "sycophantic": 20, "hallucinating": 16}'))
TRAITS = list(SEL_LAYER.keys())

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
    # Followup 001: this step ONLY adds the last-prompt-token projection column to each
    # monitor CSV and saves it. C2 Pearson r is computed later from the paper-exact
    # rejudge sidecars (gpt_<trait>) vs this projection column (see compute_results.py),
    # because inline judging is disabled (PV_NO_JUDGE=1) so the inline trait col is empty.
    results = {}
    for trait in TRAITS:
        layer = SEL_LAYER[trait]
        vecs = torch.load(f"{VECDIR}/{trait}_response_avg_diff.pt", weights_only=False)
        vector = vecs[layer].float()
        col = f"{SLUG}_{trait}_response_avg_diff_prompt_last_proj_layer{layer}"
        for ptype in ["pos", "neg"]:
            path = f"{EVALDIR}/{trait}_monitor_{ptype}.csv"
            df = pd.read_csv(path)
            df[col] = add_projection(model, tokenizer, df, vector, layer)
            df["ptype"] = ptype
            df.to_csv(path, index=False)
            print(f"{trait} {ptype}: projected {len(df)} rows -> {col}", flush=True)
        results[trait] = dict(layer=layer, proj_col=col)
    json.dump(results, open(f"{OUTDIR}/monitor_proj_columns.json", "w"), indent=2)
    print("saved projection columns to", OUTDIR, flush=True)

if __name__ == "__main__":
    main()
