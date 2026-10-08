"""Step 8: sample-level separability (Figure 9 / C13).

Compute per-sample response-avg projection of each training sample onto the
persona vectors for the intentionally trait-eliciting datasets
(evil/misaligned_2, sycophancy/misaligned_2, hallucination/misaligned_2) and an
EM-like dataset (mistake_opinions/misaligned_2), each vs its Normal control.
Build overlaid per-trait histograms (trait-inducing vs normal) to assess how
separable the two distributions are.

Uses the base Qwen2.5-7B-Instruct model (same as cal_projection --projection_type
proj). Projection = a_proj_b(mean response-token hidden state at selected layer,
persona vector).
"""
import os, json, random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("PV_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VECDIR = os.environ.get("PV_VECDIR", "persona_vectors/Qwen2.5-7B-Instruct")
ANALYSIS = os.environ.get("PV_ANALYSIS", "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/001-living-update/results/step8")
SLUG = os.environ.get("PV_SLUG", "Qwen2.5-7B-Instruct")
SEL_LAYER = json.loads(os.environ.get("PV_SEL_LAYER", '{"evil": 20, "sycophantic": 20, "hallucinating": 16}'))
# dataset folder -> the trait vector it primarily targets
_DS_TRAIT_ALL = {"evil": "evil", "sycophancy": "sycophantic",
            "hallucination": "hallucinating", "mistake_opinions": "hallucinating"}
# keep only datasets whose target trait has a selected layer (vector available)
DS_TRAIT = {ds: t for ds, t in _DS_TRAIT_ALL.items() if t in SEL_LAYER}
DATASETS = list(DS_TRAIT.keys())
N_SUB = int(os.environ.get("PV_N_SUB", "500"))
SEED = 0


def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    os.makedirs(ANALYSIS, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    vecs = {t: torch.load(f"{VECDIR}/{t}_response_avg_diff.pt", weights_only=False) for t in SEL_LAYER}

    def sample_proj(prompt, answer, vector, layer):
        inputs = tok(prompt + str(answer), return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        ra = out.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu().float()
        return a_proj_b(ra, vector).item()

    proj_data = {}  # (ds, version) -> list of projections onto DS_TRAIT[ds]
    for ds in DATASETS:
        trait = DS_TRAIT[ds]; layer = SEL_LAYER[trait]; vector = vecs[trait][layer].float()
        for v in ["misaligned_2", "normal"]:
            data = load_jsonl(f"dataset/{ds}/{v}.jsonl")
            random.Random(SEED).shuffle(data)
            data = data[:N_SUB]
            projs = []
            for d in data:
                prompt = tok.apply_chat_template(d["messages"][:-1], tokenize=False, add_generation_prompt=True)
                answer = d["messages"][-1]["content"]
                projs.append(sample_proj(prompt, answer, vector, layer))
            proj_data[f"{ds}/{v}"] = projs
            print(f"{ds}/{v} (vec={trait} L{layer}): n={len(projs)} mean={np.mean(projs):.3f}", flush=True)
        json.dump({k: v for k, v in proj_data.items()}, open(f"{ANALYSIS}/step8_projections.json", "w"))

    # separability metric + histograms
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.5), squeeze=False)
    axes = axes.ravel()
    summary = []
    for ax, ds in zip(axes, DATASETS):
        trait = DS_TRAIT[ds]
        mis = np.array(proj_data[f"{ds}/misaligned_2"])
        nor = np.array(proj_data[f"{ds}/normal"])
        lo = min(mis.min(), nor.min()); hi = max(mis.max(), nor.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(nor, bins=bins, alpha=0.5, label="normal", color="#4477AA")
        ax.hist(mis, bins=bins, alpha=0.5, label="misaligned_2", color="#EE6677")
        ax.set_title(f"{ds}\n(proj onto {trait} L{SEL_LAYER[trait]})")
        ax.set_xlabel("response-avg projection"); ax.legend()
        # simple separability: AUC via rank statistic + threshold accuracy
        from scipy.stats import mannwhitneyu
        try:
            U, _ = mannwhitneyu(mis, nor, alternative="greater")
            auc = U / (len(mis) * len(nor))
        except Exception:
            auc = float("nan")
        summary.append(dict(dataset=ds, trait=trait, mean_misaligned=float(mis.mean()),
                            mean_normal=float(nor.mean()), auc_separability=float(auc)))
    plt.tight_layout()
    fig.savefig(f"{ANALYSIS}/step8_separability_hist.png", dpi=120)
    pd.DataFrame(summary).to_csv(f"{ANALYSIS}/step8_separability_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(), flush=True)


if __name__ == "__main__":
    main()
