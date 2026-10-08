"""Step 7: (a) finetuning-shift correlation (Fig 6 / C3, C11) and
(b) dataset projection-difference prediction (Fig 8 / C4).

(a) finetuning shift: for the base model and each finetuned model, compute the
    mean last-prompt-token activation on a FIXED set of eval-set prompts (the 20
    persona eval questions per trait, chat-templated, no system prompt), project
    onto the unit-normalized persona vector at the selected layer, and take
    finetuned_mean - base_mean. Correlate the per-model shift with the
    post-finetuning target-trait score (same-trait) and with off-trait scores
    (cross-trait baseline). LoRA adapters are merged via load_model (cal_projection
    itself uses AutoModelForCausalLM without PEFT; we merge instead of patching it).

(b) projection difference: for each training (dataset,version) subsample N rows,
    project the training RESPONSES onto the persona vector (response_avg, layer L)
    with the base model -> P_train; generate the base model's natural responses to
    the same prompts and project those -> P_base; deltaP = P_train - P_base.
    Correlate deltaP with the post-finetuning trait score across models.

    b_gen: base vLLM generates natural responses for the sampled dataset prompts.
    b_proj: base HF computes response-avg projections for train + base responses,
            then Pearson r vs post-FT trait score.

Judge scores for post-FT come from the Step 6 eval CSVs.
"""
import os, sys, json, argparse, random
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVALROOT = "eval_persona_eval"
BASE_EVALDIR = "eval_persona_eval/Qwen2.5-7B-Instruct"
VECDIR = "persona_vectors/Qwen2.5-7B-Instruct"
ANALYSIS = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/analysis"
STEP7B_DIR = os.path.join(ANALYSIS, "step7b")

TRAITS = ["evil", "sycophantic", "hallucinating"]
SEL_LAYER = {"evil": 20, "sycophantic": 20, "hallucinating": 16}
# training-dataset folder name -> trait score column used for that folder's target
DATASETS = ["evil", "sycophancy", "hallucination", "mistake_medical",
            "mistake_opinions", "mistake_math", "mistake_gsm8k", "insecure_code"]
VERSIONS = ["normal", "misaligned_1", "misaligned_2"]
# folder trait name -> persona-score column name
FOLDER2TRAIT = {"evil": "evil", "sycophancy": "sycophantic", "hallucination": "hallucinating"}
N_SUB = 256  # subsample per dataset for deltaP mean estimate (Monte-Carlo mean)
SEED = 0


def a_proj_b(a, b):
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def model_name(ds, v):
    return f"qwen-{ds}_{v}"


def postft_score(ds, v, trait):
    """mean trait score of finetuned model qwen-<ds>_<v> on <trait> eval CSV."""
    p = os.path.join(EVALROOT, model_name(ds, v), f"{trait}.csv")
    if not os.path.exists(p):
        return np.nan
    return pd.read_csv(p)[trait].mean()


# ---------------- eval-set fixed prompts ----------------
def eval_prompts(tokenizer, trait):
    from eval.eval_persona import load_persona_questions
    qs = load_persona_questions(trait, version="eval")
    prompts = []
    for q in qs:
        for para in q.paraphrases:
            msgs = [dict(role="user", content=para)]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return prompts


def mean_last_prompt_proj(model, tokenizer, prompts, vector, layer):
    vals = []
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        last = out.hidden_states[layer][:, -1, :].detach().cpu().float()
        vals.append(a_proj_b(last, vector).item())
    return float(np.mean(vals))


# ---------------- part (a) ----------------
def part_a():
    from eval.model_utils import load_model
    os.makedirs(ANALYSIS, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    vecs = {t: torch.load(f"{VECDIR}/{t}_response_avg_diff.pt", weights_only=False) for t in TRAITS}
    prompts = {t: eval_prompts(tok, t) for t in TRAITS}

    rows = []  # per (model) mean_proj per trait
    # base
    print("Loading BASE model...", flush=True)
    base, btok = load_model(MODEL)
    base.eval()
    base_proj = {t: mean_last_prompt_proj(base, btok, prompts[t], vecs[t][SEL_LAYER[t]].float(), SEL_LAYER[t]) for t in TRAITS}
    print("base_proj:", base_proj, flush=True)
    del base; torch.cuda.empty_cache()

    for ds in DATASETS:
        for v in VERSIONS:
            name = model_name(ds, v)
            ckpt = f"ckpt/Qwen2.5-7B-Instruct/{name}"
            if not os.path.isdir(ckpt):
                print(f"[warn] missing ckpt {ckpt}", flush=True); continue
            print(f"Loading {name} (merge LoRA)...", flush=True)
            m, mtok = load_model(ckpt)
            m.eval()
            for t in TRAITS:
                mp = mean_last_prompt_proj(m, mtok, prompts[t], vecs[t][SEL_LAYER[t]].float(), SEL_LAYER[t])
                shift = mp - base_proj[t]
                rows.append(dict(model=name, ds=ds, version=v, trait=t,
                                 mean_proj=mp, base_proj=base_proj[t], shift=shift,
                                 postft_score=postft_score(ds, v, t)))
            del m; torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(f"{ANALYSIS}/step7a_shift.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(f"{ANALYSIS}/step7a_shift.csv", index=False)
    # correlations
    result = {}
    for t in TRAITS:
        sub = df[df.trait == t].dropna(subset=["shift", "postft_score"])
        r, p = pearsonr(sub["shift"], sub["postft_score"])
        result[f"same_trait::{t}"] = dict(r=float(r), p=float(p), n=int(len(sub)))
    # cross-trait baseline: shift on trait t vs post-FT score of a different trait t2
    for t in TRAITS:
        for t2 in TRAITS:
            if t2 == t:
                continue
            shift_t = df[df.trait == t].set_index("model")["shift"]
            score_t2 = df[df.trait == t2].set_index("model")["postft_score"]
            j = pd.concat([shift_t, score_t2], axis=1, keys=["shift", "score"]).dropna()
            if len(j) >= 3:
                r, p = pearsonr(j["shift"], j["score"])
                result[f"cross::shift_{t}__vs__score_{t2}"] = dict(r=float(r), p=float(p), n=int(len(j)))
    json.dump(result, open(f"{ANALYSIS}/step7a_correlations.json", "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)


# ---------------- part (b_gen) ----------------
def load_dataset_prompts(ds, v, tokenizer, n_sub):
    from eval.cal_projection import load_jsonl
    data = load_jsonl(f"dataset/{ds}/{v}.jsonl")
    random.Random(SEED).shuffle(data)
    data = data[:n_sub]
    prompts, train_answers = [], []
    for d in data:
        prompts.append(tokenizer.apply_chat_template(d["messages"][:-1], tokenize=False, add_generation_prompt=True))
        train_answers.append(d["messages"][-1]["content"])
    return prompts, train_answers


def part_b_gen():
    os.makedirs(STEP7B_DIR, exist_ok=True)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, enable_prefix_caching=True, tensor_parallel_size=torch.cuda.device_count(),
              max_num_seqs=64, gpu_memory_utilization=0.9, max_model_len=4096, enforce_eager=True)
    sp = SamplingParams(temperature=1.0, top_p=1, max_tokens=600, min_tokens=1, skip_special_tokens=True,
                        stop=[tok.eos_token])
    for ds in DATASETS:
        for v in VERSIONS:
            out = f"{STEP7B_DIR}/{ds}_{v}.jsonl"
            if os.path.exists(out):
                print(f"[skip] {out}", flush=True); continue
            prompts, train_answers = load_dataset_prompts(ds, v, tok, N_SUB)
            comps = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
            base_answers = [c.outputs[0].text for c in comps]
            with open(out, "w") as f:
                for p, ta, ba in zip(prompts, train_answers, base_answers):
                    f.write(json.dumps(dict(prompt=p, train_answer=ta, base_answer=ba)) + "\n")
            print(f"[gen] {out}: {len(prompts)} prompts", flush=True)


def part_b_proj():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    vecs = {t: torch.load(f"{VECDIR}/{t}_response_avg_diff.pt", weights_only=False) for t in TRAITS}

    def resp_avg_proj(prompt, answer, vector, layer):
        inputs = tok(prompt + str(answer), return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tok.encode(prompt, add_special_tokens=False))
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        ra = out.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu().float()
        return a_proj_b(ra, vector).item()

    rows = []
    for ds in DATASETS:
        for v in VERSIONS:
            path = f"{STEP7B_DIR}/{ds}_{v}.jsonl"
            if not os.path.exists(path):
                print(f"[warn] missing {path}", flush=True); continue
            recs = [json.loads(l) for l in open(path) if l.strip()]
            for t in TRAITS:
                layer = SEL_LAYER[t]; vector = vecs[t][layer].float()
                p_train = np.mean([resp_avg_proj(r["prompt"], r["train_answer"], vector, layer) for r in recs])
                p_base = np.mean([resp_avg_proj(r["prompt"], r["base_answer"], vector, layer) for r in recs])
                rows.append(dict(ds=ds, version=v, trait=t, P_train=float(p_train),
                                 P_base=float(p_base), deltaP=float(p_train - p_base),
                                 postft_score=postft_score(ds, v, t)))
            pd.DataFrame(rows).to_csv(f"{ANALYSIS}/step7b_deltaP.csv", index=False)
            print(f"[proj] {ds}_{v} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(f"{ANALYSIS}/step7b_deltaP.csv", index=False)
    result = {}
    for t in TRAITS:
        sub = df[df.trait == t].dropna(subset=["deltaP", "postft_score"])
        r, p = pearsonr(sub["deltaP"], sub["postft_score"])
        result[f"deltaP_vs_postft::{t}"] = dict(r=float(r), p=float(p), n=int(len(sub)))
    json.dump(result, open(f"{ANALYSIS}/step7b_correlations.json", "w"), indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["a", "b_gen", "b_proj"])
    args = ap.parse_args()
    {"a": part_a, "b_gen": part_b_gen, "b_proj": part_b_proj}[args.part]()
