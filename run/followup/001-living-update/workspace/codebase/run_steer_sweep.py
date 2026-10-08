"""Step 4: per-layer / per-coefficient response-steering sweep on the base model.

Reproduces Figure 3 / the layer-selection procedure (Appendix B.4). For each
trait, sweep layer in {5,8,11,14,16,20,23,25} and coef in {0.5,1,1.5,2,2.5},
steering with response_avg_diff at that layer (steering_type='response').

Loads the HF policy model ONCE (steering requires HF forward hooks, not vLLM)
and uses a large generation batch size for full-scale efficiency. Writes one CSV
per (trait, layer, coef) plus a summary table of mean trait score & coherence.
"""
import os, sys, json, asyncio, argparse
import pandas as pd
import torch

from eval.eval_persona import load_persona_questions, sample_steering
from eval.model_utils import load_model

MODEL = os.environ.get("PV_MODEL", "Qwen/Qwen2.5-7B-Instruct")
JUDGE = os.environ.get("PV_JUDGE", "openai/gpt-4.1-mini-2025-04-14")
OUTDIR = os.environ.get("PV_OUTDIR", "eval_persona_eval/Qwen2.5-7B-Instruct")
VECDIR = os.environ.get("PV_VECDIR", "persona_vectors/Qwen2.5-7B-Instruct")
N_PER_Q = int(os.environ.get("PV_N_PER_Q", "10"))
BS = int(os.environ.get("PV_BS", "200"))
# max_tokens reduced to 512 for the layer-selection SWEEP only (default eval is 1000).
# HF batched generation runs until the longest of the 200 sequences hits max_tokens,
# so cost scales with max_tokens; 512 tokens is more than enough to judge trait
# expression and does not change which layer maximizes the trait score. Final trait
# magnitude measurements (e.g. Step 9) keep max_tokens=1000.
MAX_TOKENS = int(os.environ.get("PV_MAX_TOKENS", "512"))
LAYERS = [int(x) for x in os.environ.get("PV_LAYERS", "5,8,11,14,16,20,23,25").split(",")]
COEFS = [float(x) for x in os.environ.get("PV_COEFS", "0.5,1.0,1.5,2.0,2.5").split(",")]

async def judge_df(questions_by_id, df, trait):
    """Judge every row of df for trait + coherence using the Question judges."""
    # one representative Question per id gives us judge objects (all share prompts)
    from judge import OpenAiJudge
    # build judges once
    any_q = next(iter(questions_by_id.values()))
    judges = any_q.judges  # {trait: judge, coherence: judge}
    sem = asyncio.Semaphore(100)
    async def run(idx, judge, q, a):
        async with sem:
            return idx, await judge(question=q, answer=a)
    for metric, judge in judges.items():
        tasks = [run(i, judge, row["question"], row["answer"]) for i, row in df.iterrows()]
        res = [None]*len(df)
        for coro in asyncio.as_completed(tasks):
            i, val = await coro
            res[i] = val
        df[metric] = res
    return df

def build_rows(questions, model, tokenizer, vector, layer, coef):
    all_paraphrases, all_conversations = [], []
    for q in questions:
        paras, convs = q.get_input(N_PER_Q)
        all_paraphrases.extend(paras)
        all_conversations.extend(convs)
    prompts, answers = sample_steering(model, tokenizer, all_conversations, vector, layer, coef,
                                       bs=BS, temperature=1.0, max_tokens=MAX_TOKENS, steering_type="response")
    df = pd.DataFrame([dict(question=q, prompt=p, answer=a)
                       for q, p, a in zip(all_paraphrases, prompts, answers)])
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+", default=["evil", "sycophantic", "hallucinating"])
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    print("Loading HF model once...", flush=True)
    model, tokenizer = load_model(MODEL)
    model.eval()
    summary = []
    for trait in args.traits:
        vecs = torch.load(f"{VECDIR}/{trait}_response_avg_diff.pt", weights_only=False)
        questions = load_persona_questions(trait, temperature=1.0, judge_model=JUDGE, version="eval")
        questions_by_id = {q.id: q for q in questions}
        for layer in LAYERS:
            vector = vecs[layer]
            for coef in COEFS:
                out_path = os.path.join(OUTDIR, f"{trait}_steer_layer{layer}_coef{coef}.csv")
                if os.path.exists(out_path):
                    df = pd.read_csv(out_path)
                    m = df[trait].mean(); cm = df["coherence"].mean()
                    print(f"[skip] {out_path} exists: {trait}={m:.2f} coh={cm:.2f}", flush=True)
                    summary.append(dict(trait=trait, layer=layer, coef=coef, trait_mean=m, coh_mean=cm))
                    continue
                df = build_rows(questions, model, tokenizer, vector, layer, coef)
                df = asyncio.run(judge_df(questions_by_id, df, trait))
                df.to_csv(out_path, index=False)
                m = df[trait].mean(); cm = df["coherence"].mean()
                summary.append(dict(trait=trait, layer=layer, coef=coef, trait_mean=float(m), coh_mean=float(cm)))
                print(f"[done] {trait} L{layer} c{coef}: {trait}={m:.2f} coh={cm:.2f}", flush=True)
        # write incremental summary after each trait
        pd.DataFrame(summary).to_csv(f"{OUTDIR}/steer_sweep_summary.csv", index=False)
    print("\n=== STEER SWEEP SUMMARY ===", flush=True)
    sdf = pd.DataFrame(summary)
    # Followup 001: inline judging is disabled (PV_NO_JUDGE=1) so trait_mean here is NaN;
    # layer selection is done later from the paper-exact rejudge sidecars. Guard the
    # convenience print so an all-NaN summary does not abort the (already-written) run.
    for trait in args.traits:
        sub = sdf[sdf.trait == trait]
        print(f"\n--- {trait} (trait_mean by layer x coef) ---", flush=True)
        try:
            piv = sub.pivot(index="layer", columns="coef", values="trait_mean")
            print(piv.round(1).to_string(), flush=True)
            best = sub.loc[sub.trait_mean.idxmax()]
            print(f"argmax over grid: layer={int(best.layer)} coef={best.coef} trait={best.trait_mean:.1f}", flush=True)
        except Exception as e:
            print(f"(summary print skipped: {e})", flush=True)

if __name__ == "__main__":
    main()
