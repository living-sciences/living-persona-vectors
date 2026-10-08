"""Step 9: post-hoc inference-time steering AGAINST the persona vector on a
trait-inducing finetuned checkpoint (Figure 7A / C9).

For each trait, take its misaligned_2 finetuned checkpoint, merge the LoRA
adapter (load_model), and steer the RESPONSE tokens with the trait's persona
vector at the selected layer using coefficients of increasing NEGATIVE magnitude.
Measure target-trait score and coherence as |coef| grows. Trait score should
decrease as the (negative) steering magnitude increases.

MMLU-degradation axis is NOT reproducible (repo ships no MMLU script); only the
trait-reduction + coherence axes are produced. This is recorded as a limitation.
"""
import os, asyncio, argparse
import pandas as pd
import torch

from eval.eval_persona import load_persona_questions, sample_steering
from eval.model_utils import load_model

JUDGE = "Qwen/Qwen2.5-32B-Instruct"
OUTROOT = "eval_persona_eval"
VECDIR = "persona_vectors/Qwen2.5-7B-Instruct"
N_PER_Q = 10
BS = 200
MAX_TOKENS = 1000  # final magnitude measurement -> full length

# trait -> (checkpoint dir, persona-vector file stem, selected 1-indexed layer)
SPEC = {
    "evil":         ("ckpt/Qwen2.5-7B-Instruct/qwen-evil_misaligned_2",          "evil",         20),
    "sycophantic":  ("ckpt/Qwen2.5-7B-Instruct/qwen-sycophancy_misaligned_2",    "sycophantic",  20),
    "hallucinating":("ckpt/Qwen2.5-7B-Instruct/qwen-hallucination_misaligned_2", "hallucinating",16),
}
COEFS = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5]


async def judge_df(questions_by_id, df, trait):
    any_q = next(iter(questions_by_id.values()))
    judges = any_q.judges
    sem = asyncio.Semaphore(100)
    async def run(idx, judge, q, a):
        async with sem:
            return idx, await judge(question=q, answer=a)
    for metric, judge in judges.items():
        tasks = [run(i, judge, row["question"], row["answer"]) for i, row in df.iterrows()]
        res = [None] * len(df)
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
    return pd.DataFrame([dict(question=q, prompt=p, answer=a)
                         for q, p, a in zip(all_paraphrases, prompts, answers)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traits", nargs="+", default=list(SPEC.keys()))
    args = ap.parse_args()
    summary = []
    for trait in args.traits:
        ckpt, vecname, layer = SPEC[trait]
        name = os.path.basename(ckpt)
        out_dir = os.path.join(OUTROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nLoading finetuned model {ckpt} (merging LoRA)...", flush=True)
        model, tokenizer = load_model(ckpt)
        model.eval()
        vecs = torch.load(f"{VECDIR}/{vecname}_response_avg_diff.pt", weights_only=False)
        vector = vecs[layer]
        questions = load_persona_questions(trait, temperature=1.0, judge_model=JUDGE, version="eval")
        questions_by_id = {q.id: q for q in questions}
        for coef in COEFS:
            out_path = os.path.join(out_dir, f"{trait}_steer_response_layer{layer}_coef{coef}.csv")
            if os.path.exists(out_path):
                df = pd.read_csv(out_path)
                m = df[trait].mean(); cm = df["coherence"].mean()
                print(f"[skip] {out_path}: {trait}={m:.2f} coh={cm:.2f}", flush=True)
                summary.append(dict(model=name, trait=trait, layer=layer, coef=coef, trait_mean=float(m), coh_mean=float(cm)))
                continue
            df = build_rows(questions, model, tokenizer, vector, layer, coef)
            df = asyncio.run(judge_df(questions_by_id, df, trait))
            df.to_csv(out_path, index=False)
            m = df[trait].mean(); cm = df["coherence"].mean()
            summary.append(dict(model=name, trait=trait, layer=layer, coef=coef, trait_mean=float(m), coh_mean=float(cm)))
            print(f"[done] {name} {trait} coef{coef}: {trait}={m:.2f} coh={cm:.2f}", flush=True)
        pd.DataFrame(summary).to_csv(f"{OUTROOT}/step9_posthoc_steer_summary.csv", index=False)
        del model
        torch.cuda.empty_cache()
    print("\n=== STEP 9 SUMMARY (target-trait score vs negative coef) ===", flush=True)
    print(pd.DataFrame(summary).to_string(), flush=True)


if __name__ == "__main__":
    main()
