"""Phase 2 of Step 6/10: evaluate finetuned LoRA checkpoints (coef=0, version=eval).

Loads the base Qwen2.5-7B-Instruct ONCE in vLLM (enable_lora) and evaluates every
requested checkpoint x trait by swapping LoRA adapters (unique adapter id per
checkpoint to avoid vLLM's adapter cache collisions). Judges trait + coherence.
Writes eval_persona_eval/<ckpt-basename>/<trait>.csv and a summary matrix.
"""
import os, sys, json, glob, asyncio, argparse, re
import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from eval.eval_persona import load_persona_questions

MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE = "Qwen/Qwen2.5-32B-Instruct"
EVALROOT = "eval_persona_eval"
TRAITS = ["evil", "sycophantic", "hallucinating"]
N_PER_Q = 10
_CKPT_RE = re.compile(r"checkpoint-(\d+)")

def latest_ckpt(path):
    ckpts = [(int(m.group(1)), p) for p in glob.glob(os.path.join(path, "checkpoint-*"))
             if (m := _CKPT_RE.search(os.path.basename(p)))]
    if ckpts:
        return max(ckpts, key=lambda x: x[0])[1]
    return path  # adapter directly in output_dir

async def judge_rows(questions_by_id, df, trait):
    any_q = next(iter(questions_by_id.values()))
    judges = any_q.judges
    sem = asyncio.Semaphore(100)
    async def run(i, judge, q, a):
        async with sem:
            return i, await judge(question=q, answer=a)
    for metric, judge in judges.items():
        res = [None] * len(df)
        tasks = [run(i, judge, r["question"], r["answer"]) for i, r in df.iterrows()]
        for coro in asyncio.as_completed(tasks):
            i, v = await coro
            res[i] = v
        df[metric] = res
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="checkpoint output_dirs")
    ap.add_argument("--traits", nargs="+", default=TRAITS)
    args = ap.parse_args()

    print("Loading base model in vLLM (enable_lora)...", flush=True)
    llm = LLM(model=MODEL, enable_prefix_caching=True, enable_lora=True,
              tensor_parallel_size=torch.cuda.device_count(), max_num_seqs=32,
              gpu_memory_utilization=0.9, max_model_len=20000, max_lora_rank=128,
              enforce_eager=True)
    tok = llm.get_tokenizer()
    tok.pad_token = tok.eos_token; tok.padding_side = "left"

    # pre-build questions per trait (judges reused)
    qcache = {t: load_persona_questions(t, temperature=1.0, judge_model=JUDGE, version="eval") for t in args.traits}

    summary = []
    for lora_id, ckpt_dir in enumerate(args.ckpts, start=1):
        name = os.path.basename(ckpt_dir.rstrip("/"))
        resolved = latest_ckpt(ckpt_dir)
        if not (os.path.exists(os.path.join(resolved, "adapter_config.json"))):
            print(f"[warn] no adapter at {resolved}, skipping {name}", flush=True); continue
        lora_req = LoRARequest(name, lora_id, lora_path=resolved)
        for trait in args.traits:
            out_dir = os.path.join(EVALROOT, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{trait}.csv")
            if os.path.exists(out_path):
                df = pd.read_csv(out_path)
                m = df[trait].mean(); cm = df["coherence"].mean()
                print(f"[skip] {out_path}: {trait}={m:.2f} coh={cm:.2f}", flush=True)
                summary.append(dict(model=name, trait=trait, mean=float(m), coh=float(cm)))
                continue
            questions = qcache[trait]
            qby = {q.id: q for q in questions}
            paras, convs = [], []
            for q in questions:
                p, c = q.get_input(N_PER_Q); paras += p; convs += c
            texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in convs]
            sp = SamplingParams(temperature=1.0, top_p=1, max_tokens=1000, min_tokens=1,
                                skip_special_tokens=True, stop=[tok.eos_token])
            comps = llm.generate(texts, sampling_params=sp, use_tqdm=False, lora_request=lora_req)
            answers = [c.outputs[0].text for c in comps]
            df = pd.DataFrame([dict(question=q, prompt=p, answer=a)
                               for q, p, a in zip(paras, texts, answers)])
            df = asyncio.run(judge_rows(qby, df, trait))
            df.to_csv(out_path, index=False)
            m = df[trait].mean(); cm = df["coherence"].mean()
            summary.append(dict(model=name, trait=trait, mean=float(m), coh=float(cm)))
            print(f"[done] {name} {trait}: {trait}={m:.2f} coh={cm:.2f}", flush=True)
        pd.DataFrame(summary).to_csv(f"{EVALROOT}/ckpt_eval_summary.csv", index=False)
    print("\n=== CKPT EVAL SUMMARY ===", flush=True)
    print(pd.DataFrame(summary).to_string(), flush=True)

if __name__ == "__main__":
    main()
