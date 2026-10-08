"""Consolidated persona-vector extraction driver (Step 2).

Loads the policy model (Qwen2.5-7B-Instruct) ONCE via vLLM and runs all
pos/neg extraction generations + judging for every requested trait, writing one
CSV per (trait, ptype). This avoids re-loading the model for each of the 6 runs.
Mirrors eval.eval_persona main() for coef=0 / version=extract.
"""
import os, sys, asyncio
import pandas as pd

from eval.eval_persona import load_persona_questions, eval_batched
from eval.model_utils import load_vllm_model

MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE = "Qwen/Qwen2.5-32B-Instruct"
OUTDIR = "eval_persona_extract/Qwen2.5-7B-Instruct"
N_PER_Q = 10

# (trait json name, ptype, assistant_name)
RUNS = [
    ("evil", "pos", "evil"),
    ("evil", "neg", "helpful"),
    ("sycophantic", "pos", "sycophantic"),
    ("sycophantic", "neg", "helpful"),
    ("hallucinating", "pos", "hallucinating"),
    ("hallucinating", "neg", "helpful"),
]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Loading policy model (once)...", flush=True)
    llm, tokenizer, lora_path = load_vllm_model(MODEL)  # coef=0 path
    for trait, ptype, aname in RUNS:
        out_path = os.path.join(OUTDIR, f"{trait}_{ptype}_instruct.csv")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} exists", flush=True)
            continue
        print(f"\n=== EXTRACT trait={trait} ptype={ptype} assistant={aname} ===", flush=True)
        questions = load_persona_questions(
            trait, temperature=1.0, persona_instructions_type=ptype,
            assistant_name=aname, judge_model=JUDGE, version="extract",
        )
        print(f"{len(questions)} question objects", flush=True)
        outputs_list = asyncio.run(eval_batched(
            questions, llm, tokenizer, coef=0, vector=None, layer=None,
            n_per_question=N_PER_Q, max_concurrent_judges=100, max_tokens=1000,
            steering_type="response", lora_path=lora_path,
        ))
        outputs = pd.concat(outputs_list)
        outputs.to_csv(out_path, index=False)
        n_ok_t = outputs[trait].notna().sum()
        n_ok_c = outputs["coherence"].notna().sum()
        print(f"saved {out_path}: rows={len(outputs)}  "
              f"{trait} mean={outputs[trait].mean():.2f} (n_ok={n_ok_t})  "
              f"coherence mean={outputs['coherence'].mean():.2f} (n_ok={n_ok_c})", flush=True)

if __name__ == "__main__":
    main()
