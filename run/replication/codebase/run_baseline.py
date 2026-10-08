"""Base-model baseline trait scores (Step 3): coef=0, version=eval, no intervention.

Loads Qwen2.5-7B-Instruct once via vLLM and evaluates all three traits on the
evaluation-set questions. Mirrors eval.eval_persona main() for coef=0.
"""
import os, sys, asyncio
import pandas as pd
from eval.eval_persona import load_persona_questions, eval_batched
from eval.model_utils import load_vllm_model

MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE = "Qwen/Qwen2.5-32B-Instruct"
OUTDIR = "eval_persona_eval/Qwen2.5-7B-Instruct"
TRAITS = ["evil", "sycophantic", "hallucinating"]
N_PER_Q = 10

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Loading policy model once...", flush=True)
    llm, tokenizer, lora_path = load_vllm_model(MODEL)
    summary = {}
    for trait in TRAITS:
        out_path = os.path.join(OUTDIR, f"{trait}_baseline.csv")
        questions = load_persona_questions(trait, temperature=1.0, persona_instructions_type=None,
                                           assistant_name=None, judge_model=JUDGE, version="eval")
        outputs_list = asyncio.run(eval_batched(questions, llm, tokenizer, coef=0, vector=None, layer=None,
                                                n_per_question=N_PER_Q, max_concurrent_judges=100,
                                                max_tokens=1000, steering_type="response", lora_path=lora_path))
        outputs = pd.concat(outputs_list)
        outputs.to_csv(out_path, index=False)
        m, s = outputs[trait].mean(), outputs[trait].std()
        cm = outputs["coherence"].mean()
        summary[trait] = (m, s, cm, len(outputs))
        print(f"saved {out_path}: {trait} mean={m:.2f} +- {s:.2f}  coherence={cm:.2f}  rows={len(outputs)}", flush=True)
    print("\n=== BASELINE SUMMARY (base Qwen2.5-7B-Instruct, coef=0) ===", flush=True)
    for t,(m,s,cm,n) in summary.items():
        print(f"{t}: {m:.2f} +- {s:.2f} (coherence {cm:.2f}, n={n})", flush=True)

if __name__ == "__main__":
    main()
