"""Step 5 (generation): prompt-induced monitoring data.

For each trait, generate responses under the shipped contrastive system-prompt
instruction pairs (persona_instruction_type = pos and neg) on the eval-set
questions, with the base model (coef=0, vLLM). These span trait-discouraging ->
trait-encouraging. Judge trait + coherence. Projection + correlation are done in
a separate step (cal_projection prompt_last_proj + scipy pearsonr).
"""
import os, asyncio
import pandas as pd
from eval.eval_persona import load_persona_questions, eval_batched
from eval.model_utils import load_vllm_model

MODEL = os.environ.get("PV_MODEL", "Qwen/Qwen2.5-7B-Instruct")
JUDGE = os.environ.get("PV_JUDGE", "openai/gpt-4.1-mini-2025-04-14")
OUTDIR = os.environ.get("PV_OUTDIR", "eval_persona_eval/Qwen2.5-7B-Instruct")
TRAITS = os.environ.get("PV_TRAITS", "evil,sycophantic,hallucinating").split(",")
N_PER_Q = int(os.environ.get("PV_N_PER_Q", "10"))

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Loading policy model once...", flush=True)
    llm, tokenizer, lora_path = load_vllm_model(MODEL)
    for trait in TRAITS:
        for ptype in ["pos", "neg"]:
            out_path = os.path.join(OUTDIR, f"{trait}_monitor_{ptype}.csv")
            if os.path.exists(out_path):
                print(f"[skip] {out_path}", flush=True); continue
            aname = trait if ptype == "pos" else "helpful"
            questions = load_persona_questions(trait, temperature=1.0, persona_instructions_type=ptype,
                                               assistant_name=aname, judge_model=JUDGE, version="eval")
            outputs_list = asyncio.run(eval_batched(questions, llm, tokenizer, coef=0, vector=None, layer=None,
                                                    n_per_question=N_PER_Q, max_concurrent_judges=100,
                                                    max_tokens=1000, steering_type="response", lora_path=lora_path))
            outputs = pd.concat(outputs_list)
            outputs.to_csv(out_path, index=False)
            print(f"saved {out_path}: rows={len(outputs)} {trait} mean={outputs[trait].mean():.2f} coh={outputs['coherence'].mean():.2f}", flush=True)

if __name__ == "__main__":
    main()
