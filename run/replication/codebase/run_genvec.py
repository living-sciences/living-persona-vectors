"""Compute persona vectors for all traits with a single model load (Step 2).

Uses the same difference-in-means procedure as generate_vec.py (threshold 50),
but loads Qwen2.5-7B-Instruct once and processes every trait.
"""
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_vec import get_persona_effective, get_hidden_p_and_r

MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXTRACT_DIR = "eval_persona_extract/Qwen2.5-7B-Instruct"
SAVE_DIR = "persona_vectors/Qwen2.5-7B-Instruct"
THRESHOLD = 50
TRAITS = ["evil", "sycophantic", "hallucinating"]

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Loading model once...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    for trait in TRAITS:
        pos_path = f"{EXTRACT_DIR}/{trait}_pos_instruct.csv"
        neg_path = f"{EXTRACT_DIR}/{trait}_neg_instruct.csv"
        (pe, ne, pos_prompts, neg_prompts, pos_resp, neg_resp) = get_persona_effective(
            pos_path, neg_path, trait, THRESHOLD)
        print(f"\n=== {trait}: effective pairs = {len(pos_prompts)} (of 1000) ===", flush=True)
        pa, pl, ra = {}, {}, {}
        pa["pos"], pl["pos"], ra["pos"] = get_hidden_p_and_r(model, tokenizer, pos_prompts, pos_resp)
        pa["neg"], pl["neg"], ra["neg"] = get_hidden_p_and_r(model, tokenizer, neg_prompts, neg_resp)
        prompt_avg_diff = torch.stack([pa["pos"][l].mean(0).float() - pa["neg"][l].mean(0).float() for l in range(len(pa["pos"]))], dim=0)
        response_avg_diff = torch.stack([ra["pos"][l].mean(0).float() - ra["neg"][l].mean(0).float() for l in range(len(ra["pos"]))], dim=0)
        prompt_last_diff = torch.stack([pl["pos"][l].mean(0).float() - pl["neg"][l].mean(0).float() for l in range(len(pl["pos"]))], dim=0)
        torch.save(prompt_avg_diff, f"{SAVE_DIR}/{trait}_prompt_avg_diff.pt")
        torch.save(response_avg_diff, f"{SAVE_DIR}/{trait}_response_avg_diff.pt")
        torch.save(prompt_last_diff, f"{SAVE_DIR}/{trait}_prompt_last_diff.pt")
        print(f"saved {trait} vectors: shape {tuple(response_avg_diff.shape)}", flush=True)

if __name__ == "__main__":
    main()
