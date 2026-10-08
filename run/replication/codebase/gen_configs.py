"""Generate training config JSONs for Step 6 (plain finetunes) and Step 10
(preventative-steer / CAFT-ablate finetunes)."""
import json, os, copy

BASE = json.load(open("configs/train_instruct_7b.json"))
STEER_BASE = json.load(open("configs/train_instruct_7b_steer.json"))
OUTDIR = "configs/generated"
os.makedirs(OUTDIR, exist_ok=True)

# Step 6: 8 datasets x 3 versions (dataset-folder names)
DATASETS = ["evil", "sycophancy", "hallucination", "mistake_medical",
            "mistake_opinions", "mistake_math", "mistake_gsm8k", "insecure_code"]
VERSIONS = ["normal", "misaligned_1", "misaligned_2"]

# map dataset-folder trait -> persona-vector file name + selected layer (1-indexed)
TRAIT_VEC = {"evil": ("evil", 20), "sycophancy": ("sycophantic", 20),
             "hallucination": ("hallucinating", 16)}

step6 = []
for ds in DATASETS:
    for v in VERSIONS:
        cfg = copy.deepcopy(BASE)
        cfg["training_file"] = [f"dataset/{ds}/{v}.jsonl"]
        cfg["output_dir"] = f"./ckpt/Qwen2.5-7B-Instruct/qwen-{ds}_{v}"
        cfg["finetuned_model_id"] = f"local/qwen-{ds}_{v}"
        cfg["push_to_private"] = False
        cfg["merge_before_push"] = False
        p = f"{OUTDIR}/qwen-{ds}_{v}.json"
        json.dump(cfg, open(p, "w"), indent=2)
        step6.append(p)

# Step 10: evil/sycophancy/hallucination misaligned_2, modes steer & ablate
step10 = []
for ds in ["evil", "sycophancy", "hallucination"]:
    vecname, layer = TRAIT_VEC[ds]
    for mode in ["steer", "ablate"]:
        cfg = copy.deepcopy(STEER_BASE)
        cfg["training_file"] = [f"dataset/{ds}/misaligned_2.jsonl"]
        cfg["output_dir"] = f"./ckpt/Qwen2.5-7B-Instruct/qwen-{ds}_m2_{mode}"
        cfg["finetuned_model_id"] = f"local/qwen-{ds}_m2_{mode}"
        cfg["push_to_private"] = False
        cfg["merge_before_push"] = False
        cfg["enable_steering_during_training"] = True
        cfg["steering_config"] = {
            "steering_vector_path": f"persona_vectors/Qwen2.5-7B-Instruct/{vecname}_response_avg_diff.pt",
            "type": mode,
            "steering_coef": 5.0,
            "layers": [layer],
        }
        p = f"{OUTDIR}/qwen-{ds}_m2_{mode}.json"
        json.dump(cfg, open(p, "w"), indent=2)
        step10.append(p)

print(f"Step6 configs: {len(step6)}")
print(f"Step10 configs: {len(step10)}")
json.dump({"step6": step6, "step10": step10}, open(f"{OUTDIR}/manifest.json", "w"), indent=2)
