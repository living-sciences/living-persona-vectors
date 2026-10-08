"""Fixed, public model-card metadata for the 5-model era ladder.

These are architecture/release facts (params, hidden dim, family, alignment
style), NOT experimental measurements. era / release / n_layers are
cross-checked against 002's aggregate.json ladder. Params are HF model-card
parameter counts; alignment style uses the 3 levels named in the 003
instruction (base-RLHF vs reasoning-distill vs newer-RLHF).
"""

MODEL_META = {
    "Mistral-7B-Instruct-v0.2": dict(
        params_b=7.24, hidden=4096, n_layers=32, family="Mistral",
        era=1, release="2023-12", alignment="base-RLHF"),
    "Llama-3.1-8B-Instruct": dict(
        params_b=8.03, hidden=4096, n_layers=32, family="Llama",
        era=2, release="2024-07", alignment="base-RLHF"),
    "Qwen2.5-7B-Instruct": dict(
        params_b=7.62, hidden=3584, n_layers=28, family="Qwen",
        era=3, release="2024-09", alignment="newer-RLHF"),
    "DeepSeek-R1-Distill-Llama-8B": dict(
        params_b=8.03, hidden=4096, n_layers=32, family="Llama",
        era=4, release="2025-01", alignment="reasoning-distill"),
    "Qwen3-8B-Nonthinking": dict(
        params_b=8.19, hidden=4096, n_layers=36, family="Qwen",
        era=5, release="2025-04", alignment="newer-RLHF"),
}

TRAITS = ["evil", "sycophantic", "hallucinating"]
