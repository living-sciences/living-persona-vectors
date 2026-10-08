# Large files omitted

To keep this repository lightweight and free of model weights, the files below were left out
during staging. The rule applied: omit every file 45 MB or larger, and omit every model,
weight, or activation artifact regardless of size (`*.safetensors`, model `*.bin`, weight or
activation `*.pt` / `*.pth`, `*.ckpt`, `*.gguf`, `*.h5`, large `*.npz` / `*.npy` / `*.csv` /
`*.parquet`, `acts/` directories, and local model snapshots). Small result CSVs, JSON,
figures, and logs were kept.

No omitted file was smaller than 45 MB except the weight and activation artifacts, which are
omitted by type regardless of size. After staging, no file 95 MB or larger remains anywhere in
the repository.

## Weight and finetune artifacts (`run/replication/codebase/ckpt/Qwen2.5-7B-Instruct/...`)

These are the LoRA finetune checkpoints used by the paper's training-time experiments
(approximately 30 finetune runs across the evil / sycophantic / hallucinating / mistake /
insecure-code conditions, plus steer and ablate variants).

- `*/checkpoint-*/adapter_model.safetensors` — 30 files, about 308 MB each, roughly 9.7 GB
  total. The trained LoRA adapter weights.
  Regenerate: rerun the finetunes with the kept code (`run_finetune.py` / `training.py`) on the
  kept `dataset/` files; roughly 9 GPU-hours on an A100-80GB for the full suite.
- `*/checkpoint-*/optimizer.pt` — 30 files, about 157 MB each, roughly 4.9 GB total. Optimizer
  states. Not needed for inference; recreated by rerunning the finetunes.
- `*/checkpoint-*/scheduler.pt`, `rng_state.pth`, `training_args.bin` — small checkpoint state
  files, omitted by type (weight or training artifacts). Recreated by rerunning the finetunes.

## Extracted persona-vector tensors (`.../codebase/persona_vectors/<model>/*.pt`)

- About 117 `*.pt` files, roughly 63 MB total across the replication and the followup
  workspaces (001, 002, 004). These are the extracted per-trait persona-vector tensors
  (prompt-avg, prompt-last, response-avg difference-of-means directions) for each model and
  trait. Activation artifacts, omitted by type regardless of their small size.
  Regenerate: `run_genvec.py` / `generate_vec.py` on each model with the kept trait data.

## Base model snapshot (`run/followup/001-living-update/workspace/models/`)

- `Mistral-7B-Instruct-v0.2-patched/` — a directory of symlinks pointing into a shared local
  model store for the Mistral-7B-Instruct-v0.2 base weights (`*.safetensors`, `*.bin`). The
  whole directory was omitted (the links pointed outside the run tree).
  Obtain: download `mistralai/Mistral-7B-Instruct-v0.2` from Hugging Face.

## Other large files

- `run/replication/codebase.diff` — about 723 MB. The full working-tree diff of the replication
  codebase against the upstream repository. Regenerate: `git diff` against the upstream
  persona-vectors repository from the kept `codebase/`.
- `run/replication/codebase/dataset.zip` — about 61 MB. A zipped copy of the trait and training
  datasets. The unzipped `dataset/` directory is kept, so this archive is redundant.
  Regenerate: `zip -r dataset.zip dataset/`.

## Environments and caches (stripped as clutter, not research data)

Python virtual environments (including `run/followup/004-living-update/workspace/.venv-tf/`),
`__pycache__/`, `unsloth_compiled_cache/`, and the nested upstream `.git/` history were also
removed during staging. Recreate environments from the kept `requirements.txt`.
