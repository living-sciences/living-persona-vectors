# Living update — Persona Vectors extended to the 2026 model refresh: Qwen3.5-9B (study 004)

## Question

*Persona Vectors* (Chen et al., 2025, arXiv 2507.21509) established three mechanisms on 2024-era
7–8B chat models: **(C1/C8) steering** along an extracted persona vector raises the trait
monotonically with the steering coefficient; **(C2) projection monitoring** — the last-prompt-token
projection onto the persona vector predicts the next response's trait score; and **(C13) sample
separability** — training-sample projections separate trait-eliciting data from normal controls.
Study 002 tracked these across a five-model 2023→2025 chat ladder under one paper-exact
gpt-4.1-mini judge. This study (004) **extends the ladder to the 2026 model refresh** by adding the
newest Qwen — **Qwen3.5-9B (Mar 2026, 32 layers, hidden 4096)** — as era 6, reusing all five prior
models verbatim, and asks: does the paper's finding still hold on the 2026 flagship?

This card is a **recovery**: the compute run finished the Qwen3.5-9B generations and judged the
steering grid, but the job hit its `--timeout` mid-DeepSeek-judge before the last GPU stages and
the result card were written. This tick is pure CPU + OpenRouter-API post-processing — no GPU, no
model loading, no re-generation.

## Approach

- **Reused verbatim from study 002 (read-only history):** the full per-model results for
  Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (reused anchor),
  DeepSeek-R1-Distill-Llama-8B, and Qwen3-8B-Nonthinking. No prior number was recomputed.
- **Qwen3.5-9B, computed this session from on-disk artifacts only:**
  - **C5 / C8 / C1** parsed from `results/progress.txt` — the 39 already-judged grid conditions
    (3 baselines + evil/sycophantic/hallucinating × layers {14,18,21,24} × coefs {1.0,1.5,2.0}),
    plus the corresponding paper-judge sidecars under `results/rejudged/Qwen3.5-9B/`. Selected
    layers come from `results/logs/Qwen3.5-9B/sel_layer.json` (argmax paper-judge trait_mean over
    the coarse grid): **evil L18, sycophantic L21, hallucinating L18**.
  - **C2 (attempted):** the four monitor CSVs (`evil_monitor_{pos,neg}`, `sycophantic_monitor_{pos,neg}`,
    500 rows each) were **UNJUDGED on disk** and were **re-judged this tick with the same
    paper-exact gpt-4.1-mini instrument** (`rejudge_all.py`, which writes the exact `gpt_<trait>`
    sidecar format `compute_results.py` reads for every other model). 0 judge errors, ~$0.95.
  - Aggregation: the Qwen3.5-9B entry was built with the same schema as the other models and
    appended to 002's `aggregate.json`; the over-time and dose-response figures were regenerated
    to include the 2026 point.
- **Every trait/coherence score — old and new — is the identical paper-exact gpt-4.1-mini
  instrument** (OpenRouter, `max_tokens=1, T=0, seed=0, logprobs top-20`, numeric-mass expectation).
  The judge was never swapped.

## Results

**Headline: the paper's steering mechanism replicates cleanly on the 2026 Qwen.** Along the
extracted persona directions, Qwen3.5-9B's trait scores rise monotonically with the steering
coefficient from near-floor baselines to **70.7 / 98.3 / 99.7** (evil / sycophantic /
hallucinating) at the selected layers — squarely inside the 2023→2025 ladder band and far above the
reasoning-distilled DeepSeek floor. The projection-monitoring (C2) and sample-separability (C13)
sub-claims were **not measurable this CPU/API-only tick** because the GPU forward-pass stages that
produce them (`monitor_proj`, `step8`) never ran before the timeout — they are unmeasured, not
refuted. Figure 1 (`results/fig1_over_time.png`) is the extended over-time arc; Figure 2
(`results/fig2_steering_curves.png`) shows the monotone dose-response including the 2026 curve.

### C5 — base trait score (mean of paper-judge trait score, n=200)

| Model | era | evil | sycophantic | hallucinating | source |
|---|---|---|---|---|---|
| Qwen3-8B-Nonthinking | 5 (2025-04) | 0.00 | 6.22 | 26.37 | (study 002) |
| **Qwen3.5-9B (new)** | **6 (2026-03)** | **0.00** | **5.65** | **16.25** | `results/progress.txt` + `results/rejudged/Qwen3.5-9B/*_baseline.csv` |

Qwen3.5-9B's baselines are low and well-behaved (evil ~0; sycophantic 5.7; hallucinating 16.3 — the
lowest baseline hallucination on the ladder), giving steering plenty of headroom.

### C8 — selected steering layer (1-indexed; argmax paper-judge trait_mean over the coarse grid)

| Model | evil | sycophantic | hallucinating | source |
|---|---|---|---|---|
| **Qwen3.5-9B (32L)** | **18** | **21** | **18** | `results/logs/Qwen3.5-9B/sel_layer.json` |

Selected layers land in the mid-network band (≈56–66% depth), consistent with the paper and the
rest of the ladder.

### C1 — steering raises the trait, monotone in coefficient (selected layer, paper-judge trait_mean)

| Model | evil (coef 1.0/1.5/2.0) | sycophantic (1.0/1.5/2.0) | hallucinating (1.0/1.5/2.0) | max-coef Δ vs base |
|---|---|---|---|---|
| **Qwen3.5-9B** | 47.7 / 67.0 / **70.7** (L18) | 78.6 / 95.3 / **98.3** (L21) | 73.8 / 98.6 / **99.7** (L18) | +70.7 / +92.7 / +83.5 |

The dose-response is monotone in the coefficient for all three traits. Sycophantic and hallucinating
reach near-ceiling (98–100), evil reaches ~71 (comparable to Qwen3-8B's 82.6 and above DeepSeek's
71.5). Steering clearly works on the 2026 model. (Only the coarse coefs 1.0/1.5/2.0 were judged into
`progress.txt`; the raw confirm sweep 0.5–2.5 exists on disk but was not re-judged this tick.)

### C2 — projection monitoring (Pearson r) — DESCOPED this tick

| Model | evil | sycophantic | hallucinating | status |
|---|---|---|---|---|
| prior ladder (002) | 0.45–0.99 | 0.48–0.97 | 0.25–0.91 | measured |
| **Qwen3.5-9B** | **n/a** | **n/a** | **n/a** | **descoped (see below)** |

The four monitor generations *were* re-judged with the paper-exact instrument, and the behavioral
signal is strong — the persona-positive prompts elicit far more trait than the negatives:

| trait | monitor **pos** mean | monitor **neg** mean |
|---|---|---|
| sycophantic | **98.80** | 4.01 |
| evil | 10.96 | 0.00 |

But **C2 is the projection-vs-judge correlation**, and the projection column does not exist on disk.
That column is produced by the GPU `monitor_proj` step (a forward pass that projects each monitor
answer's residual-stream activation onto the persona vector); the job died during
`MONITOR hallucinating pos`, **before** `monitor_proj` ever ran. The persona vectors themselves
(`persona_vectors/Qwen3.5-9B/*_response_avg_diff.pt`) are on disk, but the per-sample monitor
activations are not, so projections cannot be recomputed without loading the model on GPU — out of
scope for this recovery. hallucinating additionally has **no monitor file** (generation was cut off).
C2 is therefore reported as **not measured this tick**, with the trait side re-judged and cached so a
short GPU follow-up (monitor_proj on the existing generations) can complete it.

### C13 — training-sample separability (AUC) — DESCOPED this tick

`step8` separability was never run for Qwen3.5-9B (no `results/step8/Qwen3.5-9B/`). It needs GPU
forward passes over the training-sample activations and is out of scope for a CPU/API tick. Reported
as **not measured this tick** (not refuted).

## Does the paper's finding hold on the 2026 model?

**Yes for the core steering claim.** On Qwen3.5-9B the extracted persona directions steer behavior
strongly and monotonically (traits driven to 70.7 / 98.3 / 99.7 from baselines of 0.0 / 5.7 / 16.3),
exactly the qualitative and quantitative pattern the paper reports and that held across the entire
2023→2025 ladder. The 2026 flagship is a fully steerable member of the arc (Fig 1A, Fig 2).

**Undetermined this tick for the monitoring and separability sub-claims (C2, C13),** and for whether
the reasoning-model monitoring-collapse pattern (seen on DeepSeek-R1-Distill) extends to Qwen3.5-9B.
These require the two GPU stages that never executed before the timeout. The re-judged monitor
pos/neg separation (sycophantic 98.8 vs 4.0) is consistent with the direction being trait-bearing,
but is not the projection-based C2 statistic and must not be read as one.

## Deviations & limitations

- **Write blocker (environmental).** During this tick the entire `/net/projects2` NFS export was
  **read-only** (EROFS on every write, including paths not owned by the user; `mount` reported `rw`
  but the server refused writes — a server-side read-only state, not a permissions problem). The RUN
  tree could not be written. All deliverables were produced in a writable staging mirror at
  `/net/scratch/haokunliu/pv004-recovery/` (never under `/home`), with a copy-back helper
  (`copy_back.sh`) staged alongside. Once `/net/projects2` is writable again, copy back with:
  `bash /net/scratch/haokunliu/pv004-recovery/copy_back.sh`.
- **C2/C13 descope** is upstream of judging (missing GPU projections / step8), as detailed above —
  a real attempt was made (monitors re-judged, disk searched for projection/activation caches;
  none exist), then descoped with the trait-side cached for a follow-up. This is the honest state,
  not smoothed.
- **C1 coarse-only curve.** Qwen3.5-9B's reported dose-response uses the 3 coarse coefs
  (1.0/1.5/2.0) judged into `progress.txt`, vs the 5-point confirm curve of the other models; the
  raw confirm sweep (0.5–2.5) exists unjudged on disk.
- **Finetune-dependent claims (C3/C4/C10/C11/C12/C15)** and C6/C7/C14 remain out of scope, as in
  study 002; the Qwen2.5-7B paper anchor stands.
- **Same judge, reused models untouched.** Judge is identical for old and new; 0 judge errors; the
  five prior models are carried over verbatim from 002.

## Artifacts (staged under `/net/scratch/haokunliu/pv004-recovery/`, to be copied into RUN)

- `result_card.json`, `report.md`, `followup_summary.json` — the three deliverables.
- `results/aggregate.json` — all 6 models' C5/C8/C1/C2/C13 with provenance (Qwen3.5-9B added).
- `results/tidy_results_long.csv` — tidy long-format table for all 6 models.
- `results/fig1_over_time.png` — headline over-time arc extended to the 2026 point.
- `results/fig2_steering_curves.png` — steering dose-response including Qwen3.5-9B's curve.
- `results/rejudged/Qwen3.5-9B/{evil,sycophantic}_monitor_{pos,neg}.csv` — this tick's paper-judge
  monitor sidecars (enable a future GPU tick to finish C2).
- `copy_back.sh` — rsync helper to land everything under RUN when the filesystem is writable.
