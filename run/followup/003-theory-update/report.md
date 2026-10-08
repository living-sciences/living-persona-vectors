# Theory update — what governs persona-vector effect size across the 2023→2025 model era? (study 003)

## Question

Study 002 (the "living update", on disk at `followup/002-living-update/`) showed that the three
*Persona Vectors* mechanisms — **(C1/C8) steering**, **(C2) projection monitoring**, **(C13)
sample separability** — still fire on a five-model ladder spanning 2023→2025 (Mistral-7B-Instruct-v0.2 →
Llama-3.1-8B-Instruct → Qwen2.5-7B-Instruct → DeepSeek-R1-Distill-Llama-8B → Qwen3-8B-Nonthinking),
all scored with the *same* paper-exact gpt-4.1-mini judge. This study (003) asks **what governs their
strength**: how do steering effect size and projection-monitoring correlation move with **model size,
generation/era, layer-depth, and RLHF/alignment style**, and do persona directions **transfer/align
across families**? It fits a small, deliberately low-power "law per era" and states the updated claim.
It **reruns no models and makes no judge calls** — it is pure CPU analysis over 002's artifacts, plus a
cosine analysis over the persona vectors already on disk.

## Approach

- **Reused, read-only, from 002:** the per-(model,trait) values (baseline C5, selected layer,
  steering effect size, monitoring Pearson r C2, separability AUC C13, and the C1 dose-response
  curves) were read verbatim from `followup/002-living-update/results/aggregate.json`. The
  per-model era/release/n_layers come from that file's `ladder`. Nothing was recomputed from raw
  generations or re-judged.
- **Added model metadata** (params, hidden dim, family, alignment style) as fixed public
  model-card facts (`workspace/model_meta.py`) — these are architecture/release facts, not
  measurements. Alignment style uses the three levels named in the instruction: **base-RLHF**
  (Mistral, Llama-3.1), **newer-RLHF** (Qwen2.5, Qwen3), **reasoning-distill** (DeepSeek-R1-Distill).
- **Analyses (all CPU):** (1) univariate + multivariate OLS of steering effect and monitoring r on
  log-params, era, selected-layer depth-fraction, and alignment factor, plus an era-invariance test
  of depth-fraction; (2) cross-model cosine of the persona vectors
  (`persona_vectors/<slug>/<trait>_response_avg_diff.pt`, indexed at each model's selected layer),
  restricted to the same-hidden-dim (4096) set; (3) a reasoning-model deviation summary; (4) a small
  parametric fit per era (linear monitoring law + per-era logistic steering dose-response).
- **Not run:** the *optional* portability confirmation generation (analysis 2). **No GPU is present
  in this environment** (`nvidia-smi` → "No devices found"), so no model can be run; per the
  guardrail this study is therefore **0 GPU / 0 judge**. The portability question is answered only by
  the cosine geometry (a strong upper-bound proxy), not by a live transfer generation.

Data provenance: every value traces to `followup/002-living-update/results/aggregate.json` (itself
sourced from 002's paper-judge sidecars / the reused Qwen2.5 replication artifacts). Persona vectors:
`002-living-update/workspace/codebase/persona_vectors/` (4 models) and
`replication/codebase/persona_vectors/Qwen2.5-7B-Instruct/`.

## The 5-model ladder (metadata + reused measurements)

| Model | era | release | params | hidden | n_layers | alignment | mean sel-layer frac | mean steer Δ | mean monitor r |
|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | 1 | 2023-12 | 7.24B | 4096 | 32 | base-RLHF | 0.479 | 79.1 | 0.888 |
| Llama-3.1-8B-Instruct | 2 | 2024-07 | 8.03B | 4096 | 32 | base-RLHF | 0.594 | 82.6 | 0.949 |
| Qwen2.5-7B-Instruct | 3 | 2024-09 | 7.62B | 3584 | 28 | newer-RLHF | 0.667 | 87.9 | 0.850 |
| DeepSeek-R1-Distill-Llama-8B | 4 | 2025-01 | 8.03B | 4096 | 32 | reasoning-distill | 0.438 | 52.0 | 0.393 |
| Qwen3-8B-Nonthinking | 5 | 2025-04 | 8.19B | 4096 | 36 | newer-RLHF | 0.583 | 82.4 | 0.834 |

(Full 15-row table: `results/theory_table.csv`. Steering Δ = trait score at max coefficient minus
baseline; monitor r = C2 Pearson.)

## Results

### 1. Effect size vs {size, era, depth, alignment}

**Univariate marginal slopes** (15 model×trait points; `results/regression_results.json`):

| Predictor | Steering Δ slope (R², p) | Monitoring r slope (R², p) |
|---|---|---|
| log10(params) | −161 (R²=0.02, p=0.58) | −3.04 (R²=0.07, p=0.33) |
| era (ordinal) | −2.4 (R²=0.03, p=0.55) | −0.066 (R²=0.19, p=0.11) |
| **selected-layer depth-fraction** | **+123 (R²=0.33, p=0.024)** | **+1.31 (R²=0.33, p=0.025)** |
| alignment (one-way ANOVA) | F=3.68, p=0.057 | **F=27.7, p<0.001** |

- **Size has no leverage** — all five models are 7.24–8.19B, so log-params barely varies and both
  slopes are non-significant. The ladder cannot speak to a genuine size law.
- **Era alone is not a significant driver** of either outcome. The apparent negative era slope is an
  artifact of DeepSeek-R1-Distill (era 4) sitting far below its neighbours; eras 1,2,3,5 are flat.
- **Layer-depth is the one robust continuous predictor:** models whose selected layer sits deeper
  (toward ~0.7 of depth) show larger steering effect *and* higher monitoring r (both R²≈0.33,
  p≈0.02). This is consistent with the paper's mid/late-layer locus.
- **Alignment style governs monitoring.** ANOVA group means: base-RLHF **0.919**, newer-RLHF
  **0.842**, reasoning-distill **0.393**. In the full multivariate model (`monitor_r ~ log-params +
  era + depth + align`), the *only* significant coefficient is the reasoning-distill dummy
  (−0.524, p=0.015; overall R²=0.85, condition number ≈1.7e4 → treat coefficients as descriptive).

**Era-invariance of the effective locus (landscape hypothesis 2606.26161).** Selected-layer
depth-fraction regressed on era: slope **+0.005, p=0.79, R²=0.006**; mean **0.552 ± 0.099**, range
0.44–0.71. The fraction does **not** trend with generation — the effective steering layer is a
stable ~0.55 depth fraction across 2023→2025, confirming the hypothesis (Figure 3, left).

### 2. Cross-family direction transfer / alignment

Per-trait cosine between persona vectors at each model's selected layer, restricted to the four
hidden-dim-4096 models (Mistral, Llama-3.1, DeepSeek-R1-Distill, Qwen3). Qwen2.5 (3584-dim) cannot
be compared and is reported as excluded, per the guardrail — no cross-dim alignment is fabricated.

| Pair type | n pairs | mean cosine | range |
|---|---|---|---|
| **same-family (Llama ↔ DeepSeek-R1-Distill)** | 3 (one/trait) | **0.478** | 0.371 – 0.568 |
| cross-family (all other 4096-dim pairs) | 15 | **0.002** | −0.019 – +0.029 |

Same-family directions align ~200× more strongly than cross-family. Robustness: putting Llama and
DeepSeek at a *matched* layer (DeepSeek's selected layer 14) raises the cosine further to
0.54 / 0.72 / 0.73 (syco/evil/hallu). Full matrices + heatmap: `results/transfer_results.json`,
`results/cosine_pairs.csv`, `results/fig1_transfer_heatmap.png`.

**Interpretation / caveat.** DeepSeek-R1-Distill-Llama-8B is a reasoning-distillation *of*
Llama-3.1-8B, so the two share the same residual-stream coordinate basis; a ~0.5 cosine means the
persona direction is largely preserved through distillation. Independently-trained families
(Mistral, Qwen3) do **not** share a basis, so their raw-cosine ≈ 0 measures basis non-alignment,
**not** conceptual dissimilarity — off-the-shelf raw-cosine transfer across architectures is not
expected to work without a learned basis map. This bounds the "off-the-shelf vectors rival
targeted" idea (2605.21006) to the *within-architecture* case here. The optional live portability
generation was not run (no GPU); the cosine is the (favourable) geometric proxy.

### 3. Reasoning-model deviation (DeepSeek-R1-Distill-Llama-8B)

DeepSeek-R1-Distill is the outlier on essentially every strength axis (Figure 3, right):

| Axis | DeepSeek | other 4 models | 
|---|---|---|
| Monitoring r (C2) | **0.25–0.48** | 0.65–0.99 |
| Steering Δ over baseline | **16–71** (mean 52) | 54–98 (means 79–88) |
| Steering logistic fit R² (per-era) | **0.33** | 0.75–0.90 |
| Steering slope k / midpoint x0 | **1.02 / 1.73** (flattest, latest) | 2.3–3.7 / 0.8–1.25 |
| Selected depth-fraction | **0.438** (shallowest, all 3 traits) | 0.55–0.71 |
| Baseline hallucination C5 | **72.2** (inflated) | 22.8–43.5 |

Only sample-separability (C13 AUC ≥ 0.996) is unaffected. **`<think>`-token caveat:** the last
prompt-token projection that C2 monitors is computed *before* a long intervening reasoning trace, so
prompt-time state decouples from the final answer — this mechanistically explains both the monitoring
collapse and the inflated baseline hallucination score (the `<think>` scaffolding is scored as
hallucinatory). The deviation is best read as *alignment-style* (reasoning-distill), not as an era
effect: it is a property of the training recipe, not of 2025 per se (the 2025 Qwen3-Nonthinking
behaves like a standard chat model).

### 4. Small parametric model per era (deliberately low-power, ~5 points)

**Monitoring law** (OLS, 15 points): `r ≈ 0.232 − 0.074·era + 1.398·depth_frac`, **R² = 0.56**
(era coef p=0.029, depth coef p=0.008). Read descriptively: monitoring rises with layer-depth and
carries a small negative era term that is *almost entirely* the DeepSeek dip — drop the reasoning
model and the era slope is ≈0 (standard models span r 0.65–0.99 with no generation trend).

**Steering dose-response law** (3-parameter logistic `L/(1+e^{−k(coef−x0)})` fit per model to its 15
curve points): midpoints x0 = 1.22, 1.12, 1.25, **1.73**, 0.80 for eras 1–5. Regressing midpoint on
era gives slope −0.023, p=0.86, R²=0.01 → **the steering coefficient needed for half-max is
era-stable at ~1.2**, with DeepSeek (1.73, flat, poor fit) the lone exception. Saturation ceilings
L ≈ 94–110 everywhere.

## Updated claim

> **Across 2023→2025 chat models, projection monitoring is era-stable for standard RLHF models
> (Pearson r 0.65–0.99, no significant era slope) but is governed by *alignment style* rather than by
> size or generation: it collapses to r ≈ 0.25–0.48 under reasoning-distillation (DeepSeek-R1-Distill),
> the one qualitative break. Steering effect size is uniformly large and roughly flat across
> generation (Δ 54–98 over baseline for standard models; half-max coefficient era-stable at ~1.2),
> depending more on selected-layer depth than on era. The effective steering layer sits at a constant
> ~0.55 depth fraction (range 0.44–0.71) regardless of generation — the late-layer locus is
> era-invariant (depth-vs-era slope ≈ 0, p=0.79). Persona directions align within architecture family
> (Llama ↔ DeepSeek cosine ≈ 0.48, up to 0.73 at matched layers) but are ~orthogonal across
> independently-trained families in the raw hidden basis (≈ 0). Model *size* is untested — all five
> models are 7–8B.**

## Comparison to the 002 (baseline) values

Every input number is reused verbatim from 002; this study adds only derived statistics. The table
below shows the reused per-model summaries this analysis rests on (baseline side read from
`followup/002-living-update/results/aggregate.json`).

| Quantity (per model, evil/syco/hallu unless noted) | 002 value (aggregate.json) | This study's derived statistic |
|---|---|---|
| Monitoring r range across ladder | 0.25–0.99 | era slope −0.066 (p=0.11, ns); alignment ANOVA F=27.7 (p<0.001) |
| Selected-layer fraction across ladder | 0.44–0.71 | era-invariant: slope +0.005, p=0.79, mean 0.552±0.099 |
| Steering Δ across ladder | 16–98 | depth slope +123 (R²=0.33, p=0.024); midpoint era-stable ~1.2 |
| DeepSeek monitoring r | 0.452/0.482/0.246 | flagged outlier (−0.52 reasoning-distill dummy, p=0.015) |
| Llama & DeepSeek persona vectors | on disk (.pt) | same-family cosine 0.478 vs cross-family 0.002 |

## Deviations & limitations

- **No GPU in this environment**, so the *optional* portability confirmation generation (analysis 2)
  was not run; the study stays at **0 GPU / 0 judge** as the default guardrail allows. Transfer is
  answered by cosine geometry alone — a favourable proxy, not a behavioural test.
- **Low statistical power, stated plainly.** There are 5 models = 5 eras (era is 1:1 with model, and
  partly confounded with alignment style); the 15 model×trait points cluster into just 5 independent
  model-level draws. All regressions are **descriptive summaries of ~5 points**, not powered
  hypothesis tests; the multivariate fit is near-collinear (condition number ≈1.7e4). p-values are
  reported for transparency, not as confirmatory inference.
- **Size axis is essentially untested:** every model is 7.24–8.19B, so log-params has almost no
  variance and any "size law" is out of reach on this ladder.
- **Cross-family cosine is basis-limited:** raw-hidden cosine is only meaningful within a shared
  architecture basis (Llama↔DeepSeek). The ≈0 cross-family values are non-alignment of independently
  learned bases, not evidence that the concepts differ — reported as such rather than as a transfer
  failure. Qwen2.5 (3584-dim) is excluded from every cosine per the same-dim guardrail.
- **Alignment-style labels** are the analyst's mapping onto the three levels named in the instruction;
  DeepSeek's deviation is equally describable as "reasoning-distill recipe" or "era-4 point" — the
  data cannot separate the two, and the report says so.

## Artifacts (under `results/`)

- `theory_table.csv` — 15-row per-(model,trait) analysis table.
- `regression_results.json` — all univariate/multivariate fits, era-invariance test, parametric laws.
- `transfer_results.json`, `cosine_pairs.csv` — cross-model cosine matrices and pair list.
- `fig1_transfer_heatmap.png` — per-trait cross-model cosine heatmaps.
- `fig2_partial_dependence.png` — steering Δ and monitoring r vs {era, depth}, colored by alignment.
- `fig3_depth_invariance_deepseek.png` — depth-fraction era-invariance + DeepSeek deviation bars.
