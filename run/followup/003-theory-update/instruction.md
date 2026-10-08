# 002 — Theory update: what governs persona-vector effect size across the model era?

Runs AFTER 001. Consumes 001's artifacts only (`followup/002-living-update/results/` — the tidy
long-format table, per-model selected layers, over-time figure CSV). **Reruns no models and no
judge calls** unless a named gap forces one small confirmation. Invocation mirrors 001
(`followup ... --name theory-update`, reuses `002-theory-update` reading `002-living-update/results/`).

## Question
001 shows the three mechanisms still fire on 2023–2025 chat models. 002 asks *what governs their
strength*: how do steering effect size and projection-monitoring correlation move with **model
size, generation/era, layer-depth, and RLHF/alignment style**, and do persona directions
**transfer/align across families**? Fit a small parametric "law per era" and state the updated claim.

## Inputs (from 001, read-only)
- Per (model, trait): baseline C5, selected layer (+ as fraction of depth), steering effect size
  (Panel A), monitoring Pearson r (C2), separability AUC (C13).
- Per model: params, n_layers, release date, family, alignment style (base-RLHF vs
  reasoning-distill vs newer-RLHF), from 001's model table.
- The persona vectors `persona_vectors/<slug>/{trait}_response_avg_diff.pt` at each selected layer
  (already on disk from 001) — needed only for the transfer/alignment analysis (cheap CPU cosine).

## Analyses
1. **Effect size vs {size, era, depth, alignment}.** Regress steering effect size and monitoring r
   on log-params, era ordinal, selected-layer-fraction, and an alignment-style factor. Report
   coefficients + a partial-dependence plot. Hypothesis from the landscape (2606.26161): the
   effective locus is a stable *late-layer* fraction (~0.5–0.7 depth), roughly constant across
   generation — test whether selected-layer-fraction is era-invariant.
2. **Cross-family direction transfer/alignment.** For each trait, cosine-similarity matrix between
   per-model persona vectors (aligned by hidden dim where equal; for mismatched dims report only
   same-dim pairs — Llama/DeepSeek/Qwen3/Mistral are all 4096, Qwen2.5 is 3584). Do same-family
   (Llama↔DeepSeek) directions align more than cross-family? Optionally test *portability*: apply
   model A's vector to model B (same hidden dim) at B's selected layer for one trait/coef and
   measure whether the trait still rises (one small confirmation gen allowed, budgeted ≤0.3 GPU-h +
   ≤$1 judge — foreground). Connects to "off-the-shelf vectors rival targeted" (2605.21006).
3. **Reasoning-model deviation.** Is DeepSeek-R1-Distill an outlier on any axis (effect size,
   monitoring r, selected depth, coherence collapse)? Report with the `<think>`-token caveat.
4. **Small parametric model per era.** Fit the simplest defensible model (e.g. monitoring
   `r ≈ a + b·era + c·depth_frac`; steering effect `≈ logistic in coef` with per-era midpoint) and
   report parameters + R². Keep it a *description of ~5 points*, clearly labelled low-power — do not
   over-claim significance.

## Deliverables
- `report.md` — the four analyses, the fitted parametric summary, and the **updated claim** stated
  plainly (e.g. "Across 2023→2025 chat models, projection monitoring is era-stable (r 0.7–0.9,
  slope ≈ 0), steering effect size is [rising/flat/…], the effective steering layer sits at a
  constant ~X depth fraction regardless of generation, and persona directions align within family
  more than across").
- `followup_summary.json`, `result_card.json` (schema `sai.followup.result_card/v1`: headline with
  the key coefficient/number; metrics = regression slopes and the cross-family cosine; ≤2 tables;
  figures = transfer heatmap + partial-dependence plot under `results/`).
- Figures under `results/`: per-trait cross-model cosine heatmap; effect-size/monitoring-r vs
  {era, depth} partial-dependence.

## Guardrails
- Same ANTI-GOALS as 001: foreground only; no invented numbers; every value traces to a 001
  artifact or a disclosed small confirmation run scored by the SAME paper-exact judge; disclose the
  ~5-point low statistical power explicitly; report same-hidden-dim constraint on the cosine
  analysis rather than fabricating cross-dim alignment.
- Budget: primarily CPU analysis; ≤0.3 GPU-h and ≤$1 judge only if the optional portability
  confirmation (analysis 2) is run. Otherwise 0 GPU / 0 judge.