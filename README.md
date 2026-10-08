# Persona Vectors, a living paper

This repository is a living-paper companion to:

**Persona Vectors: Monitoring and Controlling Character Traits in Language Models.**
Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. 2025.
Anthropic Fellows Program; UT Austin; Constellation; Truthful AI; UC Berkeley; Anthropic.
Preprint, arXiv:2507.21509. https://arxiv.org/abs/2507.21509

The paper is a preprint and does not carry a journal DOI; cite it by its arXiv identifier.

## What a living paper is

A living paper starts from an honest, end-to-end replication of a published result and
then keeps going. After the original experiments are reproduced, a series of follow-up
studies revisit the paper's claims on newer models, push on the theory behind them, and
record what still holds and what breaks. Everything here is organized so you can see both
the original reproduction and each later update, with the code, logs, and result summaries
that produced every number. The companion web page tracks the same work as it grows:

Living page:
https://livingscience.ai/safety/living-persona-vectors

## What this repo contains

### `run/replication/`
A full ten-step reproduction of the paper on Qwen2.5-7B-Instruct (with Llama-3.1-8B-Instruct
for the monitoring step). All three core mechanisms reproduce: steering along an extracted
persona vector raises the trait, last-prompt-token projection monitors the next response's
trait (Pearson r about 0.90 / 0.88 / 0.82 for evil / sycophantic / hallucinating), and
training-sample projections separate trait data from controls (separability AUC about
1.00 / 1.00 / 0.995). The one divergence is single-layer concept-ablation finetuning, which
was ineffective here while preventative steering worked. The extraction and evaluation judge
was first run locally (Qwen2.5-32B-Instruct, no OpenAI key at run time) and then re-scored
uniformly with the paper-exact judge (gpt-4.1-mini); see `replication/REJUDGE_NOTE.md` and
`replication/evidence_summary.json`.

### `run/followup/`
The follow-up studies form a continuation chain that extends the replication across a
model-era ladder and then analyzes what governs the mechanisms.

- **001-living-update** — The initial era-ladder compute tick: runs the three mechanisms on
  a 2023 to 2025 ladder of 7 to 8B chat models with the same paper-exact judge for old and
  new models. This directory has no `result_card.json` of its own; it holds the raw ladder
  artifacts (logs, re-judged steering and monitor CSVs, step-8 separability) that study 002
  formalizes. Kept for its genuine workspace and results.
- **002-living-update** — Across a five-model 2023 to 2025 chat ladder scored by the same
  paper-exact judge, persona-vector steering (traits pushed to 72 to 100 out of 100) and
  sample separability (AUC at or above 0.996) replicate everywhere, and projection monitoring
  stays strong (Pearson r 0.65 to 0.99) on every standard model but collapses to r 0.25 to 0.48
  on the reasoning-distilled DeepSeek-R1.
- **003-theory-update** — Across the 2023 to 2025 ladder the effective persona-steering layer
  is era-invariant at a constant depth fraction near 0.55; monitoring strength is governed by
  alignment style and collapses only on the reasoning-distilled DeepSeek; and persona directions
  align far more within an architecture family than across families (cosine 0.48 vs 0.00).
  A secondary analysis that reuses study 002's artifacts, with no new generation or judging.
- **004-living-update** — On the newest 2026 Qwen (Qwen3.5-9B) the paper's core steering
  mechanism replicates cleanly: traits rise monotonically with the steering coefficient from
  near-floor baselines to 70.7 (evil) / 98.3 (sycophantic) / 99.7 (hallucinating) out of 100
  under the same judge. The projection-monitoring and separability sub-claims could not be
  measured on this CPU/API-only recovery tick because their GPU stages did not run; they are
  unmeasured, not refuted.

Each study directory keeps its `instruction.md`, `workspace/` (the code and config used),
`results/` (figures, JSON, CSV), and, where the study produced one, `report.md`,
`result_card.json`, and `followup_summary.json`.

## Large files

Model weights, LoRA adapters, optimizer states, extracted persona-vector tensors, and other
very large artifacts are not included here. Each omission is listed, with its size and how to
regenerate or obtain it, in `LARGE_FILES_OMITTED.md`.

## A note on terms

The original paper and its upstream code are the work of the authors listed above and remain
under their terms; the upstream code carries its own license, retained inside the `codebase/`
directories. This living-paper repository is an independent replication and extension built
for research transparency. Please cite the original paper and respect the authors' terms when
using anything derived from their work.
