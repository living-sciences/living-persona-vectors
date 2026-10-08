# Persona-Vectors living-update — CONTINUATION (supersedes the timed-out 001-living-update)

A prior run of this exact living-update (`followup/001-living-update/`) produced genuine,
complete per-model outputs for most of the ladder but was killed by a session timeout before it
ran the last model and before it wrote its deliverables. Your job is to FINISH it: reuse all the
completed work, run only what's missing, and write the three deliverables. Do NOT redo completed
models. Follow the ORIGINAL spec for methodology, the model table, claim IDs, and the exact
result_card format — read it in full first:
  /net/projects2/chai-lab-models/haokunliu/alignment-batch/followup_plans/persona-vectors/001-living-update.instruction.md

## What 001 ALREADY completed (reuse verbatim; read from this dir, it is READ-ONLY history)
`FU001=/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/001-living-update`
- **Mistral-7B-Instruct-v0.2** — FULLY done: all stages 1–10, `$FU001/results/rejudged/Mistral-7B-Instruct-v0.2/` (46 files), `$FU001/results/monitor_proj/Mistral-7B-Instruct-v0.2/`, `$FU001/results/logs/Mistral-7B-Instruct-v0.2/` (sel_layer.json, coarse_grid.json, active_traits.txt, all 1–10 logs).
- **Llama-3.1-8B-Instruct** (paper model) — FULLY done: same structure, 46 rejudged, monitor_proj present.
- **DeepSeek-R1-Distill-Llama-8B** — steering + judge + monitor_proj done (39 rejudged, monitor_proj present); only the final step8/rejudge2 may be missing — finish ONLY those two cheap stages for DeepSeek if they're absent, else leave as-is with a note.
- **Qwen2.5-7B-Instruct** (paper model) — REUSE the replication's on-disk baseline/steering results (no compute), exactly as the original spec says.
The workspace scripts that produced all this are at `$FU001/workspace/` (run_one_model.sh, env.sh, pick_layers.py, judge_extract.py, rejudge_all.py). COPY them into YOUR `workspace/` and reuse them unchanged.

## What you must DO (in the FOREGROUND — never background a job and end your turn)
1. Copy `$FU001/workspace/*` into your study `workspace/`, and copy/symlink `$FU001/results/*` into your `results/` so your aggregation sees all completed models. Reuse the replication venv at `run/.venv` (VERITAS_VENV_DIR) if present; else rebuild from the original spec's pinned set.
2. Run the ONE missing model **Qwen3-8B-Nonthinking** (`/net/projects2/chai-lab/shared_models/Qwen/Qwen3-8B-Nonthinking`, 36 layers) through the SAME `run_one_model.sh` pipeline with the SAME reduced grid + the SAME paper-exact gpt-4.1-mini judge (OpenRouter) the original spec mandates. If it hangs on the monitor/judge stage (a known issue with reasoning-style models — the 001 run hit it on DeepSeek and worked around it by running monitor generation with the judge disabled first, then judging), apply the same workaround. **Budget guard:** if Qwen3 cannot finish within ~2 hours, DESCOPE it with a documented note and finalize the study with the 4 completed models — a 4-model living-update is a valid, complete deliverable; DO NOT fail to write the card over the 5th model.
3. Aggregate ALL available models and write the three REQUIRED deliverables per the original card format: `report.md`, `followup_summary.json`, `result_card.json` (schema sai.followup.result_card/v1). The comparison table's original/baseline side must be read from the replication artifacts / 001 outputs on disk (cite files), never from memory.

## Discipline
- Every number from THIS or 001's actual execution; the judge must be the SAME paper-exact gpt-4.1-mini for old and new (never swap judges). Disclose exactly which models are included and any descoped (Qwen3, DeepSeek step8) with reasons.
- Foreground only. Write followup_summary.json incrementally so a cut-off still leaves a record.