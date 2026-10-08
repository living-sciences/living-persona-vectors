# JUDGE RE-SCORE (2026-09-04) — paper-exact judge applied

Every per-response trait and coherence score in eval_persona_eval/ has been
RE-JUDGED with the paper's exact judge — gpt-4.1-mini-2025-04-14, 0-100
logprob-expectation (top-20 logprobs, numeric-token weighted mean, refusal if
numeric mass < 0.25) — replacing the interim local Qwen2.5-32B judge that was used
during the replication run because no OpenAI key was available then (documented in
assess/fix_severity.json as a major deviation).

- 104,401 judge calls via OpenRouter, 0 errors. Sidecars with raw re-judged scores:
  alignment-batch/rejudge-calibration/rejudged/ (row-aligned per CSV).
- Original Qwen-judged CSVs preserved alongside as *.qwen32b_backup.csv.
- The three summary CSVs (steer_sweep_summary, ckpt_eval_summary,
  step9_posthoc_steer_summary) were regenerated from the re-judged data.
- Any OTHER derived artifact (e.g. files under replication/analysis/, correlation
  matrices, plots) predates the re-judge and still reflects Qwen-judge numbers —
  recompute from the CSVs before relying on absolute values.
- Generations, checkpoints, vectors are untouched; only judge scores changed.

Graders: absolute trait scores are now directly comparable to the paper's reported
numbers (same judge, same aggregation).
