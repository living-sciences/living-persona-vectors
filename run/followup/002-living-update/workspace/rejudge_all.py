"""Full uniform re-judge of every PV generation with the paper's exact judge
(gpt-4.1-mini-2025-04-14, logprob-expectation, via OpenRouter).

For each eval CSV, writes a sidecar under REJ/<relpath> with the original row order:
row_idx, gpt_<trait>, gpt_coherence. Resumable: complete sidecars are skipped.
Tracks real token usage/cost in usage.jsonl. Run detached; progress via progress.txt.
"""
import asyncio, glob, json, math, os, re, sys, time
import pandas as pd
from openai import AsyncOpenAI

CB = os.environ.get("REJ_CB", "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update/workspace/codebase")
BASE = os.environ.get("REJ_BASE", "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/followup/002-living-update/results")
REJ = os.environ.get("REJ_OUT", f"{BASE}/rejudged")
COST_CAP = float(os.environ.get("REJ_COST_CAP", "30"))
MODEL = "openai/gpt-4.1-mini-2025-04-14"
CONCURRENCY = 100
CSV_PARALLEL = 4
META = {"question", "prompt", "answer", "coherence"}

def read_key():
    for line in open(os.environ.get("OPENROUTER_ENV_FILE", os.path.expanduser("~/.openrouter.env"))):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("no key")

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=read_key())
sem = asyncio.Semaphore(CONCURRENCY)
usage = {"prompt": 0, "completion": 0, "calls": 0, "errors": 0, "lat_sum": 0.0}

def prompts():
    ns = {}
    exec(open(f"{CB}/eval/prompts.py").read(), ns)
    coh = ns["Prompts"]["coherence_0_100"]
    traits = {}
    for f in glob.glob(f"{CB}/data_generation/trait_data_extract/*.json"):
        t = os.path.basename(f)[:-5]
        traits[t] = json.load(open(f))["eval_prompt"]
    return traits, coh

TRAITS, COH = prompts()

def aggregate(top):
    total = 0.0; sum_ = 0.0
    for tok, lp in top:
        try:
            k = int(tok)
        except ValueError:
            continue
        if 0 <= k <= 100:
            p = math.exp(lp); sum_ += k * p; total += p
    return None if total < 0.25 else sum_ / total

async def judge_one(template, question, answer):
    msg = [dict(role="user", content=template.format(question=str(question), answer=str(answer)))]
    for attempt in range(5):
        try:
            async with sem:
                t0 = time.time()
                c = await client.chat.completions.create(
                    model=MODEL, messages=msg, max_tokens=1, temperature=0,
                    logprobs=True, top_logprobs=20, seed=0)
                usage["lat_sum"] += time.time() - t0
            u = getattr(c, "usage", None)
            if u:
                usage["prompt"] += u.prompt_tokens or 0
                usage["completion"] += u.completion_tokens or 0
            usage["calls"] += 1
            lc = c.choices[0].logprobs
            if not lc or not lc.content:
                return None
            return aggregate([(t.token, t.logprob) for t in lc.content[0].top_logprobs])
        except Exception as e:
            print(f"retry a{attempt} {type(e).__name__}: {str(e)[:140]}", file=sys.stderr, flush=True)
            if attempt == 4:
                usage["errors"] += 1
                return None
            await asyncio.sleep(min(2 ** attempt, 20))

def trait_of(df):
    cand = [c for c in df.columns if c not in META and c in TRAITS]
    return cand[0] if len(cand) == 1 else None

async def run_csv(path):
    rel = os.path.relpath(path, f"{CB}/eval_persona_eval")
    out = f"{REJ}/{rel}"
    df = pd.read_csv(path)
    t = trait_of(df)
    if t is None or "answer" not in df.columns:
        return f"SKIP {rel} (no single trait col)"
    if os.path.exists(out):
        try:
            if len(pd.read_csv(out)) == len(df):
                return f"done {rel} (cached)"
        except Exception:
            pass
    tp = TRAITS[t]
    ts = await asyncio.gather(*[judge_one(tp, r.question, r.answer) for r in df.itertuples()])
    cs = await asyncio.gather(*[judge_one(COH, r.question, r.answer) for r in df.itertuples()])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = pd.DataFrame({"row_idx": range(len(df)), f"gpt_{t}": ts, "gpt_coherence": cs})
    tmp = out + ".tmp"
    res.to_csv(tmp, index=False); os.replace(tmp, out)
    return f"DONE {rel} n={len(df)} gpt_{t}_mean={pd.Series(ts).mean():.2f}"

async def main():
    # Followup 001: REJ_GLOB lets us target only the new model's CSVs.
    pattern = os.environ.get("REJ_GLOB", f"{CB}/eval_persona_eval/**/*.csv")
    csvs = sorted(glob.glob(pattern, recursive=True))
    os.makedirs(REJ, exist_ok=True)
    os.makedirs(BASE, exist_ok=True)
    t0 = time.time()
    done = [0]

    def _cur_cost():
        return usage["prompt"] * 4e-7 + usage["completion"] * 1.6e-6

    async def one(i, p):
        msg = await run_csv(p)
        done[0] += 1
        cost = _cur_cost()
        avg = usage["lat_sum"] / max(usage["calls"], 1)
        line = (f"[{done[0]}/{len(csvs)}] {msg} | calls={usage['calls']} errors={usage['errors']} "
                f"avg_lat={avg:.1f}s cost=${cost:.2f} elapsed={int(time.time()-t0)}s")
        print(line, flush=True)
        with open(f"{BASE}/progress.txt", "a") as f:
            f.write(line + "\n")
        # hard cost cap: persist usage and abort if exceeded
        with open(f"{BASE}/usage.json", "w") as f:
            json.dump({**usage, "cost_usd": cost}, f, indent=1)
        if cost > COST_CAP:
            raise SystemExit(f"COST CAP EXCEEDED: ${cost:.2f} > ${COST_CAP}")

    csem = asyncio.Semaphore(CSV_PARALLEL)
    async def gated(i, p):
        async with csem:
            await one(i, p)
    await asyncio.gather(*[gated(i, p) for i, p in enumerate(csvs)])
    with open(f"{BASE}/usage.json", "w") as f:
        json.dump({**usage, "cost_usd": usage["prompt"] * 4e-7 + usage["completion"] * 1.6e-6}, f, indent=1)
    print("ALL DONE", usage, flush=True)

asyncio.run(main())
