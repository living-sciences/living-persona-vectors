#!/usr/bin/env python
"""Judge extraction pos/neg CSVs (trait + coherence) in a CLEAN process (no vLLM),
writing scores back INTO the CSV's <trait> and coherence columns so genvec's
get_persona_effective() can filter contrastive pairs.

Uses the SAME paper-exact instrument as rejudge_all.py: gpt-4.1-mini via OpenRouter,
max_tokens=1, temperature=0, seed=0, logprobs top_logprobs=20, numeric-mass expectation
(refusal if numeric mass < 0.25). Decoupling judging from the vLLM generation process
avoids the httpx/async breakage vLLM causes when both live in one process.

Usage: judge_extract.py <EXTRACT_DIR> [trait1 trait2 ...]
"""
import asyncio, glob, json, math, os, sys, time
import pandas as pd
from openai import AsyncOpenAI

CB = os.environ.get("REJ_CB", os.path.dirname(os.path.abspath(__file__)) + "/codebase")
MODEL = "openai/gpt-4.1-mini-2025-04-14"
CONCURRENCY = 100

def read_key():
    for line in open(os.environ.get("OPENROUTER_ENV_FILE", os.path.expanduser("~/.openrouter.env"))):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("no key")

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=read_key(), max_retries=8, timeout=90.0)
sem = asyncio.Semaphore(CONCURRENCY)

def load_prompts():
    ns = {}
    exec(open(f"{CB}/eval/prompts.py").read(), ns)
    coh = ns["Prompts"]["coherence_0_100"]
    traits = {}
    for f in glob.glob(f"{CB}/data_generation/trait_data_extract/*.json"):
        t = os.path.basename(f)[:-5]
        traits[t] = json.load(open(f))["eval_prompt"]
    return traits, coh

TRAITS, COH = load_prompts()

def aggregate(top):
    total = sum_ = 0.0
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
    async with sem:
        try:
            c = await client.chat.completions.create(
                model=MODEL, messages=msg, max_tokens=1, temperature=0,
                logprobs=True, top_logprobs=20, seed=0)
            lc = c.choices[0].logprobs
            if not lc or not lc.content:
                return None
            return aggregate([(t.token, t.logprob) for t in lc.content[0].top_logprobs])
        except Exception as e:
            print(f"[judge_extract] fail: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr, flush=True)
            return None

async def main():
    extract_dir = sys.argv[1]
    traits = sys.argv[2:] if len(sys.argv) > 2 else ["evil", "sycophantic", "hallucinating"]
    for trait in traits:
        tp = TRAITS[trait]
        for ptype in ["pos", "neg"]:
            path = f"{extract_dir}/{trait}_{ptype}_instruct.csv"
            if not os.path.exists(path):
                print(f"[skip] missing {path}", flush=True); continue
            df = pd.read_csv(path)
            if trait in df.columns and df[trait].notna().mean() > 0.9:
                print(f"[skip] {path} already judged ({trait}_ok={df[trait].notna().sum()})", flush=True); continue
            t0 = time.time()
            ts = await asyncio.gather(*[judge_one(tp, q, a) for q, a in zip(df["question"], df["answer"])])
            cs = await asyncio.gather(*[judge_one(COH, q, a) for q, a in zip(df["question"], df["answer"])])
            df[trait] = ts
            df["coherence"] = cs
            df.to_csv(path, index=False)
            n_ok = pd.Series(ts).notna().sum()
            print(f"judged {path}: n={len(df)} {trait}_ok={n_ok} mean={pd.Series(ts).mean():.2f} "
                  f"coh_mean={pd.Series(cs).mean():.2f} ({time.time()-t0:.0f}s)", flush=True)

asyncio.run(main())
