"""Parallel finetune orchestrator for Steps 6 & 10.

Runs training.py as one subprocess per config, up to len(GPUS) concurrently,
each pinned to a distinct GPU via CUDA_VISIBLE_DEVICES. Foreground/blocking:
this driver returns only when every run has finished. Skips configs whose
checkpoint already exists, so it is resumable. Writes per-run logs and an
incrementally-updated status JSON so partial progress survives interruption.

Usage: python run_finetune_parallel.py <manifest_key|config.json ...>
  e.g. python run_finetune_parallel.py step6 step10
       python run_finetune_parallel.py configs/generated/qwen-evil_normal.json
"""
import os, sys, json, glob, time, subprocess

CODEBASE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/logs/finetune"
STATUS = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/persona-vectors/run/replication/logs/finetune_status.json"
GPUS = [0, 1, 2, 3]
STAGGER = 25  # seconds between initial launches to avoid simultaneous model-load disk thrash

os.makedirs(LOGDIR, exist_ok=True)


def ckpt_done(out):
    cks = glob.glob(os.path.join(out, "checkpoint-*"))
    return any(os.path.exists(os.path.join(c, "adapter_config.json")) for c in cks) or \
        os.path.exists(os.path.join(out, "adapter_config.json"))


def resolve_configs(args):
    manifest = json.load(open("configs/generated/manifest.json"))
    out = []
    for a in args:
        if a in manifest:
            out.extend(manifest[a])
        else:
            out.append(a)
    # de-dup preserving order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def main():
    cfgs = resolve_configs(sys.argv[1:])
    todo = []
    status = {}
    for p in cfgs:
        cfg = json.load(open(p))
        out = cfg["output_dir"]
        name = os.path.basename(out.rstrip("/"))
        if ckpt_done(out):
            status[name] = {"config": p, "state": "already_done"}
            print(f"[skip] {name} already trained", flush=True)
        else:
            todo.append((p, out, name))
            status[name] = {"config": p, "state": "pending"}
    json.dump(status, open(STATUS, "w"), indent=2)
    print(f"Total configs: {len(cfgs)}, to train: {len(todo)}", flush=True)

    running = {}   # gpu -> (proc, name, out, logf, t0)
    idx = 0
    last_launch = 0.0

    def launch(gpu, item):
        p, out, name = item
        logpath = os.path.join(LOGDIR, f"{name}.log")
        logf = open(logpath, "w")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen([sys.executable, "training.py", p],
                                stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=CODEBASE)
        status[name].update(state="running", gpu=gpu, log=logpath, t0=time.time())
        json.dump(status, open(STATUS, "w"), indent=2)
        print(f"[launch] GPU{gpu} <- {name} (pid {proc.pid})", flush=True)
        return (proc, name, out, logf, time.time())

    while idx < len(todo) or running:
        # launch on free GPUs (respect stagger)
        free = [g for g in GPUS if g not in running]
        now = time.time()
        if free and idx < len(todo) and (now - last_launch) >= STAGGER:
            g = free[0]
            running[g] = launch(g, todo[idx])
            idx += 1
            last_launch = time.time()
        # poll
        for g, (proc, name, out, logf, t0) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                logf.close()
                dt = time.time() - t0
                ok = rc == 0 and ckpt_done(out)
                status[name].update(state=("done" if ok else f"FAIL(rc={rc})"),
                                    duration_s=round(dt, 1))
                json.dump(status, open(STATUS, "w"), indent=2)
                print(f"[{'done' if ok else 'FAIL'}] {name} rc={rc} in {dt:.0f}s", flush=True)
                del running[g]
        time.sleep(10)

    done = sum(1 for v in status.values() if v["state"] in ("done", "already_done"))
    print(f"\n=== FINETUNE COMPLETE: {done}/{len(cfgs)} ok ===", flush=True)
    for name, v in status.items():
        if v["state"] not in ("done", "already_done"):
            print(f"  !! {name}: {v['state']}", flush=True)


if __name__ == "__main__":
    main()
