"""Phase 1 of Steps 6/10: run training.py for each config as a subprocess.

Each finetune is a separate process (unsloth patches torch, so it must not share
a process with the vLLM-based eval). Skips a config whose checkpoint already
exists. Logs per-run status.
"""
import os, sys, json, subprocess, time, glob

def ckpt_done(output_dir):
    # training saves a checkpoint-XXX dir (save_strategy epoch) + training_config.json
    if not os.path.isdir(output_dir):
        return False
    ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    has_adapter = any(os.path.exists(os.path.join(c, "adapter_config.json")) for c in ckpts) or \
                  os.path.exists(os.path.join(output_dir, "adapter_config.json"))
    return has_adapter

def main():
    configs = sys.argv[1:]
    if not configs:
        print("usage: run_finetune.py <config.json> [config2.json ...]")
        return
    for cfg_path in configs:
        cfg = json.load(open(cfg_path))
        out = cfg["output_dir"]
        if ckpt_done(out):
            print(f"[skip] {out} already has a checkpoint", flush=True)
            continue
        print(f"\n===== TRAIN {cfg_path} -> {out} =====", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "training.py", cfg_path])
        dt = time.time() - t0
        status = "OK" if r.returncode == 0 and ckpt_done(out) else f"FAIL(rc={r.returncode})"
        print(f"===== {status} {cfg_path} in {dt:.0f}s =====", flush=True)

if __name__ == "__main__":
    main()
