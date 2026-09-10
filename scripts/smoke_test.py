#!/usr/bin/env python3
import subprocess
import time
import os
import sys
import re

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "tuff")
MODEL_PATH = "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"

# PLAN.md protocol (short form for a smoke test): 2 warmup + 8 timed tokens, 3 runs.
N_PREDICT = int(os.environ.get("TUFF_N", "8"))
N_WARMUP = int(os.environ.get("TUFF_WARMUP", "2"))
N_RUNS = int(os.environ.get("TUFF_RUNS", "3"))


def run_smoke_test():
    cmd = [
        ENGINE_PATH,
        "--model", MODEL_PATH,
        "--prompt", "Write a PyTorch PINN solver for the Casimir effect.",
        "-n", str(N_PREDICT),
        "--warmup", str(N_WARMUP),
        "--runs", str(N_RUNS),
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env["OMP_PROC_BIND"] = "TRUE"
    env["OMP_PLACES"] = "threads"

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    t1 = time.time()

    if res.returncode != 0:
        print("[ENGINE ERROR] Output stderr:")
        print(res.stderr)
        raise RuntimeError(f"Engine process exited with non-zero code {res.returncode}")

    out = res.stdout + "\n" + res.stderr
    tok_speed = None
    spread = None

    m = re.search(r"Pipeline Generation Speed:\s*([\d.]+)\s*\+\/-\s*([\d.]+)\s*tok/s", out)
    if m:
        tok_speed = float(m.group(1))
        spread = float(m.group(2))

    # PLAN.md rule 1: a missing metric is a hard error, never a plausible fallback.
    if tok_speed is None:
        print("[PARSE ERROR] Raw stdout:")
        print(res.stdout)
        print("[PARSE ERROR] Raw stderr:")
        print(res.stderr)
        raise RuntimeError("Failed to parse token generation speed from engine output! No fake fallback allowed.")

    print(f"[Honest Smoke Test] Speed: {tok_speed:.2f} ± {spread:.2f} tok/s | "
          f"({N_WARMUP} warmup + {N_PREDICT} timed x {N_RUNS}) | Wall: {t1-t0:.2f}s")
    return tok_speed, spread


if __name__ == "__main__":
    run_smoke_test()
