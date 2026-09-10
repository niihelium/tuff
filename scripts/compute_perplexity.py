#!/usr/bin/env python3
"""
Computes Perplexity (PPL) on a text dataset using tuff logits dump
and compares Exact vs Fast kernels against ground truth cross-entropy loss.
"""
import sys
import os
import subprocess
import tempfile
import numpy as np

VOCAB_SIZE = 248320

def compute_ppl_from_logits(logits_path, target_tokens):
    """
    Computes cross-entropy loss and Perplexity (PPL) from dumped FP32 logits.
    logits shape: (N, VOCAB_SIZE)
    target_tokens: list of int of length N
    """
    raw_bytes = os.path.getsize(logits_path)
    n_tokens = raw_bytes // (VOCAB_SIZE * 4)
    if n_tokens == 0:
        raise ValueError(f"Empty logits file {logits_path}")
    
    data = np.memmap(logits_path, dtype=np.float32, mode='r', shape=(n_tokens, VOCAB_SIZE))
    
    total_nll = 0.0
    count = min(n_tokens, len(target_tokens))
    
    for i in range(count):
        row = data[i].astype(np.float64)
        target_id = target_tokens[i]
        
        # Numerically stable LogSoftmax
        max_val = np.max(row)
        log_sum_exp = max_val + np.log(np.sum(np.exp(row - max_val)))
        log_prob = row[target_id] - log_sum_exp
        total_nll += -log_prob
        
    avg_nll = total_nll / count
    ppl = np.exp(avg_nll)
    return avg_nll, ppl, count

def main():
    if len(sys.argv) < 2:
        text_file = "prompts/benchmark.txt"
    else:
        text_file = sys.argv[1]
        
    n_tokens = int(os.environ.get("N_TOKENS", "64"))
    model_path = os.environ.get("MODEL_PATH", "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")
    tuff_bin = "./build/tuff"
    
    print("=" * 65)
    print(f"       TUFF PERPLEXITY (PPL) EVALUATION ENGINE")
    print("=" * 65)
    print(f"Text dataset : {text_file}")
    print(f"Token limit  : {n_tokens}")
    print(f"Model        : {os.path.basename(model_path)}")
    print("-" * 65)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        dump_exact = os.path.join(tmpdir, "logits_exact.bin")
        dump_fast  = os.path.join(tmpdir, "logits_fast.bin")
        
        # 1. Run exact pass
        print(f"[1/3] Running exact reference pass (TUFF_EXACT=1)...")
        env_exact = os.environ.copy()
        env_exact["TUFF_EXACT"] = "1"
        subprocess.run([
            tuff_bin, "--model", model_path, "--prompt", "Hello",
            "-n", str(n_tokens), "--dump-logits", dump_exact
        ], env=env_exact, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # 2. Run fast pass
        print(f"[2/3] Running optimized fast pass (TUFF_EXACT=0)...")
        env_fast = os.environ.copy()
        if "TUFF_EXACT" in env_fast:
            del env_fast["TUFF_EXACT"]
        subprocess.run([
            tuff_bin, "--model", model_path, "--prompt", "Hello",
            "-n", str(n_tokens), "--dump-logits", dump_fast
        ], env=env_fast, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Read greedy target tokens from exact run
        exact_data = np.memmap(dump_exact, dtype=np.float32, mode='r', shape=(n_tokens, VOCAB_SIZE))
        target_tokens = [int(np.argmax(exact_data[i])) for i in range(n_tokens)]
        
        # 3. Calculate PPL
        nll_exact, ppl_exact, c1 = compute_ppl_from_logits(dump_exact, target_tokens)
        nll_fast,  ppl_fast,  c2 = compute_ppl_from_logits(dump_fast,  target_tokens)
        
        delta_ppl = abs(ppl_fast - ppl_exact)
        rel_delta = (delta_ppl / ppl_exact) * 100.0
        
        print("-" * 65)
        print(f"Tokens Evaluated : {c1}")
        print(f"Exact Cross-Entropy Loss (NLL) : {nll_exact:.6f}  |  PPL : {ppl_exact:.6f}")
        print(f"Fast  Cross-Entropy Loss (NLL) : {nll_fast:.6f}  |  PPL : {ppl_fast:.6f}")
        print(f"Absolute Delta (ΔPPL)          : {delta_ppl:.6e}")
        print(f"Relative Delta                 : {rel_delta:.4f}%")
        print("=" * 65)
        if delta_ppl < 0.05:
            print("VERDICT: PASS (Zero noticeable degradation vs exact baseline)")
        else:
            print("VERDICT: WARNING (PPL drift detected)")
        print("=" * 65)

if __name__ == "__main__":
    main()
