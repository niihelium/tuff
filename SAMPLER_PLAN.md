# H60: High-Performance Dual-Mode Stochastic Sampler Pipeline

## 1. Executive Summary & Core Invariant

This document details the architectural design and implementation plan for the **Dual-Mode Stochastic Sampler Pipeline** in Tuff.

### Strict Performance Invariant
> **Zero-Overhead Greedy Invariant (`temperature <= 0.0`)**:
> Greedy decoding is the default mode for coding, mathematical reasoning, and automated benchmarks. When `temperature <= 0.0`, the inference engine **MUST** use the existing ultra-fast fused register-level argmax kernel (`gpu_fused_output_norm_quant_argmax_dev` / `mmvq_q6_K_dense_argmax_warprow_kernel`).
> In this mode:
> - **0** bytes of vocabulary logits are materialized to VRAM.
> - **0** extra CUDA kernel launches or memory transfers occur.
> - Throughput remains at the theoretical maximum roofline.

When stochastic sampling is explicitly enabled (`temperature > 0.0`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--seed`), the engine switches to a dedicated, decoupled sampling pipeline.

---

## 2. Dual-Mode Architecture

```
                                  [Layer 41: Final State d_norm_x]
                                                 |
                                                 v
                                  +------------------------------+
                                  | Is temperature <= 0.0f?      |
                                  +------------------------------+
                                         /                \
                         YES (Default Greedy)          NO (Stochastic Sampling)
                                       /                    \
                                      v                      v
        +-----------------------------------------+   +---------------------------------------+
        | Fused Q6_K Output Projection + Warp     |   | Fused Q6_K Output Projection          |
        | Register-Level Argmax Reduction         |   | -> Materialize Logits in g_d_logits   |
        +-----------------------------------------+   +---------------------------------------+
                               |                                         |
                               v                                         v
        +-----------------------------------------+   +---------------------------------------+
        | Single int32 Token Emitted              |   | Sampler Pipeline (GPU/Host):          |
        | (0 VRAM writes, 0 latency overhead)     |   | 1. Repetition / Frequency Penalty     |
        +-----------------------------------------+   | 2. Temperature Scaling (z_i = L_i/T)  |
                                                      | 3. Softmax Probabilities              |
                                                      | 4. Min-P / Top-K / Top-P Truncation   |
                                                      | 5. Fast RNG Sample (XorShift128+)     |
                                                      +---------------------------------------+
                                                                         |
                                                                         v
                                                      +---------------------------------------+
                                                      | Sampled int32 Token Emitted           |
                                                      +---------------------------------------+
```

---

## 3. Mathematical Specifications

### A. Repetition & Frequency Penalty
Applied to logits $L_i$ of tokens present in the recent context window of size $W$ (default $W=64$):
$$L'_i = \begin{cases} L_i / r & \text{if } L_i > 0 \\ L_i \times r & \text{if } L_i \le 0 \end{cases} \quad \text{for } r \ge 1.0$$

### B. Temperature Scaling
$$z_i = \frac{L'_i}{T}, \quad \text{for } T > 0$$

### C. Softmax & Normalization
$$P_i = \frac{\exp(z_i - z_{\max})}{\sum_j \exp(z_j - z_{\max})}$$

### D. Min-P Cutoff (Recommended by Qwen/Llama communities)
Let $p_{\max} = \max_i P_i$. Any candidate token $i$ with $P_i < p_{\max} \times p_{\min}$ is pruned ($P_i \leftarrow 0$).

### E. Top-K & Top-P (Nucleus) Truncation
- **Top-K**: Retain only the $K$ largest probabilities.
- **Top-P**: Sort remaining candidates descending, retain the smallest prefix with cumulative mass $\sum_{k=1}^M P_k \ge p$, and renormalize.

### F. Pseudo-Random Number Generation (PRNG)
- Uses a deterministic 64-bit `XorShift128+` generator with configurable `--seed`.
- Draw uniform float $u \sim \mathcal{U}(0, 1)$ and select index $t$ where $\sum_{k=1}^{t-1} P'_k \le u < \sum_{k=1}^t P'_k$.

---

## 4. Interfaces & CLI Parameters

### CLI Flags
```bash
./build/tuff -m <model.gguf> -p "Write a poem" \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 40 \
    --min-p 0.05 \
    --repeat-penalty 1.1 \
    --seed 42
```

### OpenAI Compatible API (`/v1/chat/completions`)
```json
{
  "model": "qwen3.6-35b",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "temperature": 0.7,
  "top_p": 0.8,
  "presence_penalty": 0.1,
  "seed": 42
}
```

---

## 5. Mandatory 4-Gate Quality Verification Protocol

All code entering `master` must clear:
1. **Gate 1 (Oracle Test)**: `scripts/oracle_check.sh` $\to$ 32/32 bit-identical tokens in greedy mode.
2. **Gate 2 (Coherence & Diversity Check)**:
   - Repeated runs at `temp = 0.0` produce 100% identical tokens.
   - Runs at `temp = 0.7` with `--seed 42` vs `--seed 43` produce fluent, distinct, and creative text.
3. **Gate 3 (CUPTI Hardware Profiler)**: `TUFF_TRACE=1` confirms 0 extra logit memory allocations, Regs $\le 64$, 0 spills during greedy mode.
4. **Gate 4 (Multi-Domain A/B Benchmark)**: 5 domains across 2 randomized passes confirm **0.0% speed degradation** in greedy decoding.
