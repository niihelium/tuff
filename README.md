# Tuff

Single-stream, single-machine greedy decoder for **Qwen3.6-35B-A3B** (`Q4_K_S` GGUF, 19.91 GiB,
35.51 B params) on an **RTX 3070 Laptop 8 GB + Ryzen 9 5900HX + 64 GB DDR4**. Measured
**60.15 tok/s base** (5-prompt corpus, 32 warmup + 128 timed, 3 runs) against **llama.cpp
`885c5bb` 43.86 tok/s same-session → +38%**; from a 0.39 tok/s CPU-only floor (**154×**).
Speculative: free n-gram draft `--spec-draft=ngram --spec-ngram 3` **+2.4%** (61.61), paid MTP
head draft `--spec-mtp` **−12.4%** (52.67).

The model is 41 blocks: **40 trunk** (10 full-attention every 4th layer + 30 gated delta-net /
Qwen3-Next linear attention) and one **MTP draft head** (`blk.40`, excluded from the trunk pass);
256 routed experts, top-8 active; vocab 248,320; declared context 262,144. Distribution is split
by locality: **dense 1.92 GiB on GPU** — Q8_0 projections (attn_q/k/v, attn_output, delta
qkv/gate/ssm_out, shared experts), the **Q6_K output head** (398 MB), F32 norms/router/ssm
scalars; the Q8_0 `token_embd` (0.5 GiB) stays in RAM (one-row lookup). The **17.49 GiB of routed
experts** (88% of the model) are **Q4_K** gate/up/down (down is Q6_K on layers 34/38/39), mmap'd
in RAM and pulled on demand into a **per-expert LFRU VRAM cache**: 2500 slots × 1.945 MiB =
4863 MiB. KV is F32 and, because only 11 of 41 layers carry attention, just 44 KiB/token (176 MiB
at 4k).

Kernels are llama.cpp's mmvq numerics (ported, MIT — see `ATTRIBUTION.md`): `__dp4a`
`vec_dot_q8_0_q8_1` for dense Q8_0 (300–351 GB/s), `vec_dot_q6_K_q8_1` for the output head
(351 GB/s), and `vec_dot_q4_K_q8_1`/`q6_K_q8_1` with a fused gate+up+SwiGLU epilogue for routed
experts. The 30 delta layers run a fused autoregressive gated-delta-net kernel (L2-norm + gate +
recurrent update + per-head RMSNorm + SiLU) captured into **CUDA graphs** (~20 launches → 1 each);
shared expert + router are device-resident with host top-k on the post-attention norm. The f32
router (206 GB/s) is the one kernel below bus speed. Cache policy: **hit → GPU mmvq from the
resident slot; miss → CPU compute from the mmap** (AVX2 int8, split across 7 pinned spin-wait
workers) with asynchronous background fill (budget 1). Hit rate 71.9%, flat 2500→2600 slots —
the routing is a large rotating working set (top 25% cover 68% of accesses), so the cache is
locality-bound, not capacity-bound. Speculative decode is lossless (two-column verify + recurrent
state snapshot/rollback); the marginal-column break-even is α > 0.51. The three dominant kernels
run ~91% of the 384 GB/s bus, so the engine is at the practical batch-1 GEMV ceiling (~103 tok/s
roofline), not bandwidth-starved.

## Methodology

Ground rules (all learned by being burned; each is a hard stop, not a preference):

1. **Never report a number the code did not measure.** No fallback constants, no synthetic
   figures. A missing metric is a hard error.
2. **Every CUDA call goes through `CUDA_CHECK` / `CUDA_CHECK_LAST()`** — a silent sticky-error
   context is how the original bogus ~38 tok/s arose.
3. **Correctness gates performance.** No optimization lands before the oracle re-passes.
4. **Read hyperparameters and shapes from the GGUF.** No hardcoded 2048/512/41.
5. **One variable per experiment.**
6. **The gate and the benchmark must run the same build in the same configuration.** Same flags,
   same env — a gate that does not exercise the benchmarked path is not a gate.
7. **Quote the verdict, not the favourable line.** If a gate's criterion is wrong, change the
   criterion — do not report around it.

Correctness contract: token-identical greedy continuation against llama.cpp (the oracle),
tracking the identical-prefix length **N** against a 128-token reference; determinism 5/5 at ≥64
tokens (assert non-empty); `compute-sanitizer` clean. `TUFF_EXACT=1` / `TUFF_CPU_DELTA=1`
are bisection tools, not correctness references.

Measurement rules:

- Real decode through the real tokenizer, greedy sampling; ≥3 runs, **32 warmup + 128 timed**;
  report mean ± spread. Same prompt/context as the llama.cpp baseline.
- `scripts/check_power_state.sh` must exit 0 (AC, uncapped power, VRAM 6001 MHz under load).
  A tok/s number is conditional on machine state: on battery this binary measures 8.09 where it
  measures 23.03 on AC, and the profile *shape* inverts.
- Compare **paired and order-balanced** (`scripts/ab_interleaved.sh`), never against a recorded
  number — decode drifts ~0.6–0.8%/min on a throttling laptop, so which tree runs second loads
  the result. Treat any single-change claim above ~5% as provisional until it survives an
  order-balanced re-measurement.
- A measurement that cannot distinguish success from absence is not a measurement (empty input
  hashes identically; a gate that ran a different config than the benchmark is vacuous).

## Open work (see `PLAN.md`)

The two remaining levers are **KV quantization for long context** and a **purpose-trained draft
model** — neither is an engine change to the single-token path, which is at the bus ceiling.

## Build & run

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j8
./scripts/check_power_state.sh && ./build/tuff --model <model.gguf> --prompt "Hello" -n 128 --warmup 32 --runs 3
```

Env knobs (defaults): `TUFF_FILL_BUDGET=1`, `TUFF_MOE_WORKERS=7`, `TUFF_EXPERT_SLOTS=2500`,
`TUFF_EXACT=0`. Spec: `--spec-draft=ngram --spec-ngram 3` (the only profitable mode).
