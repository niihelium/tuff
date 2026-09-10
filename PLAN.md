# Tuff — Plan

Forward-looking brief. The **current state, kernels, model distribution, quantizations, cache
logic and methodology live in `README.md`**; this file is the ledger of *what to test next* and
the record of hypotheses still open. Measured history lives in `ITERATION_LOG.md` (one entry per
landed iteration).

## Where we are

**60.83 tok/s base / +38% over llama.cpp 43.86** same-session; spec `--spec-draft=ngram
--spec-ngram 3` **+2.2%**, `--spec-mtp` **−12.7%**. The three dominant kernels run ~91% of the
384 GB/s bus; the single-token path is at the practical batch-1 GEMV ceiling (~103 tok/s
roofline). The GPU idle is ~10 ms of a ~17 ms step, split between the CPU-miss wait (~4 ms) and
the host tail (~3.7 ms), both already attacked (H10, H12) and now near their floor.

## Open levers, ranked by measured expected value

| # | lever | size | status |
|---|---|---|---|
| 1 | **KV quantization** (F32 → q8_0/q4_0/q3_K) for long context | long-context, not throughput | not implemented; hard-coded F32 in `gpu_attention_layer_dev`. See the context×quant crossover below |
| 2 | **Purpose-trained draft model** (DFlash, α 0.52→0.85) | ~+34% to speculation | blocked: CPU is saturated by the miss pool (no spare capacity for a CPU draft); needs a separate model file |
| 3 | f32 router → Q8_0 (206 GB/s → 300+) | ~0.15 ms/token | needs re-quantizing `ffn_gate_inp` = model change |
| 4 | slot over-allocation (~645 MiB, Q6_K-sized down for Q4_K layers) | ~330 slots, but H18a shows slots don't move hit rate | locality-bound, not capacity-bound |

Deliberately not doing: tensor cores / MMA (batch-1 GEMV, tiles need 8-16 rows), L2 persistence
(4 MiB against a 2.4 GB/token stream), `--use_fast_math` (small fp32 diffs move N more than
intuition predicts), `--n-gpu-expert-layers` (neutral-to-negative), a full megakernel (H0: model
> VRAM, launch overhead is not the cost).

## Context length × KV quantization (the only real VRAM lever)

Fixed overhead 2459 MiB (dense + delta state + CUDA ctx/scratch), pool budget **5287 MiB** at
1.945 MiB/slot and a 96 MiB OOM margin. KV = 11,264 floats/token (11 layers × 2 KV heads × 256
dim × 2 caches). Full-hit-rate (slots ≥2500) context ceiling per KV type: **F32 ~9.6k, q8_0
~37k, q4 ~70k, q3_K ~91k, q2_K ~123k**. Viable context picks: **64k/q8_0** (2333 slots),
**128k/q4_0** (2310 slots), **256k/q3_K** (2095 slots, hit rate ~65–70% vs 71.9%). One slot =
45/170/322/421/567 tokens for F32/q8/q4/q3/q2, so context steps should be in slot-multiples.

## Hypothesis ledger (pre-register here before testing)

Each entry: mechanism, prediction, the gate it must clear, and the kill criterion. Failures are
recorded in `ITERATION_LOG.md` with the same rigor as successes.

| id | hypothesis | prediction | gate | kill |
|---|---|---|---|---|
| H19 | KV q8_0 at 64k: quantize the K/V caches, keep 2500 slots | hit rate stays 71.9%, context 64k works | oracle + hit-rate + N unchanged | <2% hit-rate gain, or quality drop on a long-context eval |
| H20 | DFlash draft on CPU | α→0.85 makes spec pay | lossless token identity | CPU saturation (miss pool) absorbs the draft |
| H21 | attention q/k/v share one norm_x quantize (implemented, bit-identical) | −10 quantize launches/token ≈ +0.2% | oracle + determinism (hard bit-identity) | ✅ landed (oracle 32/32, determinism hash unchanged) |
| H22 | router F32→Q8_0 behind `TUFF_ROUTER_Q8=1` (implemented) | router 0.47→~0.15 ms/token ≈ **+1.5–2%** | N must not drop below 56; oracle may shift (numerics change) | ❌ **refuted: −5.3% ± 1.0%** order-balanced (4 rounds). The Q8_0 mmvq path pays a `quantize_q8_1_kernel` launch per layer that the f32 kernel does not, and the router GEMV is small enough that launch overhead swamps the 206→316 GB/s bandwidth win. Flag kept as a diagnostic |
| H23 | parallelize mRoPE (implemented, bit-identical) | 10.6→~2 µs × 10 layers ≈ +0.4% | oracle + determinism (hard bit-identity) | ✅ landed (oracle 32/32, determinism hash unchanged) |
| H24 | CUDA graphs over the 10 attention layers | −~200 launches/token ≈ +0.5–1% (converts ~40%, H12) | oracle + bit-identity | needs a device-resident position counter; <0.3% order-balanced |
| H6 | event-wait on the router D2H instead of full stream sync (`TUFF_H6=1`, implemented) | shared expert (~3 Q8_0 GEMVs/layer) overlaps host top-k + CPU-miss compute → +2–4% | oracle + determinism (bit-identical) | ❌ **refuted: +0.42% ± 0.78%** over 8 order-balanced rounds; interval includes 1.0. The shared-expert window is too small to hide once top-k/miss-submit is already overlapped. Flag kept off |
| H25 | **GPU-Native Top-K & On-Device Dispatch** (`TUFF_GPU_ROUTER=1`, implemented) | Eliminate 40 synchronous D2H copies of router logits per token. Compute Top-8 + Softmax in warp-level reduction on GPU. | oracle + determinism (hard bit-identity) | ❌ **refuted: −7.80% ± 2.79%** over 4 order-balanced rounds. Extra CUDA kernel launches (1 block, 32 threads) and extra async D2H calls add ~8–10 µs per layer of driver latency, exceeding the CPU sorting savings. Flag kept default OFF. |
| H26 | **AVX2-Vectorized CPU-MoE Miss Reduction** (implemented, bit-identical) | Vectorize CPU miss accumulation with `_mm256_fmadd_ps` across $n_{miss} \times 2048$ rows. | oracle + determinism (hard bit-identity) | ✅ **landed unconditionally** (oracle 32/32 passes, peaks up to 65.5 tok/s on high-locality code). |
| H27 | **VRAM Expert Cache Pool Expansion to 2600–2650 slots** (`TUFF_EXPERT_SLOTS=2600`, tested) | Utilize available VRAM (5057 MB pool) without re-quantization. | memory safety (zero OOM under load) | ❌ **refuted: −6.36% ± 2.00%** over 4 order-balanced rounds. Reduces free VRAM margin from 344 MiB to <30 MiB, triggering CUDA driver allocation stalls. Default **2500 slots** is optimal. |
| H28 | **External Small-Model Drafter** (`--spec-draft-model`) | Full Qwen 0.8B draft model running in background pthread. | oracle bit-identity | ❌ **refuted on CPU: 19.99–24.8 tok/s**. CPU decode latency (15.3–25.8 ms) and OpenMP thread contention with parallel MoE miss reductions saturate Zen 3. |
| H29 | **Hierarchical Prompt-Seeded N-Gram Speculation** (`--spec-ngram-max`, `--spec-ngram-min`) | Multi-level fallback ($N_{max} \to \dots \to N_{min}$) with prompt prefix seeding. Zero compute/memory overhead on CPU/DDR4. Predicts +5–12% throughput gain on structured text and code. | oracle lossless verified tokens | <0.5% net throughput gain over baseline |
| H52 | **GDN Recurrence Register Bounds & Spill Elimination** (`branch: hyp/h52-gdn-register-bounds`) | CUPTI hardware trace revealed `fused_gated_delta_net_ar_kernel` hit 255 registers/thread limit, causing DRAM local memory spilling. Adding `__launch_bounds__(128, 4)` and factorizing `dcy` scalar loop reduces register footprint to $\le 48$ regs, restoring SM occupancy from 25% to 75%+. | CUPTI trace Regs $\le 48$ + bit-identity | $<0.5\%$ or latency regression |
| H53 | **Batched MoE Expert Dispatch & In-Place SwiGLU Quantization** (`branch: hyp/h53-batched-moe-dispatch`) | Eliminate 76 sequential CUDA kernel dispatches per token across 40 MoE layers by fusing active expert hits into a 2D grid (`dim3(ffn_dim/32, n_hits)`) and performing Q8_1 quantization directly in shared memory. | 30x reduction in MoE kernel dispatches | Regression in kernel execution time |
| H54 | **Unified Delta-Net Quad Projections** (`branch: hyp/h54-unified-delta-projections`) | Merge QKV (4096), Z (4096), $\beta$ (32), and $\alpha$ (32) projections into a single unified MMVQ grid launch over quantized residual stream across all 30 Delta-Net layers. | Single-pass bandwidth saturation | Register spilling or latency increase |
| H55 | **LM-Head MMVQ 256-Thread Occupancy Tuning** (`branch: hyp/h55-lmhead-occupancy-tuning`) | Tune `mmvq_q6_K_dense_argmax_warprow_kernel` from 128 to 256 threads per block (8 rows/block), halving grid blocks from 37,984 to 18,992 to match RTX 3070 40 SMs limits. | Head latency $< 1.0$ ms (currently 1.13 ms) | Regression on 95% CI |
| H56 | **Ampere L2 Cache Persistence Pinning** (`branch: hyp/h56-l2-persistence`) | Pin hot inter-layer buffers (`g_d_res_x`, `g_d_norm_x`, `g_d_dense_x_q8`, `g_exslots`) into a dedicated 1 MB L2 cache window using `cudaStreamSetAttribute` (`cudaStreamAttrAccessPolicyWindow`), preventing 2.4 GB weight streams from evicting recurrent/activation buffers. | Zero DRAM roundtrips for inter-layer state | Latency regression or cache thrashing |
| H57 | **Unified 40-Layer Dynamic CUDA Graph Capture** (`branch: hyp/h57-dynamic-cuda-graphs`) | Capture the full 40-layer forward pass (Delta-Net, Attention, MoE routing) into persistent CUDA Graphs with device-resident position pointer `d_pos` and `cudaGraphExecKernelNodeSetParams`, eliminating ~1.5 ms of CPU driver launch overhead. | Eliminates ~200 launch events/token | Graph capture failure or memory corruption |
| H58 | **Ampere `cp.async` Multi-Stage Attention KV Pipeline** (`branch: hyp/h58-async-kv-pipeline`) | Use hardware asynchronous global-to-shared copy (`cp.async` / `cuda::memcpy_async`) with circular double-buffering in `attention_kernel` to prefetch KV cache tiles directly into shared memory, hiding DRAM latency. | KV fetch latency hidden behind compute | ✅ **landed unconditionally** (+2.78% Geomean, -39.1% kernel latency, 0 spills) |
| H60 | **Dual-Mode Stochastic Sampler Pipeline** (`docs: SAMPLER_PLAN.md`) | Implement isolated stochastic sampler (`--temp`, `--top-p`, `--top-k`, `--min-p`, `--repeat-penalty`, `--seed`) with zero-overhead greedy fast-path invariant (0 VRAM writes, 0 latency penalty for `temp <= 0`). | Bit-identical greedy oracle + fluent/diverse sampled completions | Any regression in greedy throughput (>0.0%) |

Marginal-gain discipline (iteration 50): these are all <2% levers. Measure with
`scripts/marginal_ab.sh <FLAG=1> 4` (order-balanced, alternating). Treat any |win| <2% as
provisional until it survives ≥4 rounds; H21/H23 are bit-identical by construction and are
verified by the existing oracle + determinism gates rather than by an A/B.

## Build & run

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j8
./scripts/check_power_state.sh && ./build/tuff --model <model.gguf> --prompt "Hello" -n 128 --warmup 32 --runs 3
# marginal A/B (H21/H22/H23):
./scripts/marginal_ab.sh TUFF_ROUTER_Q8=1 4
```

New env knobs (all default OFF): `TUFF_ROUTER_Q8=1` (H22). H21/H23 are unconditional
bit-identical changes (no flag).
