# LFM2.5 Implementation Roadmap & Engineering Architecture for Tuff

This document specifies the updated architectural plan, phased implementation milestones, validation gates, and optimization roadmap for executing **Liquid AI LFM2.5 models** (both Dense backbones and the **LFM2.5-24B-A2B** Sparse MoE) within the [`tuf`](file:///home/nii/workspace/tuf) runtime.

Target Hardware Profile:
* **GPU**: NVIDIA GeForce RTX 3070 Laptop (8 GB VRAM / ~7,880 MB usable, GA104, sm_86, ~330 GB/s sustained bandwidth).
* **CPU**: AMD Ryzen 9 5900HX (8 cores / 16 threads, Zen 3, AVX2 / FMA).
* **RAM**: 64 GB DDR4-3200 dual-channel (~45 GB/s host bandwidth).
* **Storage**: Dual NVMe PCIe SSDs.

Reference Documentation:
* [Architecture & Dispatch Manual](file:///home/nii/workspace/tuf/wiki/architecture.md)
* [Comparative Analysis of Runtimes](file:///home/nii/workspace/tuf/wiki/lfm25/other_runtimes.md)
* [LFM2.5-1.2B Specification](file:///home/nii/workspace/tuf/wiki/lfm25/1_2B.md)
* [LFM2.5-2.6B Specification](file:///home/nii/workspace/tuf/wiki/lfm25/2_6b.md)
* [LFM2.5-8B-A1B Specification](file:///home/nii/workspace/tuf/wiki/lfm25/8b-a1b.md)
* [LFM2.5-24B-A2B Specification](file:///home/nii/workspace/tuf/wiki/lfm25/24b-a2b.md)

---

## 1. Core Engineering Principles

1. **The Separation Principle & Inner-Loop Purity**:
   Do **not** introduce runtime conditionals (`if (model->arch == LFM) ... else ...`) inside the timed token loop or inner CUDA kernels. Qwen and LFM have completely different layer topologies (associative scans over DeltaNet recurrent states vs. depthwise 1D short-convolutions; 40 recurrent/GQA layers vs. 30 conv / 10 GQA). We separate models into dedicated compilation units:
   * Common infrastructure (GGUF parsing, memory arenas, sampler, server): shared.
   * Model layout, compute kernels, and execution loop: dedicated (`src/lfm25_*`, `include/lfm25_*`).
   * Performance over code reusability: If fusing two kernels or specializing a matrix-vector dot product yields higher bandwidth saturation, we write a dedicated kernel without forcing reuse from Qwen.

2. **Single Unified Binary with Init-Time VTable Dispatch**:
   `tuf` compiles into a **single binary (`build/tuff`)**.
   * When loading a model, `model_detect_arch()` reads the GGUF metadata string (`general.architecture`).
   * It binds an architecture-specific virtual dispatch table (`engine_vtable_t`), instantiating either `qwen_engine_vtable` or `lfm25_engine_vtable`.
   * During the decode loop, the engine invokes `vtable->forward_token(engine, token, pos)`.
   * **Zero Performance Tradeoff**: Exactly one indirect function call per token ($\sim 1 - 2$ ns overhead vs. a $2.5$ ms decode step, or $<0.0001\%$).

3. **The Tuff Correctness Contract**:
   * **Oracle Ground Truth**: Every stage must pass the three-stage oracle gate `tests/verify_lfm25_oracle.sh` against a reproducible `llama.cpp` reference (see *Gate Design* below).
   * **Zero Unchecked CUDA Calls**: All kernel launches, graph captures, and memory allocations are wrapped in `CUDA_CHECK` and `CUDA_CHECK_LAST()`.
   * **Strict Interleaved A/B Benchmarking**: All optimizations must be validated using `scripts/ab_interleaved.sh` (paired, order-balanced, 32 warmup + 128 timed, AC power locked via `scripts/check_power_state.sh`).

### Gate Design (why not 128/128 exact match)

A long greedy exact-match run is not a meaningful correctness test across runtimes. Measured on LFM2.5-1.2B Q4_K_M (prompt `Hello`):
* tuff and llama.cpp both use Q8_1-quantized activations, but carry different rounding noise: logit RMS vs an FP64 reference is **0.094 for tuff, 0.133 for llama.cpp**.
* The FP64 top-1/top-2 logit margin is **0.014 at step 19**, a coin flip under that noise. After the first such near-tie the streams fork and every later token is noise.
* llama.cpp disagrees with itself: GPU vs CPU diverge at token 5, and the original reference (recipe unrecorded) could not be regenerated even at the same commit.

The gate therefore has three stages, all required:
1. **Exact prefix**: first 16 greedy tokens identical to the reference.
2. **Teacher-forced top-1**: all 128 reference steps are fed to tuff; a disagreement is allowed only where the reference itself is a near-tie (llama.cpp margin < 0.5).
3. **FP64 tolerance**: over 32 teacher-forced steps, tuff's logit RMS error vs an FP64 numpy reference (`tools/lfm25_fp64_ref.py`) must be ≤ 1.10× llama.cpp's own, worst step ≤ 2.0×.

**MoE variant of stage 3 (robust statistic).** On the 8B-A1B the router's 4th/5th expert scores are nearly tied (FP64 gaps 4e-5..0.037, median 0.017), so quantization noise flips which expert runs and produces discrete, heavy-tailed per-step error spikes in *both* runtimes; which one gets a flip on a given step is luck, and flips cascade through the conv/attention state. Measured over 32 steps, the tuff/llama.cpp RMS ratio has mean 1.21 but median 0.96, p75 1.05, p90 1.11 and worst step 5.54 (step 7: tuff flipped 5 layers, llama.cpp 2; at step 8 llama.cpp was worse, 0.69 vs 0.30). Mean and worst-step are therefore ill-posed for MoE; `--stat robust` (used for `moe8`/`moe`) requires median ratio ≤ 1.10 and p90 ratio ≤ 1.5 (`MAX_P90_RATIO`), printing mean and worst step for information only. Dense 1.2B keeps mean/worst-step (mean 0.71, median 0.69, p90 0.66, worst 0.95).

**24B-A2B gate (`verify_lfm25_oracle.sh moe`).** Reference `tests/ref_lfm25_moe_128.*` was generated with llama.cpp `--cpu-experts` (13.5 GiB does not fit 8 GB), so the stage-3 llama.cpp logits run uses the same flag; stage 3 is `--stat robust`. The FP64 reference (`tools/lfm25_fp64_ref.py`, on-demand expert dequant, LRU slab cache) takes ~7.5 min for 32 steps (peak ~20 GB RSS) and is cached in `/tmp/tuff_oracle`. Sanity check of that reference against llama.cpp (independent of tuff): per-step RMS(llama - fp64) mean 0.47, median 0.44, max 0.94, no extreme spikes; FP64 argmax agrees with the reference token on 28/32 steps, the 4 misses all being near-ties (margins 0.32, 0.42, 0.055, 0.26 < 0.5).

**FP64 adjudication of stage-2 hard misses (24B).** With the CPU-expert reference being noisy (logit RMS vs FP64 ~0.44 median, ~3x the 8B's GPU reference), the reference's own top1-top2 margin is unreliable as a near-tie test. Over 72 FP64 steps the FP64 argmax agrees with tuff 69/72 but with the llama.cpp reference only 63/72; of tuff's 11 misses FP64 sides with tuff on 9. The lone hard miss (step 68, reference margin 0.954) is a near-tie in ground truth: FP64 logit(ref token) - logit(tuff token) = 0.328. `lfm25_gate.py teacher --fp64` therefore reclassifies a hard miss as a near-tie only if |FP64 gap| < `MIN_MARGIN` (0.5) and the step is covered by the FP64 file (`N_FP`: 72 for `moe`, 32 otherwise); uncovered steps stay hard misses. This is a real test, not a waiver: with `--min-margin 0.3` step 68 is not excused.

**Locked thresholds (2026-10-16).** Gate parameters live in `tests/lfm25_gate_thresholds.txt` (sha256 `4023c970b5df` (was `996f77a5ea01` before the dense26 keys were added in `855baaa`; no value changed), printed by every run). An environment override makes the run EXPLORATORY (exit 3, never a pass). A change needs its own commit with the rationale, written before looking at the result it would flip.

| Parameter | Value | Applies to |
| :--- | :---: | :--- |
| N (teacher-forced / golden steps) | 128 | all |
| N_EXACT (max exact prefix) | 16 | stage 1 (cut at first ref margin < MIN_MARGIN) |
| MIN_MARGIN | 0.5 | stages 1, 2 and FP64 adjudication |
| MAX_RATIO | 1.10 | stage 3: dense mean / MoE median |
| MAX_STEP_RATIO | 2.0 | stage 3 dense worst step |
| MAX_P90_RATIO | 1.5 | stage 3 MoE p90 |
| N_FP | 32 / 32 / 72 | dense / moe8 / moe |
| CTX | 1024 | all tuff gate runs |

**Golden self-regression gate (byte-identity).** The oracle gate cannot see bit-level regressions: a reversed warp-reduction order passes it (FP64 ratio 0.80) while changing every logit row. `tests/verify_lfm25_golden.sh` freezes tuff's validated output per model (128 greedy tokens + sha256 of each teacher-forced logit row, `tests/golden/*.json`). Every optimization/refactor must reproduce it byte-for-byte with the graph on, the graph off, and (8B) the forced expert cache. Goldens are regenerated only with `--update --reason`, after a full oracle-gate pass. Negative controls run automatically with `tests/lfm25_negative_controls.sh` (8 controls); `tests/verify_lfm25_all.sh --controls` runs everything.

The reference is produced by `tests/gen_lfm25_ref.sh` (`build/lfm25-llama-ref`: libllama, all layers on GPU, single-token decode, greedy). It is byte-reproducible; the command, llama.cpp commit and model SHA-256 are recorded in `tests/ref_lfm25_dense_128.recipe`. Negative control: swapping two conv taps fails all three stages (1/16 prefix, 55 hard misses, 17.9× FP64 error).

---

## 2. Incremental Stepping-Stone Progression

To ensure rock-solid correctness and eliminate debugging ambiguity, implementation proceeds incrementally across four stepping stones:

```
[Step 1: LFM2.5-1.2B] ──► [Step 2: LFM2.5-2.6B] ──► [Step 3: LFM2.5-8B-A1B] ──► [Step 4: LFM2.5-24B-A2B]
  (Dense Baseline)          (Full Dense Trunk)        (100% VRAM MoE)            (Hybrid LFRU Cache MoE)
   695 MiB / 100% VRAM       1.59 GiB / 100% VRAM       4.80 GiB / 100% VRAM       13.53 GiB / Split RAM+VRAM
   Target: 350-400 tok/s     Target: 175-190 tok/s      Target: 95-110 tok/s       Target: 95-110 tok/s
```

1. **Step 1: LFM2.5-1.2B-Thinking (Dense)**:
   * **Isolates**: ShortConv1D kernel ($K=3$), GQA QK-norm + RoPE, dense SwiGLU FFN, output norm resolution.
   * **Advantage**: Fast load (0.2s), tiny footprint (695 MiB), zero MoE complexity.
2. **Step 2: LFM2.5-2.6B (Dense)**:
   * **Isolates**: Scaling to 30 layers and doubled vocabulary (128,000 tokens).
3. **Step 3: LFM2.5-8B-A1B (MoE)**:
   * **Isolates**: Top-4 router kernel over 32 experts with `exp_probs_b` bias, expert gathering, SwiGLU, and weighted accumulation.
   * **Advantage**: At 4.8 GiB, **100% fits in 8 GB VRAM**. Zero PCIe paging or cache eviction noise.
4. **Step 4: LFM2.5-24B-A2B (Hybrid MoE)**:
   * **Isolates**: Enabling the 64-expert LFRU VRAM cache (~6,206 MiB holding ~47% of all experts), host mmap fallback for cache misses, and async PCIe streams.

---

## 3. Hardware Footprint & Memory Budgets (RTX 3070 8 GB)

Usable VRAM: **~7,880 MiB**

| Model | Weights Footprint | Resident in VRAM | LFRU Cache Size | Host Mmap RAM | VRAM Hit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LFM2.5-1.2B** | 694.8 MiB | **100%** | N/A (Dense) | 0 MiB | 100% |
| **LFM2.5-2.6B** | 1,589.0 MiB | **100%** | N/A (Dense) | 0 MiB | 100% |
| **LFM2.5-8B-A1B**| 4,908.9 MiB | **100%** | N/A (Full VRAM) | 0 MiB | 100% |
| **LFM2.5-24B-A2B**| 13,845.5 MiB | Trunk (623 MiB) | **6,206 MiB** (~47%) | ~7,015 MiB | **88% – 92%** |

---

## 4. Phased Implementation Roadmap

```
[Phase 0: Tooling & VTable Setup] ───► [Phase 1: Dense Baseline (1.2B)] ───► [Phase 2: In-VRAM MoE (8B)]
                                                                                       │
[Phase 5: Speculative Decoding]   ◄─── [Phase 4: CUDA Graphs & Fusion]  ◄─── [Phase 3: Hybrid MoE (24B)]
```

---

### Phase 0: Infrastructure & VTable Architecture
**Objective**: Establish tooling, oracle verification streams, and the unified architecture dispatch interface.

- [x] **Task 0.1: GGUF Metadata & Tensor Inspector**
  - Implement `tools/inspect_lfm25_gguf.py` extracting tensor layouts, block types, quant types, and memory simulations.
  - Verified exact tensor naming: `blk.N.shortconv.in_proj.weight`, `conv.weight`, `out_proj.weight`; `attn_q_norm.weight`, `attn_k_norm.weight`; `ffn_gate_inp.weight`, `exp_probs_b.bias`.
  - Identified critical Liquid AI naming quirk: final output RMSNorm is keyed as **`token_embd_norm.weight`**.
- [x] **Task 0.2: Oracle Ground Truth Harness**
  - Generated greedy ($T=0, N=128$) reference token sequences:
    * `tests/ref_lfm25_dense_128.txt` (from LFM2.5-1.2B), **regenerated reproducibly in Phase 1** via `tests/gen_lfm25_ref.sh` with `.margins.txt` and `.recipe` alongside
    * `tests/ref_lfm25_moe_128.txt` (from LFM2.5-24B-A2B), regenerated reproducibly in Task 3.2 (`tests/gen_lfm25_ref.sh moe`, llama.cpp experts on its CPU backend)
  - Created automated test harness `tests/verify_lfm25_oracle.sh` (now the three-stage gate, see *Gate Design*).
- [x] **Task 0.3: Architecture VTable & Source Tree Structure**
  - Create `include/tuff_vtable.h` defining `engine_vtable_t`.
  - Create dedicated LFM headers and translation units:
    * `include/lfm25.h`, `include/lfm25_model.h`, `include/lfm25_engine.h`
    * `src/lfm25_model.c`, `src/lfm25_gpu.cu`, `src/lfm25_engine.c`
  - Update `src/main.c` and `CMakeLists.txt` to compile unified binary `tuff` with auto-dispatch.

---

### Phase 1: Dense Baseline Implementation (LFM2.5-1.2B)
**Objective**: Implement an un-optimized but bit-exact decode pipeline for LFM2.5-1.2B.

- [x] **Task 1.1: 1D Causal Short-Convolution Decode Kernel**
  - Implement CUDA kernel `lfm_conv1d_decode`:
    * Maintain rolling state ring-buffer $S_l \in \mathbb{R}^{2 \times 2048}$ per layer.
    * In-projection splits $y_{in} \in \mathbb{R}^{6144} \to [b, c, v]$.
    * Compute $u = b \odot v$.
    * Depthwise 3-tap dot product: $\text{conv\_out}[i] = S[0, i] \cdot K[0, i] + S[1, i] \cdot K[1, i] + u[i] \cdot K[2, i]$.
    * Update state: $S[0, i] \leftarrow S[1, i], S[1, i] \leftarrow u[i]$.
    * Output gating: $g = c \odot \text{conv\_out}$.
  - Done: `lfm_shortconv_step_kernel` in `src/lfm25_gpu.cu` (fused $u = b \odot x$, 3-tap conv, state roll, gating, Q8_1 quantization for out_proj). Layer-0 output matches llama.cpp to 2e-4 relative; dense pipeline passes the oracle gate.
- [x] **Task 1.2: GQA Attention Kernel Integration**
  - Wire the 6 attention layers to `tuf`'s GQA attention kernel.
  - Implement per-head QK-RMSNorm (`attn_q_norm`, `attn_k_norm`, head dim 64) and RoPE ($\theta = 10^7$).
- [x] **Task 1.3: Dense SwiGLU FFN Kernel**
  - Implement fused SwiGLU: $\text{DownProj}(\text{SiLU}(\text{GateProj}(x)) \odot \text{UpProj}(x))$.
- [x] **Task 1.4: Dense Pipeline Assembly & Oracle Gate 1**
  - Assemble `lfm25_engine_forward_token()`.
  - Run `./tests/verify_lfm25_oracle.sh dense`.
  - **Validation Gate 1**: `verify_lfm25_oracle.sh dense` passes all three stages; `compute-sanitizer` clean (0 errors).
  - **Gate 1 passed**: 16/16 prefix, 124/128 teacher-forced (4 near-tie misses, 0 hard), FP64 error ratio 0.71; `compute-sanitizer --tool memcheck --leak-check full` 0 errors, 0 bytes leaked (found and fixed a double `cudaFree` of the tied LM head in `lfm25_gpu_shutdown`).
  - Sanitizer runs need `cmake -S . -B build-sanitize -DTUFF_PER_THREAD_STREAM=OFF`: the distro compute-sanitizer 2024.1 (CUDA 12.4) deadlocks on kernel launch + blocking `cudaMemcpy` under `--default-stream per-thread`. A newer CUDA toolkit should remove the need.
  - Oracle and eval runs pass `-c 1024`; tuff's default context (131072) allocates ~3.1 GiB of FP32 KV cache on the 1.2B model.

---

### Phase 2: In-VRAM MoE Baseline (LFM2.5-8B-A1B)
**Objective**: Implement top-4 routing and expert accumulation in 100% VRAM without caching noise.

- [x] **Task 2.1: Router Kernel (Top-4 over 32 Experts)**
  - Implement `lfm_router_top4`:
    * Gating is SIGMOID (`expert_gating_func = 2`), not softmax: $p = \sigma(x \cdot W_{\text{gate\_inp}}) \in \mathbb{R}^{32}$ on the FP32 FFN-norm output.
    * Selection score $s = p + b_{\text{exp\_probs}}$ (bias used for selection ONLY); top-4 by $s$, ties to the lower expert index.
    * Weights are the unbiased probs, normalized: $w_k = p_{id_k} / \max(\sum_k p_{id_k}, 6.1035\text{e-}5)$, no extra scale; output in descending-$s$ order.
    * Warp-reduction top-4 selection (single block, one warp per expert logit).
  - Done: `lfm_router_top4_kernel` / `lfm25_gpu_router_top4` in `src/lfm25_gpu.cu`. `build/lfm25-router-test` checks it against 176 llama.cpp cases (22 MoE layers × 8 decode steps of LFM2.5-8B-A1B, `tests/fixtures/lfm25_8b_router_cases.bin`, regenerated with `tools/lfm25_router_fixtures.c` + `tools/lfm25_pack_router_fixtures.py`): 176/176 identical top-4 sets and order, max |Δw| 1.2e-7, max |Δlogit| 1.9e-6. memcheck/racecheck/synccheck clean. Negative controls caught: biased weights (0/176), missing selection bias (57/176).
  - Known gap: the tie rule (lower index wins) is not exercised by real data and not yet confirmed against ggml's CUDA `argsort_top_k`; exact FP32 ties in `sigmoid + bias` are not expected in practice.
- [x] **Task 2.2: In-VRAM Expert SwiGLU & Weighted Accumulation**
  - Dispatch the 4 active experts per token directly from resident VRAM slots.
  - Per expert: $act_k = \text{silu}(\text{gate}_e \cdot x_q) \cdot (\text{up}_e \cdot x_q)$ (Q4_K gate/up, Q8_1 input), each $act_k$ quantized to Q8_1 on its own, then $y_k = \text{down}_e \cdot Q8_1(act_k)$. Down type is Q4_K in some layers and Q6_K in others (dispatched per layer on the host). Expert ids are read from device memory (no host sync).
  - Weights are applied **after** the down projection and summed in fixed order: $x_{\text{moe}} = ((w_0 y_0 + w_1 y_1) + w_2 y_2) + w_3 y_3$, then stored or added to the residual.
  - Done: `lfm25_gpu_moe_ffn` (+ `lfm25_gpu_quantize_q8_1`) in `src/lfm25_gpu.cu`. `lfm25-moe-ffn-test` against llama.cpp fixtures (`tests/fixtures/lfm25_8b_moe_ffn_cases.bin`, all 22 MoE layers × 2 steps, 12 Q4_K-down / 10 Q6_K-down layers; packed by `tools/lfm25_pack_moe_fixtures.py`): 44/44, max act error 1.6e-7, max output error 1.7e-7 (relative to max |ref|), residual-accumulate path verified. memcheck/racecheck/synccheck clean. Negative controls caught: wrong expert in gate/up, wrong activation slice in down, unweighted sum, Q6_K down dispatched as Q4_K (fails exactly the 20 Q6_K cases).
  - Note for Task 2.3: `lfm25_engine.c` hardcodes the dense FFN width as 8192; the 8B model's dense layers 0–1 use `feed_forward_length = 7168` (and rope base is 5e6, vocab 128000).
- [x] **Task 2.3: MoE Pipeline Assembly & Oracle Gate 2**
  - Assembled the LFM2.5-8B-A1B decode pipeline: router + expert FFN wired into `lfm25_engine_forward_token` (no host syncs in the layer loop); dense width from `feed_forward_length` (cross-checked vs `ffn_gate` dims; 7168 on the 8B); MoE scratch buffers; LFM vtable BOS/EOS from GGUF; tokenizer `add_bos` defaults to true for llama3-group pre-tokenizers (incl. `lfm2`) when the key is absent, as in llama.cpp (Qwen's `qwen2`/`qwen35` unaffected). Engine init rejects MoE shapes other than 32 experts / top-4 (24B needs Phase 3).
  - Reference: `tests/gen_lfm25_ref.sh moe8` → `tests/ref_lfm25_moe8_128.*` (byte-reproducible). `tools/lfm25_fp64_ref.py` handles `lfm2moe` (on-demand expert dequant, LRU; 32 steps ≈ 3 min).
  - **Gate 2 passed** (`./tests/verify_lfm25_oracle.sh moe8`): exact prefix 2/2 (cut at the reference near-tie at step 2), teacher-forced 127/128 (1 near-tie miss, 0 hard), FP64 robust stat median ratio 0.96, p90 1.11. memcheck on a full 8B decode: 0 errors, 0 leaks. Negative control (router ignoring the selection bias): stage 2 one hard miss, stage 3 median ratio 3.92 → FAIL.
  - Un-optimized baseline (128 timed tokens, mean of 3; power profile NOT locked to performance, so indicative): **8B 216.4 tok/s** (llama.cpp tg128 218.0), **1.2B 322.3 tok/s** (llama.cpp 330.8).

---

### Phase 3: Hybrid MoE with LFRU Cache (LFM2.5-24B-A2B)
**Objective**: Scale MoE routing to 64 experts with host mmap fallback and VRAM caching.

- [x] **Task 3.1: 64-Expert VRAM Expert Cache** (baseline: LRU, upload-on-miss)
  - 3.1a: router templated on 32/64 experts; expert-FFN kernels take per-tensor byte strides (contiguous GGUF tensor or slot pool). Tests: router 24B 304/304 vs llama.cpp; moe_ffn 24B 76/76 vs a numpy model of the CUDA path (`tools/lfm25_pack_moe_fixtures.py --simulate-gpu`, matches llama.cpp CUDA on the 8B to 4e-7 / 2e-6; 73/76 cases < 1e-6, 3 sporadic Q8_1 rounding flips up to 3e-5).
  - 3.1b: `src/lfm25_expert_cache.c` — three VRAM slot pools (gate/up/down, one slot index), capacity = free VRAM − `LFM25_VRAM_RESERVE_MB` (512) or `LFM25_EXPERT_CACHE_MB`; LRU; the current layer's experts can't evict each other; a miss uploads the 3 slabs on the compute stream. Per MoE layer: router → ids D2H (host sync) → lookup/fill → slot ids H2D → same validated kernels on the pools. On the 24B: 1087 slots / 6344 MiB, 1087 of 2432 slabs.
  - Correctness by construction: 8B forced through an 88-slot cache (`LFM25_FORCE_EXPERT_CACHE=1 LFM25_EXPERT_CACHE_MB=600`, 3240 misses) gives **byte-identical** logits to the resident path; 24B 51-slot vs 1087-slot also byte-identical. memcheck on a 24B decode with evictions: 0 errors, 0 leaks.
  - Deviation from the original plan: misses are uploaded over PCIe and computed on the GPU (identical numerics) instead of AVX2 CPU evaluation (different numerics, needs its own validation) — CPU-miss evaluation, frequency-aware LFRU, pinned staging (cudaHostRegister on the read-only GGUF mmap fails on this driver; pageable copies run ~4–6 GB/s; opt-in `LFM25_TRY_HOST_REGISTER=1`) and async prefetch move to Phase 4.
- [x] **Task 3.2: MoE Assembly & Oracle Gate 3**
  - Reference regenerated reproducibly (`tests/gen_lfm25_ref.sh moe`: llama.cpp with experts on its CPU backend, `--cpu-experts`); `./tests/verify_lfm25_oracle.sh moe` (FP64 window 72 steps, ~25 min to compute once, cached).
  - **Gate 3 passed**: exact prefix 1/1 (cut at reference near-tie, step 1), teacher-forced 114/128 with 13 reference near-ties + 1 FP64-adjudicated (step 68: reference margin 0.954 but FP64 prefers the reference token by only 0.328), 0 hard misses; FP64 robust stat median ratio 0.69, p90 0.55 (tuff is closer to FP64 than the CPU-expert reference: FP64 argmax agrees with tuff 69/72, with the reference 63/72).
  - Baseline speed (power profile not locked; short repetitive generation, cache hit rate 95.5%): **24B 110.1 tok/s** vs llama.cpp with CPU experts 34.8 tok/s (tg128). Expect lower hit rates on diverse text.

---

### Stepping stone 2: LFM2.5-2.6B (dense, 30 layers) — done; **verdict: NOT oracle-validated (stage 3 FAIL, recorded, not waived)**
- tuff runs it (177 tok/s, `-c 1024`). Reference `tests/ref_lfm25_dense26_128.*` (llama.cpp `620d2225`, byte-reproducible), `verify_lfm25_oracle.sh dense26` and `verify_lfm25_golden.sh dense26` modes, thresholds `N_FP_DENSE26=32`, `STAT_DENSE26=mean` locked **before** the first run.
- Result: stage 1 PASS (near-tie at step 0), stage 2 PASS (126/128, 2 near-ties), stage 3 **FAIL** on the worst-step limit only: mean ratio 0.95 ≤ 1.10, worst step 2.44 > 2.0 (step 1: tuff RMS 0.239 vs llama.cpp 0.098).
- Analysis (not acted on): the step-1 error is almost entirely a **uniform shift of all logits** (tuff −0.225 vs llama.cpp −0.019), which softmax/argmax ignore. Shift-free RMS at step 1 is tuff 0.082 vs llama.cpp 0.096, worst shift-free ratio over 32 steps 1.33, mean 0.93. No systematic bias: mean |shift| is 0.056 for both runtimes and tuff's signed mean is −0.005. On the 128k-vocab models the shift is 56–66 % of the squared error in both runtimes (1.2B: 4 % / 27 %).
- **Closed with the gate unchanged** (2026-10-16): the locked thresholds stand, and the 2.6B is recorded as supported and benchmarked (172 tok/s; spec up to 2.1×) but **not oracle-validated**. It has no golden, because a golden is only recorded after a full oracle pass, and it stays out of the `all` targets. `verify_lfm25_oracle.sh dense26` keeps reporting the FAIL.
- Proposal for the user, not applied: a shift-invariant stage-3 RMS for all modes, decided in its own commit. Because this result has already been seen, it is the user's call (lock rule). If approved, re-run all gates and record the 2.6B golden.

### Phase 4: Fused Kernels & CUDA Graph Capture
**Objective**: Eliminate global memory round-trips and drive launch latency to zero.

- [x] **Task 4.1: Whole-Trunk Static CUDA Graph (1.2B)**
  - Capture all 16 layers of LFM2.5-1.2B into a single static CUDA Graph.
  - Target: **$\ge 350 - 400$ tok/s** on RTX 3070 Laptop.
  - Status: implemented (dense 1.2B + resident 8B MoE; 24B expert-cache path stays un-graphed). GPU Q6_K embedding, device `d_tokpos`, one explicit stream, logits D2H only when capture is on; `LFM25_CUDA_GRAPH=0` disables. Graph logits byte-identical to un-graphed. 1.2B ~346 tok/s graph vs ~328 off. Orchestrator check: graph logits byte-identical to the pre-change engine (`8bfb4ed` worktree) on 1.2B and 8B; all three gates unchanged; memcheck clean (1.2B, 8B). Formal A/B (2026-10-16, `ab_interleaved.sh`, `-c 1024 -n 128 --warmup 32 --runs 3`, 6 rounds, `8bfb4ed` vs `e133b44`, power profile `balanced` accepted by the user — run with `FORCE=1`; AC on, GPU limit 95 W, VRAM 6001 MHz): 1.2B 321.8 → 343.4 tok/s, **+6.69% ± 0.17** (95% CI +6.34..+7.04%); 8B 214.2 → 229.2 tok/s, **+7.00% ± 0.08** (CI +6.85..+7.16%). The 350 tok/s target is reached together with Task 4.2 (361.7 tok/s). Runbook: [wiki/lfm25/devlog.md](wiki/lfm25/devlog.md).
- [x] **Task 4.2: Fused Depthwise Conv1D + Gating + `Q8_1` Requantization**
  - Single warp-cooperative kernel:
    * In-projection outputs $[b, c, x]$ kept in shared memory.
    * Depthwise 3-tap dot product with rolling state buffer.
    * Inline output gating ($c \odot \text{conv\_out}$).
    * Direct shared-memory quantization to `Q8_1` for out-projection GEMV.
  - Done (plus multi-block RMSNorm+Q8_1). `lfm_inproj_shortconv_kernel`: a block owns 32 channels, computes its 96 in_proj rows [b|c|x] with the exact per-row MMVQ code of `lfm_mmvq_q4_K_dense_kernel` (24 warps, 4 rows each) into shared memory, then warp 0 runs the unchanged conv/gate/Q8_1 math. `lfm_fused_rms_norm_quant_kernel` now runs on 8 blocks: each recomputes the identical sum of squares, then quantizes only its own Q8_1 blocks (it was one 256-thread block, latency-bound, ~10% of 1.2B decode).
  - **Byte-identical** teacher-forced logits vs `e133b44` on 1.2B, 8B, 24B with graph on and `LFM25_CUDA_GRAPH=0`, and on the 8B forced through an 88-slot expert cache. All three gates unchanged to the printed digit; router 176/176 + 304/304, moe_ffn 44/44 + 76/76; memcheck 0 errors / 0 leaks (1.2B, 8B, 24B), racecheck 0 hazards, synccheck 0 errors.
  - Negative controls (dense): b/c rows swapped → gate FAIL (prefix 0/10, 79 hard misses, ratio 19.8); norm blocks all writing the first share → FAIL (0/128 TF); x channel off by one → FAIL (101 hard misses); reversed warp-reduction order (bit-level only) → **byte-identity FAILS but the oracle gate PASSES** (ratio 0.80) — motivates the golden self-regression gate.
  - A/B (`ab_interleaved.sh`, 6 rounds, `e133b44` vs 4.2, balanced profile, `FORCE=1`): 1.2B 342.9 → 361.7 tok/s **+5.46% ± 0.21**; 8B 228.7 → 241.6 **+5.65% ± 0.08**; 24B 110.2 → 115.5 **+4.79% ± 0.10**.
- [x] **Task 4.2b: Long-K dense matvec efficiency (Q6_K / Q4_K `ffn_down`)**
  - `#pragma unroll 16` (Q6_K) / `unroll 4` (Q4_K) on the per-lane block loop of the dense MMVQ kernels: the next blocks' loads issue before the dependent float accumulation; per-lane accumulation order unchanged ⇒ byte-identical (golden PASS ×3).
  - nsys (1.2B, per call): Q6_K down 51.4 → 41.0 µs (268 → 336 GB/s); Q4_K down ~35 → 29.3 µs (270 → 322 GB/s).
  - `verify_lfm25_all.sh --controls` ALL PASS (8/8 controls); memcheck 0 errors. A/B vs `e0efcc1` (balanced, `FORCE=1`): 1.2B **+4.26% ± 0.06** (≈378 tok/s), 8B +0.17%, 24B +0.28% (their down projections run in the expert kernels).
- [x] **Task 4.3: Async Prefetch & Dual-Stream MoE Execution (24B)**
  - Overlap layer $L$ expert computation with layer $L+1$ PCIe miss transfers.
  - Target: **$\ge 95 - 110$ tok/s** on RTX 3070 Laptop.
  - **Hardware finding**: the GPU link trains at PCIe gen3 **x8** under load (nvidia-smi `Current 8x, Max 16x`); a standalone cudaMemcpy test gives 6.6 GB/s pinned *and* pageable. Miss traffic is bus-bound, so the gains come from fewer bytes and less host blocking, not faster copies.
  - Done (default on): misses on a dedicated non-blocking copy stream with a 512-entry event ring; the FFN stream only `cudaStreamWaitEvent`s on the slots it reads (the extra host sync before the FFN is gone). Host path: **pinned expert store** (anonymous mapping, 16-thread copy from the mmap, `cudaHostRegister`; 12.9 GiB in ~2 s with a warm page cache; auto when MemAvailable > experts + 8 GiB, `LFM25_PINNED_EXPERTS=0/1`), else a pinned staging ring, else pageable. **LFRU** eviction (lowest decayed use count among the older half by recency; `LFM25_CACHE_POLICY=lru|lfru`).
  - Next-layer router-predicted **prefetch** implemented and fenced but a **net loss** (default off, `LFM25_PREFETCH=1`): prediction accuracy 75–78 % at top-4, but a slab takes ~0.9 ms on the bus vs ~0.1 ms of compute before the next lookup, and wrong predictions cost bus time (translate prompt: 53 GB uploaded vs 29 GB).
  - Byte-identical: golden PASS ×3, incl. new configs `expert-cache-tiny` (8B, 300 MB ≈ 44 slots), `expert-cache-tiny-alt` (prefetch + staging ring + LRU) and 24B `prefetch`. `verify_lfm25_all.sh --controls` ALL PASS; memcheck 0 errors / 0 leaks (24B pinned store, 24B `LFM25_PINNED_EXPERTS=0`, 8B tiny cache ± prefetch). New negative controls (automated): FFN not waiting for uploads → golden moe8 FAIL; hits unprotected from their own lookup's misses → golden moe8 FAIL. (A mutation dropping the tick protection inside the LFRU candidate filter is a semantic no-op — current-tick slots are always the most recent — and is not caught, as expected.)
  - Bench (`scripts/lfm25_bench_moe.sh`, `-n 256 --runs 3`; tok/s, hit rate): Hello 112.9 → 115.6 (97.0 %); code 124.9 → 126.8 (96.2 %); history 96.5 → 101.2 (93.4 %); translate 51.9 → 57.3 (89.1 %). A/B (`ab_interleaved.sh`, `-n 128`, 6 rounds vs `93e2a9c`, balanced, `FORCE=1`): Hello 116.6 → 120.7 **+3.44% ± 0.24**; translate 39.2 → 44.1 **+12.49% ± 0.11**. Target met on low-miss text; high-miss text stays bus-bound (see the follow-up report).

---

### Phase 5: Speculative Decoding Adaptations
**Objective**: Lossless acceleration tailored to LFM's hybrid architecture.

- [x] **Task 5.1: Fast Rolling Ring-Buffer Rollback**
  - Conv rolling state is only $2 \times 2048$ floats (16 KiB/layer).
  - State rollback after draft token rejection is an $O(1)$ pointer/memcpy operation.
  - Done: `lfm25_engine_forward_batch(tokens, n ≤ 8, pos0)` (optional vtable entry) + `lfm25_engine_rollback(n_keep)`. Batched Q4_K/Q6_K dense MMVQ, fused SwiGLU and Q6_K head kernels read each weight block once for all columns, with per-(row, column) accumulation order identical to the single-token kernels. Per-token ops (norm, RoPE, KV write, causal attention ≤ pos0+j, MoE router/FFN) run per token. Conv: batched in_proj, then a strictly sequential per-channel recurrence writing per-token state snapshots `d_conv_snap[layer][B][2d]`. Rollback = one `cudaMemcpy2DAsync` of snapshot n_keep−1; KV is position-based and needs none.
  - **Byte-identical**: new golden configs `--tf-batch 4` (dense, moe8, moe), `--tf-batch 8 --tf-rollback` (dense), `--tf-batch 4 --tf-rollback` with an 8B 300 MB forced cache all reproduce the existing goldens (B = 2, 3, 5, 7 also `cmp`-identical on dense). `verify_lfm25_all.sh --controls` ALL PASS with 12/12 controls, including the new `batch-attn-sees-next-token` and `rollback-off-by-one`. memcheck 0/0 (1.2B, 8B), racecheck 0 hazards. Single-token path unchanged (A/B 1.2B 0.00 %, 8B +0.05 %, CI ∋ 0).
  - Verify cost `forward_batch(n)` / (n × graphed single): 1.2B n=2 0.64, n=4 **0.49** (4 tokens ≈ 1.9× one token), n=8 0.47; 8B 0.90 / 0.84 / 0.83; 24B 0.89 / 0.85 / 0.95. On the MoE models the experts run per token, so only the trunk is shared.
- [x] **Task 5.2: Speculative Suite & Acceptance Rate Benchmark**
  - Run `scripts/bench_ngram_suite.sh` on LFM2.5.
  - Target: **$\ge 110 - 125$ tok/s** decode speed.
  - Done: greedy prompt-lookup n-gram speculation for the LFM engines (`--spec-draft ngram`, `--spec-ngram`, `--spec-ngram-min`, `--spec-k` ≤ 7; opt-in; temp > 0 disables it). The verify step is one `forward_batch([t, d1..dk])`, accepting leading matches, emitting m+1 tokens and calling `rollback(m+1)`; steps without a draft use the graphed single token. Adaptive k maximizes the expected (1+p+…+p^k)/cost(k+1) against a single token, with per-match-length acceptance EMAs. The cost table is measured for dense and static (0.9/token) for MoE, live-rescaled, and a loss guard bounds the worst case.
  - **Lossless**: `tests/verify_lfm25_spec.sh` (6 prompts × 1.2B/8B/24B, adaptive and forced k=7, -n 256) shows token streams `cmp`-identical to non-speculative greedy in 18/18 cases. New golden configs (`spec-ngram`, `spec-ngram-k7`, 8B tiny-cache spec, 24B spec) PASS. `verify_lfm25_all.sh --controls --spec` ALL PASS with 13/13 controls (new: `spec-accept-one-too-many`). memcheck 0/0 on 1.2B and 8B with real drafts and rejections.
  - Bench (`scripts/lfm25_bench_spec.sh`, 9 prompts, `-n 256 --runs 3`; spec/baseline ratio): **1.2B 0.996–1.092** (py-csv 358 → 391, orchestrator re-run 360 → 396); **2.6B 1.04–2.12** (fibonacci 167 → 353, re-run 167 → 356; this model's greedy outputs repeat a lot); **8B 0.991–1.007** (verify costs ~0.85 of a token per position because experts run per token); **24B 0.979–1.144** (Hello 116 → 119, py-csv 127 → 145, history 101 → 112; miss-bound prompts such as fibonacci stay at 32 tok/s). Worst overall 0.979. The ≥ 110–125 tok/s target is met on cache-friendly prompts only; miss-bound 24B text is PCIe-bound (see the follow-up report).

---

## 5. Performance Milestones & Target Rooflines

Measured and projected decode throughput on **RTX 3070 Laptop (8 GB VRAM, 330 GB/s)**:

| Configuration | Active Params | VRAM Residency | Decode Speed (tok/s) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Qwen3.6-35B-A3B (Baseline)** | 3.0 B | 71.9% hit rate | **60.15** | Current production speed in `tuf` |
| **LFM2.5-1.2B (Phase 1: Baseline)** | 1.2 B | 100% VRAM | **~250 – 280** | Unfused kernels, individual launches |
| **LFM2.5-1.2B (Phase 4: CUDA Graph)** | 1.2 B | 100% VRAM | **~350 – 400** | Full static graph, fused conv+gate |
| **LFM2.5-2.6B (Dense 30 Layers)** | 2.6 B | 100% VRAM | **~175 – 190** | Static graph, 128k vocab |
| **LFM2.5-8B-A1B (Phase 2: In-VRAM MoE)**| 1.6 B | 100% VRAM | **~95 – 110** | 100% resident, zero PCIe misses |
| **LFM2.5-24B-A2B (Phase 3: Baseline)** | 2.45 B | ~88% hit rate | **~68 – 75** | Unfused kernels, no graphs |
| **LFM2.5-24B-A2B (Phase 4: CUDA Graphs)**| 2.45 B | ~91% hit rate | **~95 – 110** | Zero launch overhead + async DMA |
| **LFM2.5-24B-A2B (Phase 5: + N-Gram)** | 2.45 B | ~91% hit rate | **~110 – 125** | Lossless speculative decode |

---

## 6. Engineering Checklist & Sign-Off Criteria

Before merging each iteration into `main`:
1. [ ] **Clean Compilation**: Zero compiler warnings under `-Wall -Wextra -Werror` for C, C++, and CUDA (`sm_86`).
2. [ ] **Memory & Pointer Safety**: Full run under `compute-sanitizer --tool memcheck` with **0 errors**.
3. [ ] **Determinism**: 5 identical generations of $\ge 64$ tokens on fixed prompt and seed.
4. [ ] **Oracle Verification**: Three-stage oracle gate passes (`tests/verify_lfm25_oracle.sh`): exact prefix, margin-aware teacher-forced top-1, FP64 tolerance.
5. [ ] **Thermal / Power State Check**: `scripts/check_power_state.sh` verified AC connected and GPU clocks locked at 6001 MHz VRAM.
6. [ ] **Paired A/B Measurement**: Interleaved runs via `scripts/ab_interleaved.sh` confirming statistically significant speedup without regressions.
