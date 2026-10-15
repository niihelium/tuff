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
   * **Oracle Ground Truth**: Every stage must produce greedy token-identical continuations against `llama.cpp` up to $N = 128$ tokens (`tests/verify_lfm25_oracle.sh`).
   * **Zero Unchecked CUDA Calls**: All kernel launches, graph captures, and memory allocations are wrapped in `CUDA_CHECK` and `CUDA_CHECK_LAST()`.
   * **Strict Interleaved A/B Benchmarking**: All optimizations must be validated using `scripts/ab_interleaved.sh` (paired, order-balanced, 32 warmup + 128 timed, AC power locked via `scripts/check_power_state.sh`).

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
    * `tests/ref_lfm25_dense_128.txt` (from LFM2.5-1.2B)
    * `tests/ref_lfm25_moe_128.txt` (from LFM2.5-24B-A2B)
  - Created automated test harness `tests/verify_lfm25_oracle.sh` enforcing exact 128/128 token parity.
- [x] **Task 0.3: Architecture VTable & Source Tree Structure**
  - Create `include/tuff_vtable.h` defining `engine_vtable_t`.
  - Create dedicated LFM headers and translation units:
    * `include/lfm25.h`, `include/lfm25_model.h`, `include/lfm25_engine.h`
    * `src/lfm25_model.c`, `src/lfm25_gpu.cu`, `src/lfm25_engine.c`
  - Update `src/main.c` and `CMakeLists.txt` to compile unified binary `tuff` with auto-dispatch.

---

### Phase 1: Dense Baseline Implementation (LFM2.5-1.2B)
**Objective**: Implement an un-optimized but bit-exact decode pipeline for LFM2.5-1.2B.

- [ ] **Task 1.1: 1D Causal Short-Convolution Decode Kernel**
  - Implement CUDA kernel `lfm_conv1d_decode`:
    * Maintain rolling state ring-buffer $S_l \in \mathbb{R}^{2 \times 2048}$ per layer.
    * In-projection splits $y_{in} \in \mathbb{R}^{6144} \to [b, c, v]$.
    * Compute $u = b \odot v$.
    * Depthwise 3-tap dot product: $\text{conv\_out}[i] = S[0, i] \cdot K[0, i] + S[1, i] \cdot K[1, i] + u[i] \cdot K[2, i]$.
    * Update state: $S[0, i] \leftarrow S[1, i], S[1, i] \leftarrow u[i]$.
    * Output gating: $g = c \odot \text{conv\_out}$.
- [ ] **Task 1.2: GQA Attention Kernel Integration**
  - Wire the 6 attention layers to `tuf`'s GQA attention kernel.
  - Implement per-head QK-RMSNorm (`attn_q_norm`, `attn_k_norm`, head dim 64) and RoPE ($\theta = 10^7$).
- [ ] **Task 1.3: Dense SwiGLU FFN Kernel**
  - Implement fused SwiGLU: $\text{DownProj}(\text{SiLU}(\text{GateProj}(x)) \odot \text{UpProj}(x))$.
- [ ] **Task 1.4: Dense Pipeline Assembly & Oracle Gate 1**
  - Assemble `lfm25_engine_forward_token()`.
  - Run `./tests/verify_lfm25_oracle.sh dense`.
  - **Validation Gate 1**: Achieve **128/128 exact token match**; `compute-sanitizer` clean (0 errors).

---

### Phase 2: In-VRAM MoE Baseline (LFM2.5-8B-A1B)
**Objective**: Implement top-4 routing and expert accumulation in 100% VRAM without caching noise.

- [ ] **Task 2.1: Router Kernel (Top-4 over 32 Experts)**
  - Implement `lfm_router_top4`:
    * Linear projection + bias: $z = x \cdot W_{\text{gate\_inp}} + b_{\text{exp\_probs}} \in \mathbb{R}^{32}$.
    * Warp-reduction top-4 selection and normalized softmax weights.
- [ ] **Task 2.2: In-VRAM Expert SwiGLU & Weighted Accumulation**
  - Dispatch the 4 active experts per token directly from resident VRAM slots.
  - Accumulate weighted outputs: $x_{\text{moe}} = \sum_{k=1}^4 w_k \cdot y_k$.
- [ ] **Task 2.3: MoE Pipeline Assembly & Oracle Gate 2**
  - Assemble LFM2.5-8B-A1B decode pipeline.
  - **Validation Gate 2**: Greedy generation exact match on 8B; benchmark un-optimized baseline speed.

---

### Phase 3: Hybrid MoE with LFRU Cache (LFM2.5-24B-A2B)
**Objective**: Scale MoE routing to 64 experts with host mmap fallback and VRAM caching.

- [ ] **Task 3.1: 64-Expert LFRU VRAM Cache**
  - Allocate ~6,200 MiB VRAM for expert cache slots (holding ~47% of all expert slices).
  - Cache hits: GPU mmvq dispatch.
  - Cache misses: AVX2 CPU worker thread pool evaluation from mmap memory or PCIe stream.
- [ ] **Task 3.2: MoE Assembly & Oracle Gate 3**
  - Assemble full 24B decode loop.
  - Run `./tests/verify_lfm25_oracle.sh moe`.
  - **Validation Gate 3**: Achieve **128/128 exact token match** on LFM2.5-24B-A2B.

---

### Phase 4: Fused Kernels & CUDA Graph Capture
**Objective**: Eliminate global memory round-trips and drive launch latency to zero.

- [ ] **Task 4.1: Whole-Trunk Static CUDA Graph (1.2B)**
  - Capture all 16 layers of LFM2.5-1.2B into a single static CUDA Graph.
  - Target: **$\ge 350 - 400$ tok/s** on RTX 3070 Laptop.
- [ ] **Task 4.2: Fused Depthwise Conv1D + Gating + `Q8_1` Requantization**
  - Single warp-cooperative kernel:
    * In-projection outputs $[b, c, x]$ kept in shared memory.
    * Depthwise 3-tap dot product with rolling state buffer.
    * Inline output gating ($c \odot \text{conv\_out}$).
    * Direct shared-memory quantization to `Q8_1` for out-projection GEMV.
- [ ] **Task 4.3: Async Prefetch & Dual-Stream MoE Execution (24B)**
  - Overlap layer $L$ expert computation with layer $L+1$ PCIe miss transfers.
  - Target: **$\ge 95 - 110$ tok/s** on RTX 3070 Laptop.

---

### Phase 5: Speculative Decoding Adaptations
**Objective**: Lossless acceleration tailored to LFM's hybrid architecture.

- [ ] **Task 5.1: Fast Rolling Ring-Buffer Rollback**
  - Conv rolling state is only $2 \times 2048$ floats (16 KiB/layer).
  - State rollback after draft token rejection is an $O(1)$ pointer/memcpy operation.
- [ ] **Task 5.2: Speculative Suite & Acceptance Rate Benchmark**
  - Run `scripts/bench_ngram_suite.sh` on LFM2.5.
  - Target: **$\ge 110 - 125$ tok/s** decode speed.

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
4. [ ] **Oracle Verification**: Token-identical match with reference text (`tests/verify_lfm25_oracle.sh`).
5. [ ] **Thermal / Power State Check**: `scripts/check_power_state.sh` verified AC connected and GPU clocks locked at 6001 MHz VRAM.
6. [ ] **Paired A/B Measurement**: Interleaved runs via `scripts/ab_interleaved.sh` confirming statistically significant speedup without regressions.
