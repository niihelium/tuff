# Tuff Code Architecture & Optimization Manual

This document details the software architecture, runtime dispatch mechanism, kernel fusion hierarchy, and performance optimization contracts for running diverse model families (both **Qwen3.6 DeltaNet** and **Liquid AI LFM2.5**) within a **single unified `tuff` binary**.

---

## 1. Architectural Philosophy: The Separation Principle

The primary engineering constraint in high-performance LLM engines is **inner-loop purity**:
* **Rule 1**: Never introduce conditional branches (`if (model->arch == LFM) ... else ...`) inside the token generation loop or inside CUDA kernel bodies. Branch divergence and register bloat destroy kernel occupancy on NVIDIA architectures.
* **Rule 2**: Favour maximum performance over superficial code reusability. If combining two operations into a fused kernel improves memory bus saturation, implement a dedicated kernel even if it duplicates similar logic from another architecture.
* **Rule 3**: Retain a **single unified binary** for the user (`build/tuff`). All model family dispatching occurs **once per load**, binding an indirect execution table (`engine_vtable_t`).

---

## 2. Unified Binary & Init-Time Architecture Dispatch

```
                        [ CLI Invocation ]
                   ./build/tuff --model <path>
                               │
                               ▼
                    [ GGUF Header Inspect ]
                    Read `general.architecture`
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   "qwen35moe" (DeltaNet)                "lfm2" / "lfm2moe" (LFM)
            │                                     │
   Bind `qwen_engine_vtable`              Bind `lfm25_engine_vtable`
   Init `model_t` (Qwen layout)          Init `lfm25_model_t` (LFM layout)
   Prepare `gpu.cu` buffers               Prepare `lfm25_gpu.cu` buffers
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
               [ Timed Token Generation Loop ]
               while (step < n_predict) {
                   // Exactly ONE indirect function call per token (~1 ns overhead):
                   token = vtable->forward_token(engine, token, pos);
               }
```

### The Engine VTable Interface (`include/tuff_vtable.h`)

```c
typedef struct engine_vtable {
    const char *arch_name;
    
    // Model lifecycle
    void* (*model_load)(const char *path);
    void  (*model_free)(void *model);
    void  (*model_print_inspection)(const void *model);
    
    // GPU lifecycle
    int   (*gpu_prepare)(void *model, int *max_context, int fill_ram);
    void  (*gpu_shutdown)(void *model);
    
    // Engine lifecycle
    void* (*engine_init)(void *model, int max_context);
    void  (*engine_free)(void *engine);
    void  (*engine_set_sampler)(void *engine, const sampler_params_t *params);
    
    // Decode step (the inner loop call)
    int32_t (*forward_token)(void *engine, int32_t token, int pos);
} engine_vtable_t;
```

#### Why This Has Zero Performance Penalty
* An indirect function pointer call costs $\sim 1 - 2$ nanoseconds.
* A single token generation step takes $\sim 2,500,000$ nanoseconds (at 400 tok/s).
* The dispatch overhead is **$0.00008\%$** of the decode timeline—completely undetectable against hardware clock jitter.
* In exchange, each model family has its own completely dedicated C and CUDA translation units (`src/lfm25_*`), compiled with optimal register counts and shared memory configurations.

---

## 3. Kernel Optimization & Fusion Hierarchy

### 3.1 Kernel Fusion Strategy for LFM2.5

```
Standard Generic Path (llama.cpp):
RMSNorm ──► [Global VRAM] ──► GEMV (W_in) ──► [Global VRAM] ──► Mul (b*x) ──► [Global VRAM] ──► Conv1D ──► [Global VRAM] ──► Mul (c*conv) ──► [Global VRAM] ──► GEMV (W_out)
(5 global memory round-trips!)

Tuff Fused Path:
RMSNorm ──► GEMV (W_in) ──► [Shared Memory] ──► Fused Conv1D + Gate + Q8_1 Requant ──► GEMV (W_out)
(Zero intermediate global memory traffic!)
```

1. **Fused Conv1D Depthwise Dot-Product + Gate**:
   * Takes the $[b, c, x]$ projected vector directly in registers/shared memory.
   * Modulates $u = b \odot x$.
   * Computes the 3-tap depthwise dot product with rolling state buffer $S_l[0..1]$ in L1/shared memory.
   * Multiplies by output gate $c$.
   * Immediately quantizes the activation to `Q8_1` format in shared memory so the subsequent $W_{\text{out}}$ GEMV reads quantized blocks without an extra conversion pass.

2. **Fused SwiGLU Activation**:
   * Evaluates inline $\text{SiLU}(gate) \odot up = \frac{gate}{1 + e^{-gate}} \odot up$.
   * Quantizes the resulting row to `Q8_1` directly in shared memory for down-projection.

3. **CUDA Graph Capture (Whole-Trunk Graph)**:
   * For dense models like LFM2.5-1.2B, all 16 layers have static tensor shapes during decode.
   * We capture the entire 16-layer graph into a single static CUDA Graph (`cudaGraphLaunch`).
   * Eliminates driver launch overhead ($>120\ \mu\text{s} \to <5\ \mu\text{s}$).

---

## 4. Source Tree Layout

```
tuf/
├── include/
│   ├── tuff.h                 # Unified public API
│   ├── tuff_vtable.h          # Architecture dispatch table
│   ├── model.h                # Qwen3.6 model definitions
│   ├── engine.h               # Qwen3.6 engine definitions
│   ├── lfm25.h                # LFM2.5 public API
│   ├── lfm25_model.h          # LFM2.5 dedicated tensor layout
│   └── lfm25_engine.h         # LFM2.5 dedicated engine state
├── src/
│   ├── main.c                 # Unified CLI driver with init-time arch switch
│   ├── model.c                # Qwen GGUF loader
│   ├── engine.c               # Qwen decode engine
│   ├── gpu.cu                 # Qwen CUDA kernels & DeltaNet
│   ├── lfm25_model.c          # LFM2.5 GGUF loader & layer classifier
│   ├── lfm25_gpu.cu           # LFM2.5 dedicated CUDA kernels (Conv1D, QK-Norm, SwiGLU)
│   └── lfm25_engine.c         # LFM2.5 decode engine & graph replay
├── tools/
│   └── inspect_lfm25_gguf.py  # GGUF tensor layout & VRAM budget inspector
├── tests/
│   ├── ref_lfm25_dense_128.txt# Oracle ground truth for 1.2B
│   ├── ref_lfm25_moe_128.txt  # Oracle ground truth for 24B
│   └── verify_lfm25_oracle.sh # Exact token matching verification harness
└── wiki/
    ├── architecture.md        # This manual
    └── lfm25/
        ├── other_runtimes.md  # Comparative study of llama.cpp, colibri, Strata
        ├── 1_2B.md            # LFM2.5-1.2B spec (Dense)
        ├── 2_6b.md            # LFM2.5-2.6B spec (Dense)
        ├── 8b-a1b.md          # LFM2.5-8B-A1B spec (100% VRAM MoE)
        └── 24b-a2b.md         # LFM2.5-24B-A2B spec (Hybrid MoE)
```

---

## 5. Engineering Sign-Off Criteria

Before any code is committed to `main`:
1. **Compilation**: Zero warnings under `-Wall -Wextra -Werror` for C, C++, and CUDA (`sm_86`).
2. **Oracle Gate**: 128/128 greedy token exact match against reference (`./tests/verify_lfm25_oracle.sh`).
3. **Memory Safety**: `compute-sanitizer --tool memcheck` passes with **0 errors**.
4. **Performance Measurement**: Paired, order-balanced A/B test via `scripts/ab_interleaved.sh` confirming statistically significant speedup without Qwen regression.
