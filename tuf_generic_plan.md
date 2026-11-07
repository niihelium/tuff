# tuf Generic Decoder: Design & Plan

**Status:** design only, nothing implemented. **Date:** 2026-10-18. **Branch at writing:** `lfm25-phase1` @ `59ec59b`.

**Goal:** a new quantized GGUF model (Qwen2/2.5, Qwen3 0.6B–8B, Qwen3.5 0.8B/4B/9B, Llama-style, Nanbeige, later Gemma) runs in tuf in **hours, not days**, with **no hand-tuning**, passes the llama.cpp oracle gate with **one command**, and decodes **at or above llama.cpp tg128** on the RTX 3070 Laptop (8 GB, sm_86).

**Not the goal:** a general tensor compiler, or replacing the hand-tuned Qwen3.6-35B-A3B and LFM2.5 engines. Those stay as fast paths (§9).

---

## 1. Why: where the days go today

Adding LFM2.5 took ~4,000 lines over ~3 days (`git log`: Phase 0 on 2026-10-15, through Task 6.1 on 2026-10-17). Most of that was not new math. It was per-model wiring plus duplicated kernels:

| Cost | Evidence |
|---|---|
| **A hand-written forward pass per architecture** | Qwen: `engine_forward_token` is one 1,130-line function ([src/engine.c:771](src/engine.c:771)–1902). LFM: `lfm25_enqueue_step` ([src/lfm25_engine.c:350](src/lfm25_engine.c:350)) |
| **A model struct per architecture, with hparams read under a hardcoded key prefix** | `model_t` reads `"qwen35moe.*"` ([src/model.c:30](src/model.c:30)); `lfm25_model_t` ([include/lfm25_model.h](include/lfm25_model.h)) is a separate layout |
| **Duplicated kernels** | `vec_dot_q4_K_q8_1_c` / `vec_dot_q6_K_q8_1_c` exist twice: [src/gpu/mmvq.cuh:36](src/gpu/mmvq.cuh:36) and [src/lfm25_gpu.cu:64](src/lfm25_gpu.cu:64). Same for RMSNorm+Q8_1 quant, head+argmax and embedding (`lfm_*` vs `gpu/*`) |
| **Shape and quant-type constants baked in** | LFM attention is head_dim 64 only ([src/lfm25_model.c:325](src/lfm25_model.c:325)). Qwen3 needs 128, Qwen3.5 needs 256. The embedding must be Q6_K ([src/lfm25_model.c:340](src/lfm25_model.c:340)), but Qwen3 GGUFs ship Q4_K embeddings. Short conv is K=3 only ([src/lfm25_model.c:331](src/lfm25_model.c:331)) |
| **Narrow quant coverage** | Device dot products exist for Q4_K, Q6_K and Q8_0 only. Real GGUFs mix more types per layer (see §2) |
| **Tokens and chat format hardcoded** | BOS/EOS `151644/151645` ([src/tuff_vtable.c:30](src/tuff_vtable.c:30)); ChatML strings in [src/chat.c:352](src/chat.c:352) |
| **Tuning by env flags and A/B runs** | 19 `getenv` knobs in `src/engine.c` alone (`TUFF_H6`, `TUFF_MOE_SPLIT`, `TUFF_FILL_STRIDE`, …), each settled by interleaved A/B runs |
| **A per-architecture FP64 oracle** | [tools/lfm25_fp64_ref.py](tools/lfm25_fp64_ref.py) re-implements the LFM2 forward in numpy |

What is already reusable and must be kept:

* **Kernel bodies.** The llama.cpp-derived mmvq dot products, fused gate+up+SwiGLU, flash-decoding attention, the gated delta-net ([src/gpu/delta.cu](src/gpu/delta.cu)), short conv ([src/lfm25_gpu.cu:287](src/lfm25_gpu.cu:287)), the MoE router and expert kernels, and the batched (`*_batch`) mmvq variants.
* **The LFM step structure.** `lfm25_enqueue_step` is already almost a generic decoder: norm → token mixer (conv | attention) → norm → FFN (dense | MoE), with quant-type dispatch at enqueue time ([src/lfm25_engine.c:264](src/lfm25_engine.c:264)), captured once into a CUDA graph. The generic path generalizes exactly this.
* **The oracle tool.** [tools/lfm25_llama_ref.c](tools/lfm25_llama_ref.c) drives libllama and is already architecture-agnostic.
* **GGUF parser, sampler, server, tokenizer core.** The BPE engine and the llama3-group pre-tokenizer list are at [src/tokenizer.c:598](src/tokenizer.c:598).

## 2. What the target models need (read from their GGUFs)

| Model (file) | arch | Layers | Token mixer | Special | Weight types present |
|---|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | `qwen3` | 28 | GQA 16/8, d_head 128 | QK-norm, NeoX RoPE | Q4_K, Q6_K (v/down mixed per layer), out Q6_K, embd Q4_K |
| Qwen3-8B Q4_K_M | `qwen3` | 36 | GQA 32/8, d_head 128 | same | same |
| Qwen3.5-9B Q4_K_M | `qwen35` | 32 | 3× gated delta-net : 1× attention (`full_attention_interval=4`) | attention output gate `attn_gate`, partial M-RoPE (64 of 256 dims, sections [11,11,10,0]), `post_attention_norm` | Q4_K, Q5_K, Q6_K, Q8_0 |
| Qwen3.5-4B UD-Q4_K_XL | `qwen35` | 32 | same | **tied** LM head (no `output.weight`) | Q4_K, Q5_K, Q6_K, Q8_0, **IQ4_XS**, **F16** |
| LFM2.5-1.2B Q4_K_M | `lfm2` | 16 | conv (K=3) or GQA 32/8 d_head 64, chosen per layer by `head_count_kv[l]==0` | `token_embd_norm` as final norm, tied head | Q4_K, Q6_K |
| Nanbeige4.2-3B Q4_K_M | `nanbeige` | 22 × `num_loops=2` | GQA 48/8, d_head 128 | **looped** layer stack, no QK-norm | Q4_K, Q6_K |
| gemma-4-E4B Q4_0 | `gemma4` | 42 | SWA (512) / global attention, shared-KV layers | per-layer embeddings, logit softcap, post-norms, d_head 256/512 | **Q4_0**, Q6_K, F16 |

Conclusions:

1. **Quant type is per tensor, not per model.** Even plain Q4_K_M mixes Q4_K and Q6_K inside the same tensor role. Kernel choice must follow each tensor's `info.type`, and coverage must include Q4_0, Q5_K, IQ4_XS and F16, not just Q4_K/Q6_K/Q8_0.
2. Most differences are **flags on a common decoder**: QK-norm, QKV bias (Qwen2), tied head, RoPE style/theta/partial dims, extra post-norms, output gate, layer looping.
3. Genuinely new ops are few: gated delta-net (exists), short conv (exists), MoE (exists), and for Gemma4: sliding-window/shared-KV attention, per-layer embeddings and logit softcap.

## 3. Architecture overview

```
 GGUF ──► Arch descriptor (C table, one entry per general.architecture)
            │  hparams from "<arch>.*" keys, tensor-name patterns,
            │  per-layer kind rule, flags
            ▼
        Graph builder ──► Layer graph (IR): flat list of typed nodes over named buffers
            ▼
        Planner: fusion patterns · buffer arena (liveness) · KV/state sizing from VRAM budget
                 · per-node kernel variant from the autotune cache
            ▼
        Executor: enqueue nodes on one stream → capture once into a CUDA graph
                  (single token; batched-n graphs for prefill / spec verify)
            ▼
        engine_vtable_t "generic"  (src/tuff_vtable.c)
```

**The inner-loop rule holds.** [wiki/architecture.md](wiki/architecture.md) Rule 1 forbids per-token architecture branches. All dispatch over node kinds and quant types happens **once, at capture**. Per token, the executor does exactly what the LFM engine does today: write `(token,pos)` into the device buffer and run `cudaGraphLaunch`. Kernel bodies stay specialised through templates on quant type, head_dim and epilogue, so there are no runtime type branches inside kernels.

## 4. Arch descriptor

One static C struct per architecture in `src/generic/arch_table.c`. It is pure data, plus at most a tiny hook for odd layer-kind rules.

```c
typedef struct {
    const char *arch;                 // "qwen3", "qwen2", "qwen35", "lfm2", "llama", "nanbeige", ...
    // hparam keys are "<arch>.<suffix>"; defaults only where the GGUF legitimately omits a key
    layer_kind_rule_t kind_rule;      // ALL_ATTN | INTERVAL(full_attention_interval) | KV_HEADS_ZERO_IS_CONV
    unsigned flags;                   // QK_NORM | QKV_BIAS | ATTN_OUT_GATE | POST_ATTN_NORM | POST_FFN_NORM
                                      // | TIED_HEAD_FALLBACK | EMBD_NORM_IS_FINAL | LOOPED_STACK | LOGIT_SOFTCAP
    rope_kind_t rope;                 // NEOX | NORM | MROPE_SECTIONS ; partial dims from rope.dimension_count
    ffn_kind_t ffn;                   // SWIGLU | GEGLU ; MoE detected from expert_count > 0
    const tensor_role_t *roles;       // role -> GGUF name pattern ("blk.%d.attn_q.weight"), required/optional
} arch_desc_t;
```

The graph builder reads the descriptor plus the GGUF. Each layer gets a kind (attention / delta-net / conv) and an FFN kind (dense / MoE). Every role resolves to a tensor with its own type and shape. **Validation is generic.** Every tensor's shape is checked against hparams, and every (op, quant type, shape) is checked against what the kernel library supports. A missing capability fails at load with a precise message (`attn: head_dim 512 not instantiated`), never at decode.

How the targets map:

| arch | kind_rule | flags | rope | Effort |
|---|---|---|---|---|
| `qwen3` | ALL_ATTN | QK_NORM | NEOX | table entry only |
| `qwen2` | ALL_ATTN | QKV_BIAS | NEOX | table entry + bias epilogue |
| `llama` | ALL_ATTN | — | NORM | table entry |
| `nanbeige` | ALL_ATTN | LOOPED_STACK (num_loops) | NEOX | table entry + loop in builder |
| `lfm2` / `lfm2moe` | KV_HEADS_ZERO_IS_CONV | QK_NORM, EMBD_NORM_IS_FINAL, TIED_HEAD_FALLBACK | NEOX | table entry (ops exist) |
| `qwen35` | INTERVAL(4) → delta-net else attention | QK_NORM, ATTN_OUT_GATE, POST_ATTN_NORM, TIED_HEAD_FALLBACK | MROPE_SECTIONS, partial | table entry + M-RoPE/partial in the rope kernel |
| `gemma4` | SWA pattern array | many | dual theta | new ops (§12, P7) |

**Adding an architecture whose ops already exist** = one table entry + one gate run. That is the "hours, not days" path.

## 5. IR: node set

A flat array of nodes over named device buffers. No control flow except the layer loop, which is unrolled at build time. Each node kind has a **CPU reference** (extending [src/cpu_ops.c](src/cpu_ops.c)) and a randomized unit test against its GPU kernel.

| Node | Notes / reused kernel |
|---|---|
| `EMBED(type)` | dequantize one row; any supported quant type (today Q6_K only: `lfm_embed_q6_K_kernel`) |
| `RMSNORM_Q8(eps)` | fused RMSNorm + Q8_1 quantize (`fused_rms_norm_quantize_q8_1_kernel`, [src/gpu/output.cu:155](src/gpu/output.cu:155)) |
| `MATVEC(type, epilogue)` | epilogue ∈ {store, accum-into-residual, +bias, quant-out}. One templated mmvq over quant traits (§6) |
| `MATVEC_GROUP(types…)` | Q/K/V (or gate/up) sharing one Q8_1 input: one launch when the types match, else split |
| `SWIGLU_MATVEC` | fused gate+up+SiLU·mul+Q8_1 (`lfm_fused_swiglu_mmvq_q4_K_kernel`), generalized over type |
| `QK_NORM_ROPE_KVWRITE(rope, partial, sections)` | from `lfm_fused_qk_norm_neox_rope_kernel` / `fused_qk_split_norm_rope_kernel`; template on head_dim |
| `ATTN_DECODE(head_dim, kv_type)` | flash-decoding (`flash_attn_vec_online_kernel<HEAD_DIM,…>`, [src/gpu/attention.cu:3](src/gpu/attention.cu:3)); instantiate 64/128/256 |
| `ATTN_OUT_GATE` | sigmoid gate on attention output (Qwen3.5) |
| `SHORTCONV(K)` | `lfm_inproj_shortconv_kernel`, template on K |
| `DELTANET` | `fused_gated_delta_net_ar_kernel` ([src/gpu/delta.cu:267](src/gpu/delta.cu:267)), sizes from `ssm.*` keys |
| `ROUTER_TOPK(k, sigmoid/softmax, bias)` + `MOE_FFN` | resident-expert path only (`lfm_router_top4_kernel`, `lfm_moe_*`); offload stays in tuned engines |
| `RESIDUAL_ADD`, `SCALE`, `SOFTCAP` | trivial elementwise |
| `HEAD_ARGMAX(type)` | fused LM head + argmax (`lfm_mmvq_q6_K_head_argmax_kernel`), any type; logits D2H only when capture is on |

**Graph dump.** `tuff --dump-graph g.json` prints the built graph: nodes, tensor names, types, shapes, chosen kernel variants. It is the main debugging aid when bringing up a model, and it drives the generic FP64 reference (§8).

## 6. Kernel library consolidation

Prerequisite for everything else, and where most of the "new model" time currently leaks.

1. **Quant traits.** One `quant_traits<T>` per GGML type: block size, bytes, `vec_dot_*_q8_1`, row dequant. Port from llama.cpp `ggml-cuda/vecdotq.cuh` / `mmvq.cu` (MIT, like the existing ports; record in [ATTRIBUTION.md](ATTRIBUTION.md)). Order by what the target GGUFs contain: **Q4_K, Q6_K, Q8_0 (exist) → Q5_K, IQ4_XS, F16, Q4_0 → Q5_0, Q4_1, Q3_K, Q2_K, IQ4_NL, BF16**.
2. **One mmvq template**: `mmvq<Traits, ROWS_PER_BLOCK, NCOLS, Epilogue>`. It covers the single-token, `ncols`/batch (prefill and verify) and warp-row variants that exist today as hand-copied kernels in [src/gpu/dense_gemv.cu](src/gpu/dense_gemv.cu), [src/gpu/moe.cu](src/gpu/moe.cu) and [src/lfm25_gpu.cu](src/lfm25_gpu.cu).
3. **Delete the `lfm_*` duplicates** once the LFM tuned engine calls the shared library. Gate: the LFM goldens stay byte-identical.
4. **Attention templated on head_dim** {64, 128, 256}, with KV in F32 or Q8_0 as today.
5. **Fuzz test per (op, quant type)**: random shapes, GPU vs CPU reference. Exact for integer paths, tolerance for float. Adding a quant type = traits + this test, nothing else.

## 7. Planner and autotune

**Fusion patterns.** Peephole rules over the node list, applied in a fixed order. Each one is the generic form of a fusion the hand engines already do:
`RMSNORM → Q8` · `gate,up → SWIGLU_MATVEC` · `Q,K,V → MATVEC_GROUP` · `out_proj → MATVEC(accum)` (residual add folded) · `final norm → HEAD_ARGMAX` · `QK-norm+RoPE+K-write` in one node. A rule fires only when its type and shape preconditions hold; otherwise the unfused nodes run. Correct either way, only speed differs.

**Memory.** Buffer liveness → one activation arena. KV cache and recurrent state (conv, delta-net) sized from `-c` and the free-VRAM budget after weights. If the weights do not fit, load fails with the shortfall; offload is the tuned MoE engines' job.

**Autotune instead of hand-tuning.** For each distinct (kernel, quant type, rows, cols, ncols) in the graph, time the candidate launch configs (rows-per-block, warp-row vs block-row, threads) on first load. Cache the winners in `~/.cache/tuf/autotune/<gpu-uuid>_<driver>_<tuf-commit>.json`. Typical model: tens of distinct shapes × 3–6 candidates × ~1 ms ≈ well under 5 s, once.

* **Numerics are part of the contract.** Candidates in one set must be **bit-identical** (same per-row reduction order). The tuner checks each candidate's output against the default variant on a random input and drops any that differ. So autotune can never break the oracle gate.
* Env knobs like the 19 in `src/engine.c` do not exist in the generic path. A setting is either tuned automatically or fixed.

## 8. Correctness: one generic gate

`tests/verify_model.sh <model.gguf> [--prompt …]` works for any model the generic path loads:

1. **Reference.** Generalize [tools/lfm25_llama_ref.c](tools/lfm25_llama_ref.c) → `tools/llama_ref.c`. It is already architecture-agnostic: single-token batches, all layers on GPU, greedy, with tokens/margins/logits out. The recipe is recorded next to the reference as today.
2. **Stage 1 (exact prefix)** and **stage 2 (teacher-forced top-1 with near-tie allowance)**: unchanged from [tests/verify_lfm25_oracle.sh](tests/verify_lfm25_oracle.sh). No per-model code.
3. **Stage 3 (FP64 numerics).** Replace the per-architecture numpy forward with a **generic interpreter of the dumped graph** (§5): `tools/fp64_graph_ref.py g.json model.gguf` dequantizes with gguf-py and runs the same node list in float64.
   * Stages 1–2 catch graph-construction errors, because they compare against llama.cpp's independent implementation.
   * Stage 3 then only has to judge numerics, so interpreting tuf's own graph is valid. Thresholds stay as in [tests/lfm25_gate_thresholds.txt](tests/lfm25_gate_thresholds.txt).
4. **Tokenizer and chat format.** The tokenizer must reproduce llama.cpp's prompt tokens; the gate already checks this from `llama_ref`'s stderr. BOS/EOS come from `tokenizer.ggml.*` keys instead of [src/tuff_vtable.c:30](src/tuff_vtable.c:30).
   * Pre-tokenizers are selected by `tokenizer.ggml.pre`: add `qwen2` next to the existing `qwen35` and llama3 group.
   * Chat templates: detect a small set of families (ChatML, LFM, Llama3, Gemma) from `tokenizer.chat_template` rather than hardcoding ChatML ([src/chat.c:352](src/chat.c:352)). No Jinja engine.
5. **Same-build rule (README rule 6).** The gate and the benchmark run the same binary and flags. The generic engine is selected by the same `--engine` value in both.

New-model definition of done: `verify_model.sh` passes + `compute-sanitizer` clean + `bench_model.sh` row recorded.

## 9. Coexistence with the hand-tuned engines

* Add `TUFF_ARCH_GENERIC` and `--engine auto|generic|tuned` ([include/tuff_vtable.h](include/tuff_vtable.h)). `auto` (default) picks a tuned engine when one exists for the architecture and the model's config matches what it validates (`lfm25_model_validate`), otherwise generic.
* **The tuned engines are the generic path's yardstick.** Running LFM2.5-1.2B and 8B-A1B through `--engine generic` and comparing against the tuned engine (~378 / ~242 tok/s in [LFM25_ROADMAP.md](LFM25_ROADMAP.md)) measures exactly how much the generic path leaves on the table. Each gap found becomes a planner fusion rule or a kernel variant, so every model benefits.
* What stays tuned-only: the routed-expert VRAM cache with CPU-miss compute, n-token MoE verify, MTP/DFlash drafting ([QWEN35_ROADMAP.md](QWEN35_ROADMAP.md)). These depend on host/GPU orchestration the generic executor does not attempt.
* **Long term**, `lfm2` dense could retire its tuned engine if generic reaches parity. Decide per model on measured numbers, not up front.

## 10. Later: a persistent megakernel as a second executor

Mirage MPK was evaluated on this machine (2026-10-18). On sm_86 it supports bf16 weights only (its FP8 path asserts SM100), has no quantized, short-conv or delta-net tasks, and crashed with an illegal memory access on Qwen3-0.6B. It also gives 10 of the 40 SMs to schedulers.

Its useful idea is the execution model: persistent CTAs consuming a task list with dependency counters. Once the IR exists, that can be built as **`executor_persistent`** over the same graph, reusing the same node implementations as device functions. It applies to every model at once instead of being hand-built per model. It only pays where the token time is dominated by launch and sync gaps, i.e. small models (~≤1.5B). Not in the milestones below. First measure the inter-kernel gap fraction on the generic CUDA-graph path.

## 11. What adding a model costs after this

| Case | Example | Work | Estimate |
|---|---|---|---|
| New size of a supported arch | Qwen3-4B, Qwen3.5-2B | none, run the gate | minutes (+ autotune on first load) |
| New quant mix | UD-Q4_K_XL, Q5_K_M | none if types are covered; else add traits + fuzz test | 0–4 h |
| New arch, existing ops | qwen2, llama, nanbeige | table entry (+ small flag epilogue) + gate | 1–3 h |
| New op | Gemma4 per-layer embeddings, SWA/shared KV | kernel + CPU ref + fuzz test + IR node + table entry | 1–2 days |

## 12. Phased plan

Estimates are focused engineering days. Each phase ends on a gate, following the existing rules (correctness before performance; one variable per experiment; quote the verdict).

| Phase | Scope | Exit criteria | Est. |
|---|---|---|---|
| **P0 Harness** | `tools/llama_ref.c` (generalized), `tests/verify_model.sh` stages 1–2, `scripts/bench_model.sh` (tuff vs llama-bench tg128, same session, power state recorded), baseline table for all target GGUFs | Gate runs green on LFM2.5-1.2B through the **tuned** engine (proves the harness) | 1 |
| **P1 Kernel library** | quant traits; Q5_K, IQ4_XS, F16, Q4_0 dot products; one templated mmvq (single/ncols, epilogues); attention head_dim 64/128/256; embed for any type; per-(op,type) fuzz tests; LFM tuned engine switched to the shared library, `lfm_*` duplicates deleted | Fuzz suite green; LFM goldens byte-identical; LFM tok/s unchanged (±1%, interleaved A/B) | 3 |
| **P2 Generic dense-attention path** | descriptor table, graph builder, generic validation, planner (fusion + arena), executor + CUDA graph, `--dump-graph`, vtable entry, BOS/EOS/pre-tokenizer from GGUF, chat-template families; arches `qwen3`, `qwen2`, `llama`, `nanbeige` | Qwen3-0.6B and Qwen3-8B Q4_K_M pass `verify_model.sh`; ≥ 0.95× llama.cpp tg128 | 4 |
| **P3 Autotune** | candidate sets, bit-identity filter, on-disk cache, `--autotune=off/force` | Gate still green (same tokens); Qwen3-0.6B/8B ≥ 1.0× llama.cpp tg128; first-load tuning < 10 s | 1.5 |
| **P4 Hybrid layers** | `qwen35` dense (delta-net node, output gate, partial M-RoPE, post-attn norm, tied head); `lfm2` dense via generic; generic FP64 graph interpreter (stage 3) | Qwen3.5-0.8B/4B/9B pass the full 3-stage gate; ≥ 1.0× llama.cpp; LFM2.5-1.2B generic within 10% of the tuned engine (gap list recorded) | 3 |
| **P5 Batched forward** | generic `forward_batch(n)` + rollback (prefill chunks, n-gram spec verify), reusing ncols mmvq | Byte-identical to n single steps; prefill ≥ 5× faster than token-by-token on Qwen3-8B; spec path works on any generic model | 2 |
| **P6 Resident MoE** | `lfm2moe` (8B-A1B) and other fully-resident MoEs through generic router + expert nodes | 8B-A1B passes the gate; within 10% of the tuned engine | 2 |
| **P7 Gemma4 (stretch)** | SWA + shared-KV attention, per-layer embeddings, softcap, post-norms | gemma-4-E4B Q4_0 passes the gate | 2–3 |

**Total to P4 (the user-facing goal): ~12–13 days. All phases: ~19–20 days.**

### Targets for the first milestones (measured 2026-10-18, llama.cpp `5aeac37a`, `llama-bench -ngl 99 -fa 1 -n 128 -r 3`)

| Model | llama.cpp tg128 | Weight-read ceiling* |
|---|---|---|
| Qwen3-0.6B Q4_K_M | 377.5 | ~690 |
| Qwen3-8B Q4_K_M | 65.5 | ~66 |
| Qwen3.5-0.8B Q4_K_M | 300.0 | ~650 |
| Qwen3.5-4B Q4_K_S | 93.9 | ~128 |
| Qwen3.5-9B Q4_K_M | 58.3 | ~58 |
| LFM2.5-1.2B Q4_K_M | 332.8 (tuned tuf ~378) | ~450 |
| LFM2.5-8B-A1B Q4_K_M | 222.5 (tuned tuf ~242) | — (MoE) |

\*GGUF file size / 330 GB/s sustained. This is a rough bound: it overcounts the embedding table (only one row is read per token), so small models with large vocabularies have more headroom than shown. For 8–9B dense models, llama.cpp is already at the ceiling, so the generic path's job there is parity plus tuf's extras (speculation, server). The real speed headroom is in the ≤4B models.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Generic fusions are weaker than hand-tuned ones → slower than the tuned engines | The tuned engines measure the gap (§9); each gap becomes a fusion rule or kernel variant usable by all models. Parity is not required to ship a new model, only ≥ llama.cpp |
| Oracle mismatch on new architectures from legitimate float-order differences | The existing 3-stage gate already handles near-ties and judges numerics against FP64; the generic interpreter makes stage 3 available for every arch |
| Tokenizer coverage (pre-tokenizers, SentencePiece for Gemma) | The gate checks prompt tokens against llama.cpp first; a new pre-tokenizer is a port of one llama.cpp function, done per family as it appears |
| IQ-quant dot products are intricate | Port verbatim from llama.cpp with fuzz tests; order by what real GGUFs contain (IQ4_XS first) |
| Autotune nondeterminism | Bit-identity filter (§7); cache keyed by GPU, driver and commit; `--autotune=off` gives the fixed default |
| Scope creep into a compiler | The IR has no control flow and no codegen; kernels are hand-written templates. New behavior = new node + kernel, never a lowering pass |

## 14. Decisions needed before P0

1. **Location:** `src/generic/` beside the tuned engines (recommended), or a refactor of the existing engines onto the IR from the start (more churn, P1 already shares the kernels).
2. **First milestone models:** Qwen3-0.6B + Qwen3-8B (dense, simplest), then Qwen3.5 (recommended); or Qwen3.5 first if that family matters more.
3. **Gemma4 in scope or not.** It is the only target needing several new ops.
