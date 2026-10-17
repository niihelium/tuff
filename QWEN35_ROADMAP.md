# Qwen3.5-MoE Speculative-Decoding Roadmap for Tuff (ideas borrowed from HyperQwen)

**Ordering: implement this roadmap after [LFM25_ROADMAP.md](LFM25_ROADMAP.md) is complete.** Several steps here reuse LFM work that is not done yet: Task 5.3 (MoE verify over the expert union), Task 6.1 (byte-identical CPU miss path), and Task 6.2 (`src/moe_policy.{c,h}`). Build and prove those on LFM2.5 first, where the goldens are byte-identical, then port them here.

Target: **Qwen3.6-35B-A3B** (`qwen35moe` architecture, `Q4_K_S` GGUF) on the RTX 3070 Laptop 8 GB + Ryzen 9 5900HX, the engine in [`src/engine.c`](src/engine.c) / [`src/gpu.cu`](src/gpu.cu). The starting point and methodology are in [README.md](README.md), the open-lever ledger in [PLAN.md](PLAN.md), and measured history in [ITERATION_LOG.md](ITERATION_LOG.md).

Source of the ideas: **HyperQwen** ([github.com/syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen), local checkout `~/workspace/HyperQwen`, read at `95ee3b6`). It is a patch series against vLLM 0.31 plus a model-preparation pipeline for Qwen3.8-27B (dense, hybrid GDN) on one RTX 3090. Most of it is vLLM- or batching-specific. This roadmap keeps only what transfers to a batch-1, single-stream engine with an expert cache. Main references:

* [docs/optimizations.md](https://github.com/syv-ai/HyperQwen/blob/main/docs/optimizations.md): every optimization with measurements, including the DFlash2 and lookup-drafting sections.
* [drafter/README.md](https://github.com/syv-ai/HyperQwen/blob/main/drafter/README.md): draft vocabulary, GPTQ requantization of heads and drafters, and what did not work.
* [patches/dflash2-lookup-drafting.patch](https://github.com/syv-ai/HyperQwen/blob/main/patches/dflash2-lookup-drafting.patch): lookup-augmented block drafting (LABD) and the adaptive verify length.
* [patches/dflash2-ngram-chains.patch](https://github.com/syv-ai/HyperQwen/blob/main/patches/dflash2-ngram-chains.patch): quantized candidate chains for the drafter.
* [patches/spec-decode-attn.patch](https://github.com/syv-ai/HyperQwen/blob/main/patches/spec-decode-attn.patch): split-KV attention for multi-query verify.
* [PATCHES.md](https://github.com/syv-ai/HyperQwen/blob/main/PATCHES.md): one line per patch.

---

## 1. Where we are and why speculation has not paid yet

* **60.83 tok/s base**, +38% over llama.cpp. The three dominant kernels run at ~91% of the 384 GB/s bus, so the single-token path is at the batch-1 GEMV ceiling (~103 tok/s roofline). Speculation is the remaining throughput lever.
* Measured speculation today: n-gram `--spec-draft=ngram --spec-ngram 3` **+2.2%**; MTP head `--spec-mtp` **−12.7%** (α ≈ 0.5, break-even α > 0.51); CPU draft model (H28) 20–25 tok/s, refuted.
* **The verify step is 2 tokens wide.** `engine_forward_verify(tok0, tok1, …)` and `verify_moe_stage_n2` in [`src/engine.c`](src/engine.c) verify one draft. `--spec-draft-len` allows up to 3 drafts, but they cannot be verified in one pass.
* **Experts run once per token.** One token reads ~2.5 GB: ~1.92 GiB of dense weights (shared by every token in a batch) plus ~0.6 GB of routed experts (40 layers × 8 × ~1.9 MiB). The LFM engine's `forward_batch(n)` shows the cost when each token runs its own experts: 0.85–0.9 of a full token per extra position on the MoEs ([LFM25_ROADMAP.md](LFM25_ROADMAP.md), Task 5.1).
* **H20 (DFlash on CPU) was blocked** because the CPU miss pool leaves no spare CPU. That assumed a CPU drafter. A DFlash2 drafter for exactly this target now exists and is small enough to run on the GPU (Step 4).

**The drafter for this target**: [incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) (Apache-2.0, [blog](https://inco.ai/blog/dflash2/), [z-lab/dflash](https://github.com/z-lab/dflash)), 1.05 GB bf16:

| field | value |
|---|---|
| layers | 6, all sliding-window attention (window 2048), non-causal inside the block |
| hidden / MLP | 2048 / 6144, 32 query heads, 8 KV heads × 128 |
| target taps | hidden states of target layers 1, 6, 11, 16, 22, 27, 32, 37 (`num_target_layers` 40) |
| block | 8 = 1 anchor + 7 mask tokens (`mask_token_id` 248077); 7 drafts per pass |
| selector | top-16 candidates per position, rank 256; dynamic convs, group 16, kernel 2 |
| vocab | 248,320 (scores candidates with the target's own lm_head) |

Mean accepted tokens per step reported on the model card (GB300, sampling T=1.0, thinking on): GSM8K 5.21 (MTP 4.92), HumanEval 4.90 (4.44), MBPP 4.76 (4.36), MT-Bench 4.41 (4.26). There is also an older DFlash v1 drafter, [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash), with GGUF conversions, e.g. [abhinand/Qwen3.6-35B-A3B-DFlash-GGUF](https://huggingface.co/abhinand/Qwen3.6-35B-A3B-DFlash-GGUF).

What HyperQwen measured for DFlash2 on Qwen3.8-27B (dense, 3090, [optimizations.md § DFlash2](https://github.com/syv-ai/HyperQwen/blob/main/docs/optimizations.md#dflash2-specdflash2)): the bf16 drafter was a wash (106/112 tok/s against MTP's 111–124). GPTQ int4 made it a win: **118 tok/s at default sampling, 126 greedy**, 3.14–3.34 tokens per step against MTP's 2.8–2.9. Int4 cost **nothing in acceptance at greedy** and ~5% at sampling. Tuff is greedy.

---

## 2. Principles (inherited, not relaxed)

Everything in [README.md § Methodology](README.md#methodology) applies, plus:

1. **Lossless or it does not land.** Speculative output must be token-identical to non-speculative greedy, as in `tests/verify_lfm25_spec.sh` for LFM.
2. **Qwen has no byte-identity golden yet, and its math depends on cache contents.** A miss runs `expert_mat_vec_cpu` with q8_K activations, a hit runs GPU MMVQ with q8_1 (noted under LFM Task 6.2; confirm with a forced-hit vs forced-miss logit dump). A wider verify changes which experts hit or miss, so it changes logits even when it is correct. Step 0 fixes this before anything else.
3. **Greedy is the reproducible signal.** HyperQwen found single runs of speculative greedy differ ±3–5% in tokens per step across configs, because verify batches of different widths round differently. Repeat before believing a 2% difference ([drafter/README.md § Notes that cost time](https://github.com/syv-ai/HyperQwen/blob/main/drafter/README.md#notes-that-cost-time)). In tuff, the N-column kernels must match the single-column kernels bit for bit, as the LFM engine already does for B = 2–8.
4. **Measure paired and order-balanced** (`scripts/ab_interleaved.sh`, `scripts/marginal_ab.sh`), on AC with `scripts/check_power_state.sh` passing. Report acceptance (tokens per step) as well as tok/s, per prompt class: copy-heavy, code, prose.
5. **Estimates in this file are estimates.** Every number marked *(est.)* is arithmetic from bus figures, not a measurement. It must be replaced by a measured number before it is quoted anywhere else.

---

## 3. Phased plan

```
[Q0: Qwen golden + byte-identical CPU misses]
        │
        ▼
[Q1: N-token verify, experts once per step] ──► [Q2: lookup-drafter upgrades] ──► [Q3: adaptive long block]
        │                                                                                  │
        ▼                                                                                  ▼
[Q4: DFlash2 drafter on the GPU] ◄──────────────────────────────────────────── (Q3 fuses into Q4's block)
        │
        ▼
[Q5: draft vocabulary for drafter heads] ──► [Q6: fp16 delta state] ──► [Q7: split-KV verify attention]
```

Q2 can be done before Q1 if a quick win is wanted, but most of its value only shows once the verify is wider than 2.

---

### Phase Q0: Correctness infrastructure for the Qwen engine
**Objective**: make cache behaviour numerically invisible on Qwen, so the wider verify in Q1 can be gated byte-identical, as on LFM2.5.

- [ ] **Task Q0.1: Qwen golden with a frozen cache.** Record 128 greedy tokens plus per-step logit-row hashes with `TUFF_FILL_BUDGET=0`, in the style of `tests/verify_lfm25_golden.sh` and `tests/golden/*.json`. This catches unintended changes while Q0.2 is in progress.
- [ ] **Task Q0.2: Port the LFM Task 6.1 byte-identical CPU miss path to Qwen.** The CPU computes a missed expert's gate/up and down row dots with the GPU's exact lane math: Q8_1 activations, dp4a emulation, the SASS's fmaf contractions, xor-butterfly reduction, `-ffp-contract=off`. Reference: `src/lfm25_cpu_expert.c`, [LFM25_ROADMAP.md Task 6.1](LFM25_ROADMAP.md). Qwen's experts are Q4_K gate/up and Q4_K/Q6_K down (Q6_K on layers 34/38/39), so both formats need a CPU path that matches `vec_dot_q4_K_q8_1` / `vec_dot_q6_K_q8_1`.
  - Gate: forced-hit and forced-miss logits byte-identical on all miss masks (as `lfm25-cpu-expert-test`); the Qwen golden then becomes cache-independent and is re-recorded with `--reason`.
  - Kill: if the Q4_K/Q6_K CPU path cannot match the GPU bit for bit at acceptable speed, fall back to gating Q1 on `scripts/verify_tolerance.sh` + `scripts/oracle_check.sh` and token-identity of spec vs non-spec output.
- Negative controls: CPU reduction order swapped; Q6_K down read as Q4_K (both must fail the golden).

---

### Phase Q1: N-token verify that runs each expert once per step
**Objective**: verify 1 + k tokens (k ≤ 7) in one trunk pass, sharing the dense weights and loading each distinct expert once for every token that routed to it. This is the prerequisite for every drafter in this file, and it pays off for the existing n-gram drafter on its own.

**Why**: about 1.92 of the ~2.5 GB a token reads is dense and already shared by a batch. The rest is experts. For 8 tokens with top-8 of 256 experts, the union per layer is somewhere between 8 (full overlap) and ~57 (independent routing); correlated routing should put it nearer the low end *(est. ~35)*.

- [ ] **Task Q1.1: Generalise `engine_forward_verify` from 2 columns to N.** Dense projections become N-column MMVQ (same per-column lane order as the single-token kernel, so each column is byte-identical); the 30 delta layers process the N tokens sequentially with a state snapshot per position; the 10 attention layers run N queries with a causal mask over the new positions. Reuse the design of LFM Task 5.1 (`forward_batch(n)` + O(1) rollback), [LFM25_ROADMAP.md Phase 5](LFM25_ROADMAP.md).
  - Gate: for every n in 2..8, each column's logits byte-identical to n separate single-token forwards; `compute-sanitizer` memcheck and racecheck clean.
- [ ] **Task Q1.2: Expert union per layer.** After the router, build the union of the N tokens' top-8 ids. Each distinct expert is read once: on a hit, one GPU kernel runs it for all tokens that selected it; on a miss, one CPU task reads it once from the mmap and computes all its tokens' rows. Per-(token, expert) accumulation order stays the router-rank order, so the result is byte-identical to per-token verify. This is LFM Task 5.3's "MoE verify" item; implement it there first and port it.
  - Gate: byte-identical to Q1.1's per-token verify; spec output token-identical to non-spec greedy on the 5-prompt corpus and `scripts/bench_ngram_suite.sh`.
  - Measure: verify cost `forward_verify(n) / (n × single)` for n = 2, 4, 8, per prompt class, together with the union size and hit rate per layer. *(est.: an 8-token verify at ~1.6–2× one token, the growth coming mostly from CPU misses, ~90 → ~400 per step.)* The CPU miss pool is the likely bottleneck: it already takes ~4 ms of a ~17 ms token.
- [ ] **Task Q1.3: Adaptive k for Qwen.** Port LFM's adaptive-k rule (maximise expected (1 + p + … + p^k) / cost(k+1) against one token, per-match-length acceptance EMAs, a loss guard), with Qwen's measured cost table from Q1.2.
- [ ] **Task Q1.4: Retire the n2-only paths** (`verify_moe_stage_n2`, the 2-column graphs) once Q1.1–Q1.3 match them at n = 2.
- Negative controls: batch attention sees the next token; rollback off by one; accept one draft too many.
- Kill: if Q1.2's measured 8-token verify cost is above ~3× one token, no drafter in this file can pay; stop and re-plan around the miss path (LFM Task 6.2's probability-driven cache, prefetch by predicted tokens).

---

### Phase Q2: Lookup-drafter upgrades (cheap, no model)
**Objective**: make the free n-gram drafter propose more and better tokens. Source: HyperQwen's LABD kernel, [`dflash2-lookup-drafting.patch`](https://github.com/syv-ai/HyperQwen/blob/main/patches/dflash2-lookup-drafting.patch), and [optimizations.md § Drafting from the context](https://github.com/syv-ai/HyperQwen/blob/main/docs/optimizations.md#drafting-from-the-context-lookup1).

Today `engine_ngram_draft_multi` ([`src/engine.c`](src/engine.c)) tries suffix lengths 3 → 2, takes the most recent match, requires the match to end before the suffix starts, and caps history at 65,536 tokens with an O(n) memmove.

- [ ] **Task Q2.1: Longest match, recency tie-break.** Search for the longest suffix match up to a cap (HyperQwen: 12; a longer cap made it prefer an older long match over a newer short one, and the newer one predicted better: 3.21 vs 2.69 tokens per step on quote-and-explain work), with a minimum length (`NMIN`) checked first so most candidates are rejected before the extension loop.
- [ ] **Task Q2.2: Allow the match to overlap its suffix.** A period-p repetition matches at end = t − p. That is how repeated list markers, indentation and code fences get proposed. Tuff currently excludes overlapping matches.
- [ ] **Task Q2.3: Draft up to k tokens from the match** (needs Q1), and return the match length so Q1.3's per-length acceptance EMAs and Q3's trigger can use it.
- [ ] **Task Q2.4: History buffer.** Replace the memmove-on-full history with a ring buffer or a suffix index (LFM Task 5.3 "better drafts"); make the search cost independent of history length if it shows up in the host tail.
- Gate: spec output token-identical to non-spec greedy; drafter cost per step measured (must stay well under 0.1 ms).
- Measure on `scripts/bench_ngram_suite.sh` plus a copy-heavy prompt (reproduce a document verbatim, apply an edit to code).

---

### Phase Q3: Long verify block only while the model is copying
**Objective**: verify a long block (e.g. 16) when the lookup is reproducing context, and the normal block otherwise. HyperQwen measured **+47% where the model reproduces its context** (14.97 tokens per step, 381 tok/s vs 260 with the short block) and +2–3% elsewhere ([optimizations.md table](https://github.com/syv-ai/HyperQwen/blob/main/docs/optimizations.md#drafting-from-the-context-lookup1)).

- [ ] **Task Q3.1: Trigger.** Ask for the long block only when (a) the lookup has a match with enough tokens left to fill the tail, and (b) the previous step accepted a full short block, **twice in a row**. Match length alone is a poor predictor: HyperQwen needed a threshold of 22 before it stopped losing on prose, by which point most of the copy win was gone.
- [ ] **Task Q3.2: Sticky hold.** Keep the long block for a few steps after the trigger drops. Without it, HyperQwen saw a copy run fall from 14.97 to 13.9 tokens per step on repeats, because the flag drops out mid-copy for unrelated reasons; with it, 15.21 every time. Gating the hold on "still emitting a full block" removed the effect.
- [ ] **Task Q3.3: Fusion with a drafter** (after Q4). The drafter fills its own 7 positions, the lookup fills the rest at no drafter cost. A strong match (HyperQwen `NSTRONG` = 8 tokens) overrides the drafter; a shorter one only if the drafter independently proposed the same first tokens (`AGREE` = 2).
- Note: HyperQwen reads the trigger flags one step stale from a pinned async copy, because a synchronous read cost 5%. Tuff's host already sees accept counts synchronously, so this is free here.
- Gate: token-identity; long-block cost per extra position measured at 4k and 32k context (10 attention layers, 2 KV heads; see Q7).

---

### Phase Q4: DFlash2 drafter on the GPU
**Objective**: run [incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) inside tuff, int4 on the GPU, drafting 7 tokens per step for the Q1 verify. Unblocks PLAN.md lever 2 / H20 by moving the drafter off the CPU.

- [ ] **Task Q4.1: Format and loader.** Convert the safetensors drafter to a GGUF tuff can mmap (or load safetensors directly), keeping the `dflash_config` (block size, mask token, target layer ids, selector rank/top-k, conv group/kernel). Check the existing [DFlash v1 GGUF conversions](https://huggingface.co/abhinand/Qwen3.6-35B-A3B-DFlash-GGUF) and llama.cpp's DFlash support, if any, for tensor naming.
- [ ] **Task Q4.2: Reference implementation and oracle.** A Python/FP32 reference of one drafter step (from [z-lab/dflash](https://github.com/z-lab/dflash) and vLLM PR [#52816](https://github.com/vllm-project/vllm/pull/52816)), fed with hidden states dumped from tuff. Gate the CUDA drafter against it: same candidates and same selected path on recorded steps.
- [ ] **Task Q4.3: Target hidden-state taps.** Copy the residual stream after target layers 1, 6, 11, 16, 22, 27, 32, 37 into a device buffer during the trunk pass (8 × 2048 floats per token). Accepted positions append to the drafter's context; rejected ones are dropped with the rollback.
- [ ] **Task Q4.4: Drafter kernels.** `fc` projection of the concatenated taps (8 × 2048 → 2048), 6 sliding-window layers over a 2048-token context plus an 8-token non-causal block, dynamic two-tap convolutions that reset at the block boundary, then candidate scoring and the selector walk. All batch-8 work: a small GEMM rather than GEMV, so tensor cores are allowed here (unlike the trunk). Capture the step in a CUDA graph.
- [ ] **Task Q4.5: Int4 requantization, calibrated.** GPTQ int4 group-128 on the 36-ish drafter matrices, Hessians from the drafter's **own** inputs captured while it drafts on real prompts. HyperQwen recipe: [`drafter/capture_dflash2.py`](https://github.com/syv-ai/HyperQwen/blob/main/drafter/capture_dflash2.py), [`drafter/quant_dflash2.py`](https://github.com/syv-ai/HyperQwen/blob/main/drafter/quant_dflash2.py), [`drafter/gptq_utils.py`](https://github.com/syv-ai/HyperQwen/blob/main/drafter/gptq_utils.py). Their findings: int4 kept greedy acceptance; keeping `fc` in bf16 bought nothing; blending the context-KV inputs into the k/v Hessians was **7% worse** and nearly shipped because a single sampling run looked equal. Target size *(est.)* ~0.3 GB.
- [ ] **Task Q4.6: VRAM budget.** Free margin today is ~344 MiB, and H27 showed that eating into it causes driver stalls (−6.4%). The drafter weights, its KV (6 layers × 8 KV heads × 128 × 2048 window) and scratch come out of the expert pool: ~150–200 slots of 2500. H18 found the hit rate flat from 2500 to 2600 slots; measure 2300–2350 before committing.
- Gate: token-identity of spec vs non-spec greedy on the corpus; drafter step time measured; acceptance (tokens per step) per prompt class.
- Measure against: base, n-gram (Q2), n-gram + long block (Q3). *(est.: 3.5–4.5 tokens per step at a ~2× step cost, so ~1.3–1.6× base if Q1's verify cost holds.)*
- Kill: drafter step > ~3 ms, or net gain under the n-gram path on the corpus after an order-balanced A/B.

---

### Phase Q5: Draft vocabulary for drafter heads
**Objective**: stop drafters from reading the full 248k-row output head. HyperQwen's single biggest single-stream lever ([drafter/README.md § What actually mattered](https://github.com/syv-ai/HyperQwen/blob/main/drafter/README.md#what-actually-mattered-in-order)).

- [ ] **Task Q5.1: Count the vocabulary over the model's own outputs.** Generate a few million tokens with tuff on mixed prompts (chat, code, reasoning, the user's languages) and keep the top ~40k ids. HyperQwen: a list counted over the model's outputs covered **97.5%** of generated tokens (96% on code) against 92% (83% on code) for a web-text list, worth **10% of single-stream throughput** on its own. Coverage barely improved past 40k (the model emits ~54k distinct tokens). Tools: [`prepare/build_draft_vocab.py`](https://github.com/syv-ai/HyperQwen/blob/main/prepare/build_draft_vocab.py), [`drafter/gen_data.py`](https://github.com/syv-ai/HyperQwen/blob/main/drafter/gen_data.py).
- [ ] **Task Q5.2: Sliced head for DFlash2 candidates.** Score candidates on the 40k-row slice of the Q6_K head (~64 MB instead of 398 MB). A token outside the slice can never be drafted, so coverage directly caps acceptance; the verify still uses the full head, so output is unchanged.
- [ ] **Task Q5.3: Re-test MTP with the sliced head.** The MTP head (`blk.40`) currently pays the full head per draft and loses 12.7%. With Q1 + Q5 it may change sign; HyperQwen took each MTP draft from ~3 ms to ~0.5–1 ms with the slice plus a requantized MTP module. Note that Qwen3.6's MTP layer contains its own MoE, which a sliced head does not help.
- Gate: token-identity (verify unchanged); coverage reported per prompt class.

---

### Phase Q6: fp16 delta-net state
**Objective**: halve the recurrent-state traffic. The state is f32 (`delta_state` in [`include/engine.h`](include/engine.h), 32 v-heads × 128 × 128 per layer ≈ 2 MiB × 30 layers ≈ 63 MB), read and written every token: ~126 MB of a ~2.5 GB token, *(est.)* ~2%. With Q1, every verify position needs a state snapshot, so the saving grows with k.

- [ ] **Task Q6.1:** fp16 storage with f32 arithmetic in `fused_gated_delta_net_ar_kernel`. HyperQwen's `--mamba-ssm-cache-dtype float16`: perplexity unchanged to three decimals; they chose fp16 over bf16 for the 10 mantissa bits ([optimizations.md item 3](https://github.com/syv-ai/HyperQwen/blob/main/docs/optimizations.md#in-full)).
- Gate: this changes numerics, so oracle + tolerance (`scripts/oracle_check.sh`, `scripts/verify_tolerance.sh`) and perplexity (`scripts/compute_perplexity.py`), then re-record the golden with `--reason`. Behind a flag until it passes.
- Kill: oracle N drops, or < 1% order-balanced.

---

### Phase Q7: Split-KV attention for multi-query verify
**Objective**: keep the 10 attention layers fast when they get k+1 queries at long context. HyperQwen: FlashAttention-2 splits the KV sequence only for one query token, so their 5-query verify ran on 24 of 82 SMs (1.3 ms per layer at 16k); a split-KV Triton kernel cut it to 120 µs ([`spec-decode-attn.patch`](https://github.com/syv-ai/HyperQwen/blob/main/patches/spec-decode-attn.patch)). Tuff's Qwen has only **2 KV heads**, so parallelising over heads alone leaves almost every SM idle.

- [ ] **Task Q7.1:** Check the Q1.1 verify attention kernel's grid at 4k / 16k / 32k (Nsight); if it does not split KV, add a split-KV pass with a combine step (fixed combine order, so it stays byte-identical across runs).
- [ ] **Task Q7.2:** Coordinate with PLAN.md lever 1 (KV quantization, H19): the verify kernel must read whatever KV format lands there.
- Gate: byte-identical columns vs single-query attention; memcheck clean.

---

## 4. What not to borrow (and why)

| HyperQwen item | why not here |
|---|---|
| int8-activation Marlin (W4A8) GEMMs | compute-bound batched decode and prefill only; tuff's batch-1 trunk is bandwidth-bound and avoids MMA by design (PLAN.md) |
| hybrid prefix caching, KV-group sizing, V2 CUDA-graph memory accounting | vLLM server/pool mechanics; relevant only if `src/server.c` grows multi-turn KV reuse |
| fine-tuning the MTP head | measured no gain in HyperQwen (top-1 on response tokens unchanged) |
| Marlin tile tuning | washed out end-to-end under a power cap there; tuff does not use Marlin |
| int8-QK prefill attention | specific to their head geometry and to batched prefill; revisit with LFM Phase 7 prefill work |

---

## 5. Sign-off checklist (per phase)

1. Spec output token-identical to non-spec greedy on the 5-prompt corpus and the n-gram suite.
2. Golden byte-identical (from Q0 on), graph on and off, forced-cache configs included.
3. `compute-sanitizer` memcheck clean on a run with real drafts and rejections; racecheck on new kernels.
4. Negative controls for the phase added and caught.
5. Order-balanced A/B against the previous phase, with tokens per step reported per prompt class; one entry in [ITERATION_LOG.md](ITERATION_LOG.md) and the PLAN.md ledger updated (H20 in particular).
