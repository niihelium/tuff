# Qwen3.5 0.8B / 2B decode megakernel in tuf — report

Status: **Phase 0 complete, Phase 1 correct and integrated, P1 performance target not met.**
Phases 2–4 were not started, because the brief orders the phases and forbids starting one before the
previous phase's gates pass; P1 is a Phase 1 gate and is below target (section P1). Everything below
was measured on the RTX 3070 Laptop (`sm_86`, 40 SMs) with the machine in `performance` profile;
`scripts/check_power_state.sh` exits 0.

---

## Phase 0 — scaffolding and baselines

### 0.1 Build

`CMakeLists.txt` now has one `LLAMA_CPP_DIR` cache variable (default `/home/nii/src/llama.cpp`) used
for headers, link directory and all RPATHs. The old `/home/nii/workspace/llama.cpp` paths were dead.
Two further build fixes were required and are additive:

* libllama's own `RUNPATH` is `$ORIGIN/../lib`, which does not exist in this checkout. `tuff`,
  `tuff-server` and the oracle tools now link with `-Wl,--no-as-needed` and name every ggml soname
  directly, so the loader resolves them from the executable's RPATH before libllama's broken
  `RUNPATH` is consulted. `build/tuff --help` and `build/tuff-server --help` both exit 0.
* `src/tokenizer.c` required `tokenizer.ggml.bos_token_id`. Qwen3.5 has no BOS (GPT-2 BPE); the
  tokenizer now requires EOS only, and forces `add_bos=false` when BOS is absent. Models that carry
  a BOS are unchanged: `build/qwen-tokenizer-test` still passes, including the Qwen3.6-35B
  tokenizer fixture check.

### 0.2 Bandwidth roofline (`tools/bw_sm86.cu`)

Ported from `/home/nii/src/megakernels/common/bw.cu` (MIT): plain `__ldcs` loads, a `cp.async`
16-byte ring with `cp.async.mbarrier.arrive.noinc`, optional `prefetch.global.L2`, `evict_first`
policy. Built as `build/bw-sm86`.

```
GPU: SMs=40 L2=4MB smem/block optin=99KB regs/SM=65536
spec peak = 384.1 GB/s (6001 MHz, 256-bit)

ld.global 40 x 512, no prefetch : 371.2 GB/s
ld.global 40 x 512, prefetch    : 373.3 GB/s
cp.async ring 3x16384, 544 thr  : 372.2 GB/s
cp.async ring 4x16384, 544 thr  : 361.2 GB/s
cp.async ring 5x15232, 544 thr  : 354.7 GB/s
BEST                            : 373.4 GB/s  (97.2% of spec)
```

**Roofline used for every number below: 373.4 GB/s.** Each benchmark session was bracketed by
`bw-sm86`; the machine was re-checked and re-run whenever the best value moved by more than 3%.

### 0.3 Independent fp32 reference (`tools/qwen35_ref.py`)

Reads the same GGUF, dequantizes Q8_0 exactly (interleaved `{fp16 d; int8 q[32]}` blocks), keeps
every norm tensor as stored, and runs a teacher-forced forward with numpy only (no torch, no
llama.cpp). Two GGUF facts were verified against llama.cpp's converter and graph before writing it:

* `conversion/qwen.py` adds `1.0` to every `*.norm.weight` **except** `linear_attn.norm.weight`.
  GGUF norms are therefore used directly; adding another `+1` (the HF convention) is a bug.
* `ssm_a` is stored as `-exp(A_log)`; decay is `exp(ssm_a * softplus(alpha + dt_bias))`.
* `attn_q` rows are `[q(256) | gate(256)]` per head; GDN `attn_qkv` is `q(2048)|k(2048)|v(2048)`;
  `attn_gate` is `z`; `conv1d` is `[conv_dim][4]` with chronological taps; attention is partial
  NEOX RoPE over the first 64 dims (`rope.dimension_sections` only matters for M-RoPE positions;
  text positions are all equal).

Validation against llama.cpp, same token list, Q8_0 weights:

| model | positions | top-1 | max abs dlogit | p99 |
|---|---:|---:|---:|---:|
| 0.8B | 13 (chat "Hello") | 13/13 | 0.95 | 0.30 |
| 2B | 13 (chat "Hello") | 13/13 | 1.1 | 0.34 |

The residual difference is llama.cpp's Q8_1 **activation** quantization in its mmvq path versus the
reference's exact fp32 dequantized dot; the same difference appears between llama.cpp's CUDA and CPU
backends on the same weights. On the BF16 GGUF (identical weights, no Q8 activations) the reference
matches position 0 to `5e-5`.

A real bug was found and fixed here: the first numpy `rmsnorm` used `np.mean(x*x)` over the whole
2-D tensor instead of `axis=-1`. After the fix the reference is the oracle used by G1.

### 0.4 Oracle gate (`tests/qwen35/oracle.sh MODEL [N]`)

Prompts: `"Hello"` + the 5 `prompts_corpus.json` prompts, chat-templated with thinking off:

```
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

Tokenized by `llama-tokenize` from the same GGUF. `qwen35-teacher` generates the tuf greedy stream;
`build/qwen35-llama-ref` generates llama.cpp's. An exact prefix of 32 tokens is required. A divergence
is accepted only when llama.cpp's own top-2 margin at the predicting position is below **0.15**.

**Why 0.15.** The kernel dequantizes and accumulates in fp32; llama.cpp quantizes activations to
Q8_1. Measured on the same corpus, the exact-fp32 reference vs llama.cpp has max `|dlogit|` 0.95,
p99 0.30 and std 0.03–0.09. A margin below 0.15 is inside that noise. This is deliberately permissive
in the same way the LFM oracle allows near-ties; the kernel is *more* accurate than llama.cpp, not
less, and G1 (kernel vs fp32 reference) is the strict numeric gate.

Result:

* 0.8B: 5/6 prompts exact 32/32; `fastapi_backend` diverges at token 15 with llama margin **0.0935**
  (near-tie allowed). `oracle: PASS`.
* 2B: 6/6 prompts exact 32/32. `oracle: PASS`.

---

## Phase 1 — batch-1 megakernel

### What was built

| file | lines | purpose |
|---|---:|---|
| `src/qwen35/qwen35_mk.cu` | 1111 | templated persistent decode kernel (`<1024,3584>` / `<2048,6144>`) |
| `src/qwen35/qwen35_engine.c` | 659 | GGUF load, Q8_0 repack, state/VRAM, launch API |
| `include/qwen35_kernel.h` | 117 | C ABI shared by engine and kernel |
| `include/qwen35.h` | 44 | public engine API |
| `tools/qwen35_ref.py` | 377 | fp32 GGUF reference decoder |
| `tools/qwen35_teacher.c` | 108 | teacher-forced / greedy kernel driver (G1, G3, oracle) |
| `tools/qwen35_llama_ref.c` | 115 | llama.cpp teacher-forced / greedy oracle |
| `tools/bw_sm86.cu` | 191 | sm_86 bandwidth roofline |
| `tests/qwen35/oracle.sh` | 125 | G2 |
| `tests/qwen35/g1.sh` | 101 | G1 |

### Design

* **One cooperative CTA per SM, 17 warps** (16 consumer + 1 producer), `__launch_bounds__(544,1)`.
  Grid is exactly 40; cooperative launch. Dynamic smem is 82.6 KB (0.8B) / 90.9 KB (2B), under the
  99 KB opt-in limit.
* **Weight format.** The engine repacks each GGUF Q8_0 row into `K` bytes of `q+128` followed by
  `K/32` **fp16 scales copied bit-for-bit**. Rows stay one contiguous 16-byte-aligned span.
  `dot_row_q8` dequantizes four bytes per `PRMT` + one `FADD` and accumulates fp32 with explicit
  `fmaf`. The LM head is the tied `token_embd`.
* **Norm convention.** Every GGUF norm is used directly (`w`, not `1+w`), matching llama.cpp's
  converted weights. `ssm_norm` is direct as well.
* **sm_86 ring.** The TMA producer is replaced by a 32-lane `cp.async` 16 B ring with
  `L2::cache_hint`/`evict_first`, `prefetch.global.L2` 32 KB ahead, and an sm_86-legal completion
  protocol: `mbarrier` (initialised count 32, `cp.async.mbarrier.arrive.noinc`) for "full", and
  per-slot shared sequence counters for "empty". `mbarrier.test_wait.parity` is used because
  `try_wait` is sm_90+ in CUDA 12.4. Four 16 KB stages.
* **No grid barriers.** Activations cross CTAs as `(epoch<<32)|fp32` words with
  `st.relaxed.gpu`/`ld.relaxed.gpu`; consumers spin on the epoch they need. Fixed-order reductions,
  no float atomics.
* **GDN:** 64 tasks `(head, 32 value columns)`, state `[head][v][k]` fp32, 16 warps × 2 columns,
  causal conv + L2 norm + delta rule + gated RMS norm in one op.
* **Attention:** split-K over cached positions (`MAX_SPLITS=42`), 4 query heads per KV head, f16
  KV cache, online softmax, deterministic partial merge.
* **Shapes:** HID/INTER are template parameters; the host dispatches once at load. There are no
  per-token shape branches.
* **Prefill:** the same kernel processes up to 16 consecutive token entries in one launch
  (`--chunk`), with the LM head only on the last position unless logits are requested.

### G1 — kernel vs `qwen35_ref.py`, teacher-forced (5 prompts; N=32 all, N=256 first 3)

Command: `bash tests/qwen35/g1.sh MODEL 32 OUTDIR` (and `... 256 ...` for the full-length run).

Result, clean build, all 5 corpus prompts (N=32 teacher-forced positions each, which is the
`tests/qwen35/g1.sh` default when a shorter run is needed for time):

```
PASS casimir_surrogate_solver  n=  95 argmax_mismatch=0 max|dlogit|=0.00277 p99=0.00045
PASS fastapi_backend           n=  83 argmax_mismatch=0 max|dlogit|=0.00149 p99=0.00036
PASS casimir_theory            n=  84 argmax_mismatch=0 max|dlogit|=0.00318 p99=0.00042
PASS armenia_history           n=  90 argmax_mismatch=0 max|dlogit|=0.00108 p99=0.00027
PASS dorian_gray_analysis      n=  86 argmax_mismatch=0 max|dlogit|=0.00186 p99=0.00046
G1: PASS (0 failures)
```

The full **N=256** run was also completed for the first three prompts before the session budget
ran out (the fp32 2B reference is the long pole): `casimir_surrogate_solver` 319 positions,
`fastapi_backend` 307, `casimir_theory` 308, **0 argmax mismatches, max |dlogit| 0.00318**. The
remaining two prompts were interrupted while their references were running (not a gate failure);
the N=32 table above covers all five.

2B on the 13-token chat prompt: argmax 13/13, max `|dlogit|` = **0.0010**, p99 0.00026. The kernel
and reference agree because both use the same GGUF weights, the same direct-norm convention and the
same fp32 accumulation order per row; the only differences are f16 KV and the split-K merge.

### G2 — oracle

`tests/qwen35/oracle.sh` passes for both models (section 0.4).

### G3 — determinism

For both models, `qwen35-teacher --chunk 1`, `--chunk 7`, `--chunk 16` and a repeated run produce
**bitwise identical** logits (SHA-256 of the raw file equal). This covers "one token per launch vs
all in one launch" and run-to-run determinism.

### P1 — performance (does not meet the target)

Protocol: `tuff --prompt ... -c 512 -n 128 --warmup 32 --runs 3`, performance profile, bracketed by
`bw-sm86` (373.4 GB/s).

| model | tok/s | target | % of target | effective GB/s | % of roofline |
|---|---:|---:|---:|---:|---:|
| 0.8B Q8_0 | **347.6 ± 0.3** | 380 | 91% | 296 | 79% |
| 2B Q8_0 | **159.2 ± 0.2** | 165 | 96% | 320 | 86% |

Controlled A/B (3 alternations, `bw-sm86` 373.4 GB/s before and after, clocks 1800/6001 MHz):
0.8B ctx512 250.1, 250.6, 250.2 vs ctx4096 250.3, 249.4, 250.0; 2B ctx512 142.1, 141.7 vs ctx4096
142.4, 142.4. One earlier 0.8B/2B run showed 346.7/158.8 tok/s at ctx4096; it did not reproduce in
the controlled A/B and is reported as an outlier (transient boost), not a result.

A **filled** context, measured with `--prompt-file` (3907-token / 16058-token prompts, one run):

| model | 4K filled | 16K filled |
|---|---:|---:|
| 0.8B | 302.4 tok/s | 253.5 tok/s |
| 2B | 148.3 tok/s | 136.5 tok/s |

(llama.cpp on the same machine: 229 / 130 tok/s at concurrency 1, from the brief.)

Per-op timeline (0.8B, one decode step, block-0 op boundaries, `Q35_PROF=1`):

| op | ms | bytes | GB/s |
|---|---:|---:|---:|
| in-projection + LM head | 1.36 (0.62 + 0.75) | 194 + 270 MB | 313 / 360 |
| GDN / attention core | 1.34 | state 38 MB | latency-bound |
| out-projection | 0.17 | 53 MB | 312 |
| gate/up | 0.62 | 187 MB | 302 |
| down | 0.29 | 94 MB | 324 |
| **total** | **3.79** | | |

The weight GEMVs alone run at ~300–360 GB/s (80–96% of the roofline). The step is limited by the
**GDN/attention critical path (1.34 ms, 35%)**, which is latency, not bandwidth: 64 GDN tasks over
40 CTAs leave 24 CTAs idle in the second wave, and every task waits on projection rows produced by
other CTAs. The 2B is closer to target because its weight stream is 2.5x larger and amortizes that
constant.

What was tried: 3/4/5 ring stages, 16 KB and 24.5 KB chunks, a per-row split-K GEMV (`SPLIT` up to
8 warps/row, chosen at compile time so the 16 warps stay busy), and a hybrid mbarrier/counter ring.
The split-K GEMV raised the GEMVs to the numbers above but did not move the core. The remaining
work for 380 tok/s is (a) a GDN/attention decomposition with better than 80% CTA occupancy and
(b) overlapping the core with the next layer's weight stream, neither of which is in this diff.

VRAM (measured with `-c 512`, one slot): 0.8B 1.16 GB; 2B 2.32 GB.
32 slots × 16K for 0.8B: weights 0.81 GB + unified KV 6.44 GB + GDN state/conv 0.72 GB ≈ 7.97 GB
(fits 8 GB). 16 slots × 16K for 2B: 2.01 + 3.22 + 0.36 ≈ 5.59 GB.

---

### Final gate summary (clean build)

```
tuff --help: 0            tuff-server --help: 0
power: RESULT: OK — full PERFORMANCE mode
bw-sm86 BEST: 373.4 GB/s
oracle 0.8B: PASS (0 failures, N=32, near_tie=0.15)
oracle 2B:   PASS (0 failures, N=32, near_tie=0.15)
G3 0.8B chunk1==chunk16: 0 (cmp equal)
G3 2B   chunk1==chunk16: 0 (cmp equal)
G1 0.8B N=256: PASS (5/5, 0 argmax mismatches, max|dlogit| 0.00318)
G1 2B   N=128: PASS (5/5, 0 argmax mismatches, max|dlogit| 0.00212)
G3 after the RoPE fix: chunk1 == chunk16 (cmp 0) both models
P1 0.8B ctx512: 347.58 tok/s   P1 2B ctx512: 159.17 tok/s
G4 batch (2 slots): NOT bitwise; max|dlogit| 6e-6 (step 0) .. 6e-4 (step 2)
```

## Phases 2–4 — status at the time of the original report

**Superseded by Addendum 1: Phase 2's batched decode kernel is now implemented (with the G4 and
scaling gaps documented there); Phases 3 and 4 remain unimplemented.** The text below records the
original Phase 1 stopping point.

Per the brief's ordering rule, Phase 2 (token-batch slots, continuous batching, chunked prefill),
Phase 3 (GPU sampling/logprobs) and Phase 4 (server backends, templates, tool parser) were not
started because P1 did not pass. The kernel ABI already carries the pieces Phase 1 needed
(`epoch`, multi-step launches, per-slot state allocation is centralised in `build_params`), but no
batching/tier instantiations, no sampler epilogue, no logprob output and no server backend exist.

Known gaps, explicitly:

* The batched tiers exist (`BT ∈ {2,4,8,16,32}`) but the T=1 kernel is still used for slot 1 and
  the batch path does not scale yet (Addendum 1); chunked prefill still uses the T=1 kernel per
  token, so the pp targets are not met.
* The GPU LM-head epilogue keeps only the global argmax; no top-K/online-softmax, no penalties,
  no logprob.
* `tuff-server` still uses the existing single-request path; `--parallel > 1` on backend A was not
  added, and qwen35 is not reachable from the server.
* No prompt-cache/GDN snapshots, no chat renderer, no tool-call parser, no `/tokenize` parity run.
* `sandbox/evals/batch_bench.py --engine tuf` was not added (outside `runtime/`, the one permitted
  sandbox file).

---

## What we found in llama.cpp (source notes)

* `qwen35` selects `LLAMA_ROPE_TYPE_IMROPE`, but with text-only positions the per-section angles are
  all `p`, so the cache is identical to NEOX partial RoPE (`rotate_pairs` with stride `n_dims/2`).
* `build_gdn_l2_norm` is `rms_norm(x, eps/n) * 1/sqrt(n)`, which simplifies to
  `x / sqrt(sum(x^2) + eps)`; the `1/sqrt(n)` is **not** an extra factor (the autoregressive graph
  then scales q by `1/sqrt(S_k)` separately).
* The GDN fused CPU kernel applies the decay **before** the delta update and writes the state
  transposed (`M[j][i] = S[i][j]`); the output uses the updated state.
* `build_conv_state_store` stores the **last** `kernel-1` columns, so the conv window is
  chronological `[t-3, t-2, t-1, t]`.
* The converter's `+1` for norms and `-exp(A_log)` for `ssm_a` are the two traps for anyone reading
  the GGUF directly (both handled here).

---

---

## Addendum 1 (post-review)

### Diff fixes

1. `TUFF_ARCH_QWEN35` is now the **last** enum value (`TUFF_ARCH_LFM2MOE, TUFF_ARCH_QWEN35`), so
   the existing values are unchanged. The review guide below is corrected.
2. All dead `Q35_DUMP_*` hooks were deleted from `qwen35_engine.c` (and the corresponding unused
   ABI fields from `qwen35_kernel.h`).
3. `--prompt-file PATH` was added to `main.c` (additive; reads the file into a heap buffer). It
   allowed the filled-context table above. The Linux 128 KB argv limit is why the old report had
   only an allocation run.
4. G1 was finished: **0.8B 5/5 prompts at N=256** (max |dlogit| 0.00318, 0 p99 0.00042, 0 argmax
   mismatches) and **2B 5/5 at N=128** (max 0.00212, 0 mismatches). Exact rows in the logs.

### A real bug found while starting Phase 2: RoPE tables were host memory

`alloc_rope()` used `malloc` for the cos/sin tables and passed those host pointers to the kernel.
The driver silently did zero-copy pageable access, so the math was right (G1 passed) but every
attention op read the table over PCIe. Moving the tables to `cudaMalloc` raised B=1 from
**250.0 → 347.6 tok/s** (0.8B) and **142.1 → 159.2** (2B); the filled 4K/16K numbers above are
after the fix. This closes most of the P1 gap (91% / 96% of target; 79% / 86% of the 373.4 GB/s
roofline). It was not visible in the original report because the host reads were correct, just slow.

### Phase 2: batched decode kernel (implemented; throughput scaling does not meet P2)

New in `qwen35_mk.cu`: `decode_batch_kernel<HID,INTER,BT>` with `BT ∈ {2,4,8,16,32}`, and
`q35_engine_decode_batch()`. Design:

* One entry per slot per launch (decode entries only; chunked prefill still uses the T=1 kernel
  per token, so pp targets are not met).
* Per-slot state is one contiguous allocation per kind; the kernel uses
  `slot0_layer_ptr + slot*slot_stride` (18 GDN layers / 6 attention layers per slot).
* Each GEMV row is read once and dotted with all BT input vectors; the input vectors live in a
  per-CTA global scratch (`P.vecg`), because `BT*K*4` does not fit in 99 KB.
* GDN tasks = `BT*64` `(entry, head, value slice)`; attention tasks = `Σ_t NKV*splits(pos_t)`.
  With BT=2..32 the 40 SMs are filled by the core.
* The LM head runs once per launch for all entries, keeps a per-entry argmax and can write full
  fp32 logits (`A.logits_out`) for host sampling/logprobs.
* Per-token dot grouping is identical to the T=1 kernel (`((p0+p1)+p2)+p3`, same scale order) to
  aim for bitwise invariance.

**G4 status: not bitwise.** A slot's logits with 2–32 slots differ from its solo run by
`max|dlogit|` 6e-6 at step 0 growing to ~6e-4 by step 2 (0.8B, 2 slots). The first step already
matches to a few 1e-6; the residual comes from state updates accumulating a single-ULP difference
(the batch GEMV reads unswizzled global vectors and the compiler schedules the BT accumulators
differently). This is documented as a known gap, not claimed as a pass.

**P2 engine throughput** (`qwen35_batch_bench`, 128 steps, ctx 600, median of 3):

| B | 0.8B agg tok/s | 2B agg tok/s |
|---:|---:|---:|
| 1 | 346.8 | 154.7 |
| 2 | 156.0 | 65.5 |
| 4 | 166.5 | 68.1 |
| 8 | 162.7 | 64.7 |
| 16 | 134.0 | 55.8 |
| 32 | 103.1 | 52.6 |

The batch path is **slower than B=1** and does not scale. The cause is the GEMV tiling: each
weight row re-reads the whole BT×K activation vector, so activation traffic scales as
`BT × weight bytes` (from L2, but 32× the weight stream). The next step is a K-outer / row-tile
loop that loads each activation K-chunk once and applies it to several weight rows, or an int8
`mma.sync` path; neither is in this diff. The core task decomposition itself now has enough work
(`BT*64` tasks), so the core is no longer the only limiter.

### Phase 4 (partial): backend B in `src/server.c`

Backend B is now selected when the GGUF arch is `qwen35`; backend A's code path is untouched. The
server loads `q35_model`/`q35_engine`, keeps `server_t.arch/q35m/q35e`, and branches only inside
`handle_chat`/`handle_completion` plus four new endpoint routes. Implemented and smoke-tested:

* `GET /health`, `GET /v1/models`, `GET /props` (`n_ctx` per slot, `total_slots`).
* `POST /v1/chat/completions`, streaming and non-streaming, with the Qwen3.5 template for
  system/user/assistant/tool-role **text** messages and the thinking on/off generation prompt
  (`chat_template_kwargs.enable_thinking` is scanned in the request body). Greedy and temperature
  sampling (top-k/top-p/min-p are implemented over the top-64 candidates; not yet the full
  llama.cpp chain).
* `POST /v1/completions` (streaming and non-streaming).
* `POST /tokenize`, `POST /detokenize`, `POST /apply-template`.
* `-np/--parallel N`: allocates N engine slots and splits `-c` evenly (`-c 4096 -np 2` -> 2048/slot),
  but requests are **serialized under the existing mutex**; there is no continuous batching or
  launch-boundary scheduler yet.
* `logprobs` is still `null`; tool calls are not parsed from the model output; `tools` is not yet
  rendered into the prompt; grammar/JSON-schema rejection, `X-Tuf-Ignored`, and slot snapshots are
  absent. G8/G10/G11/G12 are therefore still open.

Smoke tests (0.8B, port 8099): `/health` ok; `/props` `{n_ctx:2048,total_slots:2}`;
`/tokenize "Hello"` -> `[9419]`; `/apply-template` -> the exact thinking-off chat prompt;
chat "Hello" -> "Hello! How can I help you today?" (non-streaming and SSE); `/v1/completions`
"Paris." for "The capital of France is".

### Phase 3

Not implemented: sampling is host-side over full fp32 logits (the batch LM head can write them),
not the GPU candidate epilogue, and there is no G6/G7 logprob test. The server's `logprobs` field
stays `null`.

---

## Review guide (existing files touched)

| file | hunks | why it cannot change other models |
|---|---|---|
| `CMakeLists.txt` | `LLAMA_CPP_DIR` cache var; `--no-as-needed` + explicit ggml sonames on targets that link llama; new qwen35 sources on `tuff`/`tuff-server`; new targets `bw-sm86`, `qwen35-teacher`, `qwen35-llama-ref` | no global compile/link flag changes; only new sources and the path fix. `--no-as-needed` only affects loader resolution, not code. |
| `include/tuff_vtable.h` | new `TUFF_ARCH_QWEN35` enum value at the **end** | append-only; the existing enum values are unchanged (the first version inserted it before `QWEN35MOE`; fixed). |
| `src/tuff_vtable.c` | qwen35 adapters + table entry; weak `tuff_logit_capture*` declarations | new case in `tuff_get_vtable`; the existing Qwen/LFM tables are untouched. Weak symbols let the same file link into `tuff-server`. |
| `src/main.c` | include `qwen35.h`; if `tuff_detect_arch` returns `UNKNOWN`, probe for `general.architecture = qwen35` and select the new arch | only runs on unknown architectures. Qwen3.6/LFM detection is unchanged. |
| `src/tokenizer.c` | missing BOS no longer fatal; `add_bos=false` when BOS is absent | every existing model GGUF has `bos_token_id`, so `bos_id`/`add_bos` and all token streams are unchanged; `qwen-tokenizer-test` passes. |
| `README.md`, `ATTRIBUTION.md` | status row / server limits / credit | documentation only. |

`src/engine.c`, `src/gpu.cu`, `src/gpu/*`, `src/lfm25*`, `src/moe_policy.c`, `src/cpu_ops.c`,
`src/drafter.c`, `src/sampler.c`, `src/qwen_cpu_expert.c`, `include/gpu.h`, `include/engine.h`,
existing tests/fixtures/goldens were **not** modified.

---

## Reproduction commands

```bash
cd runtime
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j16

# roofline
./build/bw-sm86

# G1 / G3
bash tests/qwen35/g1.sh /run/media/nii/pohjola/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 256 tests/qwen35/g1_08b
bash tests/qwen35/g1.sh /run/media/nii/pohjola/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 256 tests/qwen35/g1_2b

# G2
bash tests/qwen35/oracle.sh /run/media/nii/pohjola/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf 32
bash tests/qwen35/oracle.sh /run/media/nii/pohjola/models/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf 32

# P1
./build/tuff --model <gguf> --prompt "Hello" -c 512 -n 128 --warmup 32 --runs 3
```

---

## Phase 4 (server)

Backend B (Qwen3.5 0.8B / 2B dense) is now a real continuous-batching server; backend A
(the existing Qwen3.6-MoE engine) keeps its old code path. The work lives in new files:

| file | lines | purpose |
|---|---:|---|
| `include/q35_json.h`, `src/qwen35/q35_json.c` | 470 | strict JSON reader + minja-compatible writer (ordered objects, `", "`/`": "`, shortest float) |
| `include/q35_template.h`, `src/qwen35/q35_template.c` | 340 | Qwen3.5 `tokenizer.chat_template` renderer (system/tools, messages, tool calls/results, reasoning stripping, thinking on/off) |
| `include/q35_toolparse.h`, `src/qwen35/q35_toolparse.c` | 300 | `<think>` extraction + the qwen3-coder XML tool-call parser |
| `include/q35_sample.h`, `src/qwen35/q35_sample.c` | 320 | host sampler chain (penalties, top-k/p/min-p, temperature, seed) and logprobs |
| `include/q35_server.h`, `src/qwen35/q35_server.c` | 1100 | request parsing, one GPU scheduler thread, slots, streaming, usage/timings |
| `tools/q35_template_render.c` | 170 | G8/G9 fixture replay |
| `tools/q35_parse_cli.c` | 110 | G10 parser CLI |
| `tools/q35_chatparse_ref.cpp` | 100 | G10 oracle: llama.cpp's `common_chat_session` parser |
| `tests/qwen35/*.py`, `fixtures/*` | — | G6–G12 fixtures and runners |

### 4.2 Scheduler and slots

* One detached GPU thread owns every `q35_engine_*` call. HTTP threads only parse, tokenize
  and enqueue; they wait on a condition variable. The tokenizer is read-only and thread-safe;
  `tokenizer_decode` is only called from the GPU thread.
* Admission happens at launch boundaries. All idle slots are filled from the FIFO queue at
  once, so N simultaneous benchmark clients start in one batch. `--steps-per-launch`
  (default 16) bounds how many decode launches pass before the next admission pass; a queued
  request always triggers an admission as soon as a slot is idle.
* Every decode launch is one `q35_engine_decode_batch` over **all** decoding slots, with
  per-slot positions and the token each slot has to consume. The engine argmax is used when
  no slot needs host logits; otherwise the full fp32 rows are fetched for that launch.
  (The frozen engine API has one all-or-none `logits_host` argument, so a mixed batch copies
  all n rows. That is documented as an API limitation; greedy-without-penalty is the common
  path and fetches none.)
* Prefill uses `q35_engine_steps`. The frozen engine accepts at most **16** tokens per call
  (`run_steps`: `n > 16` returns -1), so the server's chunk is 16, not 256. At most one chunk
  is issued between decode launches, so running slots are never stalled for more than 16
  prompt tokens. The last prompt token is fed by the decode launch so its logits can be used
  directly.
* Streaming happens after every launch. A slot is freed on EOS, a stop string, max tokens,
  context exhaustion, a send error (client disconnect) or, for a queued job, the polling
  request thread seeing POLLHUP.
* `n != 1`, grammar/json-schema request fields and tool calls that fail JSON parsing are
  400s. DRY/XTC/typical-p/mirostat/logit-bias/dynatemp fields are ignored and reported in
  `X-Tuf-Ignored` (and `__tuf_ignored` in the body).

### 4.3 Prompt cache

GDN state cannot rewind, so the only sound reuse is the one the brief allows: **the new
prompt must start with the slot's whole previous sequence**. A common prefix against a
*longer* old sequence is not reusable, because the recurrent state is at the end of the old
sequence. `pick_slot()` therefore sets reuse to `seq_len` only when
`common_prefix(new, seq_len) == seq_len`; otherwise the slot is reset. Slots are ranked by
that reuse length, then by LRU. The last prompt token is always re-fed (reuse is clamped to
`n_tokens - 1`) so the next-token logits are available without storing a logits row.

`cached_tokens` in `usage` is the reuse length and `timings.prompt_n` is the number of tokens
actually prefilled. `tests/qwen35/prompt_cache.py` asserts `prompt_n == prompt_tokens -
cached_tokens`, that a second agent turn reuses more than the first prompt, that a divergent
conversation reuses 0, and that repeating the conversation reuses its prefix.

**Needed engine extension (not done, engine frozen):** per-slot GDN/conv snapshots after each
prompt, so a new prompt that shares only a prefix with a longer old sequence can restore the
snapshot at the longest match, truncate the attention KV to that point and prefill the rest.
That is what the brief's Phase 4.3 originally described; it needs an engine API to
save/restore `(conv_hist, gdn_state, kv up to pos)` per slot. The current exact-prefix rule
is strictly weaker but never returns a wrong state.

### 4.4 Template and tokenizer

`q35_template.c` implements the GGUF template directly. It includes the tools preamble
(`tools | tojson` in insertion order with minja's `", "`/`": "` separators), system
trim/strip, the reverse scan for `last_query_index` (tool responses are not user queries),
old-turn reasoning stripping, `reasoning_content` or inline `<think>` extraction on assistant
turns, assistant `tool_calls` (including parsing OpenAI `arguments` strings to objects, as
llama-server does), grouped `tool` results, `add_generation_prompt` and `enable_thinking`.
`chat_template_kwargs` from the request override `--chat-template-kwargs`; `tool_choice`
none/auto/required is accepted (tools are rendered under `none`, calls are not parsed).

**G8** (47 fixtures, `tests/qwen35/fixtures/g8_template.json`, generated by
`gen_g8_fixtures.py` from llama-server `--jinja --no-prefill-assistant`
`/apply-template` + `/tokenize`):

```
q35-template-render <0.8B.gguf> tests/qwen35/fixtures/g8_template.json
G8: PASS (47 cases, 0 failures)
```

Fixtures cover unicode/emoji/code/long prompts, content arrays, system/developer roles,
multi-turn reasoning stripping, the agent's real six tool schemas (`fixtures/agent_tools.json`,
dumped from `createWorkspaceTools`), tool results and multi-call loops, argument strings,
numbers/bools/nested objects, thinking on/off, `add_generation_prompt=false` and literal
special tokens.

**Tokenizer fix.** Qwen3.5 stores `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
`<tool_response>` as `USER_DEFINED` (GGUF token type 4), not `CONTROL`. `src/tokenizer.c`
previously pre-split only type 3, so its output diverged from llama.cpp on every chat prompt.
The new code treats type 4 like llama.cpp's `tokenizer_st_partition` does, and returns the
raw piece from `tokenizer_decode` for it. This is a correctness fix for qwen35; models
without type-4 tokens are bit-identical. The 35B tokenizer-only check is in G9 below.

**G9** (22 raw texts: the 5 corpus prompts, eight languages, emoji, C/Python/JSON, whitespace,
special tokens, numbers, long text):

```
q35-template-render <0.8B.gguf> tests/qwen35/fixtures/g9_tokenizer.json
G8: PASS (22 cases, 0 failures)      # the tool prints G8 for both fixture files
```

### 4.5 Tool-call and reasoning parser

Ported from `common/parsers/qwen3-coder.cpp` and the Qwen3.5 cases in
`tests/test-chat.cpp` / `tests/test-chat-peg-parser.cpp`: `<tool_call>` blocks,
`<function=name>`, `<parameter=key>` values terminated by `\n</parameter>`, string vs
JSON-typed arguments from the request tool schema, parallel calls, empty arguments, a tool
call ending an open think block, and raw JSON fallback when a value is not a string.
Arguments are emitted compact (`{"a":1,"b":"x"}`), which is what llama.cpp's mapper emits.

**G10**: `tests/qwen35/g10_parser.py` runs tuf's parser and `q35-chatparse-ref` (libllama-common's
`common_chat_session`, the code llama-server uses) on the same recorded raw strings:

```
G10: PASS (9 cases, 0 failures)
```

Cases: integer arg, Python code string, edit tool with a value ending in a newline, empty
args, parallel calls, reasoning + tool call, reasoning + content, tool call before
`</think>`, plain content. `tests/qwen35/test_side.c` additionally contains the ported
unit-level cases (run by `build-srv/q35-side-test`).

Streaming deltas use llama-server's shape: content and reasoning deltas, then
`tool_calls:[{index,id,type,function:{name,arguments}}]` when a block completes, a final
delta with `finish_reason:"tool_calls"`, a `choices:[]` usage/timings chunk, and `[DONE]`.
llama-server may split arguments over several deltas; tuf emits a completed block's
arguments in one delta. The accumulated message and `finish_reason` are identical.

### 4.6 / 4.7 Sampling, logprobs, fields and endpoints

Implemented on the host over the full fp32 row from `decode_batch`, in llama.cpp's chain
order: penalties (`repeat_last_n` window, repeat/frequency/presence) -> top-k -> top-p ->
min-p -> temperature -> dist/greedy. Defaults match `common_params_sampling` (temp 0.8,
top-k 40, top-p 0.95, min-p 0.05, penalty_last_n 64). Temperature 0 with no penalties uses
the engine argmax and copies no logits. Penalties are applied over the same window
llama-server feeds prompt tokens into (the whole sequence, generated and prompt). The RNG is
splitmix64 seeded per request/slot; it is deterministic and batch-invariant but is not
llama.cpp's `mt19937` stream (G6 only requires determinism and a correct distribution).

Logprobs are the raw-logit log-softmax at temperature 1 over the full vocabulary, with
`top_logprobs` <= 20, exactly llama-server's default `post_sampling_probs=false` path:
`{"id","token","bytes","logprob","top_logprobs":[...]}` under `logprobs.content`.

Endpoints: `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/tokenize`,
`/detokenize`, `/apply-template`, `/props` (`n_ctx` per slot, `total_slots`). `usage` carries
prompt/completion/total tokens, `cached_tokens`, and `timings`.

**G6** (host sampler vs an independent Python reference on fixed logits, 20k draws,
chi-square df=7 at p=0.001, same-seed determinism):

```
python3 tests/qwen35/sampler_ref.py
chi-square=9.781 df=7 crit(0.999)=24.322
PASS chi-square
PASS greedy penalties
PASS determinism
G6: PASS
```

**G7** (logprobs vs llama-server on five prompts, greedy): measured, not yet passing the
0.02 bound. Starting llama-server on the **CPU** backend (`-ngl 0`, Q8_0 weights, Q8_0
activation quantization) the chosen-token logprob deltas are mostly 1e-4..1e-2; the worst
positions are 0.02–0.11. Against the CUDA llama-server the deltas are larger (up to ~0.58)
because it quantizes activations to Q8_1. The tuff kernel uses exact fp32 dequantized dots
(G1 max |dlogit| 0.003 vs the fp32 reference), so this is llama.cpp's activation-quantization
error, not a tuff arithmetic error. The exact numbers and the G7 methodology are in the gate
section below; this is recorded as a known gap rather than silently loosened.

### P2 server throughput (`batch_bench.py --engine tuf`)

`evals/batch_bench.py` now starts one `runtime/build-srv/tuff-server -np N -c 4096*N`
per concurrency, exactly like the llamacpp engine, and uses the same 128-token
`ignore_eos`, temperature 0, thinking-off requests. Command:

```
cd sandbox && UV_CACHE_DIR=/tmp/uvc-dsh uv run python evals/batch_bench.py \
  --engine tuf --model qwen3.5-0.8b --conc 1 2 4 8 16 32
```

| aggregate tok/s | conc 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| tuf 0.8B | **255.7** | 136.5 | 144.0 | 140.0 | 116.6 | 90.0 |
| llama.cpp 0.8B | 228.9 | 371.4 | 507.4 | 620.3 | 744.0 | 853.1 |
| tuf / llama | 1.12x | 0.37x | 0.28x | 0.23x | 0.16x | 0.11x |
| tuf 2B | **115.8** | 57.9 | 59.6 | 56.2 | 48.4 | 44.4 |
| llama.cpp 2B | 129.5 | 223.7 | 354.7 | 470.0 | 621.4 | 753.4 |
| tuf / llama | 0.89x | 0.26x | 0.17x | 0.12x | 0.08x | 0.06x |

Results: `sandbox/runs/batch-bench-tuf-qwen3.5-0.8b-20261108-075622/` and
`...-2b-20261108-080255/`. Conc 1 beats llama.cpp on 0.8B and is within 11% on 2B.
Above conc 1 the known batched-kernel regression dominates: the kernel is slower at
BT=2 than at BT=1 and does not scale, so the server's continuous batching cannot help.
The server targets (0.8B 330 / 1800 / 2800 / 4000; 2B 150 / 800 / 1400 / 2200) are
missed except 0.8B conc 1 (255.7 < 330 because the server's chunked prefill is the
T=1 path) and 2B conc 1 (115.8 < 150). The host scheduler itself adds no measurable
overhead beyond the engine numbers in Addendum 1 (0.8B 346.8 at B=1 vs 255.7 here is
prefill + HTTP, measured; the engine bench does no prefill).

The prefill cause is measurable: `q35_engine_steps` processes its <=16 tokens through
the T=1 kernel, so pp is only ~equal to decode tok/s (the Phase 2 addendum already
recorded this). A 100-token prompt costs ~0.3 s (0.8B) / ~0.6 s (2B) of the 128-token
benchmark's ~0.5 s / ~1.1 s generation window, which is why conc 1 falls from the
engine's 346.8 / 154.7 to the server's 255.7 / 115.8.

### G8/G9 on both models and the 35B tokenizer proof

```
q35-template-render <0.8B.gguf> fixtures/g8_template.json   -> G8: PASS (47 cases)
q35-template-render <2B.gguf>   fixtures/g8_template.json   -> G8: PASS (47 cases)
q35-template-render <0.8B.gguf> fixtures/g9_tokenizer.json  -> PASS (22 cases)
q35-template-render <2B.gguf>   fixtures/g9_tokenizer.json  -> PASS (22 cases)
```

35B tokenizer-only regression: the changed `tokenizer.c` was compiled twice (old
CONTROL-only specials, new qwen35 USER_DEFINED handling) and both binaries tokenized
the 22 G9 texts with `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` (vocab only, CPU): **22/22
identical**. The USER_DEFINED special handling is scoped to `general.architecture ==
"qwen35"` (dense), so every other architecture keeps the old token stream exactly.

### G7 measurements (does not meet the 0.02 bound)

`tests/qwen35/g7_logprobs.py`, five prompts, greedy, 16 tokens, `top_logprobs=5`,
llama-server on CPU (`-ngl 0`, same Q8_0 weights):

| model | positions compared | p50 \|dlogprob\| | p95 | max | over 0.02 | top-5 sets ok | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 0.8B | 73 | 0.0055 | 0.0747 | 0.1125 | 23 | 66/73 | FAIL |
| 2B | 73 | 0.0089 | 0.0919 | 0.1320 | 26 | 61/73 | FAIL |

All 73 compared positions have the same greedy token; prompt 0 ("Hello") diverges after
9 tokens, so only 9 positions there are compared. Against the CUDA llama-server the
deltas are larger (0.8B: max ~0.58) because CUDA quantizes activations to Q8_1.

This is **not** a tuff arithmetic error: G1 measured tuff against the independent fp32
numpy reference at max |dlogit| 0.00318, while llama.cpp's own Q8_0 CPU path quantizes
activations and differs from that same reference. The 0.02 bound would require tuff to
reproduce llama.cpp's activation quantization exactly, which lives in the frozen kernel.
The gate is recorded as failed, not loosened. The top-5 ids are identical at most
positions; the 7/12 mismatches are tokens at the 4th/5th rank with logprob deltas near
the 0.10 near-tie band.

### G11 end to end

Agent (`tests/qwen35/g11_agent.sh`, `PROFILE=env`, streaming, thinking off, temperature 0,
`GUARD=fix` so the repeat guard can finalize a weak model's loop; `-np 4`):

| model | file written | streaming tool calls | steps / calls | cache hit | wall | final reply |
|---|---|---|---:|---:|---:|---|
| 0.8B | `hello from tuff` exact | yes | 5 / 2 | 68.3% | 4.15 s | `hello.txt` |
| 2B | `hello from tuff` exact | yes | 12 / 11 (3 tool errors) | 85.2% | 14.98 s | confirms the file |

The first 0.8B run without the repeat guard wrote the file but exhausted 10 rounds
(11 calls); that is the agent's guard, not a server failure. The gate needs a completed
task and streaming tool calls, both of which the guard-fix run has.

`canonical.py gsm8k --limit 50 --concurrency 4`, thinking off, greedy
(`tests/qwen35/g11_canonical.sh`):

| model | tuf acc | llama-server acc | delta | tuf wall | llama wall | tuf trunc | llama trunc |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 0.52 | 0.46 | **+6.0 pp** | 74.9 s | 20.9 s | 0.00 | 0.00 |
| 2B | 0.70 | 0.72 | **-2.0 pp** | 228.6 s | 38.1 s | 0.02 | 0.06 |

The 2B model passes the ±4 point bound. The 0.8B run is +6 points, i.e. 3 questions out
of 50, and a repeat of the tuf run gave the identical 0.52, so it is not run-to-run
noise. The difference is the greedy-continuation divergence caused by the same
llama.cpp activation-quantization gap as G7; the 50-item binomial standard error is
about 7 points, but the literal gate is recorded as missed (by 2 points). No truncation
in either 0.8B run.

### G12 robustness

`tests/qwen35/g12_robustness.py --duration 600`, `tuff-server -np 4 -c 32768`, 64 concurrent
streaming clients, random prompts/lengths, 30% of requests disconnect after the first SSE
event, and 15% of requests ask the fixed prompt `Reply with exactly the word BANANA`:

```
solo baseline: 'BANANA'
storm stats: {"ok": 1506, "cancelled": 639, "errors": 0, "mismatch": 0, "checked": 213} elapsed 619.7
PASS post-storm 4 concurrent slots match the solo run
PASS 213 fixed-prompt streams matched the solo run
G12: PASS
```

No crash, no HTTP errors, no leaked slot (four concurrent requests after the storm all hit
a free slot and matched the solo output), and no cross-talk: all 213 fixed-prompt streams
during the storm produced exactly `BANANA`, the same as the solo run. This also shows that
the batched-kernel non-bitwiseness (G4) did not change any greedy decision at the sizes
used here.

### Backend-A prompt-renderer regression

`tests/qwen35/backend_a_prompt_diff.py` extracts the backend-A ChatML rendering block from
`git show HEAD:runtime/src/server.c` and from the working tree, compiles both into the same
fixture driver, and compares the rendered bytes over 36 fixtures (system / user / assistant
/ tool roles, unicode, code, empty content, system override, and all four
`reasoning_effort` paths):

```
old block 1024 bytes, new block 1024 bytes, textually_identical=True
backend-A prompt renderer: PASS (36 fixtures, byte-identical)
```

The block is unchanged; the `"content"` alias removal was in `parse_request` (backend-B
`/tokenize` now parses its own body) and cannot affect backend A's prompt bytes.


## Batch race + speed (Sonnet)

Scope: the batch path (`decode_batch_kernel`, tensor-core f16-activation GEMVs, no grid barriers).
The T=1 kernel and its numerics are untouched. All measurements: RTX 3070 Laptop, GPU access
under `flock /tmp/claude-1000/q35-gpu.lock`, 0.8B and 2B never mixed in one measurement. A/B =
`BENCH=batch scripts/ab_interleaved.sh` (alternating order, 64 steps, median of 3 inside each run,
5-6 rounds); the interval is the t-based 95% CI of the mean paired ratio B/A. "Gates" = det-probe
ops 2,3,4,5 and 0 at n=1,8,32 with and without `--dup` (5 reps), G4 at B=2/8/16/32 (`--steps 16`)
and DUP, both models (`/tmp/.../sonnet-w/gates.sh`).

### H1. Race fix (kept)

Root cause as established: `P.vecf` is one global buffer that every CTA rewrites in every op
prologue while slower CTAs may still be reading the previous op's contents in the GEMV.

* Fix: `g_vdone` (global, reset per launch) counts "this CTA has finished its last `vecf` read of a
  GEMV" (`vdone_arrive()` at the end of `gemv_frag`: `__syncthreads` + `red.release.gpu`). Every
  prologue calls `vdone_wait(GRID * gemvs_so_far)` (thread 0 polls with `ld.acquire.gpu`, then
  `__syncthreads`) after its inputs have arrived and *before its first `vf_put`*.
* `vecf` reads use `ld.global` (`__ldca`/`__ldcg`) instead of `__ldg` (`ld.global.nc` on data written
  in the same kernel is undefined).
* A first version used `__ldcg` for the GEMV reads and also restructured `prologue_act_b` /
  `prologue_attn_gate_b` to issue all loads before the gate: that spilled (stack 160 -> 240 B)
  and was -12% at B=32 vs the barrier build. The accepted version keeps the original loop
  structure and puts the wait at the first store (`if (t == 0 && pair == threadIdx.x) vdone_wait`).
* `Q35_DBG_GRIDSYNC` is still in place (its `vdone_*` calls compile to nothing); the final source
  was re-checked in that mode: det-probe op 0 n=1/8/32 PASS, G4 B=8/32 PASS.
* Gates: all PASS on both models for the accepted version (and again on the final clean build).
* A/B vs the old barrier binary (`ab-base/qwen35-batch-bench-gs`, 0.8B, 5 rounds):
  B=32 1.0380 [1.0304, 1.0457]; B=8 1.0327 [1.0234, 1.0419]; B=1 0.9950 [0.9908, 0.9991]
  (the +-2sd band includes 1.0; -0.5% is the cost of the per-op poll at B=1). 2B vs the barrier build
  was not measured.

### H2. Watchdog is a hard error (kept)

* `ll_wait_raw` / the new waits record the timeout in device globals (`g_wd_b` kind 2 = `ll_wait`,
  3 = `vecf` arrival wait, 4 = slice-ready wait; `g_wd_b_blk`, `g_wd_b_ep`) with an atomic CAS, and once
  one wait has timed out every other wait returns immediately (no 100 ms x ops cascade).
* The launch resets the record and the counters (`cudaMemcpyToSymbolAsync`), so one event cannot stick.
* `q35_engine_decode_batch` reads the globals after the sync (not `out_info`, which a CTA that finished
  early would have written), prints `[qwen35] WATCHDOG batch: ... (kind K, block B, epoch E = op ~N of
  the launch, n=..)`, bumps the epoch by 4096 (stale tags of the aborted launch can never match) and
  returns -7. `q35_mk_watchdog_fired` now has the signature `(int *blk, unsigned *ep)`.
* Test: kernel built with `-DQ35_DBG_STALL`, `Q35_DBG_STALL_LAUNCH=3` makes CTA 5 exit at the start of
  launch 3; `Q35_BENCH_CONTINUE=1` lets `qwen35-batch-bench` continue after a failed step. Result at
  B=1 and B=8: exactly one error (launch 3, message above), every other launch fine.

### H3. Sliced prologues (kept)

Each CTA converts only its 1/GRID slice of the activation into `vecf`; `g_vready` (reset per launch)
is the "all slices written" barrier before any GEMV reads `vecf`. `vdone_wait` still guards the first
store. `vecf` loads are `__ldcg` (the slices are written by other CTAs).

* act / attn-gate: (entry, k-pair) items split by pair slice. gated-norm: unit = (entry, head), units
  split over CTAs, one warp per unit.
* RMSNorm: the old reduction (per-thread FMA chain, warp butterfly, sum of the 16 warp partials in
  order 0..15) is kept *exactly*: CTAs 0..15 compute the 16 warp partials of every entry (CTA w plays
  warp w), exchange them through `g_ssl` (epoch-tagged), every CTA sums them in order and scales its own
  slice. So the sliced prologues are bitwise identical to the unsliced ones and to H1: teacher logits at
  N=1 and N=32 were compared against the H1 binary with `cmp` (identical), and BT=1 keeps the unsliced
  prologues (the exchange costs more than it saves at one entry).
  (An earlier variant defined a different 40-slice reduction tree; it passed G4 but made BT=1 4%
  slower than H1 because BT=1 then needed the same tree; it was dropped.)
* Gates: all PASS both models. A/B vs H1 (5 rounds, ratio [95% CI]):

| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| 0.8B | 1.0009 [0.9956, 1.0062] | 1.1678 [1.1494, 1.1862] | 1.2577 [1.2510, 1.2645] |
| 2B | 0.9943 [0.9734, 1.0151] | 1.1336 [1.1115, 1.1557] | 1.2513 [1.2211, 1.2816] |

### H4. Parallel per-entry norms (reverted)

All five prologues rewritten with EB entries' loads in flight (two passes for the norm: partial sums,
then re-load and scale). Gates passed but it is much slower: 0.8B vs H1, 5 rounds: B=1 0.7284
[0.7129, 0.7439], B=8 0.6233 [0.6206, 0.6261], B=32 0.6118 [0.6105, 0.6130] (single-run variants:
rmsnorm only 1685 tok/s, act only 1566, gated-norm only 1789, attn-gate only 1797 vs 2003 at B=32;
stack frames 330-790 B). A clock64 breakdown showed the second load pass alone cost ~65k cycles per
call at nt=32: the prologue cost is the 40x redundant L2 traffic (reads *and* the 40 CTAs writing the same
`vecf` lines), not serial latency. H3 removes the redundancy and makes the norm warp-per-entry anyway.
Source of the experiment: kept outside the tree.

### H5. B=1 and per-op profile

`Q35_PROF=1` now works on the batch path (CTA 0 stamps every op end and the end of every GEMV
prologue; engine prints the per-op averages every 64 launches). Final code, us per decode step
(24 layers), "prologue" = from the previous op end to the end of that op's prologue (includes waiting
for the previous op's slowest CTA); the rest of the op is GEMV + epilogue + barrier:

| | in | mixer | out | gate/up | down | LM head | total |
|---|---|---|---|---|---|---|---|
| 0.8B B=1 | 753 | 262 | 274 | 718 | 290 | 769 | 3066 |
| 0.8B B=1 prologue part | 232 | - | 160 | 184 | 133 | 10 | |
| 0.8B B=32 | 2566 | 4851 | 713 | 2554 | 774 | 965 | 12423 |
| 0.8B B=32 prologue part | 723 | - | 291 | 586 | 284 | 29 | |
| 2B B=1 | 1409 | 254 | 533 | 1982 | 1021 | 1530 | 6729 |
| 2B B=32 | 4156 | 5092 | 1396 | 6251 | 2595 | 1939 | 21429 |

Reading it (0.8B, B=1): the GEMV parts of in / gate-up are at the DRAM roofline (~21.7 us per op per
layer at 373 GB/s vs 521/24 and 534/24 us measured), the LM head is at the roofline (769 vs 724 us); the
remaining gap to the T=1 kernel is the ~700 us/step of prologue dependency latency (+ 262 us mixer).
At B=32 the GDN/attention mixer is 39% of the step (fp32 recurrent state, ~1.15 GB read+write per
step ~ 3.1 ms at the roofline) and in / gate-up are still ~107 us per op against a 22 us weight floor
(ring-path GEMV with a per-group cross-warp reduction and 2 MB of tagged epilogue stores): the next
lever, not done.

Change kept: **L2 weight prefetch** (`w_prefetch`): at the start of each GEMV's prologue every CTA issues
`prefetch.global.L2` for the first 48 KB of its weight slice, so the DRAM is busy during the serial
prologue. A/B vs H3 (5 rounds, ratio [95% CI]):

| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| 0.8B | 1.0799 [1.0591, 1.1007] | 1.0454 [1.0282, 1.0626] | 1.0001 [0.9950, 1.0053] |
| 2B | 1.0217 [1.0171, 1.0263] | 1.0286 [1.0094, 1.0478] | 1.0056 [0.9896, 1.0217] |

Tried and dropped (single runs, 0.8B, no formal A/B): prefetch of 32 / 72 / 80 / 128 KB (<= 48 KB),
issuing the prefetch after the previous GEMV instead of at the prologue start, and a warp-per-group direct
GEMV path for the in / gate-up projections (B=1 -12%, B=8 -8%, B=32 +4%). The remaining B=1 gap
(307 vs 335 tok/s for the fp32 T=1 kernel on 0.8B) is not closed.

### Batch-path accuracy gate (G1 on the batch path)

`qwen35-teacher` with `Q35_TEACHER_BATCH=N` runs the teacher-forced sequence as entry 0 / slot 0 of a batch
of N through `q35_engine_decode_batch`; slots 1..N-1 are fed the same token list shifted by 13*slot. `g1.sh`
takes `Q35_BUILD=<dir>` (binaries) and `G1_REUSE=1` (reuse the llama/fp32-reference files in OUTDIR).
`tests/qwen35/g1.sh MODEL 64`, 5 prompts, tolerance 0.05 unchanged:

| model | N | argmax mismatches (margin >= 0.02) | max abs dlogit | worst p99 |
|---|---|---|---|---|
| 0.8B | 1 | 0 / 5 prompts | 0.02401 | 0.00575 |
| 0.8B | 32 | 0 | 0.02401 (identical per prompt to N=1) | 0.00575 |
| 2B | 1 | 0 | 0.02425 | 0.00445 |
| 2B | 32 | 0 | 0.02425 (identical per prompt to N=1) | 0.00445 |

So the f16 activations cost ~0.024 logit worst case, inside the tolerance, and the result is independent
of the batch size.

**Pre-existing bug found by this gate and fixed:** the LM-head epilogue wrote `logits_out` only for rows
`r0` (`v[0]`, `v[1]`) and never for `r1 = r0 + 8` (`v[2]`, `v[3]`), so half of the batch-path logits were
stale (|dlogit| ~ 33 vs the reference on the first try). Argmax (`pack_score`) was not affected; G4 and the
det-probe op-0 comparison were blind to those rows before the fix (they now compare complete logits).

### Final numbers

Engine bench (`qwen35-batch-bench --bench B --steps 64 --rounds 5`, median), tok/s; baseline = racy
build as measured before this work:

| B | 0.8B now | 0.8B baseline | 2B now | 2B baseline |
|---|---|---|---|---|
| 1 | 306.8 | 275 | 139.0 | 138 |
| 2 | 564.2 | - | 271.6 | - |
| 4 | 1040.0 | - | 516.6 | - |
| 8 | 1765.6 | 1432 | 963.2 | 803 |
| 16 | 2423.7 | - | 1295.6 | - |
| 32 | 2519.6 | 2014 | 1523.0 | 1193 |

Unlike the racy baseline these numbers are bitwise deterministic and batch-invariant (G4/DUP pass).

### Files changed

* `src/qwen35/qwen35_mk.cu`: vdone counter + coherent `vecf` loads (H1); watchdog record/fail-fast/kinds,
  per-launch reset, `q35_mk_watchdog_fired` (H2); sliced prologues, `g_vready`, `g_ssl` (H3); L2 weight
  prefetch (H5); batch-path `Q35_PROF` stamps; LM-head `logits_out` fix; `Q35_DBG_STALL` test hook; removed
  the three pre-existing unused-variable warnings.
* `src/qwen35/qwen35_engine.c`: watchdog -> error (-7), epoch bump, prof printing, stall-launch test hook.
* `include/qwen35_kernel.h`: new `q35_mk_watchdog_fired(int *blk, unsigned *ep)`.
* `tools/qwen35_teacher.c`: `Q35_TEACHER_BATCH=N`.
* `tools/qwen35_batch_bench.c`: `Q35_BENCH_CONTINUE=1` (watchdog recovery test).
* `tests/qwen35/g1.sh`: `Q35_BUILD`, `G1_REUSE`.
* `scripts/ab_interleaved.sh`: additive `BENCH=batch` mode (BIN_A/BIN_B/MODEL/BATCH_B, t-based CI).
* `QWEN35_MK_REPORT.md`: this section.

### Not verified / caveats

* H1 vs the barrier build was only measured on 0.8B; H3/H5 A/Bs cover B=1/8/32 only (not B=2/4/16).
* The rejected H5 variants (other prefetch sizes/modes, direct GEMV path) are single runs, not 5-round A/Bs.
* The watchdog test needs a `-DQ35_DBG_STALL` build; the default build contains only the (inert) hook field.
* The G1 numbers were produced with the binary built just before the last cosmetic edits (debug hook,
  warning clean-up); the clean rebuild reproduces the same teacher logits bitwise (checked at N=32, 40 tokens,
  0.8B) and passes all gates.
* T=1 kernel / CLI path was not re-run here (no source change in it besides the shared `ll_wait_raw`
  fail-fast on a prior watchdog event).

---

## Batch throughput, round 2 (2026-11-08, branch `qwen35-megakernel`)

Starting point: commit 46d27f7 (sections above). Method: `Q35_PROF=1` per-op profile (CTA-0 view, now with the
mixer split into GDN / attention), one hypothesis at a time, single-run screening with alternating binaries,
then one 5-round paired A/B of the kept set against the 46d27f7 binary. 0.8B and 2B are always measured
separately. Measured read+write DRAM roofline for the in-place GDN state pattern: **353 GB/s**
(read-only 365 GB/s, same kernel shape; `scratchpad rwbw.cu`).

### Where the time went (0.8B, B=32, before)

`in` 2580 us, mixer 5012 (GDN 4302 + attention ~700), `out` 777, gate/up 2567, `down` 777, LM head 964;
total 12.68 ms/step. The `in` and gate/up GEMVs ran at ~80 us/layer against a ~30 us/layer tensor-core floor
(f16 MMA with fp32 accumulate is half rate on GA10x: ~9.2 TFMA/s at 1800 MHz).

### Hypotheses tested

| # | hypothesis | result | kept |
|---|---|---|---|
| E1 | `gemv_cta_f` reloads the activation fragments from L2 for every 16-row group although, for K <= 1024 (one chunk per group), each warp uses the same k-tiles in every group: load once per op | with E2: 0.8B B=32 2466 -> 2994 tok/s | yes |
| E2 | the per-group cross-warp reduction is serial: 2*NT `__syncthreads` per group and warp 0 sums all n-tiles; reduce all n-tiles at once (tiles 0-1 in `red`, 2-3 in the ring slot just consumed), warp n sums tile n, same 0..15 order | bitwise identical; together with E1 above (`in` GEMV part 77 -> 41 us/layer) | yes |
| E3 | GDN: issue the state loads before waiting on the projections, prefetch the next task's 16 KB slice to L2 | GDN 4302 -> 4472-4600 us (slower, with and without the prefetch) | no |
| E4 | GDN: one task per head (8 columns/warp) instead of 32-column slices to cut per-task latency | GDN 4383 us (no gain), B=1 GDN 144 -> 163 us | no (code generalized, CPW=2 default) |
| E5 | attention: 32-position split chunks make ~5 rounds of ~14 us tasks per layer at B=32; batch-path rule `splits = ceil(n/128)` (a function of the position only, so G4 holds; T=1 kernel unchanged) | attention 672 -> 461 us, `out` prologue -160 us at B=32; B=1 attention +50 us (fewer splits for one sequence); identical above n ~ 1344 (42-split cap) | yes (chunk 64: in between) |
| E6 | attention KV loads batched per warp (KVB 2/4) | raises register pressure for the whole kernel: every op slower (B=32: -3% / -14%) | no (KVB=1) |
| E7 | LM head at BT=32 spills (all batch instantiations sit at the 128-register cap); `gemv_direct_f` DEPTH 4 -> 2 | LM head 1002 -> 941 us | yes |
| E8 | int8 -> f16 dequant uses two I2F (quarter rate on sm_86) + a pack per pair; PRMT magic number (`0x64xx` = 1024 + u, one HSUB2) | exact (all 65,536 byte pairs checked equal); kernel -1.5% at B=32 | yes |
| E9 | RMSNorm prologues: every CTA recomputes the 16 partials itself instead of the `g_ssl` round trip | `in` prologue 718 -> 905 us (the 256 KB/CTA L2 read costs more than the hop) | no |
| E10 | host: 13 small synchronous transfers per batch step (3 input copies, 5 symbol resets, 3 watchdog reads, info, tokens) + a 4 KB `__constant__` params upload every step | inputs in one block (1 copy), batch state in one `BState` struct (1 reset, 1 read), params upload skipped when byte-identical (shared by T=1 and batch paths); B=1 ~+0.5% | yes |
| E11 | conv history layout `[channel][ring 8]`: every step reads and writes one 32-byte sector per channel for 4 useful bytes (~0.39 MB/slot/layer = +19% over the 2 MB fp32 state); switch to `[ring][channel]` (both kernels; addressing only) | GDN 4312 -> 3853 us (0.8B), 4521 -> 3991 us (2B) at B=32 | yes |

### Verification of the kept set (clean build, 0 warnings in qwen35 files)

* det-probe ops 2-5 and 0, n = 1/8/32, with and without `--dup`: 30/30 PASS on 0.8B and on 2B.
* G4 B = 2/8/16/32 (`--steps 16`) and DUP: PASS on both models.
* Watchdog (`-DQ35_DBG_STALL`, `Q35_DBG_STALL_LAUNCH=3`): exactly one WATCHDOG error and one failed step at B=1 and B=8.
* G1 (`g1.sh MODEL 64`, 5 prompts, tolerance 0.05): T=1 path max |dlogit| 0.0032 (0.8B) / 0.0022 (2B);
  batch path N=1 and N=32 (identical) 0.0258 / 0.0205; 0 argmax mismatches everywhere. (The batch-path
  number moved from 0.0240 to 0.0258 on 0.8B because the attention split changes the softmax merge order.)
* A/B vs 46d27f7, `BENCH=batch scripts/ab_interleaved.sh 5`, paired ratio new/old [95% CI]:

| model | B=1 | B=8 | B=32 |
|---|---|---|---|
| 0.8B | 1.0900 [1.0750, 1.1050] | 1.0806 [1.0686, 1.0926] | **1.3316** [1.3252, 1.3380] |
| 2B | 0.9998 [0.9896, 1.0100] | 1.0485 [1.0333, 1.0638] | **1.1372** [1.1316, 1.1429] |

### Engine numbers now (`qwen35-batch-bench --bench B --steps 64 --rounds 5`, median, aggregate tok/s)

| B | 0.8B | target (server) | 2B | target (server) |
|---|---|---|---|---|
| 1 | 338.5 | 380 (P1) / 330 | 146.0 | 165 (P1) / 150 |
| 2 | 602.0 | | 276.1 | |
| 4 | 1089.9 | | 526.9 | |
| 8 | 1893.7 | 1800 | 976.6 | 800 |
| 16 | 2816.3 | 2800 | 1441.3 | 1400 |
| 32 | 3322.6 | 4000 | 1674.9 | 2200 |

The server targets are end-to-end (`batch_bench.py`, prefill + HTTP); these are engine numbers, an upper bound
for them. The server benchmark was not rerun in this round.

Per-op profile now (us/step): 0.8B B=32 `in` 1654 (prologue 712), mixer 4317 (GDN 3856, attention 461),
`out` 535, gate/up 1419 (prologue 519), `down` 684, LM 922, total 9531. 0.8B B=1: total 2921, prologues
655. 2B B=32: `in` 3548, mixer 4492 (GDN 4015), `out` 1070, gate/up 5002, `down` 2444, LM 1840, total 18396.

### Distance to the floor

* 0.8B B=32: weights 0.81 GB at 373 GB/s = 2.17 ms; fp32 GDN state 1.21 GB read+write at 353 GB/s =
  3.42 ms; GEMV tensor work ~2.7 ms (overlaps the weight stream). Floor ~6.0-6.4 ms/step = ~5000-5300 tok/s.
  Now 9.53 ms: GDN is at 89% of its floor; the loss is the GEMV part of `in` / gate/up (~1.5x the tensor
  floor), ~1.9 ms of prologue dependency latency, and attention latency (0.46 ms for a short context).
* 0.8B B=1: ~0.85 GB/token -> 2.28 ms at 373 GB/s (440 tok/s). The reference megakernel
  (`megakernels/qwen3.5-0.8b`, sm_120 only) runs int8 at 88% of its card's roofline; that efficiency here
  would be ~386 tok/s. Now 2.92 ms kernel (78% of the roofline): the GEMVs are at the roofline, the loss
  is ~0.65 ms of prologue hops (~6 us each vs the reference's 3.2 us) and 0.27 ms mixer.

### Next levers (not done)

1. 2B GEMVs at K = 2048 / 6144 still reload activation fragments per chunk (E1 only covers one chunk per
   group): gate/up is 208 us/layer at B=32 vs ~87 us tensor floor. Needs a register-light way to keep B
   fragments (software-pipelined loads, or k-split teams).
2. Prologue latency (1.9 ms at B=32, 0.65 ms at B=1): each sliced RMSNorm prologue is 3 grid-wide hops
   (xl ready -> `g_ssl` partials -> `vready`); the reference hides its hops behind a producer warp that keeps
   streaming weights (TMA ring + L2 prefetch cursor), which sm_86 can only approximate with cp.async.
3. `down` / `out` have 64 row groups over 40 CTAs (25% imbalance).
4. bf16/f16 GDN state would cut 1.7 ms at B=32 (0.8B) but changes numerics over long contexts; not tried.

## Batch throughput, round 3: 2B GEMVs (2026-11-08)

Problem: at K = 2048 / 6144 (all 2B layer GEMVs; 0.8B `out` / `down`) a 16-row group spans 2-6 ring chunks of
32 k-tiles, so round 2's once-per-op activation load (E1) does not apply, and every chunk is wait -> barrier ->
L2 activation loads -> 16 MMAs per warp -> barrier. Measured on this GPU (`scratchpad mmabw.cu`): f16 MMA with
fp32 accumulate = **13.1 TFMA/s** (f16 accumulate 26.3). 2B gate/up at B=32 was ~172 us/layer for its GEMV part
vs 61 us of tensor work and 72 us of weight streaming: the weight stream ran at 42% of the roofline.

| # | hypothesis | result | kept |
|---|---|---|---|
| X3 | issue a chunk's activation loads before the ring wait + barrier (they do not depend on the ring) | 2B B=32 step -1.5% (gate/up -230 us, `in` -120 us) | yes |
| X1 | `ld.global.ca` (L1) instead of `.cg` for the activations | no gain (with or without X3) | no |
| H-ks | k-split GEMV without the cp.async ring: warp w streams the A fragments of its own k-tiles (w + 16i, the same tiles and order as the ring path) straight from global with a 2-deep register pipeline across group boundaries; the cross-warp reduction is double-buffered in the free ring smem, 1 barrier per group (was 5 for a 2B group) | bitwise identical to the ring path (teacher logits `cmp`-equal vs 80d1fed at N=16 and N=32, both models). B=32: 2B 18.0 -> 15.7 ms/step, 0.8B 9.46 -> 8.94; B=16: 2B -6.6%, 0.8B -2%; but B<=8 +4-5% (a 2-deep register pipeline holds less in flight than the 70 KB ring) | yes, for BT >= 16 (`KS_MIN_BT`); ring below |
| | pipeline depth 3 (BT >= 16) / 8 (BT <= 8) | spills: +9% / 1.5-3x slower | no |

Verification (clean build, 0 warnings): det-probe 30/30 on both models, G4 B=2/8/16/32 and DUP PASS, watchdog
one error per stalled launch at B=1/8, G1 unchanged (T=1 0.0032 / 0.0022, batch N=1 and N=32 0.0258 / 0.0205,
0 argmax mismatches). A/B vs 80d1fed, 5 rounds, ratio [95% CI]:

| model | B=1 | B=8 | B=16 | B=32 |
|---|---|---|---|---|
| 0.8B | 0.9978 [0.9863, 1.0094] | 1.0026 [1.0013, 1.0040] | 1.0207 [1.0091, 1.0323] | **1.0645** [1.0557, 1.0732] |
| 2B | 1.0162 [0.9976, 1.0349] | 0.9967 [0.9835, 1.0099] | 1.0772 [1.0645, 1.0899] | **1.1848** [1.1525, 1.2170] |

Engine now (`--bench B --steps 64 --rounds 5`, median, aggregate tok/s; target = server target):

| B | 0.8B | target | 2B | target |
|---|---|---|---|---|
| 1 | 337.4 | 380 (P1) / 330 | 149.7 | 165 (P1) / 150 |
| 8 | 1902.8 | 1800 | 995.6 | 800 |
| 16 | 2863.0 | 2800 | 1528.0 | 1400 |
| 32 | 3539.7 | 4000 | 2012.6 | 2200 |

2B B=32 per op now (us/step): `in` 2839 (prologue 1011), mixer 4582 (GDN 4067, attention 515), `out` 826,
gate/up 3800 (prologue 757), `down` 1819, LM 1972; total 15.84 ms (was 18.40 after round 2, 20.96 at 46d27f7).
Gate/up GEMV part is now ~127 us/layer vs the 72 us weight floor. The prologues (2.5 ms at B=32 on 2B) are now the
largest non-floor item after the GDN state traffic.
