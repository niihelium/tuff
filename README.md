# runtime — tuf

**Goal:** our own inference runtime that replaces llama.cpp for **specific models on specific hardware**,
with optimizations and tricks llama.cpp cannot afford to specialise for: per-expert VRAM caching of MoE
weights, CPU/GPU split by locality, lossless speculation, fused kernels captured in CUDA graphs — and next, the
hooks our research needs (mid-layer reads, in-process decision heads, state snapshots for context management).

Target machine: RTX 3070 Laptop 8 GB (sm_86) + Ryzen 9 5900HX + 64 GB DDR4. Correctness reference: llama.cpp
(token-identical greedy continuation).

## Role in the workspace

- Serves models over an **OpenAI-compatible API** (`tuff-server`), so `../agent`, `../webui`, `../alasin` and
  `../sandbox` use it exactly like llama.cpp — swapping runtimes changes no agent code.
- Until it serves the small models with tool calls, the workspace runs on llama.cpp; the sandbox is where tuf
  must match llama.cpp's outputs (temp-0 identical) and beat its tok/s before it takes over.

## Status

| Model | Engine | State |
|---|---|---|
| Qwen3.6-35B-A3B (`qwen35moe`, Q4_K_S) | hand-tuned `engine.c` | **67.5 tok/s** vs llama.cpp 43.9 (details below) |
| LFM2.5 dense / MoE (`lfm2`, `lfm2moe`) | `lfm25_engine.c` | phase 1, see `LFM25_ROADMAP.md`, `wiki/lfm25/` |
| Qwen3.5 0.8B / 2B (dense, Q8_0) | `src/qwen35/` megakernel | **Phases 1, 2 and 4 (server)**. 0.8B 348 tok/s / 2B 159 tok/s at batch 1; the batched path is bitwise batch-invariant (G4) and scales to 2511 / 1506 tok/s aggregate at B=32 (see `QWEN35_MK_REPORT.md`). `tuff-server` backend B has continuous batching, exact-prefix prompt cache, the Qwen3.5 template, tools/tool-call + reasoning parsing, the llama.cpp sampling chain and logprobs. G8/G9/G10 pass; G7 and the P2 throughput table are documented misses. |

Backend A (Qwen3.6-35B-A3B) is unchanged: greedy only, one request at a time, no `tools`, `logprobs` null.
`-np/--parallel > 1` is a startup error on backend A. Backend B (dense Qwen3.5 0.8B/2B) is selected from the
GGUF architecture (`general.architecture = qwen35`) and implements the server's parallel-slot path.

## Server

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build --target tuff-server -j8
MODEL_PATH=/path/to/model.gguf PORT=8080 ./run.sh        # builds if needed, then serves
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/completions`, `POST /v1/chat/completions` (ChatML,
`"stream": true` for SSE), plus backend B's `POST /tokenize`, `POST /detokenize`, `POST /apply-template` and
`GET /props`. Plain C11 + POSIX sockets.

Backend B (Qwen3.5 dense) runs one GPU scheduler thread with `-np/--parallel N` slots and
`-c/--ctx-size` split evenly per slot, like llama-server:

```bash
./build-srv/tuff-server --model Qwen3.5-0.8B-Q8_0.gguf --port 8097 -np 4 -c 65536 \
  --chat-template-kwargs '{"enable_thinking":false}' --reasoning-format deepseek
```

`--kv-unified` and `--jinja` are accepted as no-ops. Sampling implements penalties, top-k/top-p/min-p,
temperature and seed; `logprobs`/`top_logprobs` are supported; grammar / JSON schema return 400 and
DRY/XTC/typical/mirostat fields return `X-Tuf-Ignored`. See `QWEN35_MK_REPORT.md` for the gates and the
kernel-side limits that keep the server below llama.cpp at concurrency > 1.

## Qwen3.5 dense megakernel (Phase 1)

`src/qwen35/` is a C/CUDA-only port of the batch-1 persistent decode megakernel: one cooperative
CTA per SM (sm_86, 40 SMs), a `cp.async` weight ring with fp16-exact Q8_0 scales, flag-in-data
activation handoff, GDN recurrence and split-K gated attention. Both shapes (0.8B `1024/3584`,
2B `2048/6144`) are template instantiations dispatched once at load.

```bash
./build/tuff --model Qwen3.5-0.8B-Q8_0.gguf --prompt "Hello" -c 512 -n 128 --warmup 32 --runs 3
./build/tuff --model Qwen3.5-0.8B-Q8_0.gguf --prompt-file prompt.txt -c 16384 -n 128 --warmup 32 --runs 3
./build/qwen35-batch-bench --model <gguf> --bench 8 --steps 128 --ctx 600
./build/tuff --model Qwen3.5-0.8B-Q8_0.gguf --prompt "Hello" -n 32 --dump-tokens
bash tests/qwen35/oracle.sh <gguf> 32      # G2: greedy == llama.cpp
bash tests/qwen35/g1.sh <gguf> 256 out/    # G1: logits vs the fp32 GGUF reference
```

Design, measurements and the unmet P1 target are documented in `QWEN35_MK_REPORT.md`.

## Roadmap (Notion page 12, "Runtime" decision of Nov 6)

1. Dense Qwen3.5 0.8B / 2B (same gated-delta-net + gated-attention mixers, dense MLP) via the generic decoder
   (`tuf_generic_plan.md`: new GGUF in hours, oracle gate in one command, ≥ llama.cpp tg128).
2. Server parity for agents and evals: tool calls (Qwen template), `chat_template_kwargs` (thinking switch),
   logprobs / top-k logprobs (KD data), sampling, parallel slots.
3. GDN state + KV snapshots at segment boundaries, tail rebuild, fork → context management in `../agent`.
4. Hook API: mid-layer hidden-state reads, in-process heads (fire / InfoNCE action / IDK), logit masks and
   grammars, result injection as prefill.
5. MTP speculation for small models with Soft-MTP heads; later Dory (looped block, SWA) and LFM short conv.
6. KV quantization for long context; purpose-trained draft model (see `PLAN.md`).

Documents: `PLAN.md` (open levers, hypothesis ledger), `ITERATION_LOG.md` (measured history),
`PERFORMANCE_GUIDE.md`, `GPU_REFACTOR_PLAN.md`, `SAMPLER_PLAN.md`, `QWEN35_ROADMAP.md`, `LFM25_ROADMAP.md`,
`wiki/`.

---

## Qwen3.6-35B-A3B engine (current fast path)

Single-stream, single-machine greedy decoder for **Qwen3.6-35B-A3B** (`Q4_K_S` GGUF, 19.91 GiB,
35.51 B params) on an **RTX 3070 Laptop 8 GB + Ryzen 9 5900HX + 64 GB DDR4**. Measured
**67.51 tok/s default** (lossless n-gram speculation) / **67.03 tok/s `--no-spec`** (5-prompt
corpus geomean, 32 warmup + 128 timed, 3 runs, order-alternated, 2026-10-17; ITERATION_LOG 92-94)
against **llama.cpp `885c5bb` 43.86 tok/s** measured in an earlier session; from a 0.39 tok/s
CPU-only floor. Speculation is token-identical to `--no-spec`; it gains +0.7% on the corpus. Spec
numbers published before 2026-10-17 (incl. 90 tok/s) came from a lossy verify and are withdrawn.

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
locality-bound, not capacity-bound. Speculative decode is lossless: a two-token verify with
per-column arithmetic identical to plain decode, a per-layer expert cache so both columns see the
plain-decode cache trajectory, and a no-write second column replayed only when accepted. The three dominant kernels
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
