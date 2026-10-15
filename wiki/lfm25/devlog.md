# LFM2.5 in tuff — Development Log & Runbook

Everything needed to resume LFM2.5 work efficiently: current state, how we work, every tool, how to reproduce every
result, the hard-won facts, the pitfalls, and what is next. Companion to [LFM25_ROADMAP.md](../../LFM25_ROADMAP.md)
(the plan + per-task results) and [architecture.md](../architecture.md).

Last updated: 2026-10-16. Branch: `lfm25-phase1` (pushed to `origin`; not merged to `main`).

---

## 1. Status snapshot

| Phase / task | State | Commit | Evidence |
| :--- | :--- | :--- | :--- |
| 0 — vtable, GGUF inspector, harness | done | `878eba0` | — |
| 1 — dense 1.2B pipeline (1.1–1.4) | done, **Gate 1 PASS** | `845a7ae` | memcheck clean |
| 2.1 — router (sigmoid top-4 of 32) | done | `1fef369` | 176/176 vs llama.cpp |
| 2.2 — in-VRAM expert FFN | done | `c18607c` | 44/44 vs llama.cpp, 1.7e-7 |
| 2.3 — 8B-A1B pipeline | done, **Gate 2 PASS** | `085f7e2` | 127/128 TF, robust median 0.96 |
| 3.1a — 64-expert router, strided expert FFN | done | `7e49ac7` | router 304/304, moe_ffn 76/76 |
| 3.1b/3.2 — VRAM expert cache, 24B | done, **Gate 3 PASS** | `8bfb4ed` | 8B forced-cache logits byte-identical |
| 4.1 — whole-trunk CUDA graph (1.2B, 8B) | done; A/B +6.69% (1.2B) / +7.00% (8B) | `e133b44` | graph logits byte-identical to pre-change engine; A/B §11 |
| 4.2 — fused in_proj+conv+gate+Q8_1, multi-block RMSNorm | done; A/B +5.46% / +5.65% / +4.79% | see git log | logits byte-identical to `e133b44` (3 models, graph on/off, forced cache) |
| 4.3 — async prefetch / CPU misses (24B) | not started | | |
| 5 — speculative decoding | not started | | |

Decode speed (RTX 3070 Laptop, `-c 1024 -n 128 --runs 3`, `balanced` profile, interleaved A/B means):

| Model | tuff | llama.cpp (tg128) | Note |
| :--- | :---: | :---: | :--- |
| 1.2B dense | 322 → 343 (graph) → **362** (4.2) | 331 | Phase-4 target ≥ 350 met |
| 8B-A1B | 214 → 229 (graph) → **242** (4.2) | 218 | fully VRAM resident |
| 24B-A2B | 110 → **115** (4.2) | 34.8 (`-ot exps=CPU`) | expert cache, 95.5 % hit rate on a short repetitive run |

---

## 2. Environment & paths

| Thing | Path / value |
| :--- | :--- |
| Repo | `/home/nii/workspace/tuf` (branch `lfm25-phase1`) |
| llama.cpp (oracle + linked lib) | `/home/nii/workspace/llama.cpp`, commit `620d2225`, libs in `build/bin` |
| gguf-py (FP64 ref, packers) | `/home/nii/workspace/llama.cpp/gguf-py` (imported by path) |
| Python with numpy | `/home/nii/workspace/local-llm/.venv/bin/python` — run with `-I`. System `python3` has **no numpy**. |
| 1.2B | `/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` |
| 8B-A1B | `/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf` |
| 24B-A2B | `/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf` |
| Gate scratch / caches | `/tmp/tuff_oracle/` (FP64 caches are slow to rebuild — see §5) |
| Baseline worktree for A/B | `/home/nii/workspace/tuf-base` (git worktree at `8bfb4ed`, built) |
| CUDA toolkit | distro 12.4 (`nvcc`, `compute-sanitizer` 2024.1); driver 595.99.02 |

Machine gotchas (each one cost time):
- **Model drive is not auto-mounted after reboot** → `udisksctl mount -b /dev/nvme0n1p1` (mounts at `/run/media/nii/pohjola`).
- **Driver/library mismatch after a driver update** (NVML "Driver/library version mismatch", compute-sanitizer hangs) → reboot.
- **compute-sanitizer 2024.1 deadlocks under `--default-stream per-thread`** (kernel + blocking `cudaMemcpy`). Use the
  sanitize tree: `cmake -S . -B build-sanitize -DTUFF_PER_THREAD_STREAM=OFF`. A hang shows as 0 % CPU, GPU idle at 210 MHz,
  host thread in `ep_poll`. A newer CUDA toolkit would remove the need (user's call).
- **Default context is 131072** → ~3.1 GiB FP32 KV on the 1.2B; always pass `-c 1024` for oracle/eval/bench so two GPU jobs fit.
- **GPU is shared**: never run two tuff/llama processes concurrently (OOM, or perturbed timings). One at a time.
- **FP64 references are CPU-heavy** (12–15 cores, up to ~23 GB RSS on the 24B); don't benchmark while one runs.
- **Power profile**: `scripts/check_power_state.sh` FAILs on `balanced` and `ab_interleaved.sh` refuses to run ("machine
  throttled"). Fix (needs sudo, user action): `powerprofilesctl set performance`.
  The user accepts `balanced` for benchmarks (2026-10-16): run with `FORCE=1` if the only FAILs are the profile lines, and
  record the profile next to every number; A and B of one A/B must run in the same profile.
- `pkill -f <pattern>` matches your own shell command line when the pattern appears in it — kill by PID instead.
- `/tmp` is wiped on reboot (scratch work is lost) — anything worth keeping goes into `tools/` or `tools/debug/`.

---

## 3. How we work (orchestrator + coding agent)

- **Orchestrator** (Claude Opus): pins down exact semantics from llama.cpp, builds golden fixtures/references, writes the
  brief, reviews the diff, re-runs every validation independently, runs negative controls, commits.
- **Coding agent** (Claude Sonnet 5.5 subagent, `model: sonnet`): implements against the brief, runs the listed checks,
  reports. Fast (1–4 min per task). Never commits.
- (`agy` with Gemini 3.8 Flash was tried: headless mode auto-denies shell commands unless `--dangerously-skip-permissions`;
  `--mode auto` unavailable on the account. Dropped in favour of Sonnet subagents.)

Brief template (what made agents succeed first try):
1. Task id + goal + which files to read first.
2. **Exact semantics already verified against llama.cpp** (formulas, layouts, tie rules, quant types per layer) — the agent
   must not re-derive them.
3. API to build (signatures), test to build (fixture format), constraints (files NOT to touch, no git ops, one GPU process
   at a time, zero new warnings, no roadmap ticking).
4. Validation commands with expected results; "if it fails, debug — fixtures are correct; report numbers, don't loosen".
5. Report format (files changed, verbatim result lines, surprises).

Parallelism: two agents only when their file sets are disjoint (e.g. engine C/CUDA vs. Python/shell gate tooling).

Orchestrator review checklist per task:
- Read the diff (semantics, indexing/strides, syncs, early returns before barriers, leaks on error paths).
- Clean rebuild, re-run the agent's checks myself.
- **Negative controls**: inject 2–4 realistic bugs (sed on a backup copy), confirm the test fails, restore, `cmp` restored.
- Sanitizers: memcheck (+ leak-check), racecheck, synccheck on the sanitize tree.
- Regression: all gates + all unit tests.
- Tick the roadmap with evidence, commit on `lfm25-phase1`, push.

---

## 4. Correctness methodology

### 4.1 Layers of evidence (strongest first)
1. **Byte-identity for optimizations / refactors** — same math ⇒ same bytes. Used for: expert cache (8B forced through an
   88-slot cache vs resident), CUDA graph (graph vs no-graph vs pre-change engine), GPU embedding (all 65 536 rows vs CPU
   `dequantize_row`). `tuff --teacher-force <ref> --dump-logits <f>` + `cmp`.
2. **Kernel unit tests vs llama.cpp fixtures** (1e-7 level): router (identical expert sets *and order*), expert FFN.
3. **Three-stage oracle gate** vs llama.cpp + an FP64 numpy reference (end to end).
4. **Negative controls** proving each test can fail.

### 4.2 The oracle gate (`tests/verify_lfm25_oracle.sh dense|moe8|moe`)
Why not 128/128 exact token match vs llama.cpp: llama.cpp cannot pass it against itself (GPU vs CPU diverge at token 5;
same commit could not reproduce the old reference past token 25). FP64 top-2 margins as small as 0.004–0.014 occur within
20 tokens while Q8_1 rounding noise is ~0.1 logit RMS, so after the first near-tie the streams fork by luck.

Stages (all must pass):
1. **Exact prefix**: first min(16, first reference step with margin < 0.5) greedy tokens identical (dense: 10, 8B: 2, 24B: 1).
2. **Teacher-forced top-1** on all 128 steps: a miss is OK only if the reference's own top1−top2 margin < 0.5, or — with FP64
   coverage — if ground truth does not prefer the reference token by ≥ 0.5 (signed: `fp64[ref] − fp64[tuff] < 0.5`).
3. **FP64 tolerance**: tuff's per-step logit RMS vs FP64 compared with llama.cpp's. Dense: mean ratio ≤ 1.10 and worst step
   ≤ 2.0. MoE (`--stat robust`): median ratio ≤ 1.10 and p90 ratio ≤ 1.5 — router near-ties (4th/5th expert gap median 0.017)
   flip experts in BOTH runtimes, making mean/worst luck-dominated (8B: mean 1.21 / worst 5.54 from flips vs median 0.96).

Current results:
- dense: prefix 10/10, TF 124/128 (4 near-ties), mean ratio 0.71, worst 0.95.
- moe8: prefix 2/2, TF 127/128 (1 near-tie), median 0.96, p90 1.11.
- moe (24B, reference = llama.cpp with experts on its CPU, noisier: RMS vs FP64 median 0.44): prefix 1/1, TF 114/128
  (13 near-ties + step 68 FP64-adjudicated: ref margin 0.954 but FP64 prefers ref by only 0.328), median 0.69, p90 0.55.
  FP64 argmax agrees with tuff 69/72 vs with the reference 63/72.

### 4.3 Honest caveats (raised by the user; to be fixed — see §11)
- Thresholds (0.5 / 1.10 / 1.5 / 2.0) are judgment calls, not derived.
- The gate was changed twice **after** a failure (robust MoE stat; FP64 adjudication). Both were evidence-backed and
  followed by a negative control that still fails — but "test fails → change test" is a pattern to distrust.
- Stage 1 is nearly vacuous for MoE (2 and 1 tokens).
- Planned fix: **golden self-regression gate** (freeze tuff's validated 128 tokens + logit hashes per model, require
  byte-identical output from every later change — a true 1:1 exact test), **lock thresholds** in the roadmap, and **wire the
  negative controls into the gate script** so they run every time.

### 4.4 Negative controls that must fail (keep using them)
| Bug injected | Caught by |
| :--- | :--- |
| conv taps k0/k1 swapped | dense gate: prefix 1/16, 55 hard misses, FP64 ratio 17.9 |
| router weights from biased score | router test 0/176 |
| router ignores selection bias | router 57/176; moe8 gate stage 2 one hard miss, stage 3 median ratio 3.92 |
| gate/up always expert id[0] | moe_ffn 0/44 |
| down reads expert-0 activations | moe_ffn 0/44 |
| unweighted expert sum | moe_ffn 0/44 |
| Q6_K down dispatched as Q4_K | moe_ffn fails exactly the 20 Q6_K cases |
| reversed router tie-break | **NOT caught** (no exact ties in real data; tie rule unverified vs ggml CUDA argsort) |

---

## 5. Reproduce everything

```bash
cd /home/nii/workspace/tuf
cmake -S . -B build && cmake --build build -j16 --target tuff lfm25-llama-ref lfm25-router-test lfm25-moe-ffn-test
cmake -S . -B build-sanitize -DTUFF_PER_THREAD_STREAM=OFF && cmake --build build-sanitize -j16 --target tuff lfm25-router-test lfm25-moe-ffn-test
```

Gates (dense ~30 s; moe8 ~1 min; moe ~15 s — FP64 cached; first FP64 build: dense ~1 min, 8B ~3 min, 24B 72 steps ~25 min):
```bash
./tests/verify_lfm25_oracle.sh dense
./tests/verify_lfm25_oracle.sh moe8
./tests/verify_lfm25_oracle.sh moe
```
Env knobs: `N` (128), `N_EXACT` (16), `N_FP` (32/32/72), `MIN_MARGIN` (0.5), `MAX_RATIO` (1.10), `MAX_P90_RATIO` (1.5),
`CTX` (1024), `MODEL_PATH`, `TUFF_BIN`, `REF_BIN`, `PYTHON`. The FP64 cache key is model size+mtime, ref sha1, N_FP.

Unit tests (seconds each):
```bash
M8=/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf
M24=/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf
./build/lfm25-router-test  $M8  tests/fixtures/lfm25_8b_router_cases.bin    # 176/176
./build/lfm25-router-test  $M24 tests/fixtures/lfm25_24b_router_cases.bin   # 304/304
./build/lfm25-moe-ffn-test $M8  tests/fixtures/lfm25_8b_moe_ffn_cases.bin   # 44/44, act 1.6e-7 out 1.7e-7
./build/lfm25-moe-ffn-test $M24 tests/fixtures/lfm25_24b_moe_ffn_cases.bin  # 76/76 (3 sporadic Q8_1 flips ≤ 3e-5)
```

Byte-identity checks:
```bash
# graph vs no-graph (also: vs a pre-change worktree build)
./build/tuff --model $M --prompt Hello -c 1024 -n 128 --temp 0 --teacher-force tests/ref_lfm25_dense_128.txt --dump-logits /tmp/a.bin
LFM25_CUDA_GRAPH=0 ./build/tuff ... --dump-logits /tmp/b.bin && cmp /tmp/a.bin /tmp/b.bin
# expert cache vs resident (8B)
LFM25_FORCE_EXPERT_CACHE=1 LFM25_EXPERT_CACHE_MB=600 LFM25_CACHE_STATS=1 ./build/tuff --model $M8 ... --dump-logits /tmp/c.bin
```

Sanitizers (sanitize tree; drop the cache size on the 24B to leave headroom):
```bash
compute-sanitizer --tool memcheck --leak-check full ./build-sanitize/tuff --model $M -c 256 -n 8 --temp 0 --prompt Hello --dump-tokens
LFM25_EXPERT_CACHE_MB=1000 compute-sanitizer --tool memcheck --leak-check full ./build-sanitize/tuff --model $M24 -c 256 -n 6 ...
compute-sanitizer --tool racecheck|synccheck ./build-sanitize/lfm25-moe-ffn-test $M8 tests/fixtures/lfm25_8b_moe_ffn_cases.bin
```

Speed:
```bash
./build/tuff --model $M --prompt Hello -c 1024 -n 128 --temp 0 --runs 3          # "Pipeline Generation Speed"
/home/nii/workspace/llama.cpp/build/bin/llama-bench -m $M -ngl 99 -p 0 -n 128 -r 3   # 24B: add -ot "ffn_(up|down|gate)_exps=CPU"
# formal A/B (needs performance power profile):
MODEL=$M TUFF_ARGS="-c 1024" N=128 WARMUP=32 RUNS=3 ./scripts/ab_interleaved.sh /home/nii/workspace/tuf-base /home/nii/workspace/tuf 6
```

Regenerate references (byte-reproducible; recipe = command, llama.cpp commit, model sha256, prompt tokens, GPU):
```bash
./tests/gen_lfm25_ref.sh dense|moe8|moe      # moe adds --cpu-experts; writes tests/ref_lfm25_<mode>_128.{txt,margins.txt,recipe}
```
Regenerate fixtures:
```bash
LLAMA=/home/nii/workspace/llama.cpp
gcc -O2 tools/lfm25_router_fixtures.c -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o /tmp/fixdump
/tmp/fixdump $M8 /tmp/d8 8 ;  /tmp/fixdump $M24 /tmp/d24 8 --cpu-experts
PY="/home/nii/workspace/local-llm/.venv/bin/python -I"
$PY tools/lfm25_pack_router_fixtures.py /tmp/d8  tests/fixtures/lfm25_8b_router_cases.bin
$PY tools/lfm25_pack_router_fixtures.py /tmp/d24 tests/fixtures/lfm25_24b_router_cases.bin --layers 2-39
$PY tools/lfm25_pack_moe_fixtures.py    /tmp/d8  tests/fixtures/lfm25_8b_moe_ffn_cases.bin
$PY tools/lfm25_pack_moe_fixtures.py    /tmp/d24 tests/fixtures/lfm25_24b_moe_ffn_cases.bin --layers 2-39 --simulate-gpu $M24
```
(All four regenerate byte-identically on llama.cpp `620d2225`.)

---

## 6. Tool reference

| Tool | What it does |
| :--- | :--- |
| `tests/verify_lfm25_oracle.sh` | three-stage gate (§4.2) |
| `tests/gen_lfm25_ref.sh` | reproducible llama.cpp references + margins + recipe |
| `tools/lfm25_llama_ref.c` (`build/lfm25-llama-ref`) | llama.cpp single-token GPU decode: greedy tokens, margins, logits; `--teacher`, `--cpu-experts` |
| `tools/lfm25_fp64_ref.py` | FP64 numpy forward pass (lfm2 + lfm2moe, on-demand expert dequant, LRU) |
| `tools/lfm25_gate.py` | `exact` / `teacher` / `tolerance` stage logic |
| `tools/lfm25_router_fixtures.c` | llama.cpp cb_eval dump of router/expert tensors per layer per step |
| `tools/lfm25_pack_{router,moe}_fixtures.py` | pack dumps into test fixtures (`--simulate-gpu` = numpy model of the CUDA expert path) |
| `tests/test_lfm25_router.cu`, `tests/test_lfm25_moe_ffn.cu` | kernel unit tests (8B contiguous / 24B slot-pool modes) |
| `tools/inspect_lfm25_gguf.py` | tensor/type/memory inventory of a GGUF |
| `tools/debug/lfm25_llama_dump.c` | dump EVERY llama.cpp graph tensor of a BOS decode (layer-by-layer bisection) |
| `tools/debug/lfm25_detok.c` | token ids → text with the model vocab |
| `tools/debug/lfm25_router_check.py` | numpy router semantics vs llama.cpp dumps |
| `tools/debug/lfm25_moe_ffn_check.py` | numpy expert-FFN (Q8_1) semantics vs llama.cpp CUDA dumps |
| `scripts/ab_interleaved.sh` | order-balanced paired A/B of two trees (`MODEL`, `TUFF_ARGS`) |
| `scripts/check_power_state.sh` | power / clock gate for benchmarks |
| tuff flags | `--teacher-force <ids>` (+`--dump-logits`), `--dump-tokens`, `--dump-logits`, `--runs`, `-c` |
| tuff env | `LFM25_CUDA_GRAPH=0`, `LFM25_FORCE_EXPERT_CACHE=1`, `LFM25_EXPERT_CACHE_MB`, `LFM25_VRAM_RESERVE_MB` (512), `LFM25_CACHE_STATS=1`, `LFM25_TRY_HOST_REGISTER=1` |

---

## 7. Verified model facts & semantics (each checked against llama.cpp)

| | 1.2B | 8B-A1B | 24B-A2B |
| :--- | :---: | :---: | :---: |
| arch string | `lfm2` | `lfm2moe` | `lfm2moe` |
| layers | 16 (10 conv, 6 attn) | 24 (attn 2,6,10,14,18,21) | 40 (attn every 4th from 2) |
| dense FFN width | 8192 | 7168 (layers 0–1) | 11776 (layers 0–1) |
| experts / top-k / expert ffn | — | 32 / 4 / 1792 | 64 / 4 / 1536 |
| expert down type | — | Q4_K ×12, Q6_K ×10 layers | Q4_K ×20, Q6_K ×18 |
| rope base / vocab | 1e7 / 65536 | 5e6 / 128000 | 1e7 / 128000 |
| BOS / EOS | 1 / 2 | 124894 / 124900 | 124894 / 124900 |
| "Hello" tokens | 1 36309 | 124894 35808 | 124894 35808 |

- ShortConv: `in_proj` → [b, c, x]; u = b⊙x; y = s0·k0 + s1·k1 + u·k2 (k2 = current); state ← (s1, u); out = c⊙y.
  Conv weight GGUF [3, 2048] ⇒ `w[i*3 + tap]`. Final norm is `token_embd_norm`; LM head tied to `token_embd` (Q6_K).
- Attention: QK-RMSNorm per head (dim 64), NeoX RoPE, scale 1/√64. llama.cpp keeps K/V in F16 (tuff F32).
- Matmuls: llama.cpp CUDA quantizes activations to Q8_1 (d = amax/127, q = roundf, ds = half2(d, sum)); tuff does the
  same and agrees to ~1e-9 per matmul. Router `ffn_gate_inp` is F32 on the un-quantized FFN-norm output.
- **Router** (`expert_gating_func = 2`): p = sigmoid(W·x); top-4 by p + `exp_probs_b` (bias only for selection; ties →
  lower index assumed); w = p[top] / max(Σ, 6.103515625e-5); no scale. (The roadmap's original "softmax" was wrong.)
- **Expert FFN**: per expert act = silu(gate·Q8_1(x))·(up·Q8_1(x)); each act requantized to Q8_1 on its own;
  y = down·Q8_1(act); out = ((w0y0 + w1y1) + w2y2) + w3y3; weights after down.
- **Tokenizer**: if `tokenizer.ggml.add_bos_token` is absent, llama.cpp defaults add_bos = true for the llama3 group of
  pre-tokenizers incl. `lfm2` (the 8B/24B GGUFs have no key). Qwen uses `qwen2`/`qwen35` — unaffected.

---

## 8. Debugging techniques that worked

1. **Teacher forcing** isolates per-step accuracy from stream divergence.
2. **FP64 ground truth decides who is wrong**: (tuff − fp64) vs (llama − fp64). Found that tuff was more accurate than
   llama.cpp on the 1.2B (0.106 vs 0.151 RMS) and on the 24B (FP64 sides with tuff 69/72 vs 63/72).
3. **Isolated matmul check**: feed llama.cpp's own input tensor, compare its output with W·x vs W·Q8_1(x) → identifies the
   activation quantization scheme.
4. **Layer bisection**: `tools/debug/lfm25_llama_dump.c` + a temporary per-layer `cudaMemcpy` dump in
   `lfm25_engine_forward_token` (env-gated, reverted after) → first layer whose relative error jumps.
5. **Routing-flip analysis (MoE)**: log per-(position, layer) top-4 sets from tuff (temporary env-gated D2H of
   `d_topk_ids`), llama.cpp (`ffn_moe_topk` dumps) and FP64 (patched copy of the FP64 script logging top-4 and the 4th–5th
   selection gap) → error spikes = expert flips at gaps ≤ 0.04, in both runtimes.
6. **Sporadic vs systematic**: temporarily tighten a test's tolerance and list failing cases — few scattered outliers =
   Q8_1 rounding flips; every case on one layout = a real bug.
7. **Hang triage**: 0 % CPU + idle GPU clocks + `ep_poll`/`futex` in `/proc/<pid>/task/*/wchan` = deadlock, not slowness.
   Bisect with minimal CUDA programs (that is how the sanitizer/per-thread-stream bug was found).

---

## 9. Pitfalls log (don't repeat)

- Stale or unrecorded references: always generate with `tests/gen_lfm25_ref.sh` (recipe recorded) and re-run twice + `cmp`.
- Comparing against a dump produced by the code under test is circular — compare against a pre-change worktree build.
- Changing the gate after a failure needs evidence (ground truth) + a negative control afterwards (see §4.3).
- `cudaHostRegister` on the read-only `MAP_SHARED` GGUF mmap fails on this driver (ReadOnly: not supported; default:
  invalid argument) and pollutes sanitizer output → opt-in only.
- Engine-init early returns after allocations leak — put shape guards before the first allocation.
- Heredoc quoting: `\n` inside an unquoted heredoc that generates Python ends up as a real newline in an f-string.
- Subagent edits can collide; give parallel agents disjoint file sets and tell them to retry once on GPU OOM.

---

## 10. Design decisions (and why)

- **Separation principle**: LFM code in `src/lfm25_*` only; the Qwen expert cache (`src/gpu.cu`) was not reused.
- **24B misses = PCIe upload + GPU compute**, not AVX2 CPU compute: identical numerics by construction (byte-identity test
  possible); CPU evaluation changes activation quantization and would need its own validation → Phase 4 optimization.
- **Expert cache**: three slot pools (gate/up/down) with one slot index; down slots sized for Q6_K; kernels take per-tensor
  byte strides so contiguous tensors and pools share one code path. LRU baseline; LFRU later.
- **24B oracle reference** uses llama.cpp with experts on its CPU backend (`--cpu-experts`) — the only way it fits 8 GB;
  its noise is handled by FP64 adjudication.
- **CUDA graph** only for models without host round-trips (1.2B, resident 8B); the 24B cache path (host sync per MoE
  layer for cache lookup) stays un-graphed.

---

## 11. Next steps (in order)

1. **Task 4.1 A/B — done 2026-10-16** (balanced profile, accepted by the user; `FORCE=1`, all other power checks ok):
   1.2B 321.8 → 343.4 tok/s, +6.69% ± 0.17 (95% CI [+6.34, +7.04]); 8B 214.2 → 229.2 tok/s, +7.00% ± 0.08
   ([+6.85, +7.16]); 6 interleaved rounds, `8bfb4ed` vs `e133b44`. Gates re-run on clean `e133b44`: identical numbers to §4.2. Correctness is done: graph logits byte-identical to the pre-change
   engine on 1.2B and 8B, all three gates unchanged, memcheck clean (1.2B, 8B). Speed 346 tok/s vs target 350 — the
   remaining gap is kernel time → Task 4.2. Logits D2H is now only done when logit capture is on.
2. **Strengthen the gate** (user request): golden self-regression (byte-identical tokens + logit hashes per model vs the
   last validated commit), thresholds locked in the roadmap, negative controls automated in the gate script.
3. **Task 4.2 — done 2026-10-16**: fused in_proj+conv kernel + multi-block RMSNorm; byte-identical, A/B +5.46% (1.2B,
   361.7 tok/s), +5.65% (8B), +4.79% (24B); evidence in the roadmap. Leftovers: `lfm25_gpu_shortconv_step` and
   `eng->d_in_proj` are now unused (kept as the reference path; remove when convenient). Next speed candidate on the 1.2B:
   Q6_K `ffn_down` matvec efficiency (~270 vs ~350 GB/s); re-profile first.
4. **Task 4.3 / 24B speed**: pinned staging buffer for misses (cudaHostRegister on the mmap fails), async prefetch of
   next-layer experts, frequency-aware LFRU, measure hit rate on diverse text (not just "Hello"); optional CPU-miss
   evaluation with its own validation.
5. **Phase 5** speculative decoding (conv-state rollback is O(1): 2×2048 floats per layer).
6. Open gaps: router tie rule unverified vs ggml CUDA argsort; Qwen tokenizer change only reasoned (pre-tokenizer names),
   not regression-tested on a Qwen model; benchmarks so far not under a locked power profile; `tuf-base` worktree can be
   removed with `git worktree remove ../tuf-base` when no longer needed.
