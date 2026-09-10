# Tuff Performance & Profiling Guide

This guide documents the target hardware architecture, profiling methodology, profiling tools, and kernel performance optimization principles for Tuff on **Qwen 3.6 35B hybrid MoE**.

---

## 1. Target Hardware Specifications

### GPU: NVIDIA GeForce RTX 3070 Laptop GPU (Ampere sm_86 / GA104)
- **Streaming Multiprocessors (SMs)**: 40 SMs
- **CUDA Cores**: 5,120 FP32 cores (128 per SM)
- **Tensor Cores**: 160 3rd Gen Tensor Cores (4 per SM)
- **VRAM Total**: 8,192 MiB GDDR6 (256-bit bus @ 6000 MHz)
- **Theoretical Peak Bandwidth**: **384.0 GB/s**
- **L2 Cache**: 4.0 MiB
- **Warp Schedulers**: 4 per SM (up to 48 warps = 1,536 active threads per SM)
- **Register File**: 65,536 32-bit registers per SM
  - *Full 100% Occupancy (1536 threads)*: $\le 32$ registers per thread.
  - *50% Occupancy (768 threads)*: $\le 64$ registers per thread.
  - *Maximum limit*: 255 registers per thread (drops SM occupancy to 1 warp block and spills into DRAM Local Memory).

### CPU: AMD Ryzen 9 5900HX (Zen 3)
- **Cores / Threads**: 8 physical cores / 16 threads (Single CCX)
- **L3 Cache**: 16 MiB unified
- **SIMD / Vector ISA**: AVX2, FMA3, BMI2, F16C
- **System Memory**: DDR4-3200 dual-channel (~45–50 GB/s peak bandwidth)

---

## 2. Profiling & Measurement Toolset

Tuff provides three complementary profiling layers:

### 1. Built-in Deep CUPTI GPU Kernel Trace (`TUFF_TRACE=1`)
Captures exact nanosecond-precision hardware metrics directly from the CUDA driver for every individual kernel launch:
- Demangled kernel name, execution count, total time (ms), average/min/max duration (µs), GPU time percentage, register count per thread, grid/block dimensions, shared memory, and PCIe transfer throughput.

```bash
# Run a 32-token CUPTI hardware trace
TUFF_TRACE=1 ./build/tuff -m <model.gguf> -p "Explain Casimir effect" -n 32 --warmup 8 --runs 1
```

### 2. End-to-End Stage Timer Breakdown (`TUFF_PROFILE=1`)
Breaks down token decode latency across architectural pipeline stages:
- `t_delta`: Delta-Net RNN recurrence & projections (30 layers).
- `t_attn`: Multi-head attention & RoPE (10 layers).
- `t_expert`: MoE Expert pipeline (Shared Expert, Gate/Up, Down, CPU miss reductions, cache synchronization).
- `t_expert_sync` / `cachesync`: Driver event waits & GPU/CPU pipeline overlap.

```bash
TUFF_PROFILE=1 ./build/tuff -m <model.gguf> -p "Hello" -n 128 --warmup 32
```

### 3. Isolated Kernel Microbenchmark Suite (`./build/microbench-gpu`)
Runs isolated CUDA kernels (RMS-Norm, MMVQ, MoE Batched GEMV, Attention, GDN) in high-iteration loops (500–10,000 runs) with warmup:

```bash
./build/microbench-gpu
```

### 4. Honest Multi-Domain A/B Corpus Benchmark
5 diverse domains, 2 randomized passes, $df=4$, Student's $t$ 95% Confidence Interval:

```bash
python3 scripts/run_corpus_benchmark.py --configs "Baseline:FLAG=0" "Hypothesis:FLAG=1" --mode randomized --passes 2 -n 512 --warmup 128 --runs 1
```

---

## 3. High-Performance CUDA Best Practices on Ampere

1. **SM Grid Sizing**:
   - Ensure grid dimensions are multiples of SM count (40 SMs). A launch with $< 40$ blocks under-utilizes the GPU.
2. **Register Allocation & Launch Bounds**:
   - Always declare `__launch_bounds__(threads_per_block, min_blocks_per_sm)` on critical kernels to guarantee that register usage stays within $\le 48–64$ registers per thread, preventing register spilling into local memory.
3. **Multi-Expert Batched Dispatches**:
   - Rather than launching separate Gate/Up/Down kernels sequentially for each active expert hit ($k \in [0..8]$), use a 2D grid (`grid.y = n_hits`) to launch a single batched kernel.
4. **Vectorized LDG.128 & Warp-Shuffle Reductions**:
   - Utilize 128-bit `float4` loads to maximize memory bus throughput.
   - Use `__shfl_xor_sync` / `__shfl_down_sync` within 32-thread warps instead of shared memory buffers where possible.

---

## 4. Mathematical Roofline Model for MoE Expert Caching & MLA

Single-token autoregressive generation latency $T_{\text{token}}$ for a hybrid-memory MoE decoder is modeled by:

$$T_{\text{token}} = T_{\text{dense}} + T_{\text{routed}} + T_{\text{driver}}$$

Expanding the memory streaming terms:

$$T_{\text{token}} = \frac{\text{Bytes}_{\text{dense}}}{B_{\text{GPU}}} + H \cdot \left(\frac{\text{Bytes}_{\text{routed}}}{B_{\text{GPU}}}\right) + (1 - H) \cdot \left(\frac{\text{Bytes}_{\text{routed}}}{B_{\text{RAM\_CPU}}}\right)$$

Where:
- $B_{\text{GPU}} \approx 330\text{ GB/s}$ (Sustained Ampere GA104 GDDR6 memory bandwidth)
- $B_{\text{RAM\_CPU}} \approx 40\text{ GB/s}$ (Dual-channel DDR4-3200 sustained read bandwidth with AVX2 workers)
- $H \in [0.0, 1.0]$ is the empirical VRAM LFRU cache hit rate.

### Multi-Head Latent Attention (MLA) Memory Scaling
For architectures using low-rank latent KV compression ($d_c = 512, d_r = 64$):

$$\text{KV Memory} = N_{\text{layers}} \times (d_c + d_r) \times N_{\text{ctx}} \times \text{bytes\_per\_element}$$

Because MLA reduces KV cache size to ~576 bytes/token per layer, large context windows (64k–128k) consume only 2.3–4.5 GB VRAM in Q8/FP8, preserving significant VRAM capacity for the LFRU expert cache.
