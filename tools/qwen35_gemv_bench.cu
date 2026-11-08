// P2a gate: standalone microbench of the batched (tensor-core) Q8_0 GEMV on the real
// Qwen3.5-0.8B shapes, sm_86.
//
// Design measured here (and integrated into the megakernel afterwards):
//
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32.  M = 16 weight rows, N = 8 batch
//   entries per n-tile (BT/8 tiles, BT < 8 padded with zero columns), K = 16.
//
//   Weights are Q8_0, repacked at load time into a K-tiled fragment layout:
//
//       W[group][kt][544]     kt = k/32 (one Q8_0 block), group = 16 rows
//       within a group tile:  for h in {0,1} (k16 half of the block)
//                               for q in {0,1} (columns 0..7 / 8..15 of the half)
//                                 lane L: 4 bytes = (r0,c0)(r0,c0+1)(r1,c0)(r1,c0+1)
//                            16 fp16 block scales at +512
//
//   so a lane's 4-byte load gives the two int8 of one MMA A register, a warp's load is
//   one dense 128-byte transaction, and the 16x32 tile is exactly 544 bytes = the source
//   16 x (32 + 2) bytes.  int8 -> fp16 in registers is exact; the fp16 block scale is
//   applied in fp32 after the two k16 MMAs, blocks in ascending order.  That makes each
//   (row, entry) a fixed arithmetic sequence independent of BT and of the other columns
//   (G4).
//
//   The CTA streams whole K-tile chunks (G groups x 544 B) through a cp.async ring
//   (NS stages).  One activation k32 tile, f16, is staged in shared memory per K tile
//   (two buffers) and reused by every group in the chunk; its 8x8 sub-tiles are stored
//   row-major so a warp's B-fragment load is a conflict-free 128-byte smem read.
//
// Usage: qwen35-gemv-bench [--iters N]
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define Q35_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    exit(1); } } while (0)

namespace {

constexpr int NWARPS = 16;
constexpr int NTHREADS = NWARPS * 32;
constexpr int GROUP_BYTES = 544;
constexpr int GMAX = 32;                 // groups per row pass (2 per warp)
constexpr int SMEM_BUDGET = 90000;

__device__ __forceinline__ uint32_t smem_u32(const void *p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint32_t cvt_pair(uint32_t v) {
    __half2 h = __floats2half2_rn((float)(int8_t)(v & 0xff), (float)(int8_t)((v >> 8) & 0xff));
    return *reinterpret_cast<uint32_t *>(&h);
}

__device__ __forceinline__ void mma_m16n8k16(float d[4], uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void store_group(float *__restrict__ out, int rows, int group,
                                            int lane, const float acc[4], int e, int BT) {
    const int r0 = lane >> 2, r1 = r0 + 8;
    if (e < BT) {
        out[(size_t)e * rows + group * 16 + r0] = acc[0];
        out[(size_t)e * rows + group * 16 + r1] = acc[2];
    }
    if (e + 1 < BT) {
        out[(size_t)(e + 1) * rows + group * 16 + r0] = acc[1];
        out[(size_t)(e + 1) * rows + group * 16 + r1] = acc[3];
    }
}

__device__ __forceinline__ void issue_chunk(const uint8_t *__restrict__ W, int n_groups, int t,
                                            int gb, int G, char *dst) {
    const uint8_t *src = W + ((size_t)t * n_groups + gb) * GROUP_BYTES;
    const int n = G * GROUP_BYTES;
    for (int i = threadIdx.x * 16; i < n; i += NTHREADS * 16) cp_async16(dst + i, src + i);
    cp_commit();
}

// Direct-load GEMV: each warp owns SPLIT K-slices of GPW 16-row groups and keeps DEPTH
// k32 tiles of A fragments in registers, so the loop has DEPTH independent global loads
// in flight.  No shared-memory staging, no block barrier in the K loop.  Weights use
// __ldcs (streaming, evict-first); activation fragments use __ldg.  SPLIT is a function
// of the shape only, never BT, so every column's arithmetic is identical at every batch
// size (G4).  The activation is f16 in fragment-major global layout
// [k16][n-tile][lane][2 halves]: one coalesced 128-byte transaction per B load.
template <int BT, int SPLIT, int GPW, int DEPTH>
__global__ void __launch_bounds__(NTHREADS, 1)
gemv_direct(const uint8_t *__restrict__ W, const uint32_t *__restrict__ abf,
            int n_groups, int n_kt, int rows, float *__restrict__ out) {
    constexpr int NT = (BT + 7) / 8;
    constexpr int GPASS = NWARPS / SPLIT;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g0 = (int)((long long)n_groups * blockIdx.x / gridDim.x);
    const int g1 = (int)((long long)n_groups * (blockIdx.x + 1) / gridDim.x);
    const int r0 = lane >> 2, c0 = (lane & 3) * 2;
    __shared__ float red[NWARPS * 32 * 4 * 4];   // [warp][lane][n-tile][4]

    for (int gb = g0; gb < g1; gb += (SPLIT > 1 ? GPASS : GPW * NWARPS)) {
        int g[GPW];
        bool act[GPW];
        int sp = 0, kt0 = 0, kt1 = n_kt;
#pragma unroll
        for (int k = 0; k < GPW; k++) {
            g[k] = gb + warp + k * NWARPS;
            act[k] = g[k] < g1;
        }
        if constexpr (SPLIT > 1) {
            const int gp = warp / SPLIT;
            sp = warp % SPLIT;
            g[0] = gb + gp;
            act[0] = g[0] < g1;
            kt0 = (int)((long long)n_kt * sp / SPLIT);
            kt1 = (int)((long long)n_kt * (sp + 1) / SPLIT);
        }
        const int nk = kt1 - kt0;

        uint32_t p[DEPTH][GPW][4];
        float sc[DEPTH][GPW][2];
        auto load = [&](int u, int d) {
            const int t = kt0 + u;
#pragma unroll
            for (int k = 0; k < GPW; k++) {
                if (!act[k]) continue;
                const uint8_t *gt = W + ((size_t)g[k] * n_kt + t) * GROUP_BYTES;
                p[d][k][0] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((0 * 32 + lane) * 4)));
                p[d][k][1] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((1 * 32 + lane) * 4)));
                p[d][k][2] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((2 * 32 + lane) * 4)));
                p[d][k][3] = __ldcs(reinterpret_cast<const uint32_t *>(gt + ((3 * 32 + lane) * 4)));
                sc[d][k][0] = __half2float(*reinterpret_cast<const __half *>(gt + 512 + 2 * r0));
                sc[d][k][1] = __half2float(*reinterpret_cast<const __half *>(gt + 512 + 2 * (r0 + 8)));
            }
        };
        for (int d = 0; d < DEPTH; d++) if (d < nk) load(d, d);

        float acc[GPW][NT][4];
#pragma unroll
        for (int k = 0; k < GPW; k++)
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[k][n][j] = 0.f;

        for (int u0 = 0; u0 < nk; u0 += DEPTH) {
#pragma unroll
            for (int d = 0; d < DEPTH; d++) {
                const int u = u0 + d;
                if (u >= nk) break;
                const int t = kt0 + u;
                uint32_t b[NT][2][2];
#pragma unroll
                for (int n = 0; n < NT; n++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        b[n][h][0] = __ldg(abf + (size_t)((t * 4 + 2 * h + 0) * NT + n) * 32 + lane);
                        b[n][h][1] = __ldg(abf + (size_t)((t * 4 + 2 * h + 1) * NT + n) * 32 + lane);
                    }
#pragma unroll
                for (int k = 0; k < GPW; k++) {
                    if (!act[k]) continue;
                    float tmp[NT][4];
#pragma unroll
                    for (int n = 0; n < NT; n++)
#pragma unroll
                        for (int j = 0; j < 4; j++) tmp[n][j] = 0.f;
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const uint32_t a0 = cvt_pair(p[d][k][h * 2 + 0]);
                        const uint32_t a1 = cvt_pair(p[d][k][h * 2 + 0] >> 16);
                        const uint32_t a2 = cvt_pair(p[d][k][h * 2 + 1]);
                        const uint32_t a3 = cvt_pair(p[d][k][h * 2 + 1] >> 16);
#pragma unroll
                        for (int n = 0; n < NT; n++)
                            mma_m16n8k16(tmp[n], a0, a1, a2, a3, b[n][h][0], b[n][h][1]);
                    }
#pragma unroll
                    for (int n = 0; n < NT; n++) {
                        acc[k][n][0] = fmaf(tmp[n][0], sc[d][k][0], acc[k][n][0]);
                        acc[k][n][1] = fmaf(tmp[n][1], sc[d][k][0], acc[k][n][1]);
                        acc[k][n][2] = fmaf(tmp[n][2], sc[d][k][1], acc[k][n][2]);
                        acc[k][n][3] = fmaf(tmp[n][3], sc[d][k][1], acc[k][n][3]);
                    }
                }
                if (u + DEPTH < nk) load(u + DEPTH, d);
            }
        }
        if constexpr (SPLIT > 1) {
            __syncthreads();
#pragma unroll
            for (int n = 0; n < NT; n++) {
                float *rb = red + ((warp * 32 + lane) * NT + n) * 4;
                rb[0] = acc[0][n][0]; rb[1] = acc[0][n][1];
                rb[2] = acc[0][n][2]; rb[3] = acc[0][n][3];
            }
            __syncthreads();
            if (sp == 0 && act[0]) {
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    float fin[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int q = 0; q < SPLIT; q++) {
                        const float *rp = red + (((warp + q) * 32 + lane) * NT + n) * 4;
                        fin[0] += rp[0]; fin[1] += rp[1]; fin[2] += rp[2]; fin[3] += rp[3];
                    }
                    store_group(out, rows, g[0], lane, fin, n * 8 + c0, BT);
                }
            }
            __syncthreads();
        } else {
#pragma unroll
            for (int k = 0; k < GPW; k++) {
                if (!act[k]) continue;
#pragma unroll
                for (int n = 0; n < NT; n++) {
                    float a2[4] = {acc[k][n][0], acc[k][n][1], acc[k][n][2], acc[k][n][3]};
                    store_group(out, rows, g[k], lane, a2, n * 8 + c0, BT);
                }
            }
        }
    }
}

__global__ void spin_kernel(const uint4 *p, size_t n, int ms, unsigned *sink) {
    unsigned long long t0, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    unsigned acc = 0;
    size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    do {
        for (int u = 0; u < 64; u++) {
            const uint4 v = __ldcs(p + (i & (n - 1)));
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
            i += 1024 * 40;
        }
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - t0 < (unsigned long long)ms * 1000000ull);
    if (acc == 0xdeadbeefu) *sink = acc;
}

void warm_gpu(int ms) {
    static uint4 *buf = nullptr;
    static unsigned *sink = nullptr;
    const size_t bytes = 64u << 20;
    if (!buf) {
        Q35_CHECK(cudaMalloc(&buf, bytes));
        Q35_CHECK(cudaMalloc(&sink, 4));
        Q35_CHECK(cudaMemset(buf, 1, bytes));
    }
    spin_kernel<<<40, 256>>>(buf, bytes / 16, ms, sink);
    Q35_CHECK(cudaDeviceSynchronize());
}

__global__ void bw_read_kernel(const uint4 *__restrict__ p, size_t n, unsigned *sink) {
    unsigned acc = 0;
    for (size_t i = (size_t)blockIdx.x * 512 + threadIdx.x; i < n; i += (size_t)gridDim.x * 512) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) *sink = acc;
}

__global__ void bw_read_kernel2(const uint4 *__restrict__ p, size_t n, unsigned *sink) {
    const size_t i0 = n * blockIdx.x / gridDim.x, i1 = n * (blockIdx.x + 1) / gridDim.x;
    unsigned acc = 0;
    for (size_t i = i0 + threadIdx.x; i < i1; i += 512) {
        const uint4 v = __ldcs(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) *sink = acc;
}

void bw_curve() {
    unsigned *sink;
    Q35_CHECK(cudaMalloc(&sink, 4));
    const size_t maxb = 512u << 20;
    uint4 *p;
    Q35_CHECK(cudaMalloc(&p, maxb));
    Q35_CHECK(cudaMemset(p, 1, maxb));
    printf("\n%-10s %10s %10s %8s\n", "MB", "ms", "GB/s", "%373.4");
    for (int mb = 2; mb <= 512; mb <<= 1) {
        const size_t n = ((size_t)mb << 20) / 16;
        bw_read_kernel<<<40, 512>>>(p, n, sink);
        Q35_CHECK(cudaDeviceSynchronize());
        warm_gpu(100);
        cudaEvent_t a, b;
        Q35_CHECK(cudaEventCreate(&a));
        Q35_CHECK(cudaEventCreate(&b));
        const int it = (mb <= 16) ? 200 : (mb <= 64 ? 40 : 10);
        Q35_CHECK(cudaEventRecord(a));
        for (int i = 0; i < it; i++) bw_read_kernel<<<40, 512>>>(p, n, sink);
        Q35_CHECK(cudaEventRecord(b));
        Q35_CHECK(cudaEventSynchronize(b));
        float ms = 0, ms2 = 0;
        Q35_CHECK(cudaEventElapsedTime(&ms, a, b));
        Q35_CHECK(cudaEventRecord(a));
        for (int i = 0; i < it; i++) bw_read_kernel2<<<40, 512>>>(p, n, sink);
        Q35_CHECK(cudaEventRecord(b));
        Q35_CHECK(cudaEventSynchronize(b));
        Q35_CHECK(cudaEventElapsedTime(&ms2, a, b));
        const double secs = ms * 1e-3, secs2 = ms2 * 1e-3;
        const double gbs = (double)mb * 1e6 * it / secs / 1e9;
        const double gbs2 = (double)mb * 1e6 * it / secs2 / 1e9;
        printf("%-10d %10.4f %10.1f %8.1f%%   slice: %10.1f %8.1f%%\n", mb, secs * 1e3 / it, gbs,
               100 * gbs / 373.4, gbs2, 100 * gbs2 / 373.4);
        Q35_CHECK(cudaEventDestroy(a));
        Q35_CHECK(cudaEventDestroy(b));
    }
    cudaFree(p);
    cudaFree(sink);
}

// CTA-streaming GEMV for shapes with few 16-row groups.  The CTA walks its groups one at
// a time; the whole CTA copies that group's K tiles through a cp.async ring of CHUNKT-tile
// chunks, so each CTA reads its weight slice as ONE long sequential stream.  The ring is
// NOT drained at group boundaries: chunks of the next group are already in flight while the
// current group's partials are reduced.  All 16 warps share a group, taking tiles warp,
// warp+16, ...; partials are summed once per group in warp order.
template <int BT, int NS, int CHUNKT, bool MATH = true>
__global__ void __launch_bounds__(NTHREADS, 1)
gemv_cta(const uint8_t *__restrict__ W, const uint32_t *__restrict__ abf,
         int n_groups, int n_kt, int rows, float *__restrict__ out, int reps) {
    constexpr int NT = (BT + 7) / 8;
    extern __shared__ __align__(16) char smem[];
    char *ring = smem;
    constexpr int STAGE = CHUNKT * GROUP_BYTES;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g0 = (int)((long long)n_groups * blockIdx.x / gridDim.x);
    const int g1 = (int)((long long)n_groups * (blockIdx.x + 1) / gridDim.x);
    const int r0 = lane >> 2, c0 = (lane & 3) * 2;
    __shared__ float red[NWARPS * 32 * 4];   // one n-tile at a time
    const int nch = (n_kt + CHUNKT - 1) / CHUNKT;
    const int total = (g1 - g0) * nch;
    if (total == 0) return;
    for (int rep = 0; rep < reps; rep++) {
    const uint8_t *Wr = W + (size_t)rep * n_groups * n_kt * GROUP_BYTES;

    auto chunk_g = [&](int u) { return g0 + u / nch; };
    auto chunk_c = [&](int u) { return u % nch; };
    auto issue = [&](int u, int slot) {
        const int g = chunk_g(u), c = chunk_c(u);
        const int t0 = c * CHUNKT;
        const int nt = n_kt - t0 < CHUNKT ? n_kt - t0 : CHUNKT;
        const uint8_t *src = Wr + ((size_t)g * n_kt + t0) * GROUP_BYTES;
        char *dst = ring + slot * STAGE;
        const int nb = nt * GROUP_BYTES;
        for (int i = threadIdx.x * 16; i < nb; i += NTHREADS * 16) cp_async16(dst + i, src + i);
        cp_commit();
    };
    for (int u = 0; u < NS && u < total; u++) issue(u, u);

    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; n++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[n][j] = 0.f;

    for (int u = 0; u < total; u++) {
        if (u + NS <= total) cp_wait<NS - 1>();
        else cp_wait<0>();
        __syncthreads();
        const int c = chunk_c(u);
        const int t0 = c * CHUNKT;
        const int nt = n_kt - t0 < CHUNKT ? n_kt - t0 : CHUNKT;
        const char *slot = ring + (u % NS) * STAGE;
        // Load the B fragments for this warp's tiles first: with NT >= 2 the B-load
        // latency otherwise sits on the critical path of every tile.
        uint32_t bb[2][NT][2][2];
        int ntl = 0;
        for (int j = warp; j < nt; j += NWARPS) {
            const int t = t0 + j;
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    bb[ntl][n][h][0] = __ldg(abf + (size_t)((t * 4 + 2 * h + 0) * NT + n) * 32 + lane);
                    bb[ntl][n][h][1] = __ldg(abf + (size_t)((t * 4 + 2 * h + 1) * NT + n) * 32 + lane);
                }
            ntl++;
        }
        ntl = 0;
        for (int j = warp; j < nt; j += NWARPS) {
            const int t = t0 + j;
            const char *gt = slot + j * GROUP_BYTES;
            if constexpr (!MATH) {
                acc[0][0] += (float)(*reinterpret_cast<const uint32_t *>(gt + lane * 4));
                acc[0][1] += (float)(*reinterpret_cast<const uint32_t *>(gt + (32 + lane) * 4));
                acc[0][2] += (float)(*reinterpret_cast<const uint32_t *>(gt + (64 + lane) * 4));
                acc[0][3] += (float)(*reinterpret_cast<const uint32_t *>(gt + (96 + lane) * 4));
                continue;
            }
            const __half *s = reinterpret_cast<const __half *>(gt + 512);
            const float s0 = __half2float(s[r0]), s1 = __half2float(s[r0 + 8]);
            float tmp[NT][4];
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int q = 0; q < 4; q++) tmp[n][q] = 0.f;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const uint32_t p0 = *reinterpret_cast<const uint32_t *>(
                    gt + ((h * 2 + 0) * 32 + lane) * 4);
                const uint32_t p1 = *reinterpret_cast<const uint32_t *>(
                    gt + ((h * 2 + 1) * 32 + lane) * 4);
                const uint32_t a0 = cvt_pair(p0);
                const uint32_t a1 = cvt_pair(p0 >> 16);
                const uint32_t a2 = cvt_pair(p1);
                const uint32_t a3 = cvt_pair(p1 >> 16);
#pragma unroll
                for (int n = 0; n < NT; n++)
                    mma_m16n8k16(tmp[n], a0, a1, a2, a3, bb[ntl][n][h][0], bb[ntl][n][h][1]);
            }
            ntl++;
#pragma unroll
            for (int n = 0; n < NT; n++) {
                acc[n][0] = fmaf(tmp[n][0], s0, acc[n][0]);
                acc[n][1] = fmaf(tmp[n][1], s0, acc[n][1]);
                acc[n][2] = fmaf(tmp[n][2], s1, acc[n][2]);
                acc[n][3] = fmaf(tmp[n][3], s1, acc[n][3]);
            }
        }
        __syncthreads();
        if (u + NS < total) issue(u + NS, u % NS);
        if (c == nch - 1) {
            const int g = chunk_g(u);
#pragma unroll
            for (int n = 0; n < NT; n++) {
                float *rb = red + (warp * 32 + lane) * 4;
                rb[0] = acc[n][0]; rb[1] = acc[n][1]; rb[2] = acc[n][2]; rb[3] = acc[n][3];
                __syncthreads();
                if (warp == 0) {
                    float fin[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int q = 0; q < NWARPS; q++) {
                        const float *rp = red + (q * 32 + lane) * 4;
                        fin[0] += rp[0]; fin[1] += rp[1]; fin[2] += rp[2]; fin[3] += rp[3];
                    }
                    store_group(out, rows, g, lane, fin, n * 8 + c0, BT);
                }
                __syncthreads();
            }
#pragma unroll
            for (int n = 0; n < NT; n++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[n][j] = 0.f;
        }
    }
    cp_wait<0>();
    __syncthreads();
    }
}

struct Shape {
    const char *name;
    int rows, K;
    bool gate;
};

// Repack a row-major Q8_0 matrix (q signed bytes, fp16 scales) into the fragment layout.
void repack(uint8_t *dst, const int8_t *q, const __half *scales, int rows, int K) {
    const int nb = K / 32, ng = rows / 16;
    for (int kt = 0; kt < nb; kt++) {
        for (int g = 0; g < ng; g++) {
            uint8_t *gt = dst + ((size_t)g * nb + kt) * GROUP_BYTES;
            for (int h = 0; h < 2; h++)
                for (int qq = 0; qq < 2; qq++)
                    for (int lane = 0; lane < 32; lane++) {
                        const int rr0 = lane >> 2, cc0 = (lane & 3) * 2;
                        const int rr1 = rr0 + 8;
                        const int kb = kt * 32 + h * 16 + qq * 8 + cc0;
                        uint8_t *o = gt + ((h * 2 + qq) * 32 + lane) * 4;
                        o[0] = (uint8_t)q[(size_t)(g * 16 + rr0) * K + kb + 0];
                        o[1] = (uint8_t)q[(size_t)(g * 16 + rr0) * K + kb + 1];
                        o[2] = (uint8_t)q[(size_t)(g * 16 + rr1) * K + kb + 0];
                        o[3] = (uint8_t)q[(size_t)(g * 16 + rr1) * K + kb + 1];
                    }
            for (int r = 0; r < 16; r++)
                *reinterpret_cast<__half *>(gt + 512 + 2 * r) = scales[(size_t)(g * 16 + r) * nb + kt];
        }
    }
}

// Pack f16 activations [NT*8][K] into the fragment-major layout: one 32-lane block per
// (8-k slice, n-tile), u32 index = ((k/8)*NT + n-tile)*32 + (entry%8)*4 + (k%8)/2.
uint32_t *pack_abf(const __half *A, int K, int NT) {
    const size_t n = (size_t)(K / 8) * NT * 32;
    uint32_t *o = (uint32_t *)malloc(n * sizeof(uint32_t));
    for (int s = 0; s < K / 8; s++)
        for (int nt = 0; nt < NT; nt++)
            for (int lane = 0; lane < 32; lane++) {
                const int e = nt * 8 + lane / 4;
                const int k = s * 8 + (lane % 4) * 2;
                memcpy(&o[((size_t)s * NT + nt) * 32 + lane], A + (size_t)e * K + k, 4);
            }
    return o;
}

// True fp32 reference dot of one (row, entry): dequantize exactly and accumulate in
// ascending k with the same per-32-block scale grouping.  Used to validate the MMA math.
double cpu_dot(const int8_t *q, const __half *scales, const __half *a, int row, int K, int Ktotal,
               const __half *A, int entry) {
    double acc = 0.0;
    for (int b = 0; b < K / 32; b++) {
        double p = 0.0;
        for (int j = 0; j < 32; j++) {
            const int k = b * 32 + j;
            p += (double)q[(size_t)row * Ktotal + k] * (double)__half2float(A[(size_t)entry * Ktotal + k]);
        }
        acc += p * (double)__half2float(scales[(size_t)row * (Ktotal / 32) + b]);
    }
    (void)a;
    return acc;
}

// Pick SPLIT/GPW once from the shape (never BT) to maximise warp utilisation.
__host__ __device__ inline void choose_cfg(int gcount, int *split, int *gpw) {
    if (gcount >= 64) { *split = 1; *gpw = 4; return; }
    int best = 1;
    float bestu = 0.f;
    for (int sp = 1; sp <= 16; sp <<= 1) {
        const int gpass = NWARPS / sp;
        const int passes = (gcount + gpass - 1) / gpass;
        const float u = (float)gcount / (float)(passes * gpass);
        if (u > bestu + 1e-6f) { bestu = u; best = sp; }
    }
    *split = best;
    *gpw = 1;
}

int g_nomath = 0, g_reps = 1;

template <int BT, int SPLIT, int GPW, int DEPTH>
void launch_one(const Shape &s, uint8_t *dW, const uint32_t *dA, float *dOut) {
    gemv_direct<BT, SPLIT, GPW, DEPTH><<<40, NTHREADS>>>(dW, dA, s.rows / 16, s.K / 32, s.rows, dOut);
    Q35_CHECK(cudaGetLastError());
}

template <int BT>
void launch(const Shape &s, uint8_t *dW, const uint32_t *dA, float *dOut) {
    const int ng = s.rows / 16;
    const int gcount = (ng + 39) / 40;
    int split, gpw;
    choose_cfg(gcount, &split, &gpw);
    (void)split;
    if (gpw == 4) {
        if constexpr (BT <= 16) launch_one<BT, 1, 4, 2>(s, dW, dA, dOut);
        else launch_one<BT, 1, 2, 4>(s, dW, dA, dOut);
        return;
    }
    const int smem = 4 * 32 * GROUP_BYTES;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(gemv_cta<BT, 4, 32, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(gemv_cta<BT, 4, 32, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr = true;
    }
    if (g_nomath)
        gemv_cta<BT, 4, 32, false><<<40, NTHREADS, smem>>>(dW, dA, s.rows / 16, s.K / 32, s.rows, dOut, g_reps);
    else
        gemv_cta<BT, 4, 32, true><<<40, NTHREADS, smem>>>(dW, dA, s.rows / 16, s.K / 32, s.rows, dOut, g_reps);
    Q35_CHECK(cudaGetLastError());
}

template <int BT>
void time_bt(const Shape &s, uint8_t *dW, uint32_t *dA, uint32_t *actbase, float *dOut, int iters,
             int reps, double *secs) {
    cudaEvent_t a, b;
    Q35_CHECK(cudaEventCreate(&a));
    Q35_CHECK(cudaEventCreate(&b));
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        launch<BT>(s, dW, actbase, dOut);
        Q35_CHECK(cudaDeviceSynchronize());
        Q35_CHECK(cudaEventRecord(a));
        for (int i = 0; i < iters; i++) launch<BT>(s, dW, actbase, dOut);
        Q35_CHECK(cudaEventRecord(b));
        Q35_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        Q35_CHECK(cudaEventElapsedTime(&ms, a, b));
        if (ms * 1e-3 < best) best = ms * 1e-3;
    }
    *secs = best;
    Q35_CHECK(cudaEventDestroy(a));
    Q35_CHECK(cudaEventDestroy(b));
}

}  // namespace

int main(int argc, char **argv) {
    int iters = 20, curve = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--curve")) curve = 1;
        else if (!strcmp(argv[i], "--nomath")) g_nomath = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    cudaDeviceProp prop;
    Q35_CHECK(cudaGetDeviceProperties(&prop, 0));
    warm_gpu(300);
    if (curve) { bw_curve(); return 0; }
    printf("GPU: %s SMs=%d smem/block=%zu regs/SM=%d\n", prop.name, prop.multiProcessorCount,
           prop.sharedMemPerBlockOptin, prop.regsPerMultiprocessor);

    Shape shapes[5] = {
        {"in-proj  (8224x1024)", 8224, 1024, false},
        {"out-proj (1024x2048)", 1024, 2048, false},
        {"gate/up  (7168x1024)", 7168, 1024, true},
        {"down     (1024x3584)", 1024, 3584, true},
        {"lm-head  (248320x1024)", 248320, 1024, false},
    };
    const int BTS[4] = {1, 8, 16, 32};
    int fails = 0;
    printf("\n%-22s %4s %10s %10s %8s %10s\n", "shape", "BT", "ms", "GB/s", "%373.4", "colmatch");
    for (int si = 0; si < 5; si++) {
        const Shape &s = shapes[si];
        int8_t *hq = (int8_t *)malloc((size_t)s.rows * s.K);
        __half *hs = (__half *)malloc((size_t)s.rows * (s.K / 32) * sizeof(__half));
        uint8_t *hW = (uint8_t *)malloc((size_t)s.rows * s.K + (size_t)s.rows * (s.K / 32) * 2);
        __half *hA = (__half *)calloc((size_t)32 * s.K, sizeof(__half));
        unsigned st = 12345u + si * 7919u;
        for (size_t i = 0; i < (size_t)s.rows * s.K; i++) {
            st = st * 1664525u + 1013904223u;
            hq[i] = (int8_t)((int)(st >> 24) - 128);
        }
        for (size_t i = 0; i < (size_t)s.rows * (s.K / 32); i++) {
            st = st * 1664525u + 1013904223u;
            float sc = 0.0009f + (float)(st & 0xffff) * (0.003f / 65536.0f);
            hs[i] = __float2half(sc);
        }
        for (int e = 0; e < 32; e++)
            for (int k = 0; k < s.K; k++) {
                st = st * 1664525u + 1013904223u;
                hA[(size_t)e * s.K + k] = __float2half((float)((int)(st >> 20) - 2048) * (1.f / 1024.f));
            }
        repack(hW, hq, hs, s.rows, s.K);
        uint8_t *dW;
        uint32_t *dA[3], *dRef;   // NT = 1, 2, 4
        float *dOut;
        uint32_t *hA4 = pack_abf(hA, s.K, 4);
        uint32_t *hA2 = pack_abf(hA, s.K, 2);
        uint32_t *hA1 = pack_abf(hA, s.K, 1);
        const size_t wbytes = (size_t)s.rows * s.K + (size_t)s.rows * (s.K / 32) * 2;
        Q35_CHECK(cudaMalloc(&dW, wbytes * 8));
        for (int r = 1; r < 8; r++)
            Q35_CHECK(cudaMemcpy(dW + (size_t)r * wbytes, hW, wbytes, cudaMemcpyHostToDevice));
        int nts[3] = {1, 2, 4};
        for (int q = 0; q < 3; q++)
            Q35_CHECK(cudaMalloc(&dA[q], (size_t)(s.K / 8) * nts[q] * 32 * sizeof(uint32_t)));
        Q35_CHECK(cudaMalloc(&dRef, (size_t)(s.K / 8) * 32 * sizeof(uint32_t)));
        Q35_CHECK(cudaMalloc(&dOut, (size_t)32 * s.rows * sizeof(float)));
        Q35_CHECK(cudaMemcpy(dW, hW, (size_t)s.rows * s.K + (size_t)s.rows * (s.K / 32) * 2,
                             cudaMemcpyHostToDevice));
        Q35_CHECK(cudaMemcpy(dA[0], hA1, (size_t)(s.K / 8) * 1 * 32 * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        Q35_CHECK(cudaMemcpy(dA[1], hA2, (size_t)(s.K / 8) * 2 * 32 * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));
        Q35_CHECK(cudaMemcpy(dA[2], hA4, (size_t)(s.K / 8) * 4 * 32 * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

        const double bytes = (double)s.rows * s.K + (double)s.rows * (s.K / 32) * 2;
        float *ref = (float *)malloc((size_t)32 * s.rows * sizeof(float));
        __half *one = (__half *)calloc((size_t)8 * s.K, sizeof(__half));
        double secs;
        for (int e = 0; e < 32; e++) {
            memcpy(one, hA + (size_t)e * s.K, sizeof(__half) * s.K);
            uint32_t *h1 = pack_abf(one, s.K, 1);
            Q35_CHECK(cudaMemcpy(dRef, h1, (size_t)(s.K / 8) * 32 * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice));
            time_bt<1>(s, dW, dA[0], dRef, dOut, 1, 1, &secs);
            Q35_CHECK(cudaMemcpy(ref + (size_t)e * s.rows, dOut, sizeof(float) * s.rows,
                                 cudaMemcpyDeviceToHost));
            free(h1);
            memset(one, 0, sizeof(__half) * s.K);
        }
        free(one);
        for (int bi = 0; bi < 4; bi++) {
            const int BT = BTS[bi];
            warm_gpu(150);
            float *hout = (float *)malloc((size_t)BT * s.rows * sizeof(float));
            if (BT == 1) time_bt<1>(s, dW, dA[0], dA[0], dOut, iters, 3, &secs);
            else if (BT == 8) time_bt<8>(s, dW, dA[0], dA[0], dOut, iters, 3, &secs);
            else if (BT == 16) time_bt<16>(s, dW, dA[1], dA[1], dOut, iters, 3, &secs);
            else time_bt<32>(s, dW, dA[2], dA[2], dOut, iters, 3, &secs);
            const double best = secs;
            Q35_CHECK(cudaMemcpy(hout, dOut, (size_t)BT * s.rows * sizeof(float),
                                 cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int e = 0; e < BT && !bad; e++) {
                for (int r = 0; r < s.rows; r++) {
                    if (memcmp(&hout[(size_t)e * s.rows + r], &ref[(size_t)e * s.rows + r],
                               sizeof(float)) != 0) {
                        if (bad < 3)
                            printf("    MISMATCH BT=%d entry=%d row=%d: %.9g vs %.9g\n", BT, e, r,
                                   hout[(size_t)e * s.rows + r], ref[(size_t)e * s.rows + r]);
                        bad++;
                        break;
                    }
                }
            }
            const bool small = (s.rows / 16) < 64 * 40;
            int cpufail = 0;
            if (!g_nomath) {
                for (int e = 0; e < BT && e < 4; e++)
                    for (int r = 0; r < s.rows; r += (s.rows / 8 + 1)) {
                        const double refv = cpu_dot(hq, hs, hA, r, s.K, s.K, hA, e);
                        const double got = hout[(size_t)e * s.rows + r];
                        if (fabs(refv - got) > 3e-3 * (1.0 + fabs(refv))) {
                            if (cpufail < 3)
                                printf("    CPU MISMATCH e=%d row=%d got=%.9g ref=%.9g\n", e, r, got, refv);
                            cpufail++;
                        }
                    }
            }
            const double gbs = bytes * iters * (small ? g_reps : 1) / best / 1e9;
            printf("%-22s %4d %10.4f %10.1f %8.1f%% %10s\n", s.name, BT, best * 1e3 / iters, gbs,
                   100.0 * gbs / 373.4, bad ? "FAIL" : "bitwise");
            if (bad || cpufail) fails++;
            if (s.gate && BT <= 16 && gbs < 0.85 * 373.4) {
                printf("    GATE FAIL: %s BT=%d %.1f GB/s < 85%% of 373.4\n", s.name, BT, gbs);
                fails++;
            }
            free(hout);
        }
        free(hq); free(hs); free(hW); free(hA); free(hA4); free(hA2); free(hA1); free(ref);
        cudaFree(dW); for (int q = 0; q < 3; q++) cudaFree(dA[q]);
        cudaFree(dRef); cudaFree(dOut);
    }
    printf("\nP2a gemv microbench: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
