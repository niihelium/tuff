// Bandwidth roofline for the RTX 3070 Laptop (sm_86, 40 SMs, 4 MB L2).
//
// Ported and re-tuned from /home/nii/src/megakernels/common/bw.cu (MIT, Samuel Reeder):
// the sm_120 TMA bulk-copy ring is replaced with a cp.async (LDGSTS) ring that notifies
// an mbarrier with cp.async.mbarrier.arrive.noinc, plus an optional prefetch.global.L2
// cursor. Plain vectorized __ldcs is measured as the control. Every number this task
// reports is a fraction of the best sustained value printed here.
//
// Build: nvcc -O3 -arch=sm_86 -o build/bw-sm86 tools/bw_sm86.cu
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static __device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
static __device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
static __device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {
    // sm_86: mbarrier.test_wait.parity is the supported polling form (try_wait is sm_90+).
    unsigned ok;
    do {
        asm volatile("{\n .reg .pred p;\n"
                     " mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
    } while (!ok);
}
static __device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
static __device__ __forceinline__ void cp_async16(void* dst, const void* src, uint64_t policy) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                 ::"r"(smem_u32(dst)), "l"(src), "l"(policy) : "memory");
}
static __device__ __forceinline__ void cp_async_arrive_noinc(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
static __device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p) : "memory");
}
static __device__ __forceinline__ uint64_t evict_first() {
    uint64_t p;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
    return p;
}

// ---------------------------------------------------------------- plain loads
template <bool PF>
__global__ void read_vec(const uint4* __restrict__ p, size_t n, unsigned* out) {
    unsigned acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    if (PF) {
        // One 128-byte line per warp iteration, 4 KB ahead of the load cursor.
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
            if ((threadIdx.x & 7) == 0) prefetch_l2(p + i + 256);
            uint4 v = __ldcs(p + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    } else {
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += stride) {
            uint4 v = __ldcs(p + i);
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
    }
    if (acc == 0x12345678) out[0] = acc;
}

// ---------------------------------------------------------------- cp.async ring
// Producer = last warp. Each of its 32 lanes copies 16 B per step; when a ring slot is full
// the lane issues cp.async.mbarrier.arrive.noinc on `full[slot]` (init count 32), so the
// consumers' mbarrier wait completes exactly when every byte of the slot has landed.
template <int STAGES, int CHUNK, bool PF>
__global__ void __launch_bounds__(544, 1) read_cpasync(const char* __restrict__ p, size_t bytes, unsigned* out) {
    extern __shared__ __align__(1024) char smem[];
    uint64_t* full = (uint64_t*)(smem + STAGES * CHUNK);
    uint64_t* empty = full + STAGES;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n_consumer_warps = (blockDim.x >> 5) - 1;
    if (threadIdx.x == 0) {
        for (int s = 0; s < STAGES; s++) { mbar_init(&full[s], 32); mbar_init(&empty[s], n_consumer_warps); }
    }
    __syncthreads();  // sm_80: barrier init is visible after the CTA barrier
    const size_t per_block = bytes / gridDim.x / CHUNK * CHUNK;
    const char* base = p + per_block * blockIdx.x;
    const int n_chunks = (int)(per_block / CHUNK);
    if (warp == n_consumer_warps) {
        const uint64_t policy = evict_first();
        for (int c = 0; c < n_chunks; c++) {
            const int s = c % STAGES;
            if (c >= STAGES) mbar_wait(&empty[s], ((c / STAGES) - 1) & 1);
            char* dst = smem + s * CHUNK;
            const char* src = base + (size_t)c * CHUNK;
            if (PF) {
                // prefetch the slot after next, one line per lane
                const int cs = (c + 2) % STAGES;
                (void)cs;
                const char* pf = base + (size_t)(c + 2) * CHUNK;
                if (c + 2 < n_chunks) prefetch_l2(pf + lane * 128);
            }
            for (int i = lane * 16; i < CHUNK; i += 32 * 16)
                cp_async16(dst + i, src + i, policy);
            cp_async_arrive_noinc(&full[s]);
        }
        return;
    }
    unsigned acc = 0;
    for (int c = 0; c < n_chunks; c++) {
        const int s = c % STAGES;
        mbar_wait(&full[s], (c / STAGES) & 1);
        const uint4* q = (const uint4*)(smem + s * CHUNK);
        for (int i = threadIdx.x; i < CHUNK / 16; i += n_consumer_warps * 32) {
            uint4 v = q[i];
            acc ^= v.x ^ v.y ^ v.z ^ v.w;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }
    if (acc == 0x12345678) out[0] = acc;
}

template <typename F>
static float time_ms(F f, int iters) {
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    f();
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms; CK(cudaEventElapsedTime(&ms, a, b));
    return ms / iters;
}

template <int STAGES, int CHUNK, bool PF>
static void run_ring(const char* d, size_t bytes, unsigned* out, int sms, int threads) {
    int smem = STAGES * CHUNK + 2 * STAGES * 8;
    CK(cudaFuncSetAttribute(read_cpasync<STAGES, CHUNK, PF>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    size_t used = bytes / sms / CHUNK * CHUNK * sms;
    float ms = time_ms([&] { read_cpasync<STAGES, CHUNK, PF><<<sms, threads, smem>>>(d, bytes, out); }, 20);
    CK(cudaGetLastError());
    printf("cp.async ring stages=%2d chunk=%5d pf=%d threads=%4d smem=%6d : %7.1f GB/s\n",
           STAGES, CHUNK, (int)PF, threads, smem, used / ms / 1e6);
}

int main() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s SMs=%d L2=%dMB smem/block optin=%zuKB smem/SM=%zuKB regs/SM=%d\n",
           prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20,
           prop.sharedMemPerBlockOptin >> 10, prop.sharedMemPerMultiprocessor >> 10,
           prop.regsPerMultiprocessor);
    int clk = 0, busw = 0;
    cudaDeviceGetAttribute(&clk, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("memory clock %d MHz, bus %d bit, spec %.1f GB/s\n", clk / 1000, busw,
           2.0 * clk * 1e3 * busw / 8.0 / 1e9);
    size_t bytes = 1536ull << 20;
    char* d; unsigned* out;
    CK(cudaMalloc(&d, bytes)); CK(cudaMalloc(&out, 4)); CK(cudaMemset(d, 1, bytes));
    int sms = prop.multiProcessorCount;
    float best = 0.f;
    for (int mult : {1, 2, 4, 8}) {
        for (int threads : {256, 512, 1024}) {
            for (int pf = 0; pf < 2; pf++) {
                float ms = pf ? time_ms([&] { read_vec<true><<<sms * mult, threads>>>((const uint4*)d, bytes / 16, out); }, 20)
                              : time_ms([&] { read_vec<false><<<sms * mult, threads>>>((const uint4*)d, bytes / 16, out); }, 20);
                float gb = bytes / ms / 1e6;
                best = gb > best ? gb : best;
                printf("ld.global grid=%4d x %4d pf=%d : %7.1f GB/s\n", sms * mult, threads, pf, gb);
            }
        }
    }
    // Ring shapes around what the 99 KB / 40-SM budget allows.
    run_ring<3, 16384, false>(d, bytes, out, sms, 544);
    run_ring<3, 16384, true>(d, bytes, out, sms, 544);
    run_ring<4, 16384, false>(d, bytes, out, sms, 544);
    run_ring<4, 16384, true>(d, bytes, out, sms, 544);
    run_ring<5, 15232, false>(d, bytes, out, sms, 544);
    run_ring<5, 15232, true>(d, bytes, out, sms, 544);
    run_ring<6, 12288, false>(d, bytes, out, sms, 544);
    run_ring<8, 8192, false>(d, bytes, out, sms, 544);
    run_ring<12, 8192, false>(d, bytes, out, sms, 544);
    printf("BEST %.1f GB/s\n", best);
    return 0;
}
