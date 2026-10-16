// tools/bw_probe.cu - GPU bandwidth / hardware probe for roofline analysis.
// Build: nvcc -O3 -arch=sm_86 tools/bw_probe.cu -o /tmp/claude-1000/bw_probe
// Reports: device read BW (streaming int4 sum), D2D copy BW (read+write counted), H2D pinned/pageable
// for 6 MB (expert-slab size) and 256 MB chunks, and the theoretical peak from CUDA attributes.
// Run with no other GPU process active.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)

__global__ void read_sum(const int4 *p, size_t n, unsigned *out) {
    unsigned acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        int4 v = __ldcs(p + i);
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[0] = acc;   // defeat DCE
}

static float time_ms(cudaEvent_t a, cudaEvent_t b) { float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }

int main() {
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, dev));
    int memclk_khz = 0, buswidth = 0, smclk_khz = 0;
    CK(cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, dev));
    CK(cudaDeviceGetAttribute(&buswidth, cudaDevAttrGlobalMemoryBusWidth, dev));
    CK(cudaDeviceGetAttribute(&smclk_khz, cudaDevAttrClockRate, dev));
    printf("GPU: %s  SMs: %d  CC %d.%d  VRAM %.2f GiB\n", pr.name, pr.multiProcessorCount, pr.major, pr.minor, pr.totalGlobalMem / 1073741824.0);
    printf("CUDA attr: memory clock %.0f MHz, bus width %d bit (source: cudaDevAttrGlobalMemoryBusWidth), SM clock %.0f MHz\n",
           memclk_khz / 1000.0, buswidth, smclk_khz / 1000.0);
    // GDDR6: reported clock = command clock; data rate = 2 x clk (DDR on the data bus... NVIDIA reports 6001 MHz for 12 Gbps => 2x)
    double peak = (double)memclk_khz * 1e3 * 2.0 * (buswidth / 8.0) / 1e9;
    printf("Theoretical peak = memclk(%.0f MHz) x 2 (DDR) x busWidth(%d)/8 B = %.1f GB/s\n", memclk_khz / 1000.0, buswidth, peak);

    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    // (a) read BW
    size_t bytes = 1ull << 30; size_t n = bytes / sizeof(int4);
    int4 *d; CK(cudaMalloc(&d, bytes)); CK(cudaMemset(d, 1, bytes));
    unsigned *o; CK(cudaMalloc(&o, 4));
    int blocks = pr.multiProcessorCount * 8;
    for (int w = 0; w < 3; w++) read_sum<<<blocks, 256>>>(d, n, o);
    CK(cudaDeviceSynchronize());
    double best = 0, sum = 0; int R = 20;
    for (int r = 0; r < R; r++) {
        CK(cudaEventRecord(e0)); read_sum<<<blocks, 256>>>(d, n, o); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        double gbs = bytes / (time_ms(e0, e1) * 1e-3) / 1e9; sum += gbs; if (gbs > best) best = gbs;
    }
    printf("(a) device read (1 GiB, int4 __ldcs, %d blocks x256): mean %.1f GB/s, best %.1f GB/s (%.1f%% of peak)\n", blocks, sum / R, best, 100 * best / peak);
    // (b) D2D copy
    void *d2; CK(cudaMalloc(&d2, bytes));
    for (int w = 0; w < 3; w++) CK(cudaMemcpy(d2, d, bytes, cudaMemcpyDeviceToDevice));
    best = sum = 0;
    for (int r = 0; r < R; r++) {
        CK(cudaEventRecord(e0)); CK(cudaMemcpyAsync(d2, d, bytes, cudaMemcpyDeviceToDevice)); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        double gbs = 2.0 * bytes / (time_ms(e0, e1) * 1e-3) / 1e9; sum += gbs; if (gbs > best) best = gbs;
    }
    printf("(b) D2D copy (1 GiB, read+write counted): mean %.1f GB/s, best %.1f GB/s (%.1f%% of peak)\n", sum / R, best, 100 * best / peak);
    CK(cudaFree(d2));
    // (c) H2D
    size_t sizes[2] = {6ull << 20, 256ull << 20};
    for (int s = 0; s < 2; s++) {
        size_t sz = sizes[s]; int reps = s == 0 ? 400 : 20;
        void *dev_buf; CK(cudaMalloc(&dev_buf, sz));
        void *pin; CK(cudaMallocHost(&pin, sz)); memset(pin, 7, sz);
        void *pg = malloc(sz); memset(pg, 7, sz);
        const char *nm[2] = {"pinned", "pageable"}; void *src[2] = {pin, pg};
        for (int k = 0; k < 2; k++) {
            for (int w = 0; w < 5; w++) CK(cudaMemcpy(dev_buf, src[k], sz, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(e0));
            for (int r = 0; r < reps; r++) CK(cudaMemcpyAsync(dev_buf, src[k], sz, cudaMemcpyHostToDevice));
            CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            double ms = time_ms(e0, e1) / reps;
            printf("(c) H2D %s %3zu MB chunk: %.2f GB/s (%.3f ms/chunk, back-to-back async x%d)\n", nm[k], sz >> 20, sz / (ms * 1e-3) / 1e9, ms, reps);
        }
        // single-chunk latency incl. sync, pinned
        double tot = 0; int L = 100;
        for (int r = 0; r < L; r++) { CK(cudaEventRecord(e0)); CK(cudaMemcpyAsync(dev_buf, pin, sz, cudaMemcpyHostToDevice)); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); tot += time_ms(e0, e1); }
        printf("    H2D pinned %3zu MB isolated (event-timed, synced each): %.2f GB/s\n", sz >> 20, sz / (tot / L * 1e-3) / 1e9);
        cudaFree(dev_buf); cudaFreeHost(pin); free(pg);
    }
    return 0;
}
