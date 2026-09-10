#include "cupti_profiler.h"
#include <cupti.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cxxabi.h>

#define BUF_SIZE (32 * 1024 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) + ((align)-1)) & ~((align)-1))

struct KernelStats {
    std::string name;
    uint64_t count = 0;
    double total_time_ms = 0.0;
    double min_time_ms = 1e9;
    double max_time_ms = 0.0;
    uint32_t grid_x = 0, grid_y = 0, grid_z = 0;
    uint32_t block_x = 0, block_y = 0, block_z = 0;
    uint32_t registers = 0;
    uint32_t static_smem = 0;
    uint32_t dynamic_smem = 0;
};

struct MemcpyStats {
    std::string kind;
    uint64_t count = 0;
    uint64_t total_bytes = 0;
    double total_time_ms = 0.0;
};

static std::map<std::string, KernelStats> g_kernel_stats;
static std::map<std::string, MemcpyStats> g_memcpy_stats;
static bool g_cupti_active = false;

static std::string demangle(const char* name) {
    if (!name) return "unknown";
    int status = 0;
    char* demangled = abi::__cxa_demangle(name, NULL, NULL, &status);
    std::string res = (status == 0 && demangled) ? demangled : name;
    if (demangled) free(demangled);
    // Shorten template noise if too long
    size_t param_pos = res.find('(');
    if (param_pos != std::string::npos) {
        res = res.substr(0, param_pos);
    }
    return res;
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
    if (bfr == NULL) {
        fprintf(stderr, "[CUPTI] Error: Out of memory\n");
        exit(EXIT_FAILURE);
    }
    *size = BUF_SIZE;
    *buffer = (uint8_t *)ALIGN_BUFFER(bfr, ALIGN_SIZE);
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    CUpti_Activity *record = NULL;
    CUptiResult status;

    while (1) {
        status = cuptiActivityGetNextRecord(buffer, validSize, &record);
        if (status == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
                double dur_ms = (double)(kernel->end - kernel->start) / 1000000.0;
                std::string k_name = demangle(kernel->name);

                KernelStats &st = g_kernel_stats[k_name];
                st.name = k_name;
                st.count++;
                st.total_time_ms += dur_ms;
                st.min_time_ms = std::min(st.min_time_ms, dur_ms);
                st.max_time_ms = std::max(st.max_time_ms, dur_ms);
                st.grid_x = kernel->gridX; st.grid_y = kernel->gridY; st.grid_z = kernel->gridZ;
                st.block_x = kernel->blockX; st.block_y = kernel->blockY; st.block_z = kernel->blockZ;
                st.registers = kernel->registersPerThread;
                st.static_smem = kernel->staticSharedMemory;
                st.dynamic_smem = kernel->dynamicSharedMemory;
            } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
                CUpti_ActivityMemcpy *cpy = (CUpti_ActivityMemcpy *)record;
                double dur_ms = (double)(cpy->end - cpy->start) / 1000000.0;
                const char *copy_kind = "UNKNOWN";
                switch (cpy->copyKind) {
                    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: copy_kind = "HtoD (Host -> GPU)"; break;
                    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: copy_kind = "DtoH (GPU -> Host)"; break;
                    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: copy_kind = "DtoD (GPU -> GPU)"; break;
                    default: break;
                }
                MemcpyStats &st = g_memcpy_stats[copy_kind];
                st.kind = copy_kind;
                st.count++;
                st.total_bytes += cpy->bytes;
                st.total_time_ms += dur_ms;
            }
        } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
            break;
        } else {
            break;
        }
    }
    free(buffer);
}

void cupti_profiler_init(void) {
    if (g_cupti_active) return;
    CUptiResult res = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    if (res != CUPTI_SUCCESS) {
        fprintf(stderr, "[CUPTI] Failed to register callbacks: %d\n", res);
        return;
    }
    g_cupti_active = true;
}

void cupti_profiler_start(void) {
    if (!g_cupti_active) cupti_profiler_init();
    g_kernel_stats.clear();
    g_memcpy_stats.clear();
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
}

void cupti_profiler_stop_and_print(void) {
    if (!g_cupti_active) return;
    cudaDeviceSynchronize();
    cuptiActivityFlushAll(0);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);

    double total_gpu_kernel_time = 0.0;
    std::vector<KernelStats> sorted_kernels;
    for (auto &kv : g_kernel_stats) {
        sorted_kernels.push_back(kv.second);
        total_gpu_kernel_time += kv.second.total_time_ms;
    }
    std::sort(sorted_kernels.begin(), sorted_kernels.end(), [](const KernelStats &a, const KernelStats &b) {
        return a.total_time_ms > b.total_time_ms;
    });

    fprintf(stderr, "\n========================================================================================================\n");
    fprintf(stderr, "                                   DEEP CUDA PROFILER TRACE REPORT                                      \n");
    fprintf(stderr, "========================================================================================================\n");
    fprintf(stderr, "Total GPU Kernel Execution Time: %.3f ms\n\n", total_gpu_kernel_time);

    fprintf(stderr, "%-50s | %6s | %9s | %8s | %8s | %7s | %10s | %5s\n",
            "Kernel Name", "Calls", "Total(ms)", "Avg(us)", "Min(us)", "Max(us)", "GPU Time%", "Regs");
    fprintf(stderr, "--------------------------------------------------------------------------------------------------------\n");

    for (const auto &k : sorted_kernels) {
        double pct = (total_gpu_kernel_time > 0.0) ? (k.total_time_ms / total_gpu_kernel_time * 100.0) : 0.0;
        double avg_us = (k.count > 0) ? (k.total_time_ms / k.count * 1000.0) : 0.0;
        std::string short_name = k.name;
        if (short_name.length() > 50) short_name = short_name.substr(0, 47) + "...";
        fprintf(stderr, "%-50s | %6lu | %9.3f | %8.2f | %8.2f | %7.2f | %9.1f%% | %5u\n",
                short_name.c_str(), k.count, k.total_time_ms, avg_us, k.min_time_ms * 1000.0, k.max_time_ms * 1000.0, pct, k.registers);
    }

    fprintf(stderr, "========================================================================================================\n");
    fprintf(stderr, "                                     GPU MEMORY TRANSFERS (PCIe)                                        \n");
    fprintf(stderr, "========================================================================================================\n");
    fprintf(stderr, "%-25s | %6s | %10s | %10s | %12s\n", "Transfer Kind", "Calls", "Total MB", "Time (ms)", "Bandwidth");
    fprintf(stderr, "--------------------------------------------------------------------------------------------------------\n");
    for (const auto &kv : g_memcpy_stats) {
        const auto &m = kv.second;
        double mb = (double)m.total_bytes / (1024.0 * 1024.0);
        double gbps = (m.total_time_ms > 0.0) ? (mb / (m.total_time_ms / 1000.0) / 1024.0) : 0.0;
        fprintf(stderr, "%-25s | %6lu | %10.2f | %10.3f | %9.2f GB/s\n",
                m.kind.c_str(), m.count, mb, m.total_time_ms, gbps);
    }
    fprintf(stderr, "========================================================================================================\n\n");
}
