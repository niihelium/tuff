cmake_minimum_required(VERSION 3.18)
project(tuff C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX 3070 Laptop is sm_86. This MUST be forced: project(... CUDA) above already defines
# CMAKE_CUDA_ARCHITECTURES, so a plain `if(NOT DEFINED ...)` guard never fires and a fresh
# `cmake -S . -B <dir>` silently configures for sm_52 — where __dp4a does not exist, producing
# "__dp4a is undefined" errors that look like a broken kernel port but are a build-config bug.
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for" FORCE)

# Single source of truth for the llama.cpp checkout: headers under include/ and ggml/include/,
# shared libraries in build/bin. Override with -DLLAMA_CPP_DIR=/path/to/llama.cpp.
set(LLAMA_CPP_DIR "/home/nii/src/llama.cpp" CACHE PATH "llama.cpp checkout (headers + build/bin libraries)")
if(NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "LLAMA_CPP_DIR=${LLAMA_CPP_DIR} does not look like a llama.cpp checkout (missing include/llama.h)")
endif()

# compute-sanitizer 2024.1 (distro CUDA 12.4) deadlocks when a kernel launch is followed by a
# blocking cudaMemcpy under --default-stream per-thread (reproduced with a 10-line program on
# driver 595.99). For memcheck runs, configure a separate tree with
#   cmake -S . -B build-sanitize -DTUFF_PER_THREAD_STREAM=OFF
# The LFM2.5 path is single-threaded on stream 0, so legacy-stream semantics are equivalent there.
option(TUFF_PER_THREAD_STREAM "Compile CUDA with --default-stream per-thread" ON)
if(TUFF_PER_THREAD_STREAM)
    set(TUFF_CUDA_STREAM_FLAG --default-stream per-thread)
else()
    set(TUFF_CUDA_STREAM_FLAG --default-stream legacy)
endif()

include_directories(include ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/ggml/include)
link_directories(${LLAMA_CPP_DIR}/build/bin)

option(TUFF_MODULAR_GPU "Use modular src/gpu/*.cu sources" ON)
if(TUFF_MODULAR_GPU)
    set(GPU_SOURCES
        src/gpu/state.cu
        src/gpu/context.cu
        src/gpu/expert_cache.cu
        src/gpu/dense_gemv.cu
        src/gpu/delta.cu
        src/gpu/attention.cu
        src/gpu/router.cu
        src/gpu/moe.cu
        src/gpu/output.cu
        src/gpu/graphs.cu
        src/gpu/verify_n2.cu
        src/gpu/verify_lockstep.cu
        src/gpu/dflash.cu
        src/gpu/batch.cu
        src/gpu/mtp.cu
    )
    include_directories(src/gpu)
else()
    set(GPU_SOURCES src/gpu.cu)
endif()

find_package(OpenMP REQUIRED)

set(SOURCES
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/tokenizer.c
    ${GPU_SOURCES}
    src/cupti_profiler.cu
    src/drafter.c
    src/sampler.c
    src/engine.c
    src/tuff_vtable.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    src/lfm25_engine.c
    src/lfm25_expert_cache.c
    src/lfm25_cpu_expert.c
    src/qwen_cpu_expert.c
    src/moe_policy.c
    src/chat.c
    src/qwen35/qwen35_engine.c
    src/qwen35/qwen35_mk.cu
    src/main.c
)

add_executable(tuff ${SOURCES})

set_target_properties(tuff PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_options(tuff PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)

# Iteration-48 item 5: the AVX2 int8 expert kernels in cpu_ops.c run at -O2 while the CUDA side
# is -O3. They are the hot path for CPU-computed misses; -O3 cannot change their numerics (they
# are explicit _mm256 intrinsics, and no -ffast-math is present), so this is gated by the same
# oracle + determinism checks as any other change.
set_source_files_properties(src/cpu_ops.c PROPERTIES COMPILE_OPTIONS "-O3")
# The CPU expert kernels must reproduce the GPU's float operation sequence bit for bit: no compiler-made contractions.
set_source_files_properties(src/lfm25_cpu_expert.c PROPERTIES COMPILE_OPTIONS "-O3;-ffp-contract=off")

find_package(CUDAToolkit REQUIRED)
# --no-as-needed + the executable's DT_RPATH: libllama.so carries RUNPATH $ORIGIN/../lib,
# which does not exist in this checkout, so its own libggml lookups fail unless they are already
# loaded. Linking every ggml soname directly keeps them in NEEDED and resolved via our RPATH.
target_link_options(tuff PRIVATE -Wl,--no-as-needed)
target_link_libraries(tuff PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C llama ggml ggml-cpu ggml-cuda ggml-base m)

# OpenAI-compatible HTTP API server (C11 + POSIX sockets + pthreads; no external HTTP lib).
find_package(Threads REQUIRED)
add_executable(tuff-server
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/tokenizer.c
    ${GPU_SOURCES}
    src/drafter.c
    src/sampler.c
    src/qwen_cpu_expert.c
    src/moe_policy.c
    src/engine.c
    src/qwen35/qwen35_engine.c
    src/qwen35/qwen35_mk.cu
    src/qwen35/q35_json.c
    src/qwen35/q35_template.c
    src/qwen35/q35_toolparse.c
    src/qwen35/q35_sample.c
    src/qwen35/q35_server.c
    src/server.c
)

set_target_properties(tuff-server PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS}"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_options(tuff-server PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)

target_link_options(tuff-server PRIVATE -Wl,--no-as-needed)
target_link_libraries(tuff-server PRIVATE CUDA::cudart CUDA::cuda_driver OpenMP::OpenMP_C Threads::Threads llama ggml ggml-cpu ggml-cuda ggml-base m)

# Bandwidth roofline (Phase 0.2). Standalone: nvcc -O3 -arch=sm_86 tools/bw_sm86.cu
add_executable(bw-sm86 tools/bw_sm86.cu)
target_compile_options(bw-sm86 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

# line info for compute-sanitizer triage of the qwen35 kernels (per-file, no global change)
set_source_files_properties(src/qwen35/qwen35_mk.cu PROPERTIES COMPILE_OPTIONS "-lineinfo")

# qwen35 kernel gate tool: teacher-forced logits from the megakernel (G1/G3).
add_executable(qwen35-teacher
    tools/qwen35_teacher.c
    src/gguf_parser.c
    src/qwen35/qwen35_engine.c
    src/qwen35/qwen35_mk.cu
)
target_include_directories(qwen35-teacher PRIVATE include)
target_compile_options(qwen35-teacher PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(qwen35-teacher PRIVATE CUDA::cudart OpenMP::OpenMP_C m)

# Phase 2 engine benchmark + G4 invariance gate.
add_executable(qwen35-batch-bench
    tools/qwen35_batch_bench.c
    src/gguf_parser.c
    src/qwen35/qwen35_engine.c
    src/qwen35/qwen35_mk.cu
)
target_include_directories(qwen35-batch-bench PRIVATE include)
target_compile_options(qwen35-batch-bench PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(qwen35-batch-bench PRIVATE CUDA::cudart OpenMP::OpenMP_C m)

# Synthetic determinism bisect for the batched decode path (debug tool).
add_executable(qwen35-det-probe
    tools/qwen35_det_probe.c
    src/gguf_parser.c
    src/qwen35/qwen35_engine.c
    src/qwen35/qwen35_mk.cu
)
target_include_directories(qwen35-det-probe PRIVATE include)
target_compile_options(qwen35-det-probe PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_link_libraries(qwen35-det-probe PRIVATE CUDA::cudart OpenMP::OpenMP_C m)

# P2a gate: standalone tensor-core batched Q8_0 GEMV microbench (real 0.8B shapes).
add_executable(qwen35-gemv-bench tools/qwen35_gemv_bench.cu)
target_compile_options(qwen35-gemv-bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(qwen35-gemv-bench PRIVATE CUDA::cudart m)

# llama.cpp oracle for the qwen35 gates (same libllama as tuff).
add_executable(qwen35-llama-ref tools/qwen35_llama_ref.c)
target_include_directories(qwen35-llama-ref PRIVATE ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/ggml/include)
target_link_options(qwen35-llama-ref PRIVATE -Wl,--no-as-needed)
set_target_properties(qwen35-llama-ref PROPERTIES
    LINK_FLAGS "-Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE)
target_link_libraries(qwen35-llama-ref PRIVATE llama ggml ggml-cpu ggml-cuda ggml-base m)

# P2: CPU-only self-test + microbenchmark for the AVX2 int8 expert kernels (no CUDA).
add_executable(cpu-expert-bench
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/cpu_expert_bench.c
)
target_compile_options(cpu-expert-bench PRIVATE -g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c)
target_link_libraries(cpu-expert-bench PRIVATE OpenMP::OpenMP_C m)

# GPU Kernel Microbenchmark Suite
add_executable(microbench-gpu
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/tokenizer.c
    ${GPU_SOURCES}
    src/cupti_profiler.cu
    src/drafter.c
    src/sampler.c
    src/qwen_cpu_expert.c
    src/moe_policy.c
    src/engine.c
    src/chat.c
    src/microbench_gpu.cu
)
target_compile_options(microbench-gpu PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(microbench-gpu PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C llama ggml ggml-cpu ggml-cuda ggml-base m)

# LFM2.5 oracle: reproducible llama.cpp reference generator (tests/gen_lfm25_ref.sh,
# tests/verify_lfm25_oracle.sh). Single-token decode on GPU, mirroring tuff's decode path.
add_executable(lfm25-llama-ref tools/lfm25_llama_ref.c)
target_compile_options(lfm25-llama-ref PRIVATE -g -O2 -Wall -Wextra -Werror)
set_target_properties(lfm25-llama-ref PROPERTIES
    LINK_FLAGS "-Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_link_options(lfm25-llama-ref PRIVATE -Wl,--no-as-needed)
target_link_libraries(lfm25-llama-ref PRIVATE llama ggml ggml-cpu ggml-cuda ggml-base)

# MoE router kernel test (Task 2.1): checks lfm25_gpu_router_top4 against llama.cpp fixtures.
# Does not call lfm25_gpu_prepare; uploads only the router tensors.
add_executable(lfm25-router-test
    src/gguf_parser.c
    src/cpu_ops.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    src/lfm25_expert_cache.c   # lfm25_gpu_shutdown releases the pinned expert store
    tests/test_lfm25_router.cu
)
set_target_properties(lfm25-router-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(lfm25-router-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(lfm25-router-test PRIVATE CUDA::cudart OpenMP::OpenMP_C Threads::Threads m)

# MoE expert FFN test (Task 2.2): checks lfm25_gpu_moe_ffn against llama.cpp fixtures.
# Calls lfm25_gpu_prepare (uploads all experts, ~4.8 GB VRAM).
add_executable(lfm25-moe-ffn-test
    src/gguf_parser.c
    src/cpu_ops.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    src/lfm25_expert_cache.c   # lfm25_gpu_shutdown releases the pinned expert store
    tests/test_lfm25_moe_ffn.cu
)
set_target_properties(lfm25-moe-ffn-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(lfm25-moe-ffn-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(lfm25-moe-ffn-test PRIVATE CUDA::cudart OpenMP::OpenMP_C Threads::Threads m)

# Hybrid CPU-miss MoE FFN byte-identity test (vs the all-GPU lfm25_gpu_moe_ffn, every miss mask).
add_executable(lfm25-cpu-expert-test
    src/gguf_parser.c
    src/cpu_ops.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    src/lfm25_expert_cache.c   # lfm25_gpu_shutdown releases the pinned expert store
    src/lfm25_cpu_expert.c
    tests/test_lfm25_cpu_expert.cu
)
set_target_properties(lfm25-cpu-expert-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(lfm25-cpu-expert-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(lfm25-cpu-expert-test PRIVATE CUDA::cudart OpenMP::OpenMP_C Threads::Threads m)

# Qwen tokenizer regression test (Task 8.4)
add_executable(qwen-tokenizer-test
    src/gguf_parser.c
    src/tokenizer.c
    tests/test_qwen_tokenizer.c
)
target_compile_options(qwen-tokenizer-test PRIVATE
    -g -O2 -Wall -Wextra -march=native
)
target_link_libraries(qwen-tokenizer-test PRIVATE Threads::Threads m)

# Qwen CPU-miss MoE byte-identity test (Task Q0.2)
set_source_files_properties(src/qwen_cpu_expert.c PROPERTIES COMPILE_OPTIONS "-O3;-ffp-contract=off")

add_executable(qwen-cpu-expert-test
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    ${GPU_SOURCES}
    src/cupti_profiler.cu
    src/moe_policy.c
    src/qwen_cpu_expert.c
    tests/test_qwen_cpu_expert.cu
)
set_target_properties(qwen-cpu-expert-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(qwen-cpu-expert-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(qwen-cpu-expert-test PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C Threads::Threads llama ggml ggml-cpu ggml-cuda ggml-base m)

# Qwen N-column batch verify test (Task Q1.1)
add_executable(qwen-batch-test
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/tokenizer.c
    ${GPU_SOURCES}
    src/cupti_profiler.cu
    src/drafter.c
    src/sampler.c
    src/qwen_cpu_expert.c
    src/moe_policy.c
    src/engine.c
    src/chat.c
    tests/test_qwen_batch.cu
)
set_target_properties(qwen-batch-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(qwen-batch-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(qwen-batch-test PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C Threads::Threads llama ggml ggml-cpu ggml-cuda ggml-base m)



# ---------------------------------------------------------------------------
# Backend B host-side tests (no GPU): JSON/template/tool parser and sampler.
# ---------------------------------------------------------------------------
add_executable(q35-side-test
    src/qwen35/q35_json.c
    src/qwen35/q35_template.c
    src/qwen35/q35_toolparse.c
    tests/qwen35/test_side.c
)
target_include_directories(q35-side-test PRIVATE include)
target_compile_options(q35-side-test PRIVATE -g -O2 -Wall -Wextra -march=native)
target_link_libraries(q35-side-test PRIVATE m)

add_executable(q35-sampler-test
    src/qwen35/q35_sample.c
    src/qwen35/q35_json.c
    tests/qwen35/test_sampler.c
)
target_include_directories(q35-sampler-test PRIVATE include)
target_compile_options(q35-sampler-test PRIVATE -g -O2 -Wall -Wextra -march=native)
target_link_libraries(q35-sampler-test PRIVATE m)

# G8 template/tokenizer fixture replay (CPU only, uses the GGUF tokenizer).
add_executable(q35-template-render
    tools/q35_template_render.c
    src/qwen35/q35_json.c
    src/qwen35/q35_template.c
    src/gguf_parser.c
    src/tokenizer.c
)
target_include_directories(q35-template-render PRIVATE include)
target_compile_options(q35-template-render PRIVATE -g -O2 -Wall -Wextra -march=native)
target_link_libraries(q35-template-render PRIVATE Threads::Threads m)

add_executable(q35-parse-cli
    tools/q35_parse_cli.c
    src/qwen35/q35_json.c
    src/qwen35/q35_template.c
    src/qwen35/q35_toolparse.c
)
target_include_directories(q35-parse-cli PRIVATE include)
target_compile_options(q35-parse-cli PRIVATE -g -O2 -Wall -Wextra -march=native)
target_link_libraries(q35-parse-cli PRIVATE m)

# G10 llama.cpp parser oracle (links libllama-common, vocab-only model load).
add_executable(q35-chatparse-ref tools/q35_chatparse_ref.cpp)
target_include_directories(q35-chatparse-ref BEFORE PRIVATE ${LLAMA_CPP_DIR}/common ${LLAMA_CPP_DIR}/include ${LLAMA_CPP_DIR}/ggml/include)
target_compile_options(q35-chatparse-ref PRIVATE -g -O2 -Wall -Wextra)
set_target_properties(q35-chatparse-ref PROPERTIES
    LINK_FLAGS "-Wl,--disable-new-dtags"
    INSTALL_RPATH "${LLAMA_CPP_DIR}/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE)
target_link_options(q35-chatparse-ref PRIVATE -Wl,--no-as-needed)
target_link_libraries(q35-chatparse-ref PRIVATE llama-common llama ggml ggml-cpu ggml-cuda ggml-base m)
