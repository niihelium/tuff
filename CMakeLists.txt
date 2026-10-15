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

include_directories(include /home/nii/workspace/llama.cpp/include /home/nii/workspace/llama.cpp/ggml/include)
link_directories(/home/nii/workspace/llama.cpp/build/bin)

find_package(OpenMP REQUIRED)

set(SOURCES
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/tokenizer.c
    src/gpu.cu
    src/cupti_profiler.cu
    src/drafter.c
    src/sampler.c
    src/engine.c
    src/tuff_vtable.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    src/lfm25_engine.c
    src/chat.c
    src/main.c
)

add_executable(tuff ${SOURCES})

set_target_properties(tuff PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "/home/nii/workspace/llama.cpp/build/bin"
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

find_package(CUDAToolkit REQUIRED)
target_link_libraries(tuff PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C llama ggml ggml-cpu ggml-base m)

# OpenAI-compatible HTTP API server (C11 + POSIX sockets + pthreads; no external HTTP lib).
find_package(Threads REQUIRED)
add_executable(tuff-server
    src/gguf_parser.c
    src/model.c
    src/cpu_ops.c
    src/tokenizer.c
    src/gpu.cu
    src/drafter.c
    src/sampler.c
    src/engine.c
    src/server.c
)

set_target_properties(tuff-server PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS}"
    INSTALL_RPATH "/home/nii/workspace/llama.cpp/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_options(tuff-server PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)

target_link_libraries(tuff-server PRIVATE CUDA::cudart CUDA::cuda_driver OpenMP::OpenMP_C Threads::Threads llama ggml ggml-cpu ggml-base m)

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
    src/gpu.cu
    src/cupti_profiler.cu
    src/drafter.c
    src/sampler.c
    src/engine.c
    src/chat.c
    src/microbench_gpu.cu
)
target_compile_options(microbench-gpu PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(microbench-gpu PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C llama ggml ggml-cpu ggml-base m)

# LFM2.5 oracle: reproducible llama.cpp reference generator (tests/gen_lfm25_ref.sh,
# tests/verify_lfm25_oracle.sh). Single-token decode on GPU, mirroring tuff's decode path.
add_executable(lfm25-llama-ref tools/lfm25_llama_ref.c)
target_compile_options(lfm25-llama-ref PRIVATE -g -O2 -Wall -Wextra -Werror)
set_target_properties(lfm25-llama-ref PROPERTIES
    LINK_FLAGS "-Wl,--disable-new-dtags"
    INSTALL_RPATH "/home/nii/workspace/llama.cpp/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_link_libraries(lfm25-llama-ref PRIVATE llama)

# MoE router kernel test (Task 2.1): checks lfm25_gpu_router_top4 against llama.cpp fixtures.
# Does not call lfm25_gpu_prepare; uploads only the router tensors.
add_executable(lfm25-router-test
    src/gguf_parser.c
    src/cpu_ops.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    tests/test_lfm25_router.cu
)
set_target_properties(lfm25-router-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "/home/nii/workspace/llama.cpp/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(lfm25-router-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(lfm25-router-test PRIVATE CUDA::cudart OpenMP::OpenMP_C m)

# MoE expert FFN test (Task 2.2): checks lfm25_gpu_moe_ffn against llama.cpp fixtures.
# Calls lfm25_gpu_prepare (uploads all experts, ~4.8 GB VRAM).
add_executable(lfm25-moe-ffn-test
    src/gguf_parser.c
    src/cpu_ops.c
    src/lfm25_model.c
    src/lfm25_gpu.cu
    tests/test_lfm25_moe_ffn.cu
)
set_target_properties(lfm25-moe-ffn-test PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS} -Wl,--disable-new-dtags"
    INSTALL_RPATH "/home/nii/workspace/llama.cpp/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)
target_compile_options(lfm25-moe-ffn-test PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 ${TUFF_CUDA_STREAM_FLAG}>
)
target_link_libraries(lfm25-moe-ffn-test PRIVATE CUDA::cudart OpenMP::OpenMP_C m)
