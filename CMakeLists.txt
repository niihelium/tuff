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
    src/chat.c
    src/main.c
)

add_executable(tuff ${SOURCES})

set_target_properties(tuff PROPERTIES
    COMPILE_FLAGS "${OpenMP_C_FLAGS}"
    LINK_FLAGS "${OpenMP_C_FLAGS}"
    INSTALL_RPATH "/home/nii/workspace/llama.cpp/build/bin"
    BUILD_WITH_INSTALL_RPATH TRUE
)

target_compile_options(tuff PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-g -O2 -Wall -Wextra -march=native -mfma -mavx2 -mf16c>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --default-stream per-thread>
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
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --default-stream per-thread>
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
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --default-stream per-thread>
)
target_link_libraries(microbench-gpu PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cupti OpenMP::OpenMP_C llama ggml ggml-cpu ggml-base m)
