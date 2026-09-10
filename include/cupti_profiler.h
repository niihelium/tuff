#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void cupti_profiler_init(void);
void cupti_profiler_start(void);
void cupti_profiler_stop_and_print(void);

#ifdef __cplusplus
}
#endif
