#ifndef LFM25_H
#define LFM25_H

#include "lfm25_model.h"
#include "lfm25_engine.h"
#include "tuff_vtable.h"

#ifdef __cplusplus
extern "C" {
#endif

// Architecture detection helper
bool lfm25_is_lfm_architecture(tuff_arch_t arch);

// Global LFM2.5 engine vtable instance
extern const engine_vtable_t lfm25_engine_vtable;

#ifdef __cplusplus
}
#endif

#endif // LFM25_H
