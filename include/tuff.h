#ifndef TUFF_H
#define TUFF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define TUFF_OK 0
#define TUFF_ERROR -1

// Public configuration parameters
typedef struct {
    const char *model_path;
    float momentum;            // Alpha factor for passive router (default 0.65)
    float sticky_bias;         // Beta bonus for previously active experts (default 0.15)
    int num_experts_per_tok;   // Top K active experts (default 8)
    int max_seq_len;           // Max context window length (default 131072 / 128k)
    bool verbose;
} tuff_config_t;

// Default configuration initialiser
static inline tuff_config_t tuff_config_default(void) {
    tuff_config_t cfg;
    cfg.model_path = NULL;
    cfg.momentum = 0.65f;
    cfg.sticky_bias = 0.15f;
    cfg.num_experts_per_tok = 8;
    cfg.max_seq_len = 131072;
    cfg.verbose = true;
    return cfg;
}

#ifdef __cplusplus
}
#endif

#endif // TUFF_H
