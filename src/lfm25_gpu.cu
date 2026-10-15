#include "lfm25_model.h"
#include "lfm25_engine.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static void upload_tensor(lfm25_tensor_t *t) {
    if (!t->data || t->info.size_bytes == 0) return;
    CUDA_CHECK(cudaMalloc(&t->dev, t->info.size_bytes));
    CUDA_CHECK(cudaMemcpy(t->dev, t->data, t->info.size_bytes, cudaMemcpyHostToDevice));
}

static void free_tensor(lfm25_tensor_t *t) {
    if (t->dev) {
        cudaFree(t->dev);
        t->dev = NULL;
    }
}

int lfm25_gpu_prepare(lfm25_model_t *model, int *max_context_len, int fill_ram) {
    (void)fill_ram;
    if (!model) return -1;
    if (max_context_len && *max_context_len <= 0) {
        *max_context_len = model->context_length > 0 ? (int)model->context_length : 4096;
    }

    fprintf(stderr, "[LFM25 GPU] Uploading model weights to device VRAM...\n");

    // 1. Upload trunk tensors
    upload_tensor(&model->token_embd);
    upload_tensor(&model->token_embd_norm);
    if (model->output_weight.data != model->token_embd.data) {
        upload_tensor(&model->output_weight);
    } else {
        model->output_weight.dev = model->token_embd.dev;
    }

    // 2. Upload layers
    for (uint32_t l = 0; l < model->block_count; ++l) {
        lfm25_layer_t *L = &model->layers[l];

        // Norms
        upload_tensor(&L->attn_norm);
        upload_tensor(&L->ffn_norm);

        // Operator
        if (L->op_type == LFM25_OP_CONV) {
            upload_tensor(&L->shortconv_in_proj);
            upload_tensor(&L->shortconv_conv);
            upload_tensor(&L->shortconv_out_proj);
        } else {
            upload_tensor(&L->attn_q);
            upload_tensor(&L->attn_k);
            upload_tensor(&L->attn_v);
            upload_tensor(&L->attn_output);
            upload_tensor(&L->attn_q_norm);
            upload_tensor(&L->attn_k_norm);
        }

        // FFN
        if (L->ffn_type == LFM25_FFN_DENSE) {
            upload_tensor(&L->ffn_gate);
            upload_tensor(&L->ffn_up);
            upload_tensor(&L->ffn_down);
        } else {
            // For MoE layers, upload router and expert tensors if VRAM permits
            upload_tensor(&L->ffn_gate_inp);
            upload_tensor(&L->exp_probs_b);
            // In 8B (100% VRAM resident), upload all experts directly
            if (!model->is_moe || model->expert_count <= 32) {
                upload_tensor(&L->ffn_gate_exps);
                upload_tensor(&L->ffn_up_exps);
                upload_tensor(&L->ffn_down_exps);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    fprintf(stderr, "[LFM25 GPU] Weight upload complete.\n");
    return 0;
}

void lfm25_gpu_shutdown(lfm25_model_t *model) {
    if (!model) return;
    fprintf(stderr, "[LFM25 GPU] Freeing device weights...\n");

    free_tensor(&model->token_embd);
    free_tensor(&model->token_embd_norm);
    if (model->output_weight.dev != model->token_embd.dev) {
        free_tensor(&model->output_weight);
    }
    model->output_weight.dev = NULL;

    for (uint32_t l = 0; l < model->block_count; ++l) {
        lfm25_layer_t *L = &model->layers[l];
        free_tensor(&L->attn_norm);
        free_tensor(&L->ffn_norm);

        if (L->op_type == LFM25_OP_CONV) {
            free_tensor(&L->shortconv_in_proj);
            free_tensor(&L->shortconv_conv);
            free_tensor(&L->shortconv_out_proj);
        } else {
            free_tensor(&L->attn_q);
            free_tensor(&L->attn_k);
            free_tensor(&L->attn_v);
            free_tensor(&L->attn_output);
            free_tensor(&L->attn_q_norm);
            free_tensor(&L->attn_k_norm);
        }

        if (L->ffn_type == LFM25_FFN_DENSE) {
            free_tensor(&L->ffn_gate);
            free_tensor(&L->ffn_up);
            free_tensor(&L->ffn_down);
        } else {
            free_tensor(&L->ffn_gate_inp);
            free_tensor(&L->exp_probs_b);
            free_tensor(&L->ffn_gate_exps);
            free_tensor(&L->ffn_up_exps);
            free_tensor(&L->ffn_down_exps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
