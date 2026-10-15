#!/usr/bin/env python3
"""FP64 numpy reference forward pass for dense LFM2/LFM2.5 GGUF models.

Mirrors llama.cpp src/models/lfm2.cpp (operator RMSNorm -> ShortConv1D or GQA attention with
QK-RMSNorm + NeoX RoPE -> residual -> FFN RMSNorm -> SwiGLU -> residual; final RMSNorm keyed as
token_embd_norm; LM head tied to token_embd). Weights are dequantized with gguf-py and every
activation is kept in float64, so the only error left is the weight quantization itself. Both
tuff and llama.cpp are measured against this in tests/verify_lfm25_oracle.sh (stage 3).

The pass is teacher-forced: step i consumes token i of --tokens, so logit vector i lines up with
reference token i exactly like `tuff --teacher-force ... --dump-logits` and
`lfm25-llama-ref --teacher ... --logits-out`.

Output: raw float32 [n_steps x vocab].
"""
import argparse
import os
import sys

LLAMA_GGUF_PY = os.environ.get("LLAMA_GGUF_PY", "/home/nii/workspace/llama.cpp/gguf-py")
sys.path.insert(0, LLAMA_GGUF_PY)

import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", required=True, help="space/comma separated prompt token IDs")
    ap.add_argument("--tokens", required=True, help="reference token stream (one ID per line)")
    ap.add_argument("-n", type=int, default=32, help="number of steps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.model)
    tensors = {t.name: t for t in r.tensors}

    def field(key):
        f = r.fields[key]
        return f.parts[f.data[0]][0]

    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    if arch != "lfm2":
        sys.exit(f"unsupported architecture {arch!r} (dense lfm2 only)")
    if any(".ffn_gate_exps." in name for name in tensors):
        sys.exit("MoE models are not supported by the FP64 reference yet")

    n_layer = int(field("lfm2.block_count"))
    eps = float(field("lfm2.attention.layer_norm_rms_epsilon"))
    base = float(field("lfm2.rope.freq_base"))
    n_head = int(field("lfm2.attention.head_count"))
    D = int(field("lfm2.embedding_length"))
    hd = D // n_head

    def W(name):
        t = tensors[name]
        return dequantize(t.data, t.tensor_type).astype(np.float64)

    def rms(x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    emb = W("token_embd.weight")  # [vocab, D]; also the tied LM head
    out_norm = W("token_embd_norm.weight")
    out_w = W("output.weight") if "output.weight" in tensors else emb
    layers = []
    for i in range(n_layer):
        p = f"blk.{i}."
        layers.append({k[len(p):]: W(k) for k in tensors if k.startswith(p)})

    half = hd // 2
    inv_freq = base ** (-np.arange(half) * 2.0 / hd)

    def rope(x, pos):  # NeoX: rotate (x[:half], x[half:]) pairs
        a = pos * inv_freq
        c, s = np.cos(a), np.sin(a)
        x0, x1 = x[:, :half], x[:, half:]
        return np.concatenate([x0 * c - x1 * s, x0 * s + x1 * c], axis=1)

    prompt = [int(t) for t in args.prompt_tokens.replace(",", " ").split()]
    stream = [int(x) for x in open(args.tokens) if x.strip()]
    n = min(args.n, len(stream))
    toks = prompt + stream[:n]

    conv_state = [np.zeros((2, D)) for _ in range(n_layer)]
    k_cache = [[] for _ in range(n_layer)]
    v_cache = [[] for _ in range(n_layer)]

    with open(args.out, "wb") as fo:
        # The vector produced after consuming toks[pos] predicts toks[pos+1]; dump from the last
        # prompt token onward.
        for pos in range(len(prompt) + n - 1):
            x = emb[toks[pos]].copy()
            for i, l in enumerate(layers):
                h = rms(x, l["attn_norm.weight"])
                if "shortconv.in_proj.weight" in l:
                    bcx = l["shortconv.in_proj.weight"] @ h
                    b, c, xx = bcx[:D], bcx[D:2 * D], bcx[2 * D:]
                    u = b * xx
                    k = l["shortconv.conv.weight"]  # [D, 3], tap 2 = current token
                    st = conv_state[i]
                    y = st[0] * k[:, 0] + st[1] * k[:, 1] + u * k[:, 2]
                    conv_state[i] = np.stack([st[1], u])
                    o = l["shortconv.out_proj.weight"] @ (c * y)
                else:
                    q = (l["attn_q.weight"] @ h).reshape(n_head, hd)
                    kk = (l["attn_k.weight"] @ h).reshape(-1, hd)
                    v = (l["attn_v.weight"] @ h).reshape(-1, hd)
                    q = rope(rms(q, l["attn_q_norm.weight"]), pos)
                    kk = rope(rms(kk, l["attn_k_norm.weight"]), pos)
                    k_cache[i].append(kk)
                    v_cache[i].append(v)
                    K = np.stack(k_cache[i], 1)  # [n_kv, t, hd]
                    V = np.stack(v_cache[i], 1)
                    g = n_head // K.shape[0]
                    K = np.repeat(K, g, 0)
                    V = np.repeat(V, g, 0)
                    s = np.einsum("hd,htd->ht", q, K) / np.sqrt(hd)
                    s = np.exp(s - s.max(1, keepdims=True))
                    s /= s.sum(1, keepdims=True)
                    o = l["attn_output.weight"] @ np.einsum("ht,htd->hd", s, V).reshape(-1)
                x = x + o
                h = rms(x, l["ffn_norm.weight"])
                gt = l["ffn_gate.weight"] @ h
                up = l["ffn_up.weight"] @ h
                x = x + l["ffn_down.weight"] @ (gt / (1.0 + np.exp(-gt)) * up)
            if pos >= len(prompt) - 1:
                logits = out_w @ rms(x, out_norm)
                logits.astype(np.float32).tofile(fo)
    print(f"wrote {n} FP64-reference logit vectors to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
