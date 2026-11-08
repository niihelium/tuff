#!/usr/bin/env python3
"""Independent fp32 reference decoder for Qwen3.5 0.8B / 2B, straight from the Q8_0 GGUF.

This is the numeric oracle for the tuf qwen35 megakernel (Phase 0.3).  It reads the
same GGUF the kernel loads, dequantizes every Q8_0 block exactly (fp16 scale * int8),
and runs a teacher-forced forward over a token list, dumping one fp32 logit vector
per position.  No torch, no llama.cpp -- only numpy + gguf-py.

Important GGUF-vs-HF detail (verified in llama.cpp conversion/qwen.py): the converter
does `weight + 1` for every `*.norm.weight` EXCEPT `linear_attn.norm.weight`
(`ssm_norm`).  So GGUF norms are used directly, with no extra +1.  `ssm_a` is stored as
`-exp(A_log)`, so the decay is `exp(ssm_a * softplus(alpha + dt_bias))`.

Layout facts (verified against llama.cpp src/models/qwen35.cpp @ 80d2c0c):
  * attn_q rows are [q(256) | gate(256)] per head, 8 heads.
  * GDN in-projection is q(2048) | k(2048) | v(2048); z is attn_gate (2048).
  * conv1d is [conv_dim][4], conv[t] = sum_{tap=0..3} win[tap] * w[ch][tap].
  * attention is partial NEOX RoPE over the first 64 dims (sections are M-RoPE only;
    qwen35 selects LLAMA_ROPE_TYPE_NEOX and text-only positions are all equal).
  * GDN L2 norm is x / sqrt(sum(x^2) + 1e-6) * 1/sqrt(128), then the delta rule.

Usage:
  qwen35_ref.py --model M.gguf --tokens ids.txt --out logits.f32
                [--n-positions N] [--margin-out margins.txt] [--topk 5]
`--out` is raw little-endian float32, row-major [n_pos, vocab].  `--margin-out` writes
"<pos> <top1> <top1-top2>" per position.  Run under
  PYTHONPATH=/home/nii/src/llama.cpp/gguf-py uv run --with numpy --with pyyaml --no-project python3 ...
"""
import argparse
import os
import sys

import numpy as np
import gguf

Q8_GROUP = 32
KV_F16 = True
GDN_DUMP = None
X2_DUMP = None
ACT_DUMP = None
Q_DUMP = None
QRAW_DUMP = None
ATTN_SCALE = 1.0
ATTN_F16 = False


def dequant_q8(t):
    """GGUF Q8_0 tensor -> float32 [rows, K].

    gguf-py exposes the tensor as raw uint8 [rows, K + 2K/32], but the on-disk layout is
    32 interleaved 34-byte blocks per row: fp16 d, then 32 int8 weights.  (The q8 pack
    for the kernel moves the scales to the end of the row, so it must be repacked.)
    """
    rows, nbytes = t.data.shape
    K = int(t.shape[0])
    nb = K // Q8_GROUP
    assert nbytes == K + 2 * nb, (t.name, nbytes, K)
    raw = t.data.reshape(rows, nb, 34)
    scale = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(rows, nb)
    q = raw[:, :, 2:34].copy().view(np.int8).astype(np.float32)
    return (q * scale[:, :, None]).reshape(rows, K)


def load_weights(path, verbose=True, n_layers_limit=0):
    r = gguf.GGUFReader(path)
    meta = {}
    for k, v in r.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            val = v.contents()
        except Exception:
            continue
        if k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type"):
            continue
        meta[k] = val
    tensors = {t.name: t for t in r.tensors}

    def get(name):
        t = tensors[name]
        if t.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            return dequant_q8(t)
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            return np.array(t.data, dtype=np.float32, copy=True)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            # numpy has no bfloat16: widen the raw uint16 by 16 bits.
            u = np.frombuffer(np.ascontiguousarray(t.data).tobytes(), dtype=np.uint16)
            u = u.astype(np.uint32) << 16
            return u.view(np.float32).reshape(int(t.shape[1]), int(t.shape[0])).copy()
        raise RuntimeError(f"unsupported type {t.tensor_type} for {name}")

    def has(name):
        return name in tensors

    cfg = dict(
        hid=int(meta["qwen35.embedding_length"]),
        n_layers=int(meta["qwen35.block_count"]),
        inter=int(meta["qwen35.feed_forward_length"]),
        nq=int(meta["qwen35.attention.head_count"]),
        nkv=int(meta["qwen35.attention.head_count_kv"]),
        hd=int(meta["qwen35.attention.key_length"]),
        rot=int(meta["qwen35.rope.dimension_count"]),
        rope_base=float(meta["qwen35.rope.freq_base"]),
        eps=float(meta["qwen35.attention.layer_norm_rms_epsilon"]),
        lin_h=int(meta["qwen35.ssm.group_count"]),
        lin_k=int(meta["qwen35.ssm.state_size"]),
        inner=int(meta["qwen35.ssm.inner_size"]),
        conv_k=int(meta["qwen35.ssm.conv_kernel"]),
        interval=int(meta["qwen35.full_attention_interval"]),
        vocab=int(tensors["token_embd.weight"].shape[1]),
    )
    cfg["lin_v"] = cfg["inner"] // cfg["lin_h"]
    cfg["conv_dim"] = 2 * cfg["lin_h"] * cfg["lin_k"] + cfg["inner"]

    if verbose:
        print(f"[ref] {os.path.basename(path)}: hid={cfg['hid']} inter={cfg['inter']} "
              f"layers={cfg['n_layers']} lin_h={cfg['lin_h']} inner={cfg['inner']}", file=sys.stderr)

    w = {}
    w["embed"] = get("token_embd.weight")
    w["final_norm"] = get("output_norm.weight")
    # tied LM head: same rows as the embedding, but we only need embed for lookup and
    # the full matrix for the final logits.  One copy serves both.
    n_layers = cfg["n_layers"] if not n_layers_limit else min(cfg["n_layers"], n_layers_limit)
    cfg["n_layers"] = n_layers
    for i in range(n_layers):
        p = f"blk.{i}."
        d = {}
        d["in_norm"] = get(p + "attn_norm.weight")
        d["post_norm"] = get(p + "post_attention_norm.weight")
        d["is_attn"] = ((i + 1) % cfg["interval"] == 0)
        if d["is_attn"]:
            d["q"] = get(p + "attn_q.weight")
            d["k"] = get(p + "attn_k.weight")
            d["v"] = get(p + "attn_v.weight")
            d["o"] = get(p + "attn_output.weight")
            d["q_norm"] = get(p + "attn_q_norm.weight")
            d["k_norm"] = get(p + "attn_k_norm.weight")
        else:
            d["qkv"] = get(p + "attn_qkv.weight")
            d["z"] = get(p + "attn_gate.weight")
            d["b"] = get(p + "ssm_beta.weight")
            d["a"] = get(p + "ssm_alpha.weight")
            d["ssm_a"] = get(p + "ssm_a")
            d["dt_bias"] = get(p + "ssm_dt.bias")
            d["conv"] = get(p + "ssm_conv1d.weight")
            d["ssm_norm"] = get(p + "ssm_norm.weight")
            d["out"] = get(p + "ssm_out.weight")
        d["gate"] = get(p + "ffn_gate.weight")
        d["up"] = get(p + "ffn_up.weight")
        d["down"] = get(p + "ffn_down.weight")
        w[i] = d
        if verbose and (i % 6 == 0 or i == cfg["n_layers"] - 1):
            print(f"[ref] loaded layer {i}", file=sys.stderr)
    return cfg, w


def rmsnorm(x, w, eps):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def rope_tables(cfg, max_pos):
    half = cfg["rot"] // 2
    inv = 1.0 / (cfg["rope_base"] ** (np.arange(0, cfg["rot"], 2, dtype=np.float64) / cfg["rot"]))
    pos = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(pos).astype(np.float32), np.sin(pos).astype(np.float32)


def apply_rope(x, cos, sin, rot):
    half = rot // 2
    x1 = x[..., :half].copy()
    x2 = x[..., half:rot].copy()
    x[..., :half] = x1 * cos - x2 * sin
    x[..., half:rot] = x2 * cos + x1 * sin
    return x


class State:
    def __init__(self, cfg, max_pos):
        self.cfg = cfg
        self.pos = 0
        H, Dk, Dv = cfg["lin_h"], cfg["lin_k"], cfg["lin_v"]
        self.conv_hist = {i: np.zeros((cfg["conv_dim"], cfg["conv_k"] - 1), np.float32)
                          for i in range(cfg["n_layers"]) if (i + 1) % cfg["interval"] != 0}
        self.state = {i: np.zeros((H, Dv, Dk), np.float32)
                      for i in range(cfg["n_layers"]) if (i + 1) % cfg["interval"] != 0}
        self.kc = {i: np.zeros((cfg["nkv"], max_pos, cfg["hd"]), np.float32)
                   for i in range(cfg["n_layers"]) if (i + 1) % cfg["interval"] == 0}
        self.vc = {i: np.zeros((cfg["nkv"], max_pos, cfg["hd"]), np.float32)
                   for i in range(cfg["n_layers"]) if (i + 1) % cfg["interval"] == 0}

    def reset(self):
        for d in (self.conv_hist, self.state):
            for v in d.values():
                v[...] = 0
        for d in (self.kc, self.vc):
            for v in d.values():
                v[...] = 0
        self.pos = 0


def layer_gdn(cfg, L, st, i, x):
    hid, H, Dk, Dv, C = cfg["hid"], cfg["lin_h"], cfg["lin_k"], cfg["lin_v"], cfg["conv_dim"]
    qkv = L["qkv"] @ x
    z = (L["z"] @ x).reshape(H, Dv)
    beta = 1.0 / (1.0 + np.exp(-(L["b"] @ x)))
    alpha = L["a"] @ x
    dt = L["dt_bias"]
    sp = np.where(alpha + dt > 20.0, alpha + dt, np.log1p(np.exp(np.minimum(alpha + dt, 20.0))))
    decay = np.exp(L["ssm_a"] * sp)

    hist = st.conv_hist[i]
    win = np.concatenate([hist, qkv[:, None]], axis=1)        # [C, 4]
    st.conv_hist[i] = win[:, 1:].copy()
    conv = win * L["conv"]                                     # [C,4] * [C,4]
    conv = conv.sum(axis=1)
    conv = conv * (1.0 / (1.0 + np.exp(-conv)))                # SiLU
    q = conv[:H * Dk].reshape(H, Dk)
    k = conv[H * Dk:2 * H * Dk].reshape(H, Dk)
    v = conv[2 * H * Dk:].reshape(H, Dv)

    nr = 1.0 / np.sqrt((q * q).sum(axis=1, keepdims=True) + 1e-6) * (1.0 / np.sqrt(Dk))
    q = q * nr
    nr = 1.0 / np.sqrt((k * k).sum(axis=1, keepdims=True) + 1e-6)
    k = k * nr

    S = st.state[i]
    S *= decay[:, None, None]
    kv_mem = np.einsum("hvk,hk->hv", S, k)
    delta = (v - kv_mem) * beta[:, None]
    S += k[:, None, :] * delta[:, :, None]
    o = np.einsum("hvk,hk->hv", S, q)
    g = o * (1.0 / np.sqrt(np.mean(o * o, axis=1, keepdims=True) + cfg["eps"]))
    if GDN_DUMP is not None:
        GDN_DUMP[st.pos, i] = o.reshape(-1)
    o = g * L["ssm_norm"][None, :] * (z * (1.0 / (1.0 + np.exp(-z))))
    return L["out"] @ o.reshape(-1)


def layer_attn(cfg, L, st, cos, sin, i, x):
    nq, nkv, D = cfg["nq"], cfg["nkv"], cfg["hd"]
    qg = (L["q"] @ x).reshape(nq, 2 * D)
    if QRAW_DUMP is not None and i == 3:
        QRAW_DUMP[st.pos] = qg[:, :D].copy()
    q = qg[:, :D]
    gate = qg[:, D:]
    k = (L["k"] @ x).reshape(nkv, D)
    v = (L["v"] @ x).reshape(nkv, D)
    q = rmsnorm(q, L["q_norm"], cfg["eps"])
    k = rmsnorm(k, L["k_norm"], cfg["eps"])
    if Q_DUMP is not None and i == 3 and st.pos == 0:
        import os
        if os.environ.get("Q35_REF_DBG"):
            qq0 = qg[:, :D]
            qt = qq0 * (1.0 / np.sqrt(np.mean(qq0 * qq0, axis=1, keepdims=True) + cfg["eps"])) * L["q_norm"]
            print("[refdbg] q_norm[:8]", L["q_norm"][:8], "q-vs-formula", np.abs(q - qt).max(),
                  "q amax", np.abs(q).max(), "qraw amax", np.abs(qq0).max(), file=__import__("sys").stderr)
    c, s = cos[st.pos], sin[st.pos]
    q = apply_rope(q, c, s, cfg["rot"])
    k = apply_rope(k, c, s, cfg["rot"])
    if Q_DUMP is not None and i == 3:
        Q_DUMP[st.pos] = q.copy()
    st.kc[i][:, st.pos] = k.astype(np.float16) if KV_F16 else k
    st.vc[i][:, st.pos] = v.astype(np.float16) if KV_F16 else v
    n = st.pos + 1
    K = st.kc[i][:, :n].repeat(nq // nkv, axis=0)
    V = st.vc[i][:, :n].repeat(nq // nkv, axis=0)
    if ATTN_F16:
        qq = q.astype(np.float16).astype(np.float32)
        KK = K.astype(np.float16).astype(np.float32)
    else:
        qq, KK = q, K
    scores = np.einsum("hd,htd->ht", qq, KK) * (ATTN_SCALE / np.sqrt(D))
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    o = np.einsum("ht,htd->hd", p, V)
    o = o * (1.0 / (1.0 + np.exp(-gate)))
    return L["o"] @ o.reshape(-1)


def forward(cfg, W, tokens, max_pos, progress=True, layer_dump=None):
    st = State(cfg, max_pos)
    cos, sin = rope_tables(cfg, max_pos)
    n_pos = len(tokens)
    out = np.empty((n_pos, cfg["vocab"]), np.float32)

    for p, tok in enumerate(tokens):
        x = W["embed"][tok].copy()
        for i in range(cfg["n_layers"]):
            L = W[i]
            h = rmsnorm(x, L["in_norm"], cfg["eps"])
            mix = (layer_attn(cfg, L, st, cos, sin, i, h) if L["is_attn"]
                   else layer_gdn(cfg, L, st, i, h))
            x = x + mix
            if X2_DUMP is not None:
                X2_DUMP[p, i] = x
            h = rmsnorm(x, L["post_norm"], cfg["eps"])
            g = L["gate"] @ h
            u = L["up"] @ h
            act = g * (1.0 / (1.0 + np.exp(-g))) * u
            if ACT_DUMP is not None:
                ACT_DUMP[p, i] = act
            x = x + L["down"] @ act
            if layer_dump is not None:
                layer_dump[p, i] = x.copy()
            st.pos = p
        st.pos = p + 1
        h = rmsnorm(x, W["final_norm"], cfg["eps"])
        out[p] = W["embed"] @ h
        if progress and (p + 1) % 16 == 0:
            print(f"[ref] {p + 1}/{n_pos}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True, help="whitespace/newline separated token ids")
    ap.add_argument("--out", required=True, help="raw float32 logits [n_pos, vocab]")
    ap.add_argument("--margin-out", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n-positions", type=int, default=0)
    ap.add_argument("--attn-f16", action="store_true")
    ap.add_argument("--attn-scale", type=float, default=1.0)
    ap.add_argument("--layer-out", default=None)
    ap.add_argument("--gdn-out", default=None)
    ap.add_argument("--x2-out", default=None)
    ap.add_argument("--act-out", default=None)
    ap.add_argument("--q-out", default=None)
    ap.add_argument("--qraw-out", default=None)
    ap.add_argument("--kv-f32", action="store_true")
    ap.add_argument("--n-layers", type=int, default=0, help="debug: use only the first N layers")
    args = ap.parse_args()
    global KV_F16, ATTN_SCALE, ATTN_F16
    KV_F16 = not args.kv_f32
    ATTN_SCALE = args.attn_scale
    ATTN_F16 = args.attn_f16

    with open(args.tokens) as f:
        toks = [int(x) for x in f.read().split()]
    if args.n_positions:
        toks = toks[: args.n_positions]
    assert toks, "empty token list"

    cfg, W = load_weights(args.model, n_layers_limit=args.n_layers)
    max_pos = len(toks) + 8
    global GDN_DUMP, X2_DUMP, ACT_DUMP, Q_DUMP, QRAW_DUMP
    if args.q_out: Q_DUMP = np.zeros((len(toks), cfg["nq"], cfg["hd"]), np.float32)
    if args.qraw_out: QRAW_DUMP = np.zeros((len(toks), cfg["nq"], cfg["hd"]), np.float32)
    if args.x2_out: X2_DUMP = np.zeros((len(toks), cfg["n_layers"], cfg["hid"]), np.float32)
    if args.act_out: ACT_DUMP = np.zeros((len(toks), cfg["n_layers"], cfg["inter"]), np.float32)
    if args.gdn_out:
        GDN_DUMP = np.zeros((len(toks), cfg["n_layers"], cfg["inner"]), np.float32)
    dump = np.empty((len(toks), cfg["n_layers"], cfg["hid"]), np.float32) if args.layer_out else None
    logits = forward(cfg, W, toks, max_pos, layer_dump=dump)
    if dump is not None:
        dump.tofile(args.layer_out)
    if args.gdn_out:
        GDN_DUMP.tofile(args.gdn_out)
    if args.x2_out: X2_DUMP.tofile(args.x2_out)
    if args.act_out: ACT_DUMP.tofile(args.act_out)
    if args.q_out: Q_DUMP.tofile(args.q_out)
    if args.qraw_out: QRAW_DUMP.tofile(args.qraw_out)
    logits.astype(np.float32).tofile(args.out)
    if args.margin_out:
        with open(args.margin_out, "w") as f:
            for p in range(logits.shape[0]):
                row = logits[p]
                top = np.argpartition(row, -(args.topk + 1))[-(args.topk + 1):]
                top = top[np.argsort(-row[top])]
                f.write(f"{p} {int(top[0])} {float(row[top[0]] - row[top[1]]):.6f}\n")
    print(f"[ref] wrote {logits.shape[0]}x{logits.shape[1]} to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
