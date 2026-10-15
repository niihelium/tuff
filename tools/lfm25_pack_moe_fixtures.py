#!/usr/bin/env python3
"""Pack tools/lfm25_router_fixtures.c output into tests/fixtures/lfm25_8b_moe_ffn_cases.bin.

The dumper must have been built with the expert-FFN tensors enabled (ffn_moe_swiglu, ffn_moe_out;
see the `wanted()` list in tools/lfm25_router_fixtures.c).

usage: lfm25_pack_moe_fixtures.py <dumpdir> <out.bin> [--layers 2-23] [--steps 0,5]
                                  [--simulate-gpu <model.gguf>]

--simulate-gpu: compute act/out with a numpy model of llama.cpp's CUDA path (Q8_1 activations,
per-expert requantization, weights applied after down, ordered sum) instead of reading llama.cpp's
dumped act/out. Needed for the 24B, whose reference run keeps experts on llama.cpp's CPU backend
(different activation quantization). The simulation matches llama.cpp's CUDA output to ~1e-7 on
the 8B (checked by regenerating lfm25_8b_moe_ffn_cases.bin within that tolerance).

Format (little-endian): char[8] "LFMMOE01", u32 n_cases, u32 dim, u32 ffn_dim, u32 k;
per case: u32 layer, u32 step, f32 x[dim], i32 topk[k], f32 weights[k],
          f32 act[k * ffn_dim] (silu(gate) * up per selected expert, in topk order),
          f32 out[dim] (weighted sum of the k expert down projections).
"""
import argparse
import os
import struct
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumpdir")
    ap.add_argument("out")
    ap.add_argument("--layers", default="2-23")
    ap.add_argument("--steps", default="0,5")
    ap.add_argument("--simulate-gpu", metavar="MODEL")
    a = ap.parse_args()
    sim = None
    if a.simulate_gpu:
        sys.path.insert(0, os.environ.get("LLAMA_GGUF_PY", "/home/nii/workspace/llama.cpp/gguf-py"))
        import gguf
        from gguf.quants import dequantize
        T = {t.name: t for t in gguf.GGUFReader(a.simulate_gpu).tensors}

        def q8_1(x):  # dequantized value of block_q8_1 quantization (d stored as fp16)
            b = x.reshape(-1, 32).astype(np.float32)
            d = (np.abs(b).max(1, keepdims=True) / np.float32(127)).astype(np.float32)
            q = np.where(d > 0, np.round(b / np.where(d > 0, d, 1)), 0)
            return (q * d.astype(np.float16).astype(np.float64)).reshape(-1)

        def sim(l, x, ids, w):
            W = lambda n, e: dequantize(T[f"blk.{l}.{n}"].data[e], T[f"blk.{l}.{n}"].tensor_type).astype(np.float64)
            xq = q8_1(x.astype(np.float64))
            acts, out = [], np.zeros(x.size)
            ys = []
            for e in ids:
                g = W("ffn_gate_exps.weight", e) @ xq
                u = W("ffn_up_exps.weight", e) @ xq
                act = (g / (1.0 + np.exp(-g)) * u).astype(np.float32)
                acts.append(act)
                ys.append(W("ffn_down_exps.weight", e) @ q8_1(act.astype(np.float64)))
            out = ((w[0] * ys[0] + w[1] * ys[1]) + w[2] * ys[2]) + w[3] * ys[3]
            return np.concatenate(acts), out.astype(np.float32)
    lo, hi = (int(v) for v in a.layers.split("-"))
    steps = [int(s) for s in a.steps.split(",")]
    cases = [(l, t) for l in range(lo, hi + 1) for t in steps]
    D = a.dumpdir
    dim, k = 2048, 4
    l0, t0 = cases[0]
    ffn = (np.fromfile(f"{D}/ffn_moe_swiglu-{l0}-t{t0}.f32", np.float32).size // k) if not sim else \
        int(T[f"blk.{l0}.ffn_gate_exps.weight"].shape[1])
    with open(a.out, "wb") as f:
        f.write(b"LFMMOE01")
        f.write(struct.pack("<4I", len(cases), dim, ffn, k))
        for l, t in cases:
            x = np.fromfile(f"{D}/router_in-{l}-t{t}.f32", np.float32)
            ids = np.fromfile(f"{D}/ffn_moe_topk-{l}-t{t}.i32", np.int32)
            w = np.fromfile(f"{D}/ffn_moe_weights_norm-{l}-t{t}.f32", np.float32)
            if sim:
                act, out = sim(l, x, ids, w.astype(np.float64))
            else:
                act = np.fromfile(f"{D}/ffn_moe_swiglu-{l}-t{t}.f32", np.float32)
                out = np.fromfile(f"{D}/ffn_moe_out-{l}-t{t}.f32", np.float32)
            assert act.size == k * ffn, act.size
            f.write(struct.pack("<2I", l, t))
            f.write(x.tobytes())
            f.write(ids.tobytes())
            f.write(w.tobytes())
            f.write(act.tobytes())
            f.write(out.tobytes())
    print(f"wrote {len(cases)} cases to {a.out}")


if __name__ == "__main__":
    main()
