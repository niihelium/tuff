#!/usr/bin/env python3
"""Pack tools/lfm25_router_fixtures.c output into tests/fixtures/lfm25_8b_moe_ffn_cases.bin.

The dumper must have been built with the expert-FFN tensors enabled (ffn_moe_swiglu, ffn_moe_out;
see the `wanted()` list in tools/lfm25_router_fixtures.c).

usage: lfm25_pack_moe_fixtures.py <dumpdir> <out.bin> [--layers 2-23] [--steps 0,5]

Format (little-endian): char[8] "LFMMOE01", u32 n_cases, u32 dim, u32 ffn_dim, u32 k;
per case: u32 layer, u32 step, f32 x[dim], i32 topk[k], f32 weights[k],
          f32 act[k * ffn_dim] (silu(gate) * up per selected expert, in topk order),
          f32 out[dim] (weighted sum of the k expert down projections).
"""
import argparse
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumpdir")
    ap.add_argument("out")
    ap.add_argument("--layers", default="2-23")
    ap.add_argument("--steps", default="0,5")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.layers.split("-"))
    steps = [int(s) for s in a.steps.split(",")]
    cases = [(l, t) for l in range(lo, hi + 1) for t in steps]
    D = a.dumpdir
    dim, ffn, k = 2048, 1792, 4
    with open(a.out, "wb") as f:
        f.write(b"LFMMOE01")
        f.write(struct.pack("<4I", len(cases), dim, ffn, k))
        for l, t in cases:
            act = np.fromfile(f"{D}/ffn_moe_swiglu-{l}-t{t}.f32", np.float32)
            assert act.size == k * ffn, act.size
            f.write(struct.pack("<2I", l, t))
            f.write(np.fromfile(f"{D}/router_in-{l}-t{t}.f32", np.float32).tobytes())
            f.write(np.fromfile(f"{D}/ffn_moe_topk-{l}-t{t}.i32", np.int32).tobytes())
            f.write(np.fromfile(f"{D}/ffn_moe_weights_norm-{l}-t{t}.f32", np.float32).tobytes())
            f.write(act.tobytes())
            f.write(np.fromfile(f"{D}/ffn_moe_out-{l}-t{t}.f32", np.float32).tobytes())
    print(f"wrote {len(cases)} cases to {a.out}")


if __name__ == "__main__":
    main()
