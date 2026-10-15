#!/usr/bin/env python3
"""Pack tools/lfm25_router_fixtures.c output into tests/fixtures/lfm25_8b_router_cases.bin.

usage: lfm25_pack_router_fixtures.py <dumpdir> <out.bin> [--layers 2-23] [--steps 8]

Format (little-endian): char[8] "LFMRTR01", u32 n_cases, u32 dim, u32 n_expert, u32 k;
per case: u32 layer, u32 step, f32 x[dim], i32 topk[k], f32 weights[k], f32 logits[n_expert].
"""
import argparse
import struct

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumpdir")
    ap.add_argument("out")
    ap.add_argument("--layers", default="2-23")
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.layers.split("-"))
    cases = [(l, t) for l in range(lo, hi + 1) for t in range(a.steps)]
    D = a.dumpdir
    with open(a.out, "wb") as f:
        f.write(b"LFMRTR01")
        f.write(struct.pack("<4I", len(cases), 2048, 32, 4))
        for l, t in cases:
            f.write(struct.pack("<2I", l, t))
            f.write(np.fromfile(f"{D}/router_in-{l}-t{t}.f32", np.float32).tobytes())
            f.write(np.fromfile(f"{D}/ffn_moe_topk-{l}-t{t}.i32", np.int32).tobytes())
            f.write(np.fromfile(f"{D}/ffn_moe_weights_norm-{l}-t{t}.f32", np.float32).tobytes())
            f.write(np.fromfile(f"{D}/ffn_moe_logits-{l}-t{t}.f32", np.float32).tobytes())
    print(f"wrote {len(cases)} cases to {a.out}")


if __name__ == "__main__":
    main()
