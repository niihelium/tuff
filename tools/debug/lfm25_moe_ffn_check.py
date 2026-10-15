#!/usr/bin/env python3
"""Check the MoE expert-FFN semantics (Q8_1 activations, per-expert requant, weights after down, ordered sum)
in numpy against llama.cpp CUDA dumps (tools/lfm25_router_fixtures.c output incl. ffn_moe_swiglu/down/out).

usage: lfm25_moe_ffn_check.py <dumpdir> <model.gguf> <comma-separated layers, e.g. 2,3>"""
import sys; sys.path.insert(0,'/home/nii/workspace/llama.cpp/gguf-py')
import numpy as np, gguf
from gguf.quants import dequantize
D=sys.argv[1]; r=gguf.GGUFReader(sys.argv[2]); T={t.name:t for t in r.tensors}
ld=lambda p,dt=np.float32: np.fromfile(p,dtype=dt)
def q8_1(x):
    b=x.reshape(-1,32).astype(np.float32); d=(np.abs(b).max(1,keepdims=True)/np.float32(127)).astype(np.float32)
    q=np.where(d>0,np.round(b/np.where(d>0,d,1)),0)
    return (q*d.astype(np.float16).astype(np.float64)).reshape(-1)
def W(name,e):
    t=T[name]; return dequantize(t.data[e],t.tensor_type).astype(np.float64)
for L in [int(a) for a in sys.argv[3].split(',')]:
    for t in range(8):
        x=ld(f'{D}/router_in-{L}-t{t}.f32').astype(np.float64); ids=ld(f'{D}/ffn_moe_topk-{L}-t{t}.i32',np.int32); w=ld(f'{D}/ffn_moe_weights_norm-{L}-t{t}.f32').astype(np.float64)
        sw=ld(f'{D}/ffn_moe_swiglu-{L}-t{t}.f32').reshape(4,-1); dn=ld(f'{D}/ffn_moe_down-{L}-t{t}.f32').reshape(4,-1); out=ld(f'{D}/ffn_moe_out-{L}-t{t}.f32')
        xq=q8_1(x); ys=[]; e_sw=0; e_dn=0
        for k,e in enumerate(ids):
            g=W(f'blk.{L}.ffn_gate_exps.weight',e)@xq; u=W(f'blk.{L}.ffn_up_exps.weight',e)@xq
            a=g/(1+np.exp(-g))*u; e_sw=max(e_sw,np.abs(a-sw[k]).max()/np.abs(sw[k]).max())
            y=W(f'blk.{L}.ffn_down_exps.weight',e)@q8_1(sw[k].astype(np.float64)); e_dn=max(e_dn,np.abs(y-dn[k]).max()/np.abs(dn[k]).max()); ys.append(y)
        o=((w[0]*ys[0]+w[1]*ys[1])+w[2]*ys[2])+w[3]*ys[3]
        print(f'L{L:2d} t{t} down={T[f"blk.{L}.ffn_down_exps.weight"].tensor_type.name} swiglu relerr {e_sw:.1e}  down relerr {e_dn:.1e}  out relerr {np.abs(o-out).max()/np.abs(out).max():.1e}')
