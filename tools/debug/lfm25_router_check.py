#!/usr/bin/env python3
"""Check the MoE router semantics in numpy against llama.cpp dumps (tools/lfm25_router_fixtures.c output).

usage: lfm25_router_check.py <dumpdir> <model.gguf> [first_moe_layer last_moe_layer]  (8B: 2 23, 24B: 2 39)
Prints how many top-4 sets match llama.cpp exactly (same order) and the max logit / weight differences."""
import sys, glob, os
sys.path.insert(0,'/home/nii/workspace/llama.cpp/gguf-py')
import numpy as np, gguf
D=sys.argv[1]; r=gguf.GGUFReader(sys.argv[2]); T={t.name:t for t in r.tensors}
ld=lambda p,dt=np.float32: np.fromfile(p,dtype=dt)
ok=0; n=0; max_lg=0; max_w=0
L0, L1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2, 23)
for L in range(L0, L1 + 1):
    W=np.array(T[f'blk.{L}.ffn_gate_inp.weight'].data,dtype=np.float64)   # [32,2048]
    b=np.array(T[f'blk.{L}.exp_probs_b.bias'].data,dtype=np.float64)
    for t in range(8):
        x=ld(f'{D}/router_in-{L}-t{t}.f32').astype(np.float64)
        lg=W@x; p=1/(1+np.exp(-lg)); sel=p+b
        top=np.argsort(-sel,kind='stable')[:4]; w=p[top]/p[top].sum()
        llg=ld(f'{D}/ffn_moe_logits-{L}-t{t}.f32'); ltop=ld(f'{D}/ffn_moe_topk-{L}-t{t}.i32',np.int32); lw=ld(f'{D}/ffn_moe_weights_norm-{L}-t{t}.f32')
        max_lg=max(max_lg,np.abs(lg-llg).max()); n+=1
        if list(ltop)==list(top): ok+=1; max_w=max(max_w,np.abs(w-lw).max())
        else: print('mismatch L',L,'t',t,'llama',list(ltop),'numpy',list(top), 'sel gaps', np.round(np.sort(sel)[::-1][:6],5))
print(f'{ok}/{n} top-4 sets identical (same order); max |logit diff| {max_lg:.2e}; max |weight diff| {max_w:.2e}')
