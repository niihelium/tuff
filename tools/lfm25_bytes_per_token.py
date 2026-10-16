#!/usr/bin/env python3
"""tools/lfm25_bytes_per_token.py - weight/KV bytes streamed from VRAM per decoded token by tuff's LFM2.5 decode path.

Run:  /home/nii/workspace/local-llm/.venv/bin/python -I tools/lfm25_bytes_per_token.py [model.gguf ...]
Counted per token: every per-layer tensor except routed experts (attn q/k/v/o + norms, shortconv in/out/conv, dense FFN,
router F32 + bias, norms), the output norm, the tied LM head (token_embd, read fully), one embedding row, and for MoE
layers the expected routed-expert bytes = n_expert_used x per-expert (gate+up+down) bytes of that layer.
KV bytes at position pos = n_kv_heads*head_dim*pos*attn_layers*2(K,V)*4 B (F32). Conv state is not counted (tiny).
PCIe (expert cache models): miss_rate x n_used x n_moe_layers x slab_bytes, slab = gate+up+down per expert
(down pool sized for the largest down type across layers, reported as 'slab(max)').
"""
import sys
sys.path.insert(0, '/home/nii/workspace/llama.cpp/gguf-py')
from gguf import GGUFReader

DEFAULT = [
    ("1.2B", "/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf"),
    ("2.6B", "/run/media/nii/pohjola/models/LiquidAI/LFM2.5-2.6B/LFM2.5-2.6B-Q4_K_M.gguf"),
    ("8B-A1B", "/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"),
    ("24B-A2B", "/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"),
]
MB = 1e6

def fval(r, key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    return f.contents()

def analyse(name, path):
    r = GGUFReader(path)
    arch = fval(r, "general.architecture")
    nl = int(fval(r, f"{arch}.block_count"))
    n_used = int(fval(r, f"{arch}.expert_used_count", 0) or 0)
    n_exp = int(fval(r, f"{arch}.expert_count", 0) or 0)
    kvh = fval(r, f"{arch}.attention.head_count_kv")
    kvh = [int(x) for x in kvh] if hasattr(kvh, "__len__") else [int(kvh)] * nl
    trunk = 0; experts = 0.0; head = 0; emb_row = 0; kv_per_pos = 0
    slabs = []; n_moe = 0; down_max = 0
    for t in r.tensors:
        n = t.name; b = int(t.n_bytes)
        if n == "token_embd.weight":
            head = b; emb_row = b // int(t.shape[1]); continue
        if n.startswith("blk."):
            li = int(n.split(".")[1])
            if "_exps." in n:
                per = b / n_exp
                experts += n_used * per
                if "down" in n: down_max = max(down_max, per)
                continue
            trunk += b
            if n.endswith("attn_k.weight"):
                head_dim_kv = int(t.shape[1]); # n_kv*head_dim
                kv_per_pos += head_dim_kv * 2 * 4   # K and V, F32
        else:
            trunk += b   # output norm etc.
    # per-layer slab bytes (gate+up+down per expert)
    per_layer = {}
    for t in r.tensors:
        if "_exps." in t.name:
            li = int(t.name.split(".")[1]); per_layer[li] = per_layer.get(li, 0) + int(t.n_bytes) / n_exp
    n_moe = len(per_layer)
    slab_avg = sum(per_layer.values()) / n_moe if n_moe else 0
    gate_up = max((int(t.n_bytes) / n_exp for t in r.tensors if "_exps." in t.name and ("gate" in t.name or "up" in t.name)), default=0)
    slab_max = 2 * gate_up + down_max
    return dict(name=name, arch=arch, layers=nl, n_used=n_used, n_exp=n_exp, trunk=trunk, head=head, emb_row=emb_row,
                experts=experts, kv_per_pos=kv_per_pos, n_moe=n_moe, slab_avg=slab_avg, slab_max=slab_max,
                file=sum(int(t.n_bytes) for t in r.tensors))

def main():
    models = DEFAULT if len(sys.argv) == 1 else [(p.split("/")[-1], p) for p in sys.argv[1:]]
    res = [analyse(n, p) for n, p in models]
    print("Bytes streamed from VRAM per decoded token (MB = 1e6 B). trunk excludes routed experts; KV is F32 K+V read.")
    hdr = f"{'model':8s} {'arch':9s} {'trunk':>9s} {'head':>9s} {'emb row':>8s} {'experts':>9s} {'KV(64)':>8s} {'KV(512)':>8s} {'total@64':>10s} {'total@512':>10s} {'file MB':>9s}"
    print(hdr)
    for x in res:
        base = x['trunk'] + x['head'] + x['emb_row'] + x['experts']
        t64 = base + x['kv_per_pos'] * 64; t512 = base + x['kv_per_pos'] * 512
        print(f"{x['name']:8s} {x['arch']:9s} {x['trunk']/MB:9.2f} {x['head']/MB:9.2f} {x['emb_row']/MB:8.4f} {x['experts']/MB:9.2f} "
              f"{x['kv_per_pos']*64/MB:8.2f} {x['kv_per_pos']*512/MB:8.2f} {t64/MB:10.2f} {t512/MB:10.2f} {x['file']/MB:9.1f}")
    print("\nRoofline ceiling at a given bandwidth = BW / total bytes (tok/s):")
    for bw in (371.4, 384.1):
        print(f"  BW {bw} GB/s: " + "  ".join(f"{x['name']}={bw*1e9/(x['trunk']+x['head']+x['emb_row']+x['experts']+x['kv_per_pos']*64):.0f}" for x in res))
    for x in res:
        if x['n_moe']:
            print(f"\n{x['name']}: MoE layers {x['n_moe']}, experts/layer {x['n_exp']}, used {x['n_used']}, "
                  f"slab avg {x['slab_avg']/MB:.3f} MB, slab(max, pool-sized) {x['slab_max']/MB:.3f} MB")
            print(f"  VRAM expert bytes/token (all hits): {x['experts']/MB:.2f} MB; kv per position: {x['kv_per_pos']} B")
            for mr in (0.03, 0.06, 0.11):
                for lab, s in (("avg", x['slab_avg']), ("max", x['slab_max'])):
                    pc = mr * x['n_used'] * x['n_moe'] * s
                    print(f"  miss {mr*100:4.1f}% [{lab} slab]: PCIe {pc/MB:7.2f} MB/token -> {pc/6.6e9*1e3:6.2f} ms/token at 6.6 GB/s (ceiling {6.6e9/pc:6.0f} tok/s)")

if __name__ == "__main__":
    main()
