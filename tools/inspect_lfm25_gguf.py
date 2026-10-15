#!/usr/bin/env python3
"""
tools/inspect_lfm25_gguf.py - GGUF Metadata & Tensor Layout Inspector for Liquid AI LFM2.5 Models

Supports:
  - Dense LFM2 models (e.g. LFM2.5-1.2B, 2.6B)
  - Sparse MoE LFM2 models (e.g. LFM2.5-8B-A1B, 24B-A2B)
"""

import sys
import os
import struct
import json
import argparse
from collections import Counter, defaultdict

GGML_TYPES = {
    0:  ('F32', 4.0),
    1:  ('F16', 2.0),
    2:  ('Q4_0', 0.5 + 2.0/32),
    3:  ('Q4_1', 0.5 + 4.0/32),
    6:  ('Q5_0', 0.625 + 2.0/32),
    7:  ('Q5_1', 0.625 + 4.0/32),
    8:  ('Q8_0', 1.0 + 2.0/32),
    9:  ('Q8_1', 1.0 + 4.0/32),
    10: ('Q2_K', 2.5625 / 8),
    11: ('Q3_K', 3.4375 / 8),
    12: ('Q4_K', 4.5 / 8),       # 144 bytes per 256 elements
    13: ('Q5_K', 5.5 / 8),       # 176 bytes per 256 elements
    14: ('Q6_K', 6.5625 / 8),    # 210 bytes per 256 elements
    15: ('Q8_K', 8.5 / 8),
    16: ('IQ2_XXS', 2.0625 / 8),
    17: ('IQ2_XS', 2.3125 / 8),
    18: ('IQ3_XXS', 3.0625 / 8),
    19: ('IQ1_S', 1.5625 / 8),
    20: ('IQ4_NL', 4.5 / 8),
    21: ('IQ3_S', 3.4375 / 8),
    22: ('IQ2_S', 2.5 / 8),
    23: ('IQ4_XS', 4.25 / 8),
    24: ('I8', 1.0),
    25: ('I16', 2.0),
    26: ('I32', 4.0),
    27: ('I64', 8.0),
    28: ('F64', 8.0),
    29: ('IQ1_M', 1.75 / 8),
    30: ('BF16', 2.0),
}

def parse_gguf_metadata(f):
    magic = f.read(4)
    if magic != b'GGUF':
        raise ValueError(f"Invalid GGUF magic: {magic}")
    
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
    
    def read_str():
        slen = struct.unpack('<Q', f.read(8))[0]
        return f.read(slen).decode('utf-8', errors='replace')
    
    def read_val(vtype):
        if vtype == 0: return struct.unpack('<B', f.read(1))[0]
        elif vtype == 1: return struct.unpack('<b', f.read(1))[0]
        elif vtype == 2: return struct.unpack('<H', f.read(2))[0]
        elif vtype == 3: return struct.unpack('<h', f.read(2))[0]
        elif vtype == 4: return struct.unpack('<I', f.read(4))[0]
        elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
        elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
        elif vtype == 7: return struct.unpack('<?', f.read(1))[0]
        elif vtype == 8: return read_str()
        elif vtype == 9:
            atype, arr_len = struct.unpack('<IQ', f.read(12))
            # skip reading giant string arrays like tokenizer tokens
            if atype == 8:
                first_items = []
                for idx in range(arr_len):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    sval = f.read(slen).decode('utf-8', errors='replace')
                    if idx < 5:
                        first_items.append(sval)
                return {
                    '__type__': f'array[{arr_len} strings]',
                    'length': arr_len,
                    'sample': first_items
                }
            sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
            if atype in sizes:
                elem_size = sizes[atype]
                raw = f.read(arr_len * elem_size)
                if arr_len <= 64:
                    fmt_char = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}[atype]
                    return list(struct.unpack(f'<{arr_len}{fmt_char}', raw))
                else:
                    return f'<array of {arr_len} elements>'
            else:
                return [read_val(atype) for _ in range(arr_len)]
        elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
        elif vtype == 11: return struct.unpack('<q', f.read(8))[0]
        elif vtype == 12: return struct.unpack('<d', f.read(8))[0]
        else: raise ValueError(f"Unknown value type: {vtype}")

    kv = {}
    for _ in range(n_kv):
        k = read_str()
        vtype = struct.unpack('<I', f.read(4))[0]
        kv[k] = read_val(vtype)
        
    tensors = []
    for _ in range(n_tensors):
        tname = read_str()
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = list(struct.unpack(f'<{n_dims}Q', f.read(8 * n_dims)))
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        
        type_str, bytes_per_elem = GGML_TYPES.get(ttype, (f'TYPE_{ttype}', 1.0))
        num_elements = 1
        for d in dims:
            num_elements *= d
        est_bytes = int(num_elements * bytes_per_elem)
        
        tensors.append({
            'name': tname,
            'dims': dims,
            'type_id': ttype,
            'type_name': type_str,
            'offset': offset,
            'num_elements': num_elements,
            'est_bytes': est_bytes,
        })
        
    return version, kv, tensors

def inspect_model(path, verbose=False, dump_json=False):
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
        
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        version, kv, tensors = parse_gguf_metadata(f)

    arch = kv.get('general.architecture', 'unknown')
    prefix = arch + '.'
    
    # Extract architecture params
    block_count = kv.get(prefix + 'block_count', 0)
    embedding_length = kv.get(prefix + 'embedding_length', 0)
    context_length = kv.get(prefix + 'context_length', 0)
    head_count = kv.get(prefix + 'attention.head_count', 0)
    head_count_kv = kv.get(prefix + 'attention.head_count_kv', head_count)
    rms_eps = kv.get(prefix + 'attention.layer_norm_rms_epsilon', 1e-5)
    
    # Conv params
    conv_cache = kv.get(prefix + 'shortconv.l_cache', 0)
    
    # MoE params
    expert_count = kv.get(prefix + 'expert_count', 0)
    expert_used = kv.get(prefix + 'expert_used_count', 0)
    expert_ffn_len = kv.get(prefix + 'expert_feed_forward_length', 0)
    leading_dense = kv.get(prefix + 'leading_dense_block_count', 0)
    expert_gating_func = kv.get(prefix + 'expert_gating_func', 0)

    # Classify layers
    layer_types = defaultdict(dict)
    category_bytes = Counter()
    type_counts = Counter()
    
    for t in tensors:
        name = t['name']
        t_type = t['type_name']
        b = t['est_bytes']
        type_counts[t_type] += 1
        
        if name in ('token_embd.weight', 'token_embd_norm.weight'):
            category_bytes['Trunk: Embeddings'] += b
        elif name in ('output.weight', 'output_norm.weight'):
            category_bytes['Trunk: Output Head & Norm'] += b
        elif name.startswith('blk.'):
            parts = name.split('.')
            l_idx = int(parts[1])
            sub = '.'.join(parts[2:])
            
            if 'shortconv' in sub:
                category_bytes['Trunk: ShortConv Blocks'] += b
                layer_types[l_idx]['has_conv'] = True
            elif any(x in sub for x in ('attn_q.', 'attn_k.', 'attn_v.', 'attn_output.', 'attn_q_norm.', 'attn_k_norm.')):
                category_bytes['Trunk: Attention Blocks'] += b
                if 'attn_q.' in sub:
                    layer_types[l_idx]['has_attn'] = True
            elif 'exps' in sub or 'gate_inp' in sub or 'exp_probs_b' in sub:
                category_bytes['MoE: Routed Experts & Router'] += b
                layer_types[l_idx]['has_moe'] = True
            elif 'ffn_gate.' in sub or 'ffn_up.' in sub or 'ffn_down.' in sub:
                category_bytes['Trunk: Dense FFN Blocks'] += b
                layer_types[l_idx]['has_dense_ffn'] = True
            else:
                category_bytes['Trunk: Layer Norms'] += b
        else:
            category_bytes['Other'] += b

    attn_layers = sorted([l for l, d in layer_types.items() if d.get('has_attn')])
    conv_layers = sorted([l for l, d in layer_types.items() if d.get('has_conv')])
    moe_layers = sorted([l for l, d in layer_types.items() if d.get('has_moe')])
    dense_ffn_layers = sorted([l for l, d in layer_types.items() if d.get('has_dense_ffn')])

    # Calculate actual active KV heads across attention layers
    if isinstance(head_count_kv, list):
        active_kv_heads = max(head_count_kv) if head_count_kv else 8
    else:
        active_kv_heads = int(head_count_kv) if head_count_kv else 8

    # Output JSON if requested
    if dump_json:
        report = {
            'file': path,
            'file_size_bytes': file_size,
            'gguf_version': version,
            'architecture': arch,
            'block_count': block_count,
            'context_length': context_length,
            'embedding_length': embedding_length,
            'head_count': head_count,
            'head_count_kv': head_count_kv,
            'shortconv_cache_len': conv_cache,
            'is_moe': expert_count > 0,
            'expert_count': expert_count,
            'expert_used_count': expert_used,
            'leading_dense_blocks': leading_dense,
            'attn_layers': attn_layers,
            'conv_layers': conv_layers,
            'moe_layers': moe_layers,
            'dense_ffn_layers': dense_ffn_layers,
            'tensor_count': len(tensors),
            'type_counts': dict(type_counts),
            'category_bytes': dict(category_bytes)
        }
        if verbose:
            report['tensors'] = tensors
        print(json.dumps(report, indent=2))
        return

    # Formatted terminal report
    print("=" * 80)
    print(f"       LFM2.5 GGUF TENSOR & METADATA INSPECTOR ({os.path.basename(path)})")
    print("=" * 80)
    print(f"  File Path:            {path}")
    print(f"  File Size:            {file_size / (1024**3):.2f} GiB ({file_size:,} bytes)")
    print(f"  GGUF Format Version:  v{version}")
    print(f"  Architecture:         {arch}")
    print(f"  Total Layers:         {block_count}")
    print(f"  Embedding Dimension:  {embedding_length}")
    print(f"  Context Window:       {context_length:,} tokens")
    print(f"  Attention Heads:      {head_count} Q, {head_count_kv} KV")
    print(f"  RMS Norm Epsilon:     {rms_eps}")
    print(f"  ShortConv Window:     K = {conv_cache} (rolling state per channel: {conv_cache - 1})")
    
    if expert_count > 0:
        print(f"  MoE Configuration:    {expert_count} experts, Top-{expert_used} routing")
        print(f"  Expert FFN Dim:       {expert_ffn_len}")
        print(f"  Leading Dense Layers: {leading_dense} (Layers 0..{leading_dense-1})")
        print(f"  Gating Function:      {expert_gating_func} (Softmax)")
    else:
        print(f"  MoE Configuration:    None (100% Dense Model)")
    print("-" * 80)
    
    print("Layer Composition Breakdown:")
    print(f"  * Conv1D Layers ({len(conv_layers)}):      {conv_layers if len(conv_layers) <= 12 else f'{len(conv_layers)} layers'}")
    print(f"  * Attention Layers ({len(attn_layers)}):   {attn_layers}")
    if expert_count > 0:
        print(f"  * MoE Layers ({len(moe_layers)}):        {moe_layers if len(moe_layers) <= 12 else f'{len(moe_layers)} layers'}")
        print(f"  * Dense FFN Layers ({len(dense_ffn_layers)}):  {dense_ffn_layers}")
    print("-" * 80)
    
    print("Memory Footprint by Component:")
    total_est = sum(category_bytes.values())
    for cat, b in sorted(category_bytes.items(), key=lambda x: -x[1]):
        pct = (b / total_est) * 100
        print(f"  {cat:32s}: {b / (1024**2):9.1f} MiB  ({pct:5.1f}%)")
    print(f"  {'Total Model Weights':32s}: {total_est / (1024**2):9.1f} MiB  (100.0%)")
    print("-" * 80)
    
    print("Tensor Quantization Types Breakdown:")
    for t_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t_type:12s}: {count:4d} tensors")
    print("-" * 80)

    # 8 GB VRAM Budget Estimate
    if expert_count > 0:
        trunk_bytes = (category_bytes['Trunk: Embeddings'] +
                       category_bytes['Trunk: Output Head & Norm'] +
                       category_bytes['Trunk: ShortConv Blocks'] +
                       category_bytes['Trunk: Attention Blocks'] +
                       category_bytes['Trunk: Dense FFN Blocks'])
        moe_bytes = category_bytes['MoE: Routed Experts & Router']
        vram_budget_mib = 7880  # RTX 3070 usable
        trunk_bytes = (category_bytes['Trunk: Embeddings'] +
                       category_bytes['Trunk: Output Head & Norm'] +
                       category_bytes['Trunk: ShortConv Blocks'] +
                       category_bytes['Trunk: Attention Blocks'] +
                       category_bytes['Trunk: Dense FFN Blocks'] +
                       category_bytes['Trunk: Layer Norms'])
        moe_bytes = category_bytes['MoE: Routed Experts & Router']
        vram_budget_mib = 7880  # RTX 3070 usable
        trunk_mib = trunk_bytes / (1024**2)
        head_dim = (embedding_length // head_count) if head_count else 64
        # KV cache: 2 (K & V) * num_attn_layers * num_tokens * active_kv_heads * head_dim * sizeof(fp16)
        kv_128k_mib = (131072 * 2 * len(attn_layers) * active_kv_heads * head_dim * 2) / (1024**2)
        scratch_mib = 250
        free_for_lfru = max(0, vram_budget_mib - trunk_mib - min(kv_128k_mib, 800) - scratch_mib)
        moe_mib = moe_bytes / (1024**2)
        fit_pct = min(100.0, (free_for_lfru / moe_mib) * 100) if moe_mib > 0 else 100.0
        
        print("Target RTX 3070 (8 GB / ~7,880 MiB Usable) Budget Simulation:")
        print(f"  Trunk Weights (Resident in VRAM): {trunk_mib:7.1f} MiB")
        print(f"  Scratch & Active KV Buffers:      ~{min(kv_128k_mib, 800) + scratch_mib:5.0f} MiB")
        print(f"  Available for MoE LFRU Cache:     {free_for_lfru:7.1f} MiB")
        print(f"  Total MoE Expert Data:            {moe_mib:7.1f} MiB")
        print(f"  Projected VRAM Expert Residency:  {fit_pct:7.1f}%")
        print("=" * 80)
    else:
        print("Target RTX 3070 (8 GB) Budget Simulation:")
        print(f"  Model fits 100% resident in GPU VRAM with ~{7880 - (total_est / (1024**2)):.0f} MiB free headroom.")
        print("=" * 80)

    if verbose:
        print("\nAll Model Tensors:")
        print(f"{'Tensor Name':45s} {'Shape':22s} {'Type':8s} {'Size (MiB)':10s}")
        print("-" * 90)
        for t in tensors:
            s_dims = str(tuple(t['dims']))
            print(f"{t['name']:45s} {s_dims:22s} {t['type_name']:8s} {t['est_bytes'] / (1024**2):9.2f}")

def main():
    parser = argparse.ArgumentParser(description="GGUF Metadata & Tensor Layout Inspector for LFM2.5 Models")
    parser.add_argument("model", nargs="?", default=None,
                        help="Path to GGUF model file")
    parser.add_argument("--json", action="store_true", help="Output summary in JSON format")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print all individual tensors")
    args = parser.parse_args()

    default_model = "/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"
    model_path = args.model or default_model
    
    inspect_model(model_path, verbose=args.verbose, dump_json=args.json)

if __name__ == '__main__':
    main()
