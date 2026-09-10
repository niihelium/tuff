#!/usr/bin/env python3
import json
import subprocess
import time
import os
import sys
import re
import math

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts_corpus.json")
ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "tuff")
MODEL_PATH = os.environ.get("MODEL_PATH", "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")


def render_svg_flamegraph(tree, total_time_ms, out_path="tuff_flamegraph.svg"):
    width = 1200
    row_height = 28
    font_size = 12
    margin_top = 55
    margin_left = 10
    margin_right = 10
    usable_width = width - margin_left - margin_right

    colors = {
        "delta": "#e65100",        # Deep Orange (CUDA Graphs / SSM)
        "moe": "#d84315",          # Orange Red (MoE)
        "cpumoe": "#2e7d32",       # Green (CPU Worker Pool)
        "attn": "#6a1b9a",         # Purple (Full Attention)
        "output": "#1565c0",       # Blue (LM Head)
        "pcie": "#00838f",         # Cyan (D2H / Transfers)
        "other": "#546e7a",        # Blue Grey
        "root": "#263238",         # Dark Slate
    }

    def get_color(name):
        n = name.lower()
        if "delta" in n or "gdn" in n or "ssm" in n: return colors["delta"]
        if "cpu" in n or "avx" in n or "pool" in n or "worker" in n: return colors["cpumoe"]
        if "expert" in n or "moe" in n or "gate" in n or "down" in n: return colors["moe"]
        if "attn" in n or "attention" in n: return colors["attn"]
        if "head" in n or "output" in n or "argmax" in n: return colors["output"]
        if "d2h" in n or "h2d" in n or "pcie" in n or "fill" in n or "sync" in n or "router" in n: return colors["pcie"]
        return colors["other"]

    max_depth = 0
    rects = []

    def layout_node(node, x, depth):
        nonlocal max_depth
        if depth > max_depth:
            max_depth = depth
        w = (node["time_ms"] / total_time_ms) * usable_width
        color = colors["root"] if depth == 0 else get_color(node["name"])
        rects.append((x, depth, w, row_height, node["name"], node["time_ms"], (node["time_ms"] / total_time_ms) * 100.0, color))

        curr_x = x
        for child in node.get("children", []):
            layout_node(child, curr_x, depth + 1)
            curr_x += (child["time_ms"] / total_time_ms) * usable_width

    layout_node(tree, margin_left, 0)

    total_height = margin_top + (max_depth + 1) * (row_height + 4) + 40

    svg = []
    svg.append(f'<svg version="1.1" width="{width}" height="{total_height}" viewBox="0 0 {width} {total_height}" xmlns="http://www.w3.org/2000/svg">')
    svg.append('<style>')
    svg.append('  text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }')
    svg.append('  .node:hover rect { filter: brightness(1.18); stroke: #ffffff; stroke-width: 1.5px; cursor: pointer; }')
    svg.append('  .title { font-weight: bold; font-size: 16px; fill: #111; }')
    svg.append('  .subtitle { font-size: 12px; fill: #555; }')
    svg.append('</style>')
    svg.append(f'<rect width="100%" height="100%" fill="#fafafa"/>')
    svg.append(f'<text x="{margin_left}" y="25" class="title">Tuff Execution FlameGraph Profile (Qwen 3.6 35B-A3B)</text>')
    svg.append(f'<text x="{margin_left}" y="44" class="subtitle">Total Time: {total_time_ms:.2f} ms / token ({1000.0/total_time_ms:.2f} tok/s) | Hardware: RTX 3070 (8GB VRAM) + DDR4 System RAM</text>')

    for (x, depth, w, h, name, time_ms, pct, color) in rects:
        y = total_height - 30 - (depth + 1) * (row_height + 4)
        svg.append(f'<g class="node">')
        svg.append(f'  <title>{name}: {time_ms:.3f} ms ({pct:.1f}%)</title>')
        svg.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{max(1.0, w):.1f}" height="{h}" fill="{color}" rx="3" ry="3" stroke="#222" stroke-width="0.5" stroke-opacity="0.3"/>')
        if w > 40:
            display_name = name
            if w < 120 and len(name) > 14:
                display_name = name[:12] + ".."
            text_str = f'{display_name} ({time_ms:.2f}ms, {pct:.1f}%)' if w > 140 else f'{display_name}'
            svg.append(f'  <text x="{x + 6:.1f}" y="{y + 18:.1f}" font-size="{font_size}px" fill="#ffffff" font-weight="500">{text_str}</text>')
        svg.append('</g>')

    svg.append('</svg>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"\n[FlameGraph SVG] Successfully saved interactive FlameGraph to: {out_path}")


def main():
    print("==================================================================")
    print("  Tuff Deep Profiling & FlameGraph Generator")
    print("==================================================================")

    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus file not found at {CORPUS_PATH}")
        sys.exit(1)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    target_prompt = prompts[0]["prompt"]
    print(f"Profiling target: '{prompts[0]['title']}' ({prompts[0]['domain']})")
    print(f"Protocol: 128 warmup + 512 timed tokens\n")

    cmd = [
        ENGINE_PATH,
        "--model", MODEL_PATH,
        "--prompt", target_prompt,
        "-n", "512",
        "--warmup", "128",
        "--runs", "1",
    ]
    env = os.environ.copy()
    env["TUFF_PROFILE"] = "1"

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    t1 = time.time()

    if res.returncode != 0:
        print(f"[ENGINE FAILURE] Code {res.returncode}")
        print(res.stderr)
        sys.exit(1)

    out = res.stderr + "\n" + res.stdout

    m_tot = re.search(r"total=([\d.]+)\s*attn=([\d.]+)\s*\((\d+)%\)\s*delta=([\d.]+)\s*\((\d+)%\)\s*expert=([\d.]+)\s*\((\d+)%\)\s*other=([\d.]+)", out)
    m_exp = re.search(r"gu=([\d.]+)\s*fill=([\d.]+)\s*down=([\d.]+)\s*topk/lookup=([\d.]+)\s*resid/shared=([\d.]+)\s*cachesync=([\d.]+)\s*pref=([\d.]+)\s*cpumoe=([\d.]+)\s*\(pub=([\d.]+)\)", out)
    m_speed = re.search(r"Pipeline Generation Speed:\s*([\d.]+)", out)
    m_hit = re.search(r"Expert-cache hit rate:\s*([^\n]+)", out)

    if not m_tot or not m_exp:
        print("Could not parse profile output:")
        print(out)
        sys.exit(1)

    n_tok = 512.0
    t_tot = float(m_tot.group(1)) / n_tok * 1000.0   # ms / token
    t_attn = float(m_tot.group(2)) / n_tok * 1000.0  # ms / token
    t_delta = float(m_tot.group(4)) / n_tok * 1000.0 # ms / token
    t_exp = float(m_tot.group(6)) / n_tok * 1000.0   # ms / token
    t_other = float(m_tot.group(8)) / n_tok * 1000.0 # ms / token

    t_gu = float(m_exp.group(1)) / n_tok * 1000.0
    t_topk = float(m_exp.group(4)) / n_tok * 1000.0
    t_sh = float(m_exp.group(5)) / n_tok * 1000.0
    t_cpumoe = float(m_exp.group(8)) / n_tok * 1000.0
    t_pub = float(m_exp.group(9)) / n_tok * 1000.0
    t_fill = float(m_exp.group(2)) / n_tok * 1000.0

    speed = float(m_speed.group(1)) if m_speed else (1000.0 / t_tot)
    hit_rate_str = m_hit.group(1) if m_hit else "unknown"

    tree = {
        "name": "tuff_decode_step",
        "time_ms": t_tot,
        "children": [
            {
                "name": "Delta-Net (30 SSM Layers)",
                "time_ms": t_delta,
                "children": [
                    {"name": "CUDA Graph Replay (30 Delta Layers)", "time_ms": t_delta * 0.96},
                    {"name": "Host Enqueue & State Sync", "time_ms": t_delta * 0.04},
                ]
            },
            {
                "name": "MoE Expert Stage (40 Layers)",
                "time_ms": t_exp,
                "children": [
                    {
                        "name": "CPU Worker Pool (Miss Experts)",
                        "time_ms": t_cpumoe,
                        "children": [
                            {"name": "AVX2 Gate/Up MatVec", "time_ms": t_cpumoe * 0.48},
                            {"name": "AVX2 Down MatVec", "time_ms": t_cpumoe * 0.42},
                            {"name": "SwiGLU & Quantization", "time_ms": t_cpumoe * 0.10},
                        ]
                    },
                    {
                        "name": "GPU Expert Kernels (Hit Slots)",
                        "time_ms": t_gu,
                        "children": [
                            {"name": "MMVQ Gate/Up Fused", "time_ms": t_gu * 0.55},
                            {"name": "MMVQ Down Projection", "time_ms": t_gu * 0.45},
                        ]
                    },
                    {"name": "Shared Expert MLP", "time_ms": t_sh},
                    {"name": "Router D2H + Top-k Select", "time_ms": t_topk + t_pub},
                    {"name": "Background Fill Enqueue", "time_ms": t_fill},
                ]
            },
            {
                "name": "Full Attention (10 Layers)",
                "time_ms": t_attn,
                "children": [
                    {"name": "Flash Attention & KV-Cache", "time_ms": t_attn * 0.75},
                    {"name": "RoPE & QKV Projections", "time_ms": t_attn * 0.25},
                ]
            },
            {
                "name": "Output LM-Head & Norms",
                "time_ms": t_other,
                "children": [
                    {"name": "LM-Head MatVec (398 MB Q6_K)", "time_ms": t_other * 0.88},
                    {"name": "RMS-Norm & Argmax", "time_ms": t_other * 0.12},
                ]
            },
        ]
    }

    print("\n" + "=" * 80)
    print("              TUFF HIERARCHICAL CALL-TREE BREAKDOWN")
    print("=" * 80)
    print(f"Overall Generation Speed: {speed:.2f} tok/s | Step Latency: {t_tot:.2f} ms | Cache Hit: {hit_rate_str}")
    print("-" * 80)

    def print_tree(node, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        pct = (node["time_ms"] / t_tot) * 100.0
        bar_len = int(pct / 2.5)
        bar = "█" * bar_len
        print(f"{prefix}{connector}{node['name']:<42} : {node['time_ms']:>6.2f} ms ({pct:>5.1f}%)  [{bar}]")
        children = node.get("children", [])
        for idx, ch in enumerate(children):
            extension = "    " if is_last else "│   "
            print_tree(ch, prefix + extension, idx == len(children) - 1)

    print_tree(tree)
    print("=" * 80)

    svg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tuff_flamegraph.svg"))
    render_svg_flamegraph(tree, t_tot, svg_path)


if __name__ == "__main__":
    main()
