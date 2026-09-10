#!/usr/bin/env python3
import json
import subprocess
import time
import os
import sys
import re
import math
import argparse
import random

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts_corpus.json")
ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "tuff")
MODEL_PATH = os.environ.get("MODEL_PATH", "/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf")

# PLAN.md measurement protocol: >=3 runs, 32 warmup + 128 timed tokens.
N_PREDICT = int(os.environ.get("TUFF_N", "128"))
N_WARMUP = int(os.environ.get("TUFF_WARMUP", "32"))
N_RUNS = int(os.environ.get("TUFF_RUNS", "3"))


def run_single(prompt_text, env_overrides=None, extra_cli_args=None):
    engine_bin = ENGINE_PATH
    if env_overrides and "BIN" in env_overrides:
        engine_bin = env_overrides["BIN"]
    cmd = [
        engine_bin,
        "--model", MODEL_PATH,
        "--prompt", prompt_text,
        "-n", str(N_PREDICT),
        "--warmup", str(N_WARMUP),
        "--runs", str(N_RUNS),
    ]
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    run_env = os.environ.copy()
    if env_overrides:
        for k, v in env_overrides.items():
            run_env[k] = str(v)

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=run_env)
    t1 = time.time()

    if res.returncode != 0:
        print(f"[ENGINE FAILURE] Process exited with code {res.returncode}")
        print("Stderr:")
        print(res.stderr)
        sys.exit(1)

    out = res.stdout + "\n" + res.stderr
    speed = None
    speed_spread = None
    rss_mib = None
    vram = None
    hit_rate = None

    m = re.search(r"Pipeline Generation Speed:\s*([\d.]+)\s*\+\/-\s*([\d.]+)\s*tok/s", out)
    if m:
        speed = float(m.group(1))
        speed_spread = float(m.group(2))
    m = re.search(r"RSS:\s*([\d.]+)\s*MiB", out)
    if m:
        rss_mib = float(m.group(1))
    m = re.search(r"VRAM:\s*([^\n]+)", out)
    if m:
        vram = m.group(1).strip()
    m = re.search(r"Expert-cache hit rate:\s*([\d.]+%)\s*\(([0-9]+)\s*hits\s*/\s*([0-9]+)\s*misses\)", out)
    if m:
        hit_rate = f"{m.group(1)} ({m.group(2)} / {m.group(3)})"
    else:
        m = re.search(r"Expert-cache hit rate:\s*([^\n]+)", out)
        if m:
            hit_rate = m.group(1).strip()

    if speed is None:
        print(f"[PARSE FAILURE] Could not parse 'Pipeline Generation Speed:'!")
        print("Raw output:")
        print(out)
        sys.exit(1)

    return {
        "speed_tok_s": speed,
        "speed_spread": speed_spread,
        "rss_mib": rss_mib,
        "vram": vram,
        "hit_rate": hit_rate,
        "elapsed_sec": round(t1 - t0, 3),
    }


def parse_config_spec(spec, default_name=None):
    # Formats: "Name:ENV1=VAL1 ENV2=VAL2" or "ENV1=VAL1 ENV2=VAL2" or "--spec-draft=mtp"
    if ":" in spec:
        name, env_part = spec.split(":", 1)
        name = name.strip()
    else:
        name = default_name or spec.strip() or "Baseline"
        env_part = spec

    env_dict = {}
    cli_args = []
    for token in env_part.split():
        if token.startswith("--") or token.startswith("-"):
            cli_args.append(token)
        elif "=" in token:
            k, v = token.split("=", 1)
            if k == "TUFF_ARGS":
                cli_args.extend(v.split())
            else:
                env_dict[k] = v
        elif token.lower() in ("base", "none", "baseline", "default"):
            pass
        else:
            env_dict[token] = "1"
    return {"name": name, "env": env_dict, "cli": cli_args, "raw": spec}


def compute_ci(ratios):
    n = len(ratios)
    if n < 1:
        return {}
    mean_val = sum(ratios) / n
    variance = sum((x - mean_val) ** 2 for x in ratios) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(variance)
    sem = sd / math.sqrt(n) if n > 0 else 0.0

    geomean = math.exp(sum(math.log(max(1e-6, x)) for x in ratios) / n)
    log_var = sum((math.log(max(1e-6, x)) - math.log(geomean)) ** 2 for x in ratios) / (n - 1) if n > 1 else 0.0
    log_sem = math.sqrt(log_var / n) if n > 0 else 0.0

    t_table_95 = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042
    }
    df = max(1, n - 1)
    t_crit = t_table_95.get(df, 1.96 if df > 30 else 2.0)

    ci_lower = mean_val - t_crit * sem
    ci_upper = mean_val + t_crit * sem

    geo_ci_lower = math.exp(math.log(geomean) - t_crit * log_sem)
    geo_ci_upper = math.exp(math.log(geomean) + t_crit * log_sem)

    return {
        "n": n,
        "df": df,
        "mean": mean_val,
        "sd": sd,
        "sem": sem,
        "geomean": geomean,
        "t_crit": t_crit,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "geo_ci_lower": geo_ci_lower,
        "geo_ci_upper": geo_ci_upper,
    }


def main():
    global N_PREDICT, N_WARMUP, N_RUNS
    parser = argparse.ArgumentParser(description="Tuff Multi-Domain Multi-Hypothesis Benchmark Runner")
    parser.add_argument("--configs", nargs="+", metavar="CONFIG",
                        help="List of hypothesis configs to benchmark (e.g. --configs 'Baseline:base' 'MTP:--spec-draft=mtp' 'Ngram:--spec-draft=ngram')")
    parser.add_argument("--ab", nargs=2, metavar=("ARM_A", "ARM_B"),
                        help="Run A/B test between two configs (e.g. --ab 'base' '--spec-draft=ngram')")
    parser.add_argument("--flag", type=str,
                        help="Run A/B comparing baseline vs candidate flag (e.g. --flag TUFF_LFRU_GUARD=1)")
    parser.add_argument("--mode", choices=["randomized", "interleaved", "sequential"], default="randomized",
                        help="Benchmark execution mode: 'randomized' (shuffles all config-domain pairs each pass), 'interleaved' (per-prompt alternating order), or 'sequential'")
    parser.add_argument("--passes", type=int, default=2,
                        help="Number of full random passes across the entire set of configs and domains (default: 2)")
    parser.add_argument("-n", type=int, default=N_PREDICT, help=f"Timed tokens per prompt (default: {N_PREDICT})")
    parser.add_argument("--warmup", type=int, default=N_WARMUP, help=f"Warmup tokens per prompt (default: {N_WARMUP})")
    parser.add_argument("--runs", type=int, default=N_RUNS, help=f"Repetitions per prompt within a single run (default: {N_RUNS})")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Seconds to sleep between benchmark runs for thermal dissipation (default: 2.0)")
    args = parser.parse_args()

    N_PREDICT = args.n
    N_WARMUP = args.warmup
    N_RUNS = args.runs
    n_passes = args.passes
    cooldown_s = args.cooldown

    if not os.path.exists(CORPUS_PATH):
        print(f"Error: Corpus file not found at {CORPUS_PATH}")
        sys.exit(1)

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    raw_configs = []
    if args.configs:
        for idx, c in enumerate(args.configs):
            raw_configs.append(parse_config_spec(c, default_name=f"Config {idx+1}"))
    elif args.ab:
        raw_configs.append(parse_config_spec(args.ab[0], default_name="Arm A (Baseline)"))
        raw_configs.append(parse_config_spec(args.ab[1], default_name="Arm B (Candidate)"))
    elif args.flag:
        flag_key = args.flag.split("=")[0] if "=" in args.flag else args.flag
        raw_configs.append(parse_config_spec(f"{flag_key}=0", default_name="Baseline"))
        raw_configs.append(parse_config_spec(args.flag if "=" in args.flag else f"{flag_key}=1", default_name="Candidate"))
    else:
        extra_cli = os.environ.get("TUFF_ARGS", "")
        raw_configs.append(parse_config_spec(f"Baseline:{extra_cli}", default_name="Default"))

    mode = args.mode

    print("==================================================================")
    print(f"  Tuff Multi-Hypothesis Benchmark Runner ({len(raw_configs)} Configurations)")
    print(f"  Execution Mode: {mode.upper()} ({n_passes} full passes, {len(prompts)} domains)")
    print("==================================================================")
    for idx, c in enumerate(raw_configs):
        print(f"  [{idx+1}] {c['name']:<24} -> CLI: {c['cli']} | Env: {c['env']}")
    print(f"Model: {MODEL_PATH}")
    print(f"Protocol: {N_WARMUP} warmup + {N_PREDICT} timed tokens, {N_RUNS} runs per execution, {n_passes} random passes\n")

    # domain_id -> config_idx -> list of runs across passes
    domain_runs = {p["id"]: [[] for _ in raw_configs] for p in prompts}

    if mode == "randomized":
        total_executions = n_passes * len(raw_configs) * len(prompts)
        exec_count = 0
        for pass_idx in range(n_passes):
            print(f"\n==================================================================")
            print(f"  STARTING RANDOMIZED PASS [{pass_idx+1}/{n_passes}]")
            print(f"==================================================================")
            pairs = []
            for p_idx, item in enumerate(prompts):
                for c_idx, cfg in enumerate(raw_configs):
                    pairs.append((p_idx, c_idx))
            random.shuffle(pairs)

            for p_idx, c_idx in pairs:
                exec_count += 1
                item = prompts[p_idx]
                cfg = raw_configs[c_idx]
                domain = item["domain"]
                title = item["title"]
                prompt_text = item["prompt"]

                print(f"[{exec_count}/{total_executions}] [Pass {pass_idx+1}] [{c_idx+1}] '{domain}' -", end="", flush=True)
                res = run_single(prompt_text, env_overrides=cfg["env"], extra_cli_args=cfg["cli"])
                spr = f"±{res['speed_spread']:.2f}" if res['speed_spread'] is not None else ""
                print(f" {res['speed_tok_s']:.2f} {spr} tok/s | Hit: {res['hit_rate']}")
                domain_runs[item["id"]][c_idx].append(res)
                if cooldown_s > 0:
                    time.sleep(cooldown_s)

    elif mode == "sequential":
        for c_idx, cfg in enumerate(raw_configs):
            print(f"\n==================================================================")
            print(f"  RUNNING CONFIG [{c_idx+1}/{len(raw_configs)}]: {cfg['name']}")
            print(f"==================================================================")
            for pass_idx in range(n_passes):
                for p_idx, item in enumerate(prompts):
                    domain = item["domain"]
                    title = item["title"]
                    prompt_text = item["prompt"]

                    print(f"[{p_idx+1}/{len(prompts)}] [Pass {pass_idx+1}] [{c_idx+1}] '{domain}' -", end="", flush=True)
                    res = run_single(prompt_text, env_overrides=cfg["env"], extra_cli_args=cfg["cli"])
                    spr = f"±{res['speed_spread']:.2f}" if res['speed_spread'] is not None else ""
                    print(f" {res['speed_tok_s']:.2f} {spr} tok/s | Hit: {res['hit_rate']}")
                    domain_runs[item["id"]][c_idx].append(res)

    else: # interleaved
        for p_idx, item in enumerate(prompts):
            domain = item["domain"]
            title = item["title"]
            prompt_text = item["prompt"]
            print(f"\n[{p_idx+1}/{len(prompts)}] [Domain: '{domain}'] {title}")

            for pass_idx in range(n_passes):
                perm = list(range(len(raw_configs)))
                if (p_idx + pass_idx) % 2 == 1 and len(raw_configs) == 2:
                    perm = [1, 0]

                for c_idx in perm:
                    cfg = raw_configs[c_idx]
                    print(f"  -> [Pass {pass_idx+1}] [{c_idx+1}] '{domain}' -", end="", flush=True)
                    res = run_single(prompt_text, env_overrides=cfg["env"], extra_cli_args=cfg["cli"])
                    spr = f"±{res['speed_spread']:.2f}" if res['speed_spread'] is not None else ""
                    print(f" {res['speed_tok_s']:.2f} {spr} tok/s | Hit: {res['hit_rate']}")
                    domain_runs[item["id"]][c_idx].append(res)

    # Summary Report Table
    print("\n\n=================== HONEST MULTI-DOMAIN BENCHMARK REPORT ===================")
    header_cols = ["Domain", "Prompt Title"] + [c["name"] for c in raw_configs]
    if len(raw_configs) == 2:
        header_cols.append("Ratio (B/A)")

    print("| " + " | ".join(header_cols) + " |")
    print("| " + " | ".join([":---" if i < 2 else ":---:" for i in range(len(header_cols))]) + " |")

    config_domain_speeds = [[] for _ in raw_configs]

    for item in prompts:
        pid = item["id"]
        row = [f"**{item['domain']}**", item["title"]]
        runs_per_cfg = domain_runs[pid]

        cell_means = []
        for c_idx, r_list in enumerate(runs_per_cfg):
            if r_list:
                speeds = [r["speed_tok_s"] for r in r_list]
                m_speed = sum(speeds) / len(speeds)
                cell_means.append(m_speed)
                config_domain_speeds[c_idx].append(m_speed)
                if len(speeds) > 1:
                    variance = sum((s - m_speed) ** 2 for s in speeds) / (len(speeds) - 1)
                    s_spread = math.sqrt(variance)
                    row.append(f"{m_speed:.2f} ±{s_spread:.2f}")
                else:
                    spr = f"±{r_list[0]['speed_spread']:.2f}" if r_list[0]['speed_spread'] is not None else ""
                    row.append(f"{m_speed:.2f} {spr}")
            else:
                cell_means.append(0.0)
                row.append("—")

        if len(raw_configs) == 2 and cell_means[0] > 0 and cell_means[1] > 0:
            rat = cell_means[1] / cell_means[0]
            diff_p = (rat - 1.0) * 100.0
            sign = "+" if diff_p >= 0 else ""
            row.append(f"**{sign}{diff_p:.2f}%** ({rat:.4f})")

        print("| " + " | ".join(row) + " |")

    # Geomean row
    geo_row = ["**OVERALL GEOMEAN**", f"**All {len(prompts)} Domains**"]
    geomeans = []
    for c_idx in range(len(raw_configs)):
        speeds = config_domain_speeds[c_idx]
        if speeds:
            gm = math.exp(sum(math.log(max(1e-6, s)) for s in speeds) / len(speeds))
            geomeans.append(gm)
            geo_row.append(f"**{gm:.2f} tok/s**")
        else:
            geo_row.append("—")

    if len(raw_configs) == 2 and len(geomeans) == 2:
        overall_rat = geomeans[1] / geomeans[0]
        overall_diff = (overall_rat - 1.0) * 100.0
        sign = "+" if overall_diff >= 0 else ""
        geo_row.append(f"**{sign}{overall_diff:.2f}%** ({overall_rat:.4f})")

    print("| " + " | ".join(geo_row) + " |")
    print("=============================================================================")

    # Statistical CI analysis when comparing 2 or more configs against Baseline (Config 1)
    if len(raw_configs) >= 2:
        baseline_speeds = config_domain_speeds[0]
        for c_idx in range(1, len(raw_configs)):
            cand_name = raw_configs[c_idx]["name"]
            base_name = raw_configs[0]["name"]
            cand_speeds = config_domain_speeds[c_idx]

            pair_ratios = [c / b for b, c in zip(baseline_speeds, cand_speeds) if b > 0]
            st = compute_ci(pair_ratios)
            if not st:
                continue

            diff_pct = (st["geomean"] - 1.0) * 100.0
            sign = "+" if diff_pct >= 0 else ""
            geo_lo = (st["geo_ci_lower"] - 1.0) * 100.0
            geo_hi = (st["geo_ci_upper"] - 1.0) * 100.0

            print(f"\n--- STATISTICAL ANALYSIS: [{cand_name}] vs [{base_name}] (df={st['df']}) ---")
            print(f"  • Geometric Mean Ratio:   {st['geomean']:.4f}  ({sign}{diff_pct:.2f}%)")
            print(f"  • Arithmetic Mean Ratio:  {st['mean']:.4f} ± {st['sd']:.4f}  ({(st['mean']-1)*100:+.2f}% ± {st['sd']*100:.2f}%)")
            print(f"  • Standard Error (SEM):   ±{st['sem']:.4f}  (±{st['sem']*100:.2f}%)")
            print(f"  • 95% Confidence Interval (Student's t = {st['t_crit']:.3f}):")
            print(f"      [{(st['ci_lower']-1)*100:+.2f}%, {(st['ci_upper']-1)*100:+.2f}%]  (Ratio: [{st['ci_lower']:.4f}, {st['ci_upper']:.4f}])")
            print(f"      (Log-Scale CI: [{geo_lo:+.2f}%, {geo_hi:+.2f}%] = [{st['geo_ci_lower']:.4f}, {st['geo_ci_upper']:.4f}])")

            if st["geo_ci_lower"] > 1.0:
                print(f"  ✅ VERDICT: [{cand_name}] is STATISTICALLY SIGNIFICANTLY FASTER than [{base_name}] at 95% confidence (p < 0.05).")
            elif st["geo_ci_upper"] < 1.0:
                print(f"  ❌ VERDICT: [{cand_name}] is STATISTICALLY SIGNIFICANTLY SLOWER than [{base_name}] at 95% confidence (p < 0.05).")
            else:
                print(f"  ⚠️  VERDICT: Difference ({sign}{diff_pct:.2f}%) is NOT statistically distinguishable from noise at 95% confidence (interval includes 0.00%).")


if __name__ == "__main__":
    main()
