#!/usr/bin/env bash
# scripts/lfm25_bench_vs_llama.sh <stage> [args] - fair tuff vs llama.cpp benchmark on the four LFM2.5 models.
# One GPU process at a time (everything here is sequential). Raw logs -> $OUT (default /tmp/claude-1000/bench_final),
# machine-readable rows appended to $OUT/results.tsv: stage \t model \t config \t prompt \t metric \t value
#
# Stages:
#   env                      record power state, nvidia-smi, versions
#   tuff <1.2B|2.6B|8B|24B>  tuff decode: baseline + --spec-draft ngram, -n 128 and 512, --runs 3, all prompts
#   llama <1.2B|2.6B|8B>     llama-bench -ngl 99 -p 0 -n 128,512 -r 3, -fa 0 and 1
#   sweep24                  llama.cpp 24B config sweep (-ot all-CPU, -ncmoe N, manual -ot, thread sweep)
#   gen24 "<llama args>"     llama-completion 24B real-prompt generation (-n 256 --temp 0) over the 4 prompts
#   prefill_llama            llama-bench -p 512 -n 0 per model (24B with $LL24_ARGS)
#   prefill_tuff <model>     tuff wall(512-token prompt) - wall(1-token prompt), -n 1 --warmup 0 --runs 1
# Env: OUT, TUFF_BIN, LLAMA_BIN (dir), LL24_ARGS (llama.cpp args for the best 24B config, used by gen24/prefill_llama).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:-/tmp/claude-1000/bench_final}"; mkdir -p "$OUT"
BIN="${TUFF_BIN:-./build/tuff}"
LB="${LLAMA_BIN:-/home/nii/workspace/llama.cpp/build/bin}"
declare -A MODEL=(
  [1.2B]=/run/media/nii/pohjola/models/unsloth/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q4_K_M.gguf
  [2.6B]=/run/media/nii/pohjola/models/LiquidAI/LFM2.5-2.6B/LFM2.5-2.6B-Q4_K_M.gguf
  [8B]=/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf
  [24B]=/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf
)
# prompts of lfm25_bench_moe.sh (hello = "Hello") and the 5 others of lfm25_bench_spec.sh (24B uses all 9)
MOE_NAMES=(hello code history translate)
MOE_PROMPTS=("Hello"
  "Write a Python function that parses a CSV file and computes per-column statistics, with error handling."
  "Explain the causes of the French Revolution and its consequences for European politics in the 19th century."
  "Translate to German and French: The quick brown fox jumps over the lazy dog near the river bank.")
SPEC_NAMES=(fibonacci io-async json-schema quicksort sqrt2-proof)
SPEC_PROMPTS=(
  "Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization."
  "Explain the difference between synchronous and asynchronous I/O in distributed operating systems."
  "Generate a JSON configuration schema for a high-availability PostgreSQL cluster with replication slots."
  "Write a quicksort algorithm in C with in-place Lomuto partitioning and explain its time complexity."
  "Prove that the square root of 2 is an irrational number using proof by contradiction.")

row() { printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$OUT/results.tsv"; }
smi() { nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,pcie.link.gen.current,pcie.link.width.current,utilization.gpu --format=csv,noheader | tr '\n' ' '; }
cool() { # wait (<=60 s) until GPU temp <= 55 C so runs start from a comparable thermal state; log the sample
    for _ in $(seq 60); do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits); [ "$t" -le 55 ] && break; sleep 1; done
    echo "[smi-before] $(smi)"
}
need_idle() { if pgrep -x tuff >/dev/null || pgrep -x llama-bench >/dev/null || pgrep -x llama-complet >/dev/null; then echo "GPU process already running" >&2; exit 1; fi; }

stage_env() {
    { echo "date: $(date -Is)"; echo "git: $(git rev-parse --short HEAD) $(git status --short | wc -l) dirty files"
      ./scripts/check_power_state.sh; echo; echo "powerprofilesctl: $(powerprofilesctl get 2>&1)"
      echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile)"
      nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit,power.default_limit,pcie.link.gen.max --format=csv
      echo "llama.cpp: $(git -C "$(dirname "$LB")/.." rev-parse --short HEAD 2>/dev/null || git -C /home/nii/workspace/llama.cpp rev-parse --short HEAD)"
      nvcc --version | tail -2; nproc; free -g | head -2; lscpu | grep 'Model name'; } > "$OUT/env.txt" 2>&1
    cat "$OUT/env.txt"
}

tps_of() { sed -n 's/.*Pipeline Generation Speed: \([0-9.]*\) +\/- \([0-9.]*\) .*/\1 \2/p' "$1" | head -1; }

stage_tuff() {
    local m="$1" names prompts; names=("${MOE_NAMES[@]}"); prompts=("${MOE_PROMPTS[@]}")
    if [ "$m" = 24B ]; then names+=("${SPEC_NAMES[@]}"); prompts+=("${SPEC_PROMPTS[@]}"); fi
    for n in 128 512; do for i in "${!names[@]}"; do for mode in base spec; do
        need_idle; local args=(); [ $mode = spec ] && args=(--spec-draft ngram)
        local log="$OUT/tuff_${m}_n${n}_${names[$i]}_${mode}.log"
        { echo "CMD: $BIN --model ${MODEL[$m]} --prompt \"${prompts[$i]}\" -c 1024 -n $n --temp 0 --runs 3 ${args[*]}"; cool; } > "$log"
        LFM25_CACHE_STATS=1 "$BIN" --model "${MODEL[$m]}" --prompt "${prompts[$i]}" -c 1024 -n $n --temp 0 --runs 3 "${args[@]}" >/dev/null 2>> "$log"
        read -r v sd <<< "$(tps_of "$log")"
        echo "tuff $m n=$n ${names[$i]} $mode: ${v:-FAIL} +/- ${sd:-}"
        row tuff "$m" "n$n/$mode" "${names[$i]}" tok_s "${v:-NA}"; row tuff "$m" "n$n/$mode" "${names[$i]}" tok_s_sd "${sd:-NA}"
        if [ $mode = spec ]; then
            acc="$(sed -n 's/.*spec stats:.*acceptance=\([0-9.]*\)%.*/\1/p' "$log" | awk '{s+=$1;c++} END{if(c)printf "%.1f",s/c}')"
            row tuff "$m" "n$n/spec" "${names[$i]}" acceptance_pct "${acc:-NA}"
        fi
    done; done; done
}

# llama-bench CSV -> rows. Columns: ... n_prompt,n_gen,... avg_ts,stddev_ts
parse_bench() { # file stage model config
    python3 - "$@" <<'PY'
import csv,sys
f,stage,model,cfg=sys.argv[1:5]
with open(f) as fh:
    lines=[l for l in fh if l.count(',')>5]
for r in csv.DictReader(lines):
    t=("pp%s"%r["n_prompt"]) if int(r["n_prompt"])>0 else ("tg%s"%r["n_gen"])
    extra="fa=%s,t=%s,ncmoe=%s,ot=%s"%(r.get("flash_attn"),r.get("n_threads"),r.get("n_cpu_moe",""),r.get("tensor_buft_overrides",""))
    print("\t".join([stage,model,cfg,extra,t,r["avg_ts"],r["stddev_ts"]]))
PY
}

lbench() { # stage model cfgname logname -- llama-bench args...
    local stage="$1" m="$2" cfg="$3" logn="$4"; shift 5
    need_idle
    local log="$OUT/${logn}.log"
    { echo "CMD: $LB/llama-bench $*"; cool; } > "$log"
    "$LB/llama-bench" "$@" -o csv > "$OUT/${logn}.csv" 2>> "$log"; local rc=$?
    echo "rc=$rc" >> "$log"
    if [ $rc -ne 0 ]; then echo "llama-bench $logn FAILED rc=$rc"; row "$stage" "$m" "$cfg" - status "FAIL_rc$rc"; return $rc; fi
    parse_bench "$OUT/${logn}.csv" "$stage" "$m" "$cfg" | tee -a "$OUT/results.tsv"
}

stage_llama() {
    local m="$1"
    lbench llama "$m" "ngl99" "llama_${m}_dense" -- -m "${MODEL[$m]}" -ngl 99 -p 0 -n 128,512 -r 3 -fa 0,1
}

stage_sweep24() {
    local M="${MODEL[24B]}" EX='ffn_(up|down|gate)_exps'
    lbench sweep24 24B "ot_all_cpu t=8" sweep_ot_allcpu_t8 -- -m "$M" -ngl 99 -ot "$EX=CPU" -p 0 -n 128 -r 3 -t 8
    for t in 12 16; do lbench sweep24 24B "ot_all_cpu t=$t" sweep_ot_allcpu_t$t -- -m "$M" -ngl 99 -ot "$EX=CPU" -p 0 -n 128 -r 3 -t $t; done
    lbench sweep24 24B "ncmoe_40 t=8" sweep_ncmoe40 -- -m "$M" -ngl 99 -ncmoe 40 -p 0 -n 128 -r 3 -t 8
    for N in ${NCMOE_LIST:-34 32 30 28 27 26 25 24}; do
        lbench sweep24 24B "ncmoe_$N t=8" sweep_ncmoe$N -- -m "$M" -ngl 99 -ncmoe $N -p 0 -n 128 -r 3 -t 8 || break
    done
}

stage_gen24() {
    local extra="$*"
    for i in "${!MOE_NAMES[@]}"; do
        need_idle; local log="$OUT/llama_gen24_${MOE_NAMES[$i]}.log"
        { echo "CMD: $LB/llama-completion -m ${MODEL[24B]} $extra -c 1024 -n 256 --temp 0 -no-cnv -p \"${MOE_PROMPTS[$i]}\""; cool; } > "$log"
        # shellcheck disable=SC2086
        "$LB/llama-completion" -m "${MODEL[24B]}" $extra -c 1024 -n 256 --temp 0 -no-cnv -p "${MOE_PROMPTS[$i]}" >> "$log.out" 2>> "$log"
        v="$(grep -v 'prompt eval' "$log" | sed -n 's/.* eval time = .*(.*, *\([0-9.]*\) tokens per second).*/\1/p' | tail -1)"
        pv="$(sed -n 's/.*prompt eval time = .*, *\([0-9.]*\) tokens per second).*/\1/p' "$log" | tail -1)"
        echo "llama gen24 ${MOE_NAMES[$i]}: eval ${v:-FAIL} tok/s, prompt-eval ${pv:-NA}"
        row gen24 24B "$extra" "${MOE_NAMES[$i]}" eval_tok_s "${v:-NA}"; row gen24 24B "$extra" "${MOE_NAMES[$i]}" prompt_eval_tok_s "${pv:-NA}"
    done
}

stage_prefill_llama() {
    for m in 1.2B 2.6B 8B; do lbench prefill_llama "$m" "ngl99 fa=auto" "prefill_llama_$m" -- -m "${MODEL[$m]}" -ngl 99 -p 512 -n 0 -r 3; done
    # shellcheck disable=SC2086
    lbench prefill_llama 24B "$LL24_ARGS" prefill_llama_24B -- -m "${MODEL[24B]}" $LL24_ARGS -p 512 -n 0 -r 3
}

stage_prefill_tuff() {
    local m="$1" long=""
    while [ ${#long} -lt 2300 ]; do long+="The quick brown fox jumps over the lazy dog near the river bank while the sun sets. "; done
    for r in 1 2 3; do for kind in short long; do
        need_idle; local p="Hello"; [ $kind = long ] && p="$long"
        local log="$OUT/prefill_tuff_${m}_${kind}_$r.log"
        { echo "CMD: $BIN --model ${MODEL[$m]} --prompt <$kind> -c 1024 -n 1 --warmup 0 --runs 1 --temp 0"; cool; } > "$log"
        local t0 t1; t0=$(date +%s.%N)
        "$BIN" --model "${MODEL[$m]}" --prompt "$p" -c 1024 -n 1 --warmup 0 --runs 1 --temp 0 >/dev/null 2>> "$log"
        t1=$(date +%s.%N)
        local w ntok; w=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f",b-a}')
        ntok=$(sed -n 's/.*Prompt tokens (\([0-9]*\)).*/\1/p' "$log" | head -1)
        echo "tuff prefill $m $kind run$r: wall ${w}s prompt_tokens=$ntok"
        row prefill_tuff "$m" "$kind" "run$r" wall_s "$w"; row prefill_tuff "$m" "$kind" "run$r" prompt_tokens "$ntok"
    done; done
}

st="${1:-}"; shift || true
case "$st" in
    env) stage_env;; tuff) stage_tuff "$@";; llama) stage_llama "$@";; sweep24) stage_sweep24;;
    gen24) stage_gen24 "$@";; prefill_llama) stage_prefill_llama;; prefill_tuff) stage_prefill_tuff "$@";;
    *) sed -n 2,16p "$0"; exit 1;;
esac
