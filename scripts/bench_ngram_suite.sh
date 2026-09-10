#!/usr/bin/env bash
# N-Gram Speculative Decoding Configuration Suite & Benchmark Harness

set -uo pipefail
cd "$(dirname "$0")/.."

BIN="./build/tuff"
MODEL="/run/media/nii/pohjola/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"
N="${N:-64}"
WARMUP="${WARMUP:-16}"
RUNS="${RUNS:-2}"

if [ ! -x "$BIN" ]; then
    echo "Error: binary $BIN not found. Build with: cmake --build build -j8" >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: model $MODEL not found." >&2
    exit 1
fi

PROMPTS=(
    "Write a Python function to compute the Fibonacci sequence using dynamic programming with memoization."
    "Explain the difference between synchronous and asynchronous I/O in distributed operating systems."
    "Generate a JSON configuration schema for a high-availability PostgreSQL cluster with replication slots."
    "Write a quicksort algorithm in C with in-place Lomuto partitioning and explain its time complexity."
    "Prove that the square root of 2 is an irrational number using proof by contradiction."
)

CONFIG_NAMES=(
    "Baseline (Non-Speculative)"
    "Fixed 3-gram (N=3)"
    "Fixed 2-gram (N=2)"
    "Hierarchical Fallback (3->2)"
    "Hierarchical Fallback (4->3->2)"
    "Hierarchical Fallback (5->4->3->2)"
    "Hybrid N-gram + MTP"
)

CONFIG_ARGS=(
    ""
    "--spec-draft ngram --spec-ngram 3 --spec-ngram-min 3"
    "--spec-draft ngram --spec-ngram 2 --spec-ngram-min 2"
    "--spec-draft ngram --spec-ngram 3 --spec-ngram-min 2"
    "--spec-draft ngram --spec-ngram 4 --spec-ngram-min 2"
    "--spec-draft ngram --spec-ngram 5 --spec-ngram-min 2"
    "--spec-draft ngram+mtp --spec-ngram 3 --spec-ngram-min 2"
)

echo "========================================================================"
echo " N-Gram Speculative Decoding Configuration Suite"
echo " Binary: $BIN"
echo " Model:  $(basename "$MODEL")"
echo " Runs:   $RUNS runs ($WARMUP warmup + $N timed tokens)"
echo " Prompts: ${#PROMPTS[@]} domain test prompts"
echo " Configs: ${#CONFIG_NAMES[@]} evaluation configurations"
echo "========================================================================"

run_one() {
    local prompt="$1"
    local extra_args="$2"
    local logfile
    logfile=$(mktemp)
    
    # Run Tuff with explicit prompt
    # shellcheck disable=SC2086
    $BIN --model "$MODEL" --prompt "$prompt" -n "$N" --warmup "$WARMUP" --runs "$RUNS" $extra_args > /dev/null 2> "$logfile"
    
    # Extract tok/s
    local speed
    speed=$(grep "Pipeline Generation Speed:" "$logfile" | sed -E 's/.*Pipeline Generation Speed: ([0-9.]+) \+\/-.*/\1/' | head -1)
    if [ -z "$speed" ]; then
        speed="0.00"
    fi
    
    # Extract alpha acceptance rate if speculative
    local alpha
    alpha=$(grep -oE "alpha=[0-9.]+" "$logfile" | head -1 | cut -d'=' -f2)
    if [ -z "$alpha" ]; then
        alpha="N/A"
    else
        alpha="${alpha}%"
    fi
    
    rm -f "$logfile"
    echo "${speed}|${alpha}"
}

echo ""
echo "| Configuration | Python Code | Technical Expl. | JSON Schema | C Quicksort | Math Proof | Average Speed | vs Baseline |"
echo "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"

baseline_avg="0"

for c in "${!CONFIG_NAMES[@]}"; do
    c_name="${CONFIG_NAMES[$c]}"
    c_args="${CONFIG_ARGS[$c]}"
    
    sum="0"
    row_str="| **$c_name** | "
    
    for p in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$p]}"
        res=$(run_one "$prompt" "$c_args")
        spd=$(echo "$res" | cut -d'|' -f1)
        alp=$(echo "$res" | cut -d'|' -f2)
        
        sum=$(awk "BEGIN {print $sum + $spd}")
        
        if [ "$alp" != "N/A" ]; then
            row_str+="$(printf "%.2f (α:%s) | " "$spd" "$alp")"
        else
            row_str+="$(printf "%.2f | " "$spd")"
        fi
    done
    
    count="${#PROMPTS[@]}"
    avg=$(awk "BEGIN {printf \"%.2f\", $sum / $count}")
    
    if [ "$c" -eq 0 ]; then
        baseline_avg="$avg"
        row_str+="**$avg tok/s** | **baseline** |"
    else
        delta=$(awk "BEGIN {printf \"%+.2f%%\", (($avg - $baseline_avg) / $baseline_avg) * 100}")
        row_str+="**$avg tok/s** | **$delta** |"
    fi
    
    echo "$row_str"
done

echo ""
echo "Benchmark suite execution completed."
