#!/usr/bin/env bash
# tests/lfm25_negative_controls.sh - automated negative controls for the LFM2.5 gates.
#
# Usage: lfm25_negative_controls.sh [--quick] [--only <id>]
#
# For every control in tests/lfm25_negctl.txt (override: NEGCTL_TABLE): inject one realistic bug into a
# scratch COPY of the tree (NEGCTL_DIR, default /tmp/tuff_negctl; the real repo is never modified), rebuild
# the needed targets there, run the listed checks against the mutant binaries and require EACH to FAIL.
# Before any mutation, EVERY distinct check used by the selected controls is run once on the unmutated scratch
# build and must PASS (cached per run); otherwise "FAIL" under a mutant could just be a broken setup.
# Check outcomes: PASS; FAIL = a real gate failure recognized by the gate's own verdict line ("GOLDEN FAIL",
# "ORACLE FAIL", "SPEC FAIL", "PREFILL FAIL", "N/M cases pass" with N<M); ERROR = anything else (missing files, tracebacks,
# unexpected exit codes). CAUGHT needs every listed check FAIL. ERROR (also in the baseline) -> "ERROR (setup)",
# exit 1. A check that still passes = NOT CAUGHT (a blind spot). A control whose search string no longer
# matches exactly once is STALE. Exit 1 on any NOT CAUGHT / STALE / build or setup error.
#
# The baseline PASS of each check is cached across invocations (key: scratch-tree content hash + binaries + check; see
# "Baseline cache" below; NEGCTL_NO_CACHE=1 disables) so `--only A` then `--only B` does not redo it.
#
# Env: NEGCTL_DIR, NEGCTL_TABLE, NEGCTL_NO_CACHE, PYTHON (stdlib python; default python3), JOBS (default 16).
set -uo pipefail

cd "$(dirname "$0")/.."
SRC="$PWD"
DIR="${NEGCTL_DIR:-/tmp/tuff_negctl}"
TABLE="${NEGCTL_TABLE:-$SRC/tests/lfm25_negctl.txt}"
JOBS="${JOBS:-16}"
PY="${PYTHON:-python3}"
QUICK=0
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick) QUICK=1 ;;
        --only) shift; ONLY="${1:-}" ;;
        *) echo "Unknown argument: $1" >&2; echo "Usage: $0 [--quick] [--only <id>]" >&2; exit 2 ;;
    esac
    shift
done

# Caller LFM25_* variables must not leak into the checks (the gate scripts scrub too; this covers the unit tests).
SCRUB=(); SCRUBBED=()
while IFS= read -r v; do
    case "$v" in LFM25_*) SCRUB+=(-u "$v"); SCRUBBED+=("$v") ;; esac
done < <(compgen -e)
if [ "${#SCRUBBED[@]}" -gt 0 ]; then
    echo "WARNING: scrubbed caller environment for checks (unset): ${SCRUBBED[*]}" >&2
fi

M8="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf"
M24="/run/media/nii/pohjola/models/LiquidAI/LFM2.5-24B-A2B-GGUF/LFM2.5-24B-A2B-Q4_K_M.gguf"
REF_BIN="$SRC/build/lfm25-llama-ref"
[ -x "$REF_BIN" ] || { echo "Error: $REF_BIN not found (cmake --build build --target lfm25-llama-ref)" >&2; exit 1; }
[ -f "$TABLE" ] || { echo "Error: control table $TABLE missing" >&2; exit 1; }

# --- table helper (stdlib python): list | apply <id> <dst-root> -------------------------------------
helper() { "$PY" -I -c '
import sys
def parse(path):
    cs, cur = [], None
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            if not line.strip() and cur: cs.append(cur); cur = None
            continue
        k, _, v = line.partition(":")
        if v.startswith(" "): v = v[1:]
        if k == "id":
            if cur: cs.append(cur)
            cur = {}
        cur[k] = v
    if cur: cs.append(cur)
    return cs
def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i+1] in "n\\":
            out.append("\n" if s[i+1] == "n" else "\\"); i += 2
        else: out.append(s[i]); i += 1
    return "".join(out)
cmd, table = sys.argv[1], sys.argv[2]
cs = parse(table)
if cmd == "list":
    for c in cs:
        for k in ("id", "tags", "desc", "file", "old", "new", "targets", "checks"):
            c.setdefault(k, "")
        print("\t".join([c["id"], "1" if "quick" in c["tags"].split() else "0", c["file"], c["targets"], c["checks"], c["desc"]]))
elif cmd == "apply":
    cid, srcroot, dstroot = sys.argv[3:6]
    c = next(c for c in cs if c["id"] == cid)
    text = open(srcroot + "/" + c["file"]).read()
    old, new = unesc(c["old"]), unesc(c["new"])
    n = text.count(old)
    if n != 1:
        print("STALE: search string matches %d time(s) in %s (need exactly 1)" % (n, c["file"])); sys.exit(3)
    open(dstroot + "/" + c["file"], "w").write(text.replace(old, new))
' "$@"; }

# --- scratch tree ----------------------------------------------------------------------------------
mkdir -p "$DIR" || exit 1
# --checksum without -t: identical files keep their mtime (no needless rebuild); changed files get a fresh
# mtime (never an older one than the objects built from the previous content).
rsync -rlpgoD --checksum --delete --exclude 'build*' --exclude '.git' "$SRC"/ "$DIR"/ || { echo "Error: rsync failed" >&2; exit 1; }
if [ ! -f "$DIR/build/CMakeCache.txt" ]; then
    cmake -S "$DIR" -B "$DIR/build" -DCMAKE_CUDA_ARCHITECTURES=86 > "$DIR/cmake_configure.log" 2>&1 \
        || { tail -20 "$DIR/cmake_configure.log" >&2; echo "Error: cmake configure failed" >&2; exit 1; }
fi
FILES_TOUCHED=()
restore() {  # restore pristine copies of every file a control may have touched
    local f
    for f in ${FILES_TOUCHED[@]+"${FILES_TOUCHED[@]}"}; do cp "$SRC/$f" "$DIR/$f"; done
}
trap restore EXIT

build() {  # <targets...>
    cmake --build "$DIR/build" -j"$JOBS" --target "$@" > "$DIR/build.log" 2>&1
}

# unit_verdict <out-file> <rc> <summary-regex-prefix>: unit tests exit 0 = pass, 1 = failed cases, anything else = error.
# FAIL only if rc==1 AND the summary line is present and shows a real failure (N<M or accum FAIL).
unit_verdict() {
    local out="$1" rc="$2" line n m
    if [ "$rc" -eq 0 ]; then CHK_RES=PASS; return; fi
    line="$(grep -E '^(router|moe_ffn|cpu_expert): [0-9]+/[0-9]+ cases pass' "$out" | tail -1)"
    if [ "$rc" -eq 1 ] && [ -n "$line" ]; then
        n="$(sed -E 's/^[a-z_]+: ([0-9]+)\/([0-9]+) .*/\1/' <<< "$line")"
        m="$(sed -E 's/^[a-z_]+: ([0-9]+)\/([0-9]+) .*/\2/' <<< "$line")"
        if [ "$n" -lt "$m" ] || grep -q 'accum FAIL' <<< "$line"; then CHK_RES=FAIL; return; fi
    fi
    CHK_RES="ERROR:exit $rc, no failing 'N/M cases pass' verdict"
}

# run_check <check> -> sets CHK_LABEL, CHK_RES (PASS | FAIL = real gate failure | ERROR:<why>)
run_check() {
    local chk="$1" out rc
    out="$DIR/check.out"
    case "$chk" in
        golden:*)
            CHK_LABEL="golden-${chk#golden:}"
            TUFF_BIN="$DIR/build/tuff" "$DIR/tests/verify_lfm25_golden.sh" "${chk#golden:}" > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ] && grep -q "^GOLDEN PASS" "$out"; then CHK_RES=PASS
            elif grep -q "NOT comparable\|golden missing\|tuff run incomplete\|GOLDEN ERROR" "$out"; then CHK_RES="ERROR:golden check could not run cleanly (exit $rc)"
            elif [ $rc -eq 1 ] && grep -q "^GOLDEN FAIL" "$out"; then CHK_RES=FAIL
            else CHK_RES="ERROR:golden exit $rc without a verdict"; fi ;;
        oracle:*)
            CHK_LABEL="oracle-${chk#oracle:}"
            TUFF_BIN="$DIR/build/tuff" REF_BIN="$REF_BIN" "$DIR/tests/verify_lfm25_oracle.sh" "${chk#oracle:}" > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ] && grep -q "ORACLE PASS" "$out"; then CHK_RES=PASS
            elif [ $rc -eq 1 ] && grep -q "ORACLE FAIL" "$out"; then CHK_RES=FAIL
            else CHK_RES="ERROR:oracle exit $rc without a verdict"; fi ;;
        spec)
            CHK_LABEL="spec-dense"
            TUFF_BIN="$DIR/build/tuff" "$DIR/tests/verify_lfm25_spec.sh" dense > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ] && grep -q "^SPEC PASS" "$out"; then CHK_RES=PASS
            elif [ $rc -eq 1 ] && grep -q "^SPEC FAIL" "$out"; then CHK_RES=FAIL
            else CHK_RES="ERROR:spec exit $rc without a verdict"; fi ;;
        prefill)
            CHK_LABEL="prefill-dense"
            TUFF_BIN="$DIR/build/tuff" "$DIR/tests/verify_lfm25_prefill.sh" dense > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ] && grep -q "^PREFILL PASS" "$out"; then CHK_RES=PASS
            elif [ $rc -eq 1 ] && grep -q "^PREFILL FAIL" "$out"; then CHK_RES=FAIL
            else CHK_RES="ERROR:prefill exit $rc without a verdict"; fi ;;
        router8)   CHK_LABEL=router8;   env ${SCRUB[@]+"${SCRUB[@]}"} "$DIR/build/lfm25-router-test"  "$M8"  "$DIR/tests/fixtures/lfm25_8b_router_cases.bin"  > "$out" 2>&1; rc=$?; unit_verdict "$out" $rc ;;
        router24)  CHK_LABEL=router24;  env ${SCRUB[@]+"${SCRUB[@]}"} "$DIR/build/lfm25-router-test"  "$M24" "$DIR/tests/fixtures/lfm25_24b_router_cases.bin" > "$out" 2>&1; rc=$?; unit_verdict "$out" $rc ;;
        moeffn8)   CHK_LABEL=moeffn8;   env ${SCRUB[@]+"${SCRUB[@]}"} "$DIR/build/lfm25-moe-ffn-test" "$M8"  "$DIR/tests/fixtures/lfm25_8b_moe_ffn_cases.bin"  > "$out" 2>&1; rc=$?; unit_verdict "$out" $rc ;;
        moeffn24)  CHK_LABEL=moeffn24;  env ${SCRUB[@]+"${SCRUB[@]}"} "$DIR/build/lfm25-moe-ffn-test" "$M24" "$DIR/tests/fixtures/lfm25_24b_moe_ffn_cases.bin" > "$out" 2>&1; rc=$?; unit_verdict "$out" $rc ;;
        cpuexp8)   CHK_LABEL=cpuexp8;   env ${SCRUB[@]+"${SCRUB[@]}"} "$DIR/build/lfm25-cpu-expert-test" "$M8"  1 > "$out" 2>&1; rc=$?; unit_verdict "$out" $rc ;;
        cpuexp24)  CHK_LABEL=cpuexp24;  env ${SCRUB[@]+"${SCRUB[@]}"} "$DIR/build/lfm25-cpu-expert-test" "$M24" 1 > "$out" 2>&1; rc=$?; unit_verdict "$out" $rc ;;
        *) CHK_LABEL="$chk"; CHK_RES="ERROR:unknown check" ;;
    esac
}

total=0; caught=0; bad=0

# Files any control touches may have been built from a mutant in an earlier (interrupted) run: refresh
# their mtime so the baseline build below is guaranteed to recompile them from pristine content.
while IFS=$'\t' read -r _ _ f _; do cp "$SRC/$f" "$DIR/$f"; done < <(helper list "$TABLE")

# --- select controls, then baseline every distinct check on the unmutated scratch build ---------------------
SEL_CHECKS=(); SEL_TARGETS="tuff"
while IFS=$'\t' read -r id quick _ targets checks _; do
    [ -n "$id" ] || continue
    [ -z "$ONLY" ] || [ "$id" = "$ONLY" ] || continue
    [ "$QUICK" -eq 0 ] || [ "$quick" = "1" ] || continue
    for t in $targets; do [[ " $SEL_TARGETS " == *" $t "* ]] || SEL_TARGETS+=" $t"; done
    for c in $checks; do [[ " ${SEL_CHECKS[*]-} " == *" $c "* ]] || SEL_CHECKS+=("$c"); done
done < <(helper list "$TABLE")

# shellcheck disable=SC2086
if ! build $SEL_TARGETS; then tail -20 "$DIR/build.log" >&2; echo "[negctl] baseline: scratch build FAILED"; exit 1; fi
declare -A BASE
setup_bad=0
# Baseline cache: a PASS of a check on the unmutated scratch build is remembered in $DIR/build/negctl_baseline/<key>, where
# key = sha256(content hash of the scratch tree sources [everything except build*/logs] + sha256 of the freshly built binaries
# + the check id + the model files' size/mtime). A later invocation (e.g. `--only A` then `--only B`) with the same tree and
# binaries skips that baseline run. Only PASS is cached (a failing baseline is always re-run); NEGCTL_NO_CACHE=1 disables it.
BCACHE="$DIR/build/negctl_baseline"
mkdir -p "$BCACHE"
TREE_HASH="$(cd "$DIR" && find . \( -name 'build*' -o -name check.out -o -name build.log -o -name cmake_configure.log \) -prune -o -type f -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
BIN_HASH="$( (for t in $SEL_TARGETS; do [ -f "$DIR/build/$t" ] && sha256sum "$DIR/build/$t" | cut -d' ' -f1; done) | sha256sum | cut -d' ' -f1)"
MODEL_ID="$(stat -c '%n:%s:%Y' "$M8" "$M24" "$REF_BIN" 2>/dev/null | sha256sum | cut -d' ' -f1)"
for chk in ${SEL_CHECKS[@]+"${SEL_CHECKS[@]}"}; do
    ckey="$(printf '%s %s %s %s' "$TREE_HASH" "$BIN_HASH" "$MODEL_ID" "$chk" | sha256sum | cut -d' ' -f1)"
    if [ -z "${NEGCTL_NO_CACHE:-}" ] && [ -f "$BCACHE/$ckey" ]; then
        BASE[$chk]=PASS
        echo "[negctl] baseline: $(cat "$BCACHE/$ckey") on the unmutated scratch build -> PASS (cached: same tree + binaries, skipped)"
        continue
    fi
    run_check "$chk"
    BASE[$chk]="$CHK_RES"
    if [ "$CHK_RES" = PASS ]; then
        echo "$CHK_LABEL" > "$BCACHE/$ckey"
        echo "[negctl] baseline: $CHK_LABEL on the unmutated scratch build -> PASS"
    else
        tail -15 "$DIR/check.out"
        echo "[negctl] baseline: $CHK_LABEL on the unmutated scratch build -> ${CHK_RES} -> ERROR (setup)"
        setup_bad=1
    fi
done
if [ "$setup_bad" -ne 0 ]; then
    echo "[negctl] ERROR (setup): a check does not PASS on the unmutated build, so 'FAIL' under a mutant would be meaningless"
    echo "NEGATIVE CONTROLS FAIL (ERROR (setup))"; exit 1
fi

while IFS=$'\t' read -r id quick file targets checks desc <&3; do
    [ -n "$id" ] || continue
    [ -z "$ONLY" ] || [ "$id" = "$ONLY" ] || continue
    [ "$QUICK" -eq 0 ] || [ "$quick" = "1" ] || continue
    total=$((total + 1))
    FILES_TOUCHED+=("$file"); restore
    msg="$(helper apply "$TABLE" "$id" "$SRC" "$DIR")"; arc=$?
    if [ $arc -ne 0 ]; then
        echo "[negctl] $id: $desc: $msg -> STALE"; bad=$((bad + 1)); continue
    fi
    # shellcheck disable=SC2086
    if ! build $targets; then
        tail -15 "$DIR/build.log" >&2
        echo "[negctl] $id: $desc: mutant does not build -> BUILD ERROR"; bad=$((bad + 1)); restore; continue
    fi
    line=""; ok=1; errd=0
    for chk in $checks; do
        run_check "$chk"
        case "$CHK_RES" in
            FAIL) line+="${line:+, }$CHK_LABEL FAIL" ;;
            PASS) line+="${line:+, }$CHK_LABEL PASS"; ok=0 ;;
            *)    line+="${line:+, }$CHK_LABEL ${CHK_RES}"; ok=0; errd=1 ;;
        esac
    done
    if [ $ok -eq 1 ]; then echo "[negctl] $id: $desc: $line -> CAUGHT"; caught=$((caught + 1))
    elif [ $errd -eq 1 ]; then echo "[negctl] $id: $desc: $line -> ERROR (setup)"; bad=$((bad + 1))
    else echo "[negctl] $id: $desc: $line -> NOT CAUGHT"; bad=$((bad + 1)); fi
    restore
done 3< <(helper list "$TABLE")

echo "[negctl] $caught/$total controls caught"
if [ "$total" -eq 0 ]; then echo "[negctl] no controls selected" >&2; exit 1; fi
if [ "$bad" -ne 0 ]; then echo "NEGATIVE CONTROLS FAIL ($bad not caught/stale/error)"; exit 1; fi
echo "NEGATIVE CONTROLS PASS"
exit 0
