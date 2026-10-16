#!/usr/bin/env bash
# tests/lfm25_negative_controls.sh - automated negative controls for the LFM2.5 gates.
#
# Usage: lfm25_negative_controls.sh [--quick] [--only <id>]
#
# For every control in tests/lfm25_negctl.txt (override: NEGCTL_TABLE): inject one realistic bug into a
# scratch COPY of the tree (NEGCTL_DIR, default /tmp/tuff_negctl; the real repo is never modified), rebuild
# the needed targets there, run the listed checks against the mutant binaries and require EACH to FAIL.
# A check that still passes = NOT CAUGHT (a blind spot in the gates). A control whose search string no
# longer matches exactly once is STALE. Exit 1 on any NOT CAUGHT / STALE / build or setup error.
#
# Env: NEGCTL_DIR, NEGCTL_TABLE, PYTHON (stdlib python; default python3), JOBS (default 16).
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

# run_check <check> -> sets CHK_LABEL, CHK_RES (FAIL = the check failed = good; PASS = not caught; ERROR:<why>)
run_check() {
    local chk="$1" out rc
    out="$DIR/check.out"
    case "$chk" in
        golden:*)
            CHK_LABEL="golden-${chk#golden:}"
            TUFF_BIN="$DIR/build/tuff" "$DIR/tests/verify_lfm25_golden.sh" "${chk#golden:}" > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ]; then CHK_RES=PASS
            elif grep -q "NOT comparable" "$out"; then CHK_RES="ERROR:golden not comparable in scratch build (environment mismatch)"
            else CHK_RES=FAIL; fi ;;
        oracle:*)
            CHK_LABEL="oracle-${chk#oracle:}"
            TUFF_BIN="$DIR/build/tuff" REF_BIN="$REF_BIN" "$DIR/tests/verify_lfm25_oracle.sh" "${chk#oracle:}" > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ]; then CHK_RES=PASS
            elif [ $rc -eq 1 ]; then CHK_RES=FAIL
            else CHK_RES="ERROR:oracle exit $rc"; fi ;;
        spec)
            CHK_LABEL="spec-dense"
            TUFF_BIN="$DIR/build/tuff" "$DIR/tests/verify_lfm25_spec.sh" dense > "$out" 2>&1; rc=$?
            if [ $rc -eq 0 ]; then CHK_RES=PASS
            elif grep -q "FAIL(" "$out"; then CHK_RES=FAIL      # a token stream differed
            else CHK_RES="ERROR:spec check exit $rc without a stream difference"; fi ;;
        router8)   CHK_LABEL=router8;   "$DIR/build/lfm25-router-test"  "$M8"  "$DIR/tests/fixtures/lfm25_8b_router_cases.bin"  > "$out" 2>&1; rc=$? ;;
        router24)  CHK_LABEL=router24;  "$DIR/build/lfm25-router-test"  "$M24" "$DIR/tests/fixtures/lfm25_24b_router_cases.bin" > "$out" 2>&1; rc=$? ;;
        moeffn8)   CHK_LABEL=moeffn8;   "$DIR/build/lfm25-moe-ffn-test" "$M8"  "$DIR/tests/fixtures/lfm25_8b_moe_ffn_cases.bin"  > "$out" 2>&1; rc=$? ;;
        moeffn24)  CHK_LABEL=moeffn24;  "$DIR/build/lfm25-moe-ffn-test" "$M24" "$DIR/tests/fixtures/lfm25_24b_moe_ffn_cases.bin" > "$out" 2>&1; rc=$? ;;
        *) CHK_LABEL="$chk"; CHK_RES="ERROR:unknown check"; return ;;
    esac
    case "$chk" in
        router*|moeffn*)
            if [ $rc -eq 0 ]; then CHK_RES=PASS
            elif [ $rc -eq 1 ]; then CHK_RES=FAIL
            else CHK_RES="ERROR:exit $rc"; fi ;;
    esac
}

total=0; caught=0; bad=0

# Files any control touches may have been built from a mutant in an earlier (interrupted) run: refresh
# their mtime so the baseline build below is guaranteed to recompile them from pristine content.
while IFS=$'\t' read -r _ _ f _; do cp "$SRC/$f" "$DIR/$f"; done < <(helper list "$TABLE")

# --- baseline: the unmutated scratch build must reproduce the golden, or "FAIL" below means nothing -----
if ! build tuff; then tail -20 "$DIR/build.log" >&2; echo "[negctl] baseline: scratch build FAILED" ; exit 1; fi
if TUFF_BIN="$DIR/build/tuff" "$DIR/tests/verify_lfm25_golden.sh" dense > "$DIR/baseline.out" 2>&1; then
    echo "[negctl] baseline: unmutated scratch build reproduces the dense golden -> OK"
else
    cat "$DIR/baseline.out"; echo "[negctl] baseline: unmutated scratch build does NOT match the golden -> controls are meaningless"; exit 1
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
    line=""; ok=1
    for chk in $checks; do
        run_check "$chk"
        case "$CHK_RES" in
            FAIL) line+="${line:+, }$CHK_LABEL FAIL" ;;
            PASS) line+="${line:+, }$CHK_LABEL PASS"; ok=0 ;;
            *)    line+="${line:+, }$CHK_LABEL ${CHK_RES}"; ok=0 ;;
        esac
    done
    if [ $ok -eq 1 ]; then echo "[negctl] $id: $desc: $line -> CAUGHT"; caught=$((caught + 1))
    else echo "[negctl] $id: $desc: $line -> NOT CAUGHT"; bad=$((bad + 1)); fi
    restore
done 3< <(helper list "$TABLE")

echo "[negctl] $caught/$total controls caught"
if [ "$total" -eq 0 ]; then echo "[negctl] no controls selected" >&2; exit 1; fi
if [ "$bad" -ne 0 ]; then echo "NEGATIVE CONTROLS FAIL ($bad not caught/stale/error)"; exit 1; fi
echo "NEGATIVE CONTROLS PASS"
exit 0
