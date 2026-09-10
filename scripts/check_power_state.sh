#!/usr/bin/env bash
# Guard against benchmarking on a throttled machine.
#
# Why this exists: on 2026-08-14 a benchmark run reproduced 8.09 tok/s on the command that
# ITERATION_LOG.md records at 21.90 tok/s. The code was identical. The laptop was on battery:
# GPU power limit 30 W instead of 80 W, VRAM clock 405 MHz instead of 6001 MHz, CPU governor
# `powersave`. That is a 2.7x confound, and worse, it is not a uniform slowdown — VRAM
# bandwidth falls ~15x while host latency barely moves, so the *shape* of the profile inverts
# and the optimization ranking derived from it would be wrong.
#
# Any tok/s number recorded in ITERATION_LOG.md / HONEST_BENCHMARK_REPORT.md must come from a
# run where this script exited 0.
#
# Exit codes: 0 = safe to benchmark, 1 = throttled (numbers not admissible).

set -uo pipefail
fail=0
warn() { echo "  [FAIL] $*"; fail=1; }
ok()   { echo "  [ ok ] $*"; }

echo "== power state check =="

# --- AC adapter -------------------------------------------------------------
ac=unknown
ac_ok=0
for p in /sys/class/power_supply/*/; do
    [ "$(cat "$p/type" 2>/dev/null)" = "Mains" ] || continue
    ac=$(cat "$p/online" 2>/dev/null || echo unknown)
done
if [ "$ac" = "1" ]; then ok "AC adapter connected"; ac_ok=1
else warn "on battery (AC online=$ac) — plug in the charger"; fi

# --- GPU power limit --------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    cur=$(nvidia-smi --query-gpu=power.limit     --format=csv,noheader,nounits 2>/dev/null | head -1)
    def=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null | head -1)
    # On this driver the CSV query reports power.limit as "[N/A]" while `-q -d POWER` has the
    # real number. Fall back to parsing that rather than silently skipping the check.
    case "$cur" in ''|*N/A*)
        cur=$(nvidia-smi -q -d POWER 2>/dev/null | awk -F: '/Current Power Limit/{gsub(/ |W/,"",$2); print $2; exit}') ;;
    esac
    case "$def" in ''|*N/A*)
        def=$(nvidia-smi -q -d POWER 2>/dev/null | awk -F: '/Default Power Limit/{gsub(/ |W/,"",$2); print $2; exit}') ;;
    esac
    if [ -n "${cur:-}" ] && [ -n "${def:-}" ] && [ "$cur" != "N/A" ] && [ "$def" != "N/A" ]; then
        # awk, not bash arithmetic: these are floats ("30.00").
        if awk -v c="$cur" -v d="$def" 'BEGIN{exit !(c >= d - 0.5)}'; then
            ok "GPU power limit ${cur} W (default ${def} W)"
            plim_ok=1
        else
            warn "GPU power limit ${cur} W is below the ${def} W default — GPU is capped"
        fi
    fi

    # VRAM clock is the tell that matters most: the expert and delta-net phases are
    # bandwidth-bound, and a memory downclock destroys exactly the numbers this repo tracks.
    mcur=$(nvidia-smi --query-gpu=clocks.mem     --format=csv,noheader,nounits 2>/dev/null | head -1)
    mmax=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${mcur:-}" ] && [ -n "${mmax:-}" ]; then
        # Idle downclocking is normal, so only flag a *ceiling* that is below spec. A cold
        # idle GPU legitimately sits at its lowest P-state; what is not legitimate is a run
        # that cannot reach the max clock at all. We report both and let the caller judge,
        # but treat < 25% of max as a hard fail since that is the battery signature.
        if awk -v c="$mcur" -v m="$mmax" 'BEGIN{exit !(c < m * 0.25)}'; then
            # An idle GPU and a battery-capped GPU read identically at rest — this card sits at
            # 405 MHz idle on AC and at 405 MHz on battery. So a low reading is only decisive
            # when something else corroborates it. On AC with an uncapped power limit, treat it
            # as the idle P-state it almost certainly is; a guard that cries wolf here just
            # teaches everyone to pass FORCE=1.
            if [ "${ac_ok:-0}" = "1" ] && [ "${plim_ok:-0}" = "1" ]; then
                echo "  [ ?? ] VRAM clock ${mcur} MHz of ${mmax} MHz — GPU is idle; AC and power"
                echo "         limit are both nominal, so this should boost once the run starts."
                echo "         Confirm during a run: nvidia-smi --query-gpu=clocks.mem --format=csv -l 1"
            else
                warn "VRAM clock ${mcur} MHz vs ${mmax} MHz max — throttled (AC/power limit also bad)"
            fi
        else
            ok "VRAM clock ${mcur} MHz (max ${mmax} MHz)"
        fi
    fi
else
    warn "nvidia-smi not found"
fi

# --- CPU governor -----------------------------------------------------------
# Careful: the governor *name* is not decisive on this machine. The driver here is
# amd-pstate-epp, whose "powersave" governor still boosts on demand — measured 4.29 GHz
# against a 4.68 GHz ceiling while nominally in "powersave". Treating that as a hard failure
# would block every legitimate benchmark. So judge by achieved frequency, and only fall back
# to the governor name on drivers where it really does pin the clock (acpi-cpufreq).
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
drv=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver  2>/dev/null || echo unknown)
fmax=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null)   # kHz
fobs=$(awk '/cpu MHz/{if($4>m)m=$4}END{printf "%d", m*1000}' /proc/cpuinfo)     # kHz

if [ -n "${fmax:-}" ] && [ -n "${fobs:-}" ] && [ "$fobs" -gt 0 ]; then
    pct=$(awk -v o="$fobs" -v m="$fmax" 'BEGIN{printf "%d", o*100/m}')
    # 70% of the boost ceiling observed on at least one core means the clock is not pinned.
    # Note this is a snapshot of an idle-ish machine; it proves boost is *available*, which is
    # the thing that would otherwise silently halve the P1/P2 CPU expert path.
    if [ "$pct" -ge 70 ]; then
        ok "CPU clock ${pct}% of ceiling ($((fobs/1000)) of $((fmax/1000)) MHz), driver '$drv', governor '$gov'"
    else
        case "$drv" in
            amd-pstate*|intel_pstate)
                echo "  [ ?? ] CPU observed only ${pct}% of ceiling ($((fobs/1000)) of $((fmax/1000)) MHz)"
                echo "         — may just be idle; re-check under load before trusting a slow result" ;;
            *)
                warn "CPU clock ${pct}% of ceiling, governor '$gov' — CPU MoE path (P1/P2) throttled" ;;
        esac
    fi
else
    case "$gov" in
        performance|schedutil|ondemand) ok "CPU governor '$gov'" ;;
        powersave) warn "CPU governor 'powersave' and no frequency readback — cannot verify" ;;
        *) echo "  [ ?? ] CPU governor '$gov'" ;;
    esac
fi

# --- ACPI platform profile & system power mode -------------------------------
prof=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo unknown)
pp_daemon=$(powerprofilesctl get 2>/dev/null || echo unknown)
epp=$(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo unknown)

case "$prof" in
    performance)
        ok "ACPI platform profile 'performance'" ;;
    balanced)
        warn "ACPI platform profile is 'balanced' — CPU/GPU sustained TDP will throttle after boost. Set to 'performance' (choices: $(cat /sys/firmware/acpi/platform_profile_choices 2>/dev/null || echo 'performance balanced'))" ;;
    powersave|low-power|quiet|cool)
        warn "ACPI platform profile is '$prof' (powersave) — system is severely TDP-throttled!" ;;
    unknown)
        : ;;
    *)
        echo "  [ ?? ] ACPI platform profile '$prof'" ;;
esac

case "$pp_daemon" in
    performance)
        ok "power-profiles-daemon 'performance'" ;;
    balanced)
        warn "power-profiles-daemon is 'balanced'. Run: powerprofilesctl set performance" ;;
    power-saver)
        warn "power-profiles-daemon is 'power-saver'! Run: powerprofilesctl set performance" ;;
    unknown)
        : ;;
    *)
        echo "  [ ?? ] power-profiles-daemon '$pp_daemon'" ;;
esac

case "$epp" in
    performance)
        ok "CPU Energy Performance Preference (EPP) 'performance'" ;;
    balance_performance)
        echo "  [ ?? ] CPU EPP 'balance_performance'" ;;
    balance_power|power)
        warn "CPU EPP is '$epp' — CPU frequency scaling is power-capped!" ;;
    unknown)
        : ;;
    *)
        echo "  [ ?? ] CPU EPP '$epp'" ;;
esac

echo
if [ "$fail" -eq 0 ]; then
    echo "RESULT: OK — machine is in full PERFORMANCE mode. Measurements are admissible."
else
    echo "RESULT: THROTTLED / SUBOPTIMAL — do NOT record tok/s from this state."
    echo
    echo "To restore full performance mode:"
    echo "  1. Plug in the AC charger"
    echo "  2. echo performance | sudo tee /sys/firmware/acpi/platform_profile"
    echo "  3. powerprofilesctl set performance"
    echo "  4. for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee \$f; done"
    echo "  5. for f in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo performance | sudo tee \$f; done"
fi
exit "$fail"
