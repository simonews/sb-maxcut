#!/usr/bin/env bash
# F08-R1 on the laptop: the CUDA presets rebuilt from scratch and the probes of the GPU
# environment that decide the telemetry of F08-C2 (phase 08, part 1 of 3). Run from WSL, in the
# root of the repository, branch fase-08 with the commit of F08-R1:
#
#   bash scripts/laptop/f08r1.sh run "$NOTES"
#       1. checks; 2. build from scratch (rm -rf build/<preset>) and ctest of cpu-strict, cpu,
#       cuda-strict and cuda; 3. smoke_simt of cuda (device.txt); 4. nvidia-smi of WSL and of
#       Windows: --version, --help-query-gpu and every field in a call of its own (VALUE, NA or
#       INVALID), then the two sources sampled together every 100 ms: 5 s at rest, ctest -L gpu
#       of cuda 5 times, 5 s at rest; 5. the registry value EnableInterface of the debugger
#       interface (read only) and compute-sanitizer memcheck on smoke_simt and
#       test_curand_philox; 6. the summary;
#   bash scripts/laptop/f08r1.sh sanitizer
#       part 5 only (for example after a reboot), after a run: its lines replace those of
#       results/f08/env/summary.txt.
#
# Outputs in results/f08/env/, full logs in ~/sbmc_logs/f08r1/. The terminal gets a summary of
# about 20 lines (save it with tee), the same as results/f08/env/summary.txt. F08R1_VERDICT is
# PASS when the four builds pass and git status shows changes only in results/f08/env/: the
# probes inform, they do not decide. A problem that makes the rest meaningless stops the script
# with "STOP: <reason>". Exit status: 0 with F08R1_VERDICT PASS, 1 otherwise, 2 for a wrong
# command line, 130 when interrupted.
#
# Read only: nothing is written to the Windows registry, and no option of nvidia-smi that changes
# the state of the GPU (persistence, clocks, limits) is used. When the Windows nvidia-smi.exe of
# the sampling survives the end of its WSL process, it is ended with taskkill.exe, but only when
# no nvidia-smi.exe was running before the sampling.
#
# Only for the test of the script in the sandbox (never on the laptop): F08R1_PRESETS and
# F08R1_EXPECTED_TESTS (empty: no build), F08R1_SMI_WSL, F08R1_SMI_WIN, F08R1_REG, F08R1_TASKLIST,
# F08R1_TASKKILL, F08R1_SANITIZER and F08R1_SMOKE_SIMT replace the values below (an empty path:
# tool not found).
set -u
export LC_ALL=C

OUT=results/f08/env
LOGDIR="$HOME/sbmc_logs/f08r1"
read -r -a PRESETS <<<"${F08R1_PRESETS-cpu-strict cpu cuda-strict cuda}"
read -r -a EXPECTED_TESTS <<<"${F08R1_EXPECTED_TESTS-33 33 39 39}"
GPU_PRESET=cuda
NVCC_BUILD=V13.4.92
REST_S=5
LOAD_REPEATS=5
SAMPLE_MS=100
SANITIZER_TIMEOUT_S=600
REG_KEY='HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger'
REG_VALUE=EnableInterface
WIN_SYSTEM32=/mnt/c/Windows/System32
# The fields of F08-C1, section 5.2, then timestamp (needed by the sampling) and a few more.
FIELDS=(name driver_version pstate clocks.sm clocks.gr clocks.mem clocks.max.sm clocks.max.mem
  temperature.gpu temperature.memory power.draw power.draw.average power.draw.instant
  power.limit enforced.power.limit utilization.gpu utilization.memory pcie.link.gen.current
  pcie.link.width.current clocks_event_reasons.active clocks_throttle_reasons.active
  clocks_event_reasons.sw_power_cap clocks_event_reasons.hw_slowdown
  clocks_event_reasons.sw_thermal_slowdown timestamp clocks.max.gr power.default_limit
  power.max_limit temperature.gpu.tlimit clocks_event_reasons.gpu_idle
  clocks_event_reasons.hw_thermal_slowdown clocks_event_reasons.hw_power_brake_slowdown
  memory.used memory.total)
# Sampled when VALUE for the source, in this order.
SAMPLED=(timestamp pstate clocks.sm clocks.mem temperature.gpu power.draw utilization.gpu
  clocks_event_reasons.active)

MODE=""
BRANCH=""
TOOLKIT_ROOT=""
REG=""
SANITIZER=""
TASKLIST=""
TASKKILL=""
BUILD_VERDICT=""
GIT_OK=0
OUT_READY=0
SUMMARY=()
declare -A SMI=() FIELD_STATUS=() SAMPLER_PID=() READER_PID=()
SAMPLE_TMP=""
WIN_BEFORE=""
ORPHAN_NOTE=""

# say WORDS...: one line of the summary (the words joined by spaces), on the terminal and in
# summary.txt.
say() {
  local line="$*"
  printf '%s\n' "$line"
  SUMMARY+=("$line")
}

write_summary() {
  printf '%s\n' "${SUMMARY[@]}" >"$OUT/summary.txt"
}

stop() {
  say "STOP: $1"
  if [ -n "${2:-}" ]; then
    say "  see: $2"
    if [ -f "$2" ]; then
      local line
      while IFS= read -r line; do
        say "  | $line"
      done < <(tail -n 5 "$2" | cut -c1-150)
    fi
  fi
  say "F08R1_VERDICT: FAIL (stopped; full logs: $LOGDIR/)"
  if [ "$MODE" = run ] && [ "$OUT_READY" -eq 1 ]; then
    write_summary
  fi
  exit 1
}

# windows_tool NAME: the path of NAME in PATH (the interop of WSL appends the PATH of Windows),
# otherwise in System32; nothing when neither has it.
windows_tool() {
  if command -v "$1" 2>/dev/null; then
    return 0
  fi
  if [ -f "$WIN_SYSTEM32/$1" ]; then
    printf '%s\n' "$WIN_SYSTEM32/$1"
  fi
}

# kv KEY FILE: the value of the first line "KEY: value" of FILE.
kv() {
  sed -n "s/^$1: //p" "$2" | head -n 1
}

# first_errors LOG: up to 3 lines of LOG that name an error or a failed test.
first_errors() {
  grep -m 3 -E 'error|Error|FAILED|\*\*\*Failed|\*\*\*Exception|\*\*\*Timeout' "$1" |
    cut -c1-150
}

# csv_quote VALUE: VALUE as a CSV field.
csv_quote() {
  case "$1" in
    *[,\"]* | ' '* | *' ')
      printf '"%s"' "${1//\"/\"\"}"
      ;;
    *) printf '%s' "$1" ;;
  esac
}

# run_logged CMD...: the command line, its output without carriage returns, its exit status.
run_logged() {
  local out status
  out=$(timeout -k 5 60 "$@" 2>&1)
  status=$?
  printf '$ %s\n' "$*"
  printf '%s\n' "$out" | tr -d '\r'
  printf 'exit %d\n\n' "$status"
}

header() {
  local omp cpu commit
  omp=$(env | grep '^OMP_' | sort | tr '\n' ' ')
  omp=${omp% }
  cpu=$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -n 1)
  commit=$(git rev-parse --short HEAD)
  say "F08R1 $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$MODE branch=$BRANCH commit=$commit" \
    "cpu=\"$cpu\" nproc=$(nproc) OMP=${omp:-none}"
}

# 1. Checks common to both commands.
checks() {
  { [ -f CMakePresets.json ] && [ -e .git ] && [ -f scripts/laptop/f08r1.sh ]; } ||
    stop "not in the root of the repository (F08-R1 applied?)"
  BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  [ "$BRANCH" = fase-08 ] || stop "branch ${BRANCH:-unknown}: the procedure runs on fase-08"
  local outside
  outside=$(git status --porcelain --untracked-files=all | grep -v -E '^.. results/' |
    cut -c4- | head -n 3 | paste -sd ' ' -)
  [ -z "$outside" ] || stop "changes outside results/, commit them first: $outside"
  [ -f CMakeUserPresets.json ] ||
    stop "CMakeUserPresets.json missing: run scripts/setup_cuda_pypi.sh"
  local preset
  for preset in cuda cuda-strict; do
    grep -q "\"name\": \"$preset\"" CMakeUserPresets.json ||
      stop "CMakeUserPresets.json has no preset $preset: run scripts/setup_cuda_pypi.sh"
  done
  TOOLKIT_ROOT=$(sed -n \
    's/^[[:space:]]*"toolkit-root":[[:space:]]*"\(.*\)",\{0,1\}[[:space:]]*$/\1/p' \
    CMakeUserPresets.json)
  [ -x "$TOOLKIT_ROOT/bin/nvcc" ] ||
    stop "no nvcc in the toolkit root '$TOOLKIT_ROOT': run scripts/setup_cuda_pypi.sh"
  local nvcc
  nvcc=$("$TOOLKIT_ROOT/bin/nvcc" --version 2>&1 | grep -o 'V[0-9][0-9.]*' | tail -n 1)
  [ "$nvcc" = "$NVCC_BUILD" ] ||
    stop "nvcc ${nvcc:-unknown}, expected $NVCC_BUILD: run scripts/setup_cuda_pypi.sh"
}

# Prerequisites of the builds and of their tests (0 skipped expected).
prerequisites() {
  local tool
  for tool in cmake ctest ninja g++ git rsync unzip zip; do
    command -v "$tool" >/dev/null 2>&1 || stop "missing prerequisite: $tool"
  done
  if ! .venv/bin/python -c 'import numpy, matplotlib' >/dev/null 2>&1; then
    stop "NumPy or matplotlib missing in .venv: run scripts/setup_python.sh"
  fi
  [ -f data/raw/k2000/WK2000_1.rud ] ||
    stop "data/raw/k2000/WK2000_1.rud missing: run scripts/fetch_data.sh --k2000"
  [ "${#PRESETS[@]}" -eq "${#EXPECTED_TESTS[@]}" ] ||
    stop "F08R1_PRESETS and F08R1_EXPECTED_TESTS have different lengths"
}

# 2. build_preset PRESET EXPECTED: build from scratch and ctest; one BUILD line (more on a
# failure); sets BUILD_VERDICT.
build_preset() {
  local preset="$1" expected="$2" base="$LOGDIR/$1" start seconds
  local configure=- build=- tests=- log warnings total failed skipped passed verdict
  case "$preset" in
    '' | *[!a-z0-9-]*) stop "invalid preset name '$preset'" ;;
  esac
  start=$(date +%s)
  rm -rf -- "build/$preset"
  cmake --preset "$preset" >"${base}_configure.log" 2>&1
  configure=$?
  log="${base}_configure.log"
  if [ "$configure" -eq 0 ]; then
    cmake --build --preset "$preset" >"${base}_build.log" 2>&1
    build=$?
    log="${base}_build.log"
    if [ "$build" -eq 0 ]; then
      ctest --preset "$preset" >"${base}_ctest.log" 2>&1
      tests=$?
      log="${base}_ctest.log"
    fi
  fi
  warnings=$(cat "${base}_configure.log" "${base}_build.log" 2>/dev/null | grep -ci warning)
  total=0
  failed=0
  skipped=0
  if [ -f "${base}_ctest.log" ]; then
    total=$(sed -n 's/.* tests failed out of \([0-9]*\).*/\1/p' "${base}_ctest.log")
    failed=$(sed -n 's/.*tests passed, \([0-9]*\) tests failed out of.*/\1/p' \
      "${base}_ctest.log")
    skipped=$(grep -c '\*\*\*Skipped' "${base}_ctest.log")
  fi
  total=${total:-0}
  failed=${failed:-0}
  passed=$((total - failed - skipped))
  verdict=FAIL
  if [ "$tests" = 0 ] && [ "$total" -eq "$expected" ] && [ "$failed" -eq 0 ] &&
    [ "$skipped" -eq 0 ] && [ "$warnings" -eq 0 ]; then
    verdict=PASS
  fi
  BUILD_VERDICT=$verdict
  seconds=$(($(date +%s) - start))
  printf 'preset=%s configure=%s build=%s ctest=%s warnings=%s total=%s failed=%s' \
    "$preset" "$configure" "$build" "$tests" "$warnings" "$total" "$failed" \
    >>"$OUT/build_summary.txt"
  printf ' skipped=%s passed=%s expected=%s seconds=%s verdict=%s\n' \
    "$skipped" "$passed" "$expected" "$seconds" "$verdict" >>"$OUT/build_summary.txt"
  if [ "$verdict" = PASS ]; then
    say "BUILD_$preset: PASS ($passed/$total, skipped $skipped, warnings $warnings, $seconds s)"
    return
  fi
  say "BUILD_$preset: FAIL ($passed/$total, expected $expected; skipped $skipped," \
    "warnings $warnings, $seconds s; exit configure $configure, build $build, ctest $tests)"
  if [ "$skipped" -gt 0 ]; then
    say "  skipped: $(sed -n 's/^[[:space:]]*[0-9]* - \(.*\) (Skipped)$/\1/p' \
      "${base}_ctest.log" | paste -sd ' ' -)"
  fi
  if [ "$configure" != 0 ] || [ "$build" != 0 ] || [ "$failed" -gt 0 ] || [ "$tests" != 0 ]; then
    say "  see: $log"
    local line
    while IFS= read -r line; do
      say "  | $line"
    done < <(first_errors "$log")
  fi
  if [ "$warnings" -gt 0 ]; then
    say "  warnings: $(cat "${base}_configure.log" "${base}_build.log" 2>/dev/null |
      grep -i -m 1 warning | cut -c1-150)"
  fi
}

# 3. The device, from smoke_simt of the GPU preset.
probe_device() {
  local smoke="${F08R1_SMOKE_SIMT:-build/$GPU_PRESET/tests/smoke/smoke_simt}" status
  local file="$OUT/device.txt"
  if [ ! -x "$smoke" ]; then
    printf 'missing: %s\n' "$smoke" >"$file"
    say "DEVICE: UNAVAILABLE ($smoke missing)"
    return
  fi
  timeout -k 10 300 "$smoke" >"$file" 2>&1
  status=$?
  local name
  name=$(kv device0_name "$file")
  if [ "$status" -ne 0 ] || [ -z "$name" ]; then
    local reason
    reason=$(kv skip_reason "$file")
    [ -n "$reason" ] || reason=$(grep -v '^[[:space:]]*$' "$file" | tail -n 1 | cut -c1-120)
    say "DEVICE: UNAVAILABLE (smoke_simt exit $status: $reason)"
    return
  fi
  say "$(awk -v name="$name" -v cc="$(kv device0_compute_capability "$file")" \
    -v sm="$(kv device0_sm_count "$file")" -v l2="$(kv device0_l2_cache_bytes "$file")" \
    -v mem="$(kv device0_global_memory_bytes "$file")" \
    -v clk="$(kv device0_sm_clock_khz "$file")" '
    function mib(b, f) { return b ~ /^[0-9]+$/ ? sprintf(f, b / 1048576) : b }
    BEGIN {
      mhz = clk ~ /^[0-9]+$/ ? sprintf("%d", clk / 1000) : clk
      printf "DEVICE: %s, CC %s, SM %s, L2 %s MiB, memory %s MiB, SM clock %s MHz (attribute)\n",
        name, cc, sm, mib(l2, "%.1f"), mib(mem, "%.0f"), mhz
    }')"
}

# classify STATUS VALUE: VALUE, NA or INVALID.
classify() {
  if [ "$1" -ne 0 ]; then
    echo INVALID
    return
  fi
  case "$2" in
    *"not a valid field"*) echo INVALID ;;
    '' | N/A | '['*']') echo NA ;;
    *) echo VALUE ;;
  esac
}

# 4a. probe_fields SOURCE: --version, --help-query-gpu and every field alone; one SMI line.
probe_fields() {
  local src="$1" smi="${SMI[$1]}" label field out status class driver=""
  local -a nas=() invalids=()
  local nvalue=0
  label=${src^^}
  if [ -z "$smi" ]; then
    printf 'NOT_FOUND\n' >"$OUT/smi_help_$src.txt"
    for field in "${FIELDS[@]}"; do
      FIELD_STATUS[$src:$field]=NOT_FOUND
      printf '%s,,%s,NOT_FOUND,\n' "$src" "$field" >>"$OUT/smi_fields.csv"
    done
    say "SMI_$label: NOT_FOUND"
    return
  fi
  {
    run_logged "$smi" --version
    run_logged "$smi" --help-query-gpu
  } >"$OUT/smi_help_$src.txt"
  for field in "${FIELDS[@]}"; do
    out=$(timeout -k 5 30 "$smi" --query-gpu="$field" --format=csv,noheader,nounits -i 0 2>&1)
    status=$?
    out=$(printf '%s\n' "$out" | tr -d '\r' | grep -m 1 -v '^[[:space:]]*$' |
      sed 's/^[[:space:]]*//; s/[[:space:]]*$//')
    class=$(classify "$status" "$out")
    [ "$class" = INVALID ] && [ "$status" -ne 0 ] && out="exit $status: $out"
    FIELD_STATUS[$src:$field]=$class
    printf '%s,%s,%s,%s,%s\n' "$src" "$(csv_quote "$smi")" "$field" "$class" \
      "$(csv_quote "$out")" >>"$OUT/smi_fields.csv"
    case "$class" in
      VALUE) nvalue=$((nvalue + 1)) ;;
      NA) nas+=("$field") ;;
      INVALID) invalids+=("$field") ;;
    esac
    [ "$field" = driver_version ] && [ "$class" = VALUE ] && driver=" (driver $out)"
  done
  local line="SMI_$label: $smi$driver; VALUE $nvalue, NA ${#nas[@]}, INVALID ${#invalids[@]}"
  [ "${#nas[@]}" -gt 0 ] && line+="; NA: ${nas[*]}"
  [ "${#invalids[@]}" -gt 0 ] && line+="; INVALID: ${invalids[*]}"
  say "$line"
}

# stamp_lines PHASE_FILE: every line of stdin prefixed by the time of arrival in WSL
# (EPOCHREALTIME, seconds) and the phase, with ", " of nvidia-smi turned into ",".
stamp_lines() {
  local line t phase
  while IFS= read -r line; do
    t=${EPOCHREALTIME/,/.}
    { read -r phase <"$1"; } 2>/dev/null || phase=unknown
    line=${line%$'\r'}
    printf '%s,%s,%s\n' "$t" "${phase:-unknown}" "${line//, /,}"
  done
}

set_phase() {
  printf '%s\n' "$1" >"$SAMPLE_TMP/phase.new" && mv -f "$SAMPLE_TMP/phase.new" "$SAMPLE_TMP/phase"
}

# count_windows_smi: the number of nvidia-smi.exe processes on Windows; nothing when tasklist.exe
# is missing or fails.
count_windows_smi() {
  [ -n "$TASKLIST" ] || return 0
  local out
  out=$(timeout -k 5 30 "$TASKLIST" /FI "IMAGENAME eq nvidia-smi.exe" /NH /FO CSV 2>/dev/null) ||
    return 0
  printf '%s\n' "$out" | tr -d '\r' | grep -ci '^"nvidia-smi.exe"'
}

# start_sampler SOURCE FIELDS FILE: nvidia-smi -lms into a FIFO read by stamp_lines.
start_sampler() {
  local fifo="$SAMPLE_TMP/$1.fifo"
  mkfifo "$fifo" || return 1
  stamp_lines "$SAMPLE_TMP/phase" <"$fifo" >>"$3" &
  READER_PID[$1]=$!
  "${SMI[$1]}" --query-gpu="$2" --format=csv,noheader,nounits -i 0 -lms "$SAMPLE_MS" \
    >"$fifo" 2>"$LOGDIR/smi_sampling_$1.err" &
  SAMPLER_PID[$1]=$!
}

# gone PID TENTHS: waits up to TENTHS tenths of a second for PID to end.
gone() {
  local i
  for ((i = 0; i < $2; i++)); do
    kill -0 "$1" 2>/dev/null || return 0
    sleep 0.1
  done
  ! kill -0 "$1" 2>/dev/null
}

# stop_sampling: ends the samplers and their readers (also from the EXIT trap), then the
# nvidia-smi.exe left on Windows by the sampling, if any. Idempotent.
stop_sampling() {
  local src pid win=0
  [ "${#SAMPLER_PID[@]}" -gt 0 ] && set_phase stopping
  for src in "${!SAMPLER_PID[@]}"; do
    [ "$src" = win ] && win=1
    pid=${SAMPLER_PID[$src]}
    kill -TERM "$pid" 2>/dev/null
    gone "$pid" 30 || kill -KILL "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    pid=${READER_PID[$src]:-}
    if [ -n "$pid" ]; then
      gone "$pid" 30 || kill -KILL "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    fi
  done
  SAMPLER_PID=()
  READER_PID=()
  [ "$win" -eq 1 ] || return 0
  if [ -z "$WIN_BEFORE" ]; then
    ORPHAN_NOTE="orphan check unavailable (tasklist.exe)"
    return 0
  fi
  local after
  after=$(count_windows_smi)
  if [ "${after:-0}" -le "$WIN_BEFORE" ]; then
    ORPHAN_NOTE="children stopped"
  elif [ "$WIN_BEFORE" -eq 0 ] && [ -n "$TASKKILL" ]; then
    timeout -k 5 30 "$TASKKILL" /F /IM nvidia-smi.exe >>"$LOGDIR/taskkill.log" 2>&1
    ORPHAN_NOTE="$after nvidia-smi.exe left, taskkill: $(count_windows_smi) left"
  else
    ORPHAN_NOTE="$((after - WIN_BEFORE)) nvidia-smi.exe left (others were running before)"
  fi
}

cleanup() {
  stop_sampling >/dev/null 2>&1
  if [ -n "$SAMPLE_TMP" ]; then
    rm -rf -- "$SAMPLE_TMP"
  fi
}

# sample_stats FILE: count, phases, period, maxima of clocks.sm and temperature.gpu, power.draw.
sample_stats() {
  awk -F, -v ms="$SAMPLE_MS" '
    function num(v) { return v ~ /^-?[0-9]+(\.[0-9]+)?$/ }
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i; next }
    {
      n++
      count[$2]++
      if (n == 1) first = $1
      last = $1
      if ("clocks.sm" in col && num($(col["clocks.sm"])) &&
          (sm == "" || $(col["clocks.sm"]) + 0 > sm)) sm = $(col["clocks.sm"]) + 0
      if ("temperature.gpu" in col && num($(col["temperature.gpu"])) &&
          (temp == "" || $(col["temperature.gpu"]) + 0 > temp)) temp = $(col["temperature.gpu"]) + 0
      if ("power.draw" in col) {
        v = $(col["power.draw"])
        if (num(v)) {
          pn++
          ps += v
          if (pn == 1 || v + 0 < pmin) pmin = v + 0
          if (pn == 1 || v + 0 > pmax) pmax = v + 0
        } else pna++
      }
    }
    END {
      # Parentheses around every comparison: in the arguments of printf, ">" would redirect.
      if (n == 0) { printf "0 samples"; exit }
      period = (n > 1) ? (last - first) * 1000 / (n - 1) : 0
      printf "%d samples (rest %d, load %d, rest %d", n, count["rest_before"], count["load"],
        count["rest_after"]
      if (count["stopping"] > 0) printf ", %d after the stop", count["stopping"]
      printf "), period %.0f ms", period
      printf "; clocks.sm max %s", ((sm == "") ? "N/A" : sm " MHz")
      printf "; temperature.gpu max %s", ((temp == "") ? "N/A" : temp " C")
      if (pn > 0) {
        printf "; power.draw min/mean/max %.2f/%.2f/%.2f W", pmin, ps / pn, pmax
        if (pna > 0) printf " (%d N/A)", pna
      } else printf "; power.draw N/A"
    }' "$1"
}

# 4b. Sampling of both sources together: rest, load, rest. LOAD and SAMPLES lines.
sample_sources() {
  local src field fields file
  SAMPLE_TMP=$(mktemp -d) || stop "mktemp failed"
  set_phase rest_before
  for src in wsl win; do
    fields=""
    for field in "${SAMPLED[@]}"; do
      if [ "${FIELD_STATUS[$src:$field]:-}" = VALUE ]; then
        fields+="${fields:+,}$field"
      fi
    done
    file="$OUT/smi_samples_$src.csv"
    printf 't_wsl_s,phase%s\n' "${fields:+,$fields}" >"$file"
    if [ -n "${SMI[$src]}" ] && [ -n "$fields" ]; then
      if [ "$src" = win ]; then
        WIN_BEFORE=$(count_windows_smi)
      fi
      start_sampler "$src" "$fields" "$file" || stop "mkfifo failed in $SAMPLE_TMP"
    fi
  done
  sleep "$REST_S"
  set_phase load
  local load_line i statuses="" start
  start=$(date +%s)
  if [ -f "build/$GPU_PRESET/CTestTestfile.cmake" ]; then
    for ((i = 1; i <= LOAD_REPEATS; i++)); do
      ctest --preset "$GPU_PRESET" -L gpu >"$LOGDIR/load_ctest_$i.log" 2>&1
      statuses+="${statuses:+,}$?"
    done
    local last="$LOGDIR/load_ctest_$LOAD_REPEATS.log"
    load_line="LOAD: ctest -L gpu of $GPU_PRESET x$LOAD_REPEATS, exit $statuses, tests \
$(sed -n 's/.* tests failed out of \([0-9]*\).*/\1/p' "$last") per run, skipped \
$(grep -c '\*\*\*Skipped' "$last") in the last, $(($(date +%s) - start)) s, $REST_S s at rest \
before and after"
  else
    load_line="LOAD: SKIPPED (build/$GPU_PRESET missing), $REST_S s at rest before and after"
  fi
  set_phase rest_after
  sleep "$REST_S"
  stop_sampling
  say "$load_line"
  for src in wsl win; do
    local line
    if [ -z "${SMI[$src]}" ]; then
      line="NOT_FOUND"
    elif [ "$(head -n 1 "$OUT/smi_samples_$src.csv")" = t_wsl_s,phase ]; then
      line="none (no sampled field is VALUE)"
    else
      line=$(sample_stats "$OUT/smi_samples_$src.csv")
      [ "$src" = win ] && [ -n "$ORPHAN_NOTE" ] && line+="; $ORPHAN_NOTE"
    fi
    say "SAMPLES_${src^^}: $line"
  done
  rm -rf -- "$SAMPLE_TMP"
  SAMPLE_TMP=""
}

# 5a. The registry value of the debugger interface, read only. One REGISTRY line.
probe_registry() {
  local file="$OUT/registry.txt" out status line value_type data
  if [ -z "$REG" ]; then
    printf 'reg.exe not found\n' >"$file"
    say "REGISTRY_ENABLEINTERFACE: UNKNOWN (reg.exe not found)"
    return
  fi
  out=$(timeout -k 5 30 "$REG" query "$REG_KEY" /v "$REG_VALUE" 2>&1)
  status=$?
  out=$(printf '%s\n' "$out" | tr -d '\r')
  printf '$ %s query "%s" /v %s\n%s\nexit %d\n' "$REG" "$REG_KEY" "$REG_VALUE" "$out" \
    "$status" >"$file"
  line=$(printf '%s\n' "$out" | grep -i -m 1 "^[[:space:]]*${REG_VALUE}[[:space:]]")
  if [ "$status" -eq 0 ] && [ -n "$line" ]; then
    read -r _ value_type data <<<"$line"
    if [ "$value_type" = REG_DWORD ] && [[ "$data" =~ ^0x0*1$ ]]; then
      say "REGISTRY_ENABLEINTERFACE: ENABLED (0x1)"
    else
      say "REGISTRY_ENABLEINTERFACE: VALUE ${data:-empty} ($value_type)"
    fi
  elif [ "$status" -eq 1 ]; then
    say "REGISTRY_ENABLEINTERFACE: ABSENT ($(printf '%s\n' "$out" | grep -m 1 -v \
      '^[[:space:]]*$' | cut -c1-100))"
  else
    say "REGISTRY_ENABLEINTERFACE: UNKNOWN (exit $status: $(printf '%s\n' "$out" |
      grep -m 1 -v '^[[:space:]]*$' | cut -c1-100))"
  fi
}

# 5b. probe_sanitizer TEST EXE: compute-sanitizer memcheck; one SANITIZER line.
probe_sanitizer() {
  local test="$1" exe="$2" file="$OUT/sanitizer_$1.txt" status reason app
  if [ ! -x "$SANITIZER" ] || [ ! -x "$exe" ]; then
    [ -x "$SANITIZER" ] || reason="$SANITIZER missing"
    [ -x "$exe" ] || reason="${reason:+$reason, }$exe missing"
    printf '%s\n' "$reason" >"$file"
    say "SANITIZER_$test: ERRORS ($reason)"
    return
  fi
  printf '$ %s --tool memcheck --error-exitcode 1 %s\n' "$SANITIZER" "$exe" >"$file"
  timeout -k 10 "$SANITIZER_TIMEOUT_S" "$SANITIZER" --tool memcheck --error-exitcode 1 "$exe" \
    >>"$file" 2>&1
  status=$?
  printf 'exit %d\n' "$status" >>"$file"
  app=$(grep -m 1 -o -E '(SIMT_VERDICT|T11_GPU_VERDICT): [A-Z]+' "$file")
  reason=$(grep -m 1 -o -E 'Failed to initialize WDDM debugger interface|Device not supported' \
    "$file")
  if [ -n "$reason" ]; then
    say "SANITIZER_$test: NOT_ATTACHED (exit $status: $reason)"
  elif [ "$status" -eq 0 ] && grep -q 'ERROR SUMMARY: 0 errors' "$file"; then
    say "SANITIZER_$test: ATTACHED_CLEAN (ERROR SUMMARY: 0 errors; ${app:-no verdict line})"
  else
    if [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
      reason="timeout after $SANITIZER_TIMEOUT_S s"
    else
      reason=$(grep -m 1 -E 'Error|ERROR' "$file" | sed 's/^=* *//' | cut -c1-100)
    fi
    say "SANITIZER_$test: ERRORS (exit $status: ${reason:-see $file}; ${app:-no verdict line})"
  fi
}

probe_part5() {
  REG="${F08R1_REG-$(windows_tool reg.exe)}"
  SANITIZER="${F08R1_SANITIZER:-$TOOLKIT_ROOT/bin/compute-sanitizer}"
  probe_registry
  probe_sanitizer smoke_simt "build/$GPU_PRESET/tests/smoke/smoke_simt"
  probe_sanitizer test_curand_philox "build/$GPU_PRESET/tests/gpu/test_curand_philox"
}

# GIT_STATUS line; sets GIT_OK.
git_status_line() {
  local entries count outside
  entries=$(git status --porcelain --untracked-files=all)
  count=$(printf '%s' "$entries" | grep -c .)
  outside=$(printf '%s\n' "$entries" | grep -v -E "^.. $OUT/" | grep . | cut -c4- |
    head -n 3 | paste -sd ' ' -)
  GIT_OK=0
  if [ -z "$outside" ]; then
    GIT_OK=1
    say "GIT_STATUS: PASS ($count entries, all in $OUT/)"
  else
    say "GIT_STATUS: FAIL (changes outside $OUT/: $outside)"
  fi
}

run_command() {
  local notes="$1" start_all
  start_all=$(date +%s)
  [ -n "$notes" ] || stop "NOTES is empty: set it first, on a line of its own"
  case "$notes" in *$'\n'* | *$'\r'*) stop "NOTES must be a single line" ;; esac

  # 1. Checks.
  checks
  prerequisites
  mkdir -p "$LOGDIR" "$OUT" || stop "cannot create $LOGDIR or $OUT"
  find "$LOGDIR" -maxdepth 1 -type f -delete
  rm -f -- "$OUT/notes.txt" "$OUT/build_summary.txt" "$OUT/device.txt" "$OUT"/smi_help_*.txt \
    "$OUT/smi_fields.csv" "$OUT"/smi_samples_*.csv "$OUT/registry.txt" "$OUT"/sanitizer_*.txt \
    "$OUT/summary.txt"
  OUT_READY=1
  header
  say "NOTES: $notes"
  printf '%s\n' "$notes" >"$OUT/notes.txt"

  # 2. Builds from scratch.
  printf 'toolchain cmake=%s ninja=%s gcc=%s nvcc=%s toolkit_root=%s\n' \
    "$(cmake --version | sed -n 's/^cmake version //p')" "$(ninja --version)" \
    "$(g++ -dumpfullversion)" "$NVCC_BUILD" "$TOOLKIT_ROOT" >"$OUT/build_summary.txt"
  local i builds_ok=1
  for i in "${!PRESETS[@]}"; do
    build_preset "${PRESETS[$i]}" "${EXPECTED_TESTS[$i]}"
    if [ "$BUILD_VERDICT" != PASS ]; then
      builds_ok=0
    fi
  done

  # 3. Device.
  probe_device

  # 4. nvidia-smi.
  SMI[wsl]="${F08R1_SMI_WSL-$(command -v nvidia-smi)}"
  SMI[win]="${F08R1_SMI_WIN-$(windows_tool nvidia-smi.exe)}"
  TASKLIST="${F08R1_TASKLIST-$(windows_tool tasklist.exe)}"
  TASKKILL="${F08R1_TASKKILL-$(windows_tool taskkill.exe)}"
  printf 'source,path,field,status,value\n' >"$OUT/smi_fields.csv"
  probe_fields wsl
  probe_fields win
  sample_sources

  # 5. Registry and compute-sanitizer.
  probe_part5

  # 6. Summary.
  git_status_line
  local verdict=FAIL
  if [ "$builds_ok" -eq 1 ] && [ "$GIT_OK" -eq 1 ]; then
    verdict=PASS
  fi
  say "F08R1_VERDICT: $verdict (total $(($(date +%s) - start_all)) s; full logs: $LOGDIR/)"
  write_summary
  [ "$verdict" = PASS ]
}

sanitizer_command() {
  checks
  [ -f "$OUT/summary.txt" ] ||
    stop "$OUT/summary.txt missing: run first bash scripts/laptop/f08r1.sh run \"\$NOTES\""
  mkdir -p "$LOGDIR" || stop "cannot create $LOGDIR"
  header
  say "NOTES: $(head -n 1 "$OUT/notes.txt" 2>/dev/null)"
  local first=${#SUMMARY[@]}
  probe_part5
  say "SANITIZER_RERUN: $(date -u +%Y-%m-%dT%H:%M:%SZ) commit=$(git rev-parse --short HEAD)"
  local -a block=("${SUMMARY[@]:first}")
  git_status_line
  # The new lines replace those of the run in summary.txt, before GIT_STATUS.
  local tmp
  tmp=$(mktemp) || stop "mktemp failed"
  if ! SUMMARY_BLOCK="$(printf '%s\n' "${block[@]}")" awk '
    /^(REGISTRY_ENABLEINTERFACE|SANITIZER_[A-Za-z_]+): / { next }
    !done && /^(GIT_STATUS|F08R1_VERDICT|STOP): / { print ENVIRON["SUMMARY_BLOCK"]; done = 1 }
    { print }
    END { if (!done) print ENVIRON["SUMMARY_BLOCK"] }' "$OUT/summary.txt" >"$tmp" ||
    ! mv -f "$tmp" "$OUT/summary.txt"; then
    rm -f -- "$tmp"
    stop "cannot update $OUT/summary.txt"
  fi
  local verdict
  verdict=$(sed -n 's/^F08R1_VERDICT: \([A-Z]*\).*/\1/p' "$OUT/summary.txt")
  say "F08R1_VERDICT: ${verdict:-unknown} (that of the run: sanitizer only; summary.txt" \
    "updated; full logs: $LOGDIR/)"
}

trap cleanup EXIT
trap 'exit 130' INT TERM

case "${1:-}" in
  run)
    MODE=run
    run_command "${2:-}"
    ;;
  sanitizer)
    MODE=sanitizer
    sanitizer_command
    ;;
  *)
    echo "usage: bash scripts/laptop/f08r1.sh run \"\$NOTES\" | sanitizer" >&2
    exit 2
    ;;
esac
