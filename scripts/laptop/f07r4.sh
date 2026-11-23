#!/usr/bin/env bash
# F07-R4 on the laptop: E9, the OpenMP scaling on Windows native and in WSL2 (phase 07, part 4).
# Run from WSL, in the root of the repository, branch fase-07 with the commit of F07-R4:
#
#   bash scripts/laptop/f07r4.sh stage
#       copies HEAD to /mnt/c/Users/Simone/sbmc-win/src (scripts/windows/stage.sh) and prints the
#       line to run in a PowerShell window (scripts/windows/build.ps1: the presets win-cpu-strict
#       and win-cpu, built and tested on Windows);
#   bash scripts/laptop/f07r4.sh run "$NOTES"
#       1. checks; 2. sbmc_run.exe --version and --print-topology through the interop of WSL
#       (also under every --affinity); 3. the summary of build.ps1; 4. incremental build and
#       ctest of cpu-strict and cpu in WSL; 5. the bitwise comparison of section 5.4
#       (python/sbmc_e9.py bitwise); 6. the campaign E9 (bench/configs/f07_e9.toml, resumable),
#       status, analyze, manifest; 7. the summary.
#
# The terminal gets a summary of about 20 lines (save it with tee); the details go to
# ~/sbmc_logs/f07r4/. Verdicts: WIN_BUILD, INTEROP, TOPOLOGY, BITWISE_SA, BITWISE_DSB_STRICT, E9,
# E9_CONSISTENCY and F07R4_VERDICT (PASS when all are PASS). A problem that makes the rest
# meaningless stops the script with "STOP: <reason>" and the file to read; running it again
# resumes E9 where it stopped.
#
# Only for the test of the script in the sandbox (never on the laptop): F07R4_WIN_DEST,
# F07R4_CONFIG, F07R4_E9_CONFIGS, F07R4_EXPECTED_TOPOLOGY and F07R4_MIN_NPROC replace the values
# below.
set -u

WIN_DEST="${F07R4_WIN_DEST:-/mnt/c/Users/Simone/sbmc-win}"
WIN_RUN="$WIN_DEST/build-win-cpu/apps/sbmc_run.exe"
WIN_STRICT_RUN="$WIN_DEST/build-win-cpu-strict/apps/sbmc_run.exe"
WIN_THRESHOLDS="$WIN_DEST/build-win-cpu/tests/tools/sbmc_sa_thresholds.exe"
CONFIG="${F07R4_CONFIG:-bench/configs/f07_e9.toml}"
E9_CONFIGS="${F07R4_E9_CONFIGS:-136}"
PY=.venv/bin/python
LOGDIR="$HOME/sbmc_logs/f07r4"
MIN_NPROC="${F07R4_MIN_NPROC:-16}"
EXPECTED_TESTS=33  # WSL presets cpu-strict and cpu
# The i7-13620H of the laptop.
EXPECTED_TOPOLOGY="${F07R4_EXPECTED_TOPOLOGY:-logical=16 classes=2 p=12 e=4 p1=6}"

declare -A DURATION
PART_START=0
part() { PART_START=$(date +%s); }
part_end() { DURATION["$1"]=$(($(date +%s) - PART_START)); }

stop() {
  echo "STOP: $1"
  [ -n "${2:-}" ] && echo "  see: $2"
  [ -n "${2:-}" ] && [ -f "$2" ] && tail -n 5 "$2" | sed 's/^/  | /'
  echo "F07R4_VERDICT: FAIL (stopped; full logs: $LOGDIR/)"
  exit 1
}

# Common checks of both commands.
common_checks() {
  { [ -f CMakePresets.json ] && [ -e .git ] && [ -f "$CONFIG" ]; } ||
    stop "not in the root of the repository with $CONFIG (F07-R4 applied?)"
  BRANCH=$(git rev-parse --abbrev-ref HEAD)
  [ "$BRANCH" = "fase-07" ] || stop "branch $BRANCH: the procedure runs on fase-07"
  HEAD_HASH=$(git rev-parse HEAD)
  command -v wslpath >/dev/null 2>&1 || stop "wslpath missing: run this script in WSL"
}

stage_command() {
  common_checks
  mkdir -p "$LOGDIR"
  bash scripts/windows/stage.sh "$WIN_DEST" 2>&1 | tee "$LOGDIR/stage.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || stop "scripts/windows/stage.sh failed" "$LOGDIR/stage.log"
  local ps1
  ps1=$(wslpath -w "$WIN_DEST/src/scripts/windows/build.ps1") || stop "wslpath -w failed"
  echo "Now, in a PowerShell window (not from WSL), run:"
  echo "  powershell -NoProfile -ExecutionPolicy Bypass -File $ps1"
  echo "then, here: bash scripts/laptop/f07r4.sh run \"\$NOTES\""
}

# field FILE PREFIX KEY: the value of KEY=<value> on the first line of FILE that starts with
# PREFIX (empty when there is none).
field() {
  awk -v prefix="$2" -v key="$3" 'index($0, prefix) == 1 {
    for (i = 1; i <= NF; i++) {
      if (index($i, key "=") == 1) { print substr($i, length(key) + 2); exit }
    }
  }' "$1"
}

run_command() {
  local notes="$1"
  [ -n "$notes" ] || stop "NOTES is empty: set it first, on a line of its own (as in F07-R3)"
  case "$notes" in *$'\n'* | *$'\r'*) stop "NOTES must be a single line" ;; esac
  local start_all
  start_all=$(date +%s)

  # 1. Checks.
  part
  common_checks
  mkdir -p "$LOGDIR" || stop "cannot create $LOGDIR"
  for tool in cmake ninja g++ git; do
    command -v "$tool" >/dev/null 2>&1 || stop "missing prerequisite: $tool"
  done
  [ -x "$PY" ] || stop "$PY missing: run scripts/setup_python.sh"
  "$PY" -c "import numpy, matplotlib" 2>/dev/null || stop "NumPy or matplotlib missing in .venv"
  [ "$(nproc)" -ge "$MIN_NPROC" ] || stop "nproc = $(nproc): E9 needs $MIN_NPROC logical CPUs"
  for f in data/gset/G1 data/raw/k2000/WK2000_1.rud results/f07/e2/selected.csv; do
    [ -f "$f" ] || stop "$f missing"
  done
  local staged
  staged=$(sed -n 's/^git_hash=//p' "$WIN_DEST/src/.sbmc_build_info" 2>/dev/null)
  [ "$staged" = "$HEAD_HASH" ] ||
    stop "the copy on Windows is ${staged:-missing}, HEAD is $HEAD_HASH: run stage and build.ps1"
  for exe in "$WIN_RUN" "$WIN_STRICT_RUN" "$WIN_THRESHOLDS"; do
    [ -f "$exe" ] || stop "$exe missing: run build.ps1 in PowerShell"
  done
  local omp
  omp=$(env | grep '^OMP_' | sort | tr '\n' ' ')
  echo "F07-R4 $(date -u +%Y-%m-%dT%H:%M:%SZ) branch=$BRANCH commit=$(git rev-parse --short HEAD)" \
    "cpu=\"$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -n 1)\"" \
    "nproc=$(nproc) OMP=${omp:-none}"
  echo "NOTES: $notes"
  part_end checks

  # 2. Interop: --version and --print-topology of sbmc_run.exe.
  part
  local verdict_interop=FAIL
  "$WIN_RUN" --version >"$LOGDIR/interop_version.txt" 2>&1 ||
    stop "INTEROP: $WIN_RUN --version failed (is the interop of WSL enabled?)" \
      "$LOGDIR/interop_version.txt"
  "$WIN_STRICT_RUN" --version >>"$LOGDIR/interop_version.txt" 2>&1 ||
    stop "INTEROP: $WIN_STRICT_RUN --version failed" "$LOGDIR/interop_version.txt"
  local built
  built=$(grep -c "^sbmc_run git_hash=$HEAD_HASH dirty=0 " "$LOGDIR/interop_version.txt")
  [ "$built" -eq 2 ] || stop "INTEROP: the executables are not built from HEAD $HEAD_HASH" \
    "$LOGDIR/interop_version.txt"
  verdict_interop=PASS
  echo "INTEROP: $verdict_interop ($(head -n 1 "$LOGDIR/interop_version.txt" | cut -d' ' -f2- |
    sed 's/git_hash=\([0-9a-f]\{7\}\)[0-9a-f]*/git_hash=\1/'))"
  local topo="$LOGDIR/topology.txt"
  "$WIN_RUN" --print-topology >"$topo" 2>&1 || stop "TOPOLOGY: --print-topology failed" "$topo"
  local logical classes p e p1 masks="" applied=1
  logical=$(field "$topo" TOPOLOGY logical)
  classes=$(field "$topo" TOPOLOGY classes)
  p=$(field "$topo" "AFFINITY p " count)
  e=$(field "$topo" "AFFINITY e " count)
  p1=$(field "$topo" "AFFINITY p1" count)
  for rule in p1 p e all; do
    local file="$LOGDIR/topology_$rule.txt" wanted got procs
    wanted=$(field "$topo" "AFFINITY $rule " count)
    if [ -z "$wanted" ]; then
      masks+=" $rule unavailable"  # e with one class
      continue
    fi
    if "$WIN_RUN" --affinity "$rule" --print-topology >"$file" 2>&1; then
      got=$(field "$file" MASK count)
      procs=$(field "$file" OPENMP num_procs)
      masks+=" $rule ${got:-?}/$wanted (num_procs ${procs:-?})"
      if [ "$got" != "$wanted" ]; then
        applied=0
      fi
    else
      masks+=" $rule error"
      applied=0
    fi
  done
  local bind
  bind=$(field "$topo" OPENMP proc_bind)
  local seen="logical=$logical classes=$classes p=$p e=$e p1=$p1"
  local verdict_topology=FAIL
  if [ "$seen" = "$EXPECTED_TOPOLOGY" ] && [ "$applied" -eq 1 ]; then
    verdict_topology=PASS
  fi
  echo "TOPOLOGY: $verdict_topology ($seen; masks:$masks; proc_bind $bind)"
  [ "$verdict_topology" = PASS ] ||
    stop "TOPOLOGY: expected $EXPECTED_TOPOLOGY and every mask applied" "$topo"
  part_end interop

  # 3. The summary of build.ps1.
  local summary="$WIN_DEST/logs/build_summary.txt"
  [ -f "$summary" ] || stop "$summary missing: run build.ps1 in PowerShell"
  tr -d '\r' <"$summary" >"$LOGDIR/build_summary.txt"
  local verdict_build
  verdict_build=$(sed -n 's/^WIN_BUILD: //p' "$LOGDIR/build_summary.txt")
  local source_hash
  source_hash=$(field "$LOGDIR/build_summary.txt" source git_hash)
  grep '^preset ' "$LOGDIR/build_summary.txt" | sed 's/^/WIN /'
  [ "$source_hash" = "$HEAD_HASH" ] || verdict_build=FAIL
  echo "WIN_BUILD: ${verdict_build:-FAIL} ($(sed -n 's/^toolchain .* gcc=//p' \
    "$LOGDIR/build_summary.txt"); source ${source_hash:0:7})"
  [ "$verdict_build" = PASS ] || stop "WIN_BUILD: FAIL" "$LOGDIR/build_summary.txt"

  # 4. WSL: incremental build and ctest of cpu-strict and cpu.
  part
  for preset in cpu-strict cpu; do
    local configure="-" build_log="$LOGDIR/wsl_${preset}_build.log"
    local test_log="$LOGDIR/wsl_${preset}_ctest.log"
    if [ ! -f "build/$preset/CMakeCache.txt" ]; then
      cmake --preset "$preset" >"$LOGDIR/wsl_${preset}_configure.log" 2>&1 ||
        stop "configure $preset failed" "$LOGDIR/wsl_${preset}_configure.log"
      configure=0
    fi
    cmake --build --preset "$preset" >"$build_log" 2>&1 || stop "build $preset failed" "$build_log"
    local warnings failed total skipped status
    warnings=$(grep -ci warning "$build_log")
    ctest --preset "$preset" >"$test_log" 2>&1
    status=$?
    failed=$(sed -n 's/.*tests passed, \([0-9]*\) tests failed out of.*/\1/p' "$test_log")
    total=$(sed -n 's/.* tests failed out of \([0-9]*\).*/\1/p' "$test_log")
    skipped=$(grep -c -e '\*\*\*Skipped' -e '\*\*\*Not Run' "$test_log")
    echo "WSL preset $preset configure=$configure build=0 warnings=$warnings" \
      "tests=$((${total:-0} - ${failed:-0}))/${total:-0} skipped=$skipped"
    if [ "$status" -ne 0 ] || [ "$warnings" -ne 0 ] || [ "${total:-0}" -ne "$EXPECTED_TESTS" ]; then
      stop "WSL preset $preset: expected 0 warnings and $EXPECTED_TESTS/$EXPECTED_TESTS" "$test_log"
    fi
  done
  part_end wsl

  # 5. Bitwise comparison of the two builds (section 5.4).
  part
  local bitwise="$LOGDIR/bitwise.txt"
  "$PY" -B python/sbmc_e9.py bitwise "$CONFIG" --workdir "$LOGDIR/bitwise" \
    --win-run "$WIN_RUN" --win-strict-run "$WIN_STRICT_RUN" --win-thresholds "$WIN_THRESHOLDS" \
    --wsl-run build/cpu/apps/sbmc_run --wsl-strict-run build/cpu-strict/apps/sbmc_run \
    --wsl-thresholds build/cpu/tests/tools/sbmc_sa_thresholds >"$bitwise" 2>&1 ||
    stop "the bitwise comparison failed to run" "$bitwise"
  local verdict_sa verdict_dsb
  verdict_sa=$(sed -n 's/^BITWISE_SA: //p' "$bitwise")
  verdict_dsb=$(sed -n 's/^BITWISE_DSB_STRICT: //p' "$bitwise")
  grep '^BITWISE [a-z]' "$bitwise" | cut -c1-160
  echo "BITWISE_SA: ${verdict_sa:-FAIL}"
  echo "BITWISE_DSB_STRICT: ${verdict_dsb:-FAIL}"
  part_end bitwise

  # 6. E9: run (resumable), status, analyze, manifest.
  part
  local e9_log="$LOGDIR/e9_run.log" before=0 e9_status runs skips
  if [ -f "$e9_log" ]; then
    before=$(wc -l <"$e9_log")
  fi
  "$PY" -B python/sbmc_campaign.py run "$CONFIG" --notes "$notes" >>"$e9_log" 2>&1
  e9_status=$?
  runs=$(tail -n +"$((before + 1))" "$e9_log" | grep -c ' done instance=')
  skips=$(tail -n +"$((before + 1))" "$e9_log" | grep -c ' skip instance=')
  [ "$e9_status" -eq 0 ] || stop "E9: the runner exited with $e9_status" "$e9_log"
  "$PY" -B python/sbmc_campaign.py status "$CONFIG" >"$LOGDIR/e9_status.txt" 2>&1 ||
    stop "E9: status failed" "$LOGDIR/e9_status.txt"
  local total done_count
  total=$(field "$LOGDIR/e9_status.txt" STATUS_TOTAL configurations)
  done_count=$(field "$LOGDIR/e9_status.txt" STATUS_TOTAL "done")
  part_end e9
  echo "E9 run: $done_count of $total done (this run: $runs done, $skips skip) in" \
    "$((DURATION[e9] / 60)) min"
  part
  local analysis="$LOGDIR/e9_analyze.txt"
  "$PY" -B python/sbmc_e9.py analyze "$CONFIG" >"$analysis" 2>&1 ||
    stop "E9: analyze failed" "$analysis"
  "$PY" -B python/sbmc_campaign.py manifest "$CONFIG" >"$LOGDIR/e9_manifest.txt" 2>&1 ||
    stop "E9: manifest failed" "$LOGDIR/e9_manifest.txt"
  grep '^E9_TABLE ' "$analysis"
  local verdict_e9=FAIL
  if [ "$total" = "$E9_CONFIGS" ] && [ "$done_count" = "$E9_CONFIGS" ]; then
    verdict_e9=PASS
  fi
  echo "E9: $verdict_e9 ($done_count of $total)"
  local verdict_consistency checks
  verdict_consistency=$(sed -n 's/^E9_CONSISTENCY: //p' "$analysis")
  checks=$(sed -n 's/^E9_CHECK \([^:]*\): \([A-Z]*\).*/\1 \2/p' "$analysis" | paste -sd ';' - |
    sed 's/;/; /g')
  echo "E9_CONSISTENCY: ${verdict_consistency:-FAIL} ($checks)"
  grep '^E9_INFO ' "$analysis" | cut -c1-160
  part_end analyze

  # 7. Summary.
  local outside
  outside=$(git status --short | grep -c -v -E '^.. results/f07/e9/')
  echo "git status: $(git status --short | wc -l) entries, $outside outside results/f07/e9/" \
    "(expected 0)"
  echo "durations (s): checks ${DURATION[checks]} interop ${DURATION[interop]}" \
    "wsl ${DURATION[wsl]} bitwise ${DURATION[bitwise]} e9 ${DURATION[e9]}" \
    "analyze ${DURATION[analyze]} total $(($(date +%s) - start_all))"
  local verdict=PASS
  for v in "$verdict_build" "$verdict_interop" "$verdict_topology" "$verdict_sa" "$verdict_dsb" \
    "$verdict_e9" "$verdict_consistency"; do
    [ "$v" = PASS ] || verdict=FAIL
  done
  echo "F07R4_VERDICT: $verdict (full logs: $LOGDIR/)"
}

case "${1:-}" in
  stage) stage_command ;;
  run) run_command "${2:-}" ;;
  *)
    echo "usage: bash scripts/laptop/f07r4.sh stage | run \"\$NOTES\"" >&2
    exit 2
    ;;
esac
