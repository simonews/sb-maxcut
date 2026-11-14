#!/usr/bin/env bash
# Builds, tests and reports the project in one go on a Kaggle notebook with two Tesla T4.
#
# Usage: scripts/kaggle_run.sh [--ref LABEL] [--out DIR]
#
# Run it from the root of a clone of the repository (README.md, section "Running on Kaggle
# (2 x T4)", has the notebook cell). Steps, each with its own log:
#   1. context: UTC date, commit, git status --porcelain (expected empty), nproc, GPUs seen by
#      nvidia-smi;
#   2. scripts/setup_cuda_pypi.sh --presets t4 (toolkit from PyPI, presets t4 and t4-strict);
#   3. for t4-strict, then t4: configure, build from scratch (build/<preset> is deleted first),
#      ctest;
#   4. build/t4/tests/smoke/smoke_simt;
#   5. scripts/env_report.sh --preset t4;
#   6. summary.txt, printed on screen too.
# Everything goes to DIR/sbmc_<LABEL>_<YYYYmmdd-HHMMSS>/ (UTC time), archived as a .tar.gz next
# to that folder. --ref labels the run (default: short hash of HEAD); characters other than
# letters, digits, '.', '_' and '-' become '-' in file names. --out defaults to /kaggle/working
# when it exists, ./kaggle-out otherwise.
#
# Exit codes: 0 everything passed and smoke_simt printed "SIMT_VERDICT: PASS"; 1 a step or a test
# failed (also for a wrong command line); 2 no usable GPU (smoke_simt exited with 77) and nothing
# failed. A test that ctest reports as skipped (test_sync_from_zip without rsync, unzip or zip;
# smoke_simt without a GPU) is not a failure: summary.txt lists it.
#
# Besides build/ and results/env/, the clone gets the CMakeUserPresets.json written by step 2
# (ignored by git). No environment variable is printed or saved: only the environment report
# holds two of them, KAGGLE_DOCKER_IMAGE and KAGGLE_KERNEL_RUN_TYPE. Before the archive is made,
# the files of the run are searched for the values of the environment variables whose name
# contains TOKEN, SECRET, KEY or PASSWORD (values are never printed): if one is found, the run
# fails and no archive is made.
set -euo pipefail

usage() {
  echo "Usage: scripts/kaggle_run.sh [--ref LABEL] [--out DIR]"
}

die() {
  echo "kaggle_run.sh: error: $*" >&2
  exit 1
}

invocation_dir="$(pwd -P)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"

ref=""
out=""
while (($# > 0)); do
  case "$1" in
    --ref | --out)
      if (($# < 2)) || [[ -z "$2" ]]; then
        usage >&2
        die "$1 needs a non-empty value"
      fi
      if [[ "$1" == "--ref" ]]; then ref="$2"; else out="$2"; fi
      shift 2
      ;;
    --ref=* | --out=*)
      if [[ -z "${1#*=}" ]]; then
        usage >&2
        die "${1%%=*} needs a non-empty value"
      fi
      if [[ "$1" == --ref=* ]]; then ref="${1#*=}"; else out="${1#*=}"; fi
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument '$1'"
      ;;
  esac
done

# Output folder: absolute path, created if needed (a relative --out is relative to the folder the
# script is launched from).
if [[ -z "${out}" ]]; then
  if [[ -d /kaggle/working ]]; then
    out="/kaggle/working"
  else
    out="./kaggle-out"
  fi
fi
[[ "${out}" == /* ]] || out="${invocation_dir}/${out}"
mkdir -p -- "${out}" || die "cannot create the output folder ${out}"
out="$(cd -- "${out}" && pwd -P)"

cd -- "${repo_root}"

if [[ -z "${ref}" ]]; then
  ref="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
fi
label="$(printf '%s' "${ref}" | tr -c 'A-Za-z0-9._-' '-')"
stamp="$(date -u +%Y%m%d-%H%M%S)"
name="sbmc_${label}_${stamp}"
run_dir="${out}/${name}"
archive="${out}/${name}.tar.gz"
mkdir -- "${run_dir}" || die "cannot create ${run_dir}"

readonly NA="n/a"
readonly PRESETS=(t4-strict t4)
readonly SMOKE_BIN="build/t4/tests/smoke/smoke_simt"
SECONDS=0

failed=()      # failed steps, for the summary
failed_logs=() # their logs, whose last lines are printed at the end
no_gpu=0

# Runs a command with stdout and stderr in a file of the run folder; returns its exit status.
run_logged() {
  local log="$1"
  shift
  local status=0
  "$@" >"${run_dir}/${log}" 2>&1 </dev/null || status=$?
  return "${status}"
}

step_t0=0
step_begin() {
  step_t0=${SECONDS}
  printf '==> %s\n' "$1"
}

# Records the outcome of a step in the global variable outcome ("ok" or "FAILED (exit status N)")
# and prints it; a status other than 0 adds the step and its log to the failures.
outcome=""
step_outcome() {
  local status="$1" what="$2" log="$3"
  local elapsed=$((SECONDS - step_t0))
  if ((status == 0)); then
    printf '    ok (%d s)\n' "${elapsed}"
    outcome="ok"
  else
    printf '    FAILED, exit status %d (%d s), log: %s\n' "${status}" "${elapsed}" "${log}"
    failed+=("${what}")
    failed_logs+=("${log}")
    outcome="FAILED (exit status ${status})"
  fi
}

# Reads a field of vendor.sbmc in CMakeUserPresets.json ("toolkit-root", "venv"), or nothing.
presets_field() {
  local value=""
  if [[ -f CMakeUserPresets.json ]]; then
    value="$(sed -n "s/^[[:space:]]*\"$1\":[[:space:]]*\"\(.*\)\",\{0,1\}[[:space:]]*\$/\1/p" \
      CMakeUserPresets.json)"
  fi
  printf '%s' "${value%%$'\n'*}"
}

# Prints the value of a dynamic-section entry ("RPATH" or "RUNPATH") of readelf -d output, with
# several entries joined by ':'; prints nothing when there is no such entry.
dynamic_entry() {
  local values
  values="$(sed -n "s/^.*($1)[[:space:]]*Library [a-z]*: \[\(.*\)\][[:space:]]*\$/\1/p" <<<"$2")"
  printf '%s' "${values//$'\n'/:}"
}

# Prints "<status> <name>" for every test of a ctest log; status is passed, failed or skipped.
ctest_results() {
  awk '
    /^ *[0-9]+\/[0-9]+ +Test +#[0-9]+: / {
      line = $0
      sub(/^ *[0-9]+\/[0-9]+ +Test +#[0-9]+: +/, "", line)
      test_name = line
      sub(/ .*/, "", test_name)
      if (line ~ /\*\*\*Skipped/) status = "skipped"
      else if (line ~ /\*\*\*/) status = "failed"
      else if (line ~ /Passed/) status = "passed"
      else status = "failed"
      print status " " test_name
    }' "$1"
}

# Prints "0" or "<count> (<name>, <name>...)" for the tests of a ctest log with a given status.
ctest_list() {
  ctest_results "$1" | awk -v s="$2" '
    $1 == s { names = names sep $2; sep = ", "; n++ }
    END { if (n == 0) print "0"; else printf "%d (%s)\n", n, names }'
}

# Indents a block of text by two spaces; an empty block becomes "  n/a".
indent() {
  if [[ -z "$1" ]]; then
    printf '  %s\n' "${NA}"
  else
    printf '%s\n' "$1" | sed 's/^/  /'
  fi
}

# Searches the run folder for the value of every environment variable whose name contains TOKEN,
# SECRET, KEY or PASSWORD (in any case). Only lines of at least 8 characters of a value are
# searched; values go to grep through a pipe, never as arguments, and are never printed.
# Sets scan_searched (variables searched), scan_found (names of the variables found) and
# scan_errors (searches that grep could not complete).
scan_searched=0
scan_found=()
scan_errors=0
secret_scan() {
  local var value status
  while IFS= read -r var; do
    case "${var^^}" in
      *TOKEN* | *SECRET* | *KEY* | *PASSWORD*) ;;
      *) continue ;;
    esac
    value="${!var-}"
    ((${#value} >= 8)) || continue
    scan_searched=$((scan_searched + 1))
    status=0
    grep -rqF -f <(printf '%s\n' "${value}" | awk 'length($0) >= 8') -- "${run_dir}" || status=$?
    case "${status}" in
      0) scan_found+=("${var}") ;;
      1) ;;
      *) scan_errors=$((scan_errors + 1)) ;;
    esac
  done < <(compgen -e)
}

echo "kaggle_run.sh: run ${name}"
echo "    clone : ${repo_root}"
echo "    output: ${run_dir}"

# ---------------------------------------------------------------------------------------------
# 1. Context
# ---------------------------------------------------------------------------------------------
step_begin "[1/6] context (log: 1_context.log)"
context_status=0
date_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
commit="$(git rev-parse HEAD 2>&1)" || {
  context_status=1
  commit="${NA} (git rev-parse HEAD failed: ${commit})"
}
git_status="$(git status --porcelain 2>&1)" || {
  context_status=1
  git_status="${NA} (git status failed: ${git_status})"
}
nproc_value="$(nproc 2>/dev/null || echo "${NA}")"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="0 (nvidia-smi not found)"
elif gpu_lines="$(nvidia-smi -L 2>/dev/null)"; then
  gpu_count="$(grep -c '^GPU [0-9]' <<<"${gpu_lines}" || true) (nvidia-smi -L)"
else
  gpu_count="0 (nvidia-smi -L failed)"
fi
if ((context_status != 0)); then
  git_status_summary="${NA} (see 1_context.log)"
elif [[ -z "${git_status}" ]]; then
  git_status_summary="clean"
else
  git_status_summary="NOT CLEAN ($(wc -l <<<"${git_status}") entries, see 1_context.log)"
fi
{
  echo "date_utc: ${date_utc}"
  echo "ref: ${ref}"
  echo "commit: ${commit}"
  echo "git_status_porcelain:"
  if [[ -z "${git_status}" ]]; then echo "  (empty)"; else indent "${git_status}"; fi
  echo "nproc: ${nproc_value}"
  echo "gpus_visible: ${gpu_count}"
} >"${run_dir}/1_context.log"
step_outcome "${context_status}" "context" "1_context.log"
context_outcome="${outcome}"

# ---------------------------------------------------------------------------------------------
# 2. Toolkit and presets
# ---------------------------------------------------------------------------------------------
step_begin "[2/6] scripts/setup_cuda_pypi.sh --presets t4 (log: 2_setup.log)"
setup_status=0
run_logged 2_setup.log scripts/setup_cuda_pypi.sh --presets t4 || setup_status=$?
step_outcome "${setup_status}" "setup" "2_setup.log"
setup_outcome="${outcome}"

# ---------------------------------------------------------------------------------------------
# 3. Configure, build from scratch and test the two presets
# ---------------------------------------------------------------------------------------------
declare -A configure_outcome build_outcome ctest_outcome
for preset in "${PRESETS[@]}"; do
  configure_outcome[${preset}]="not run (setup failed)"
  build_outcome[${preset}]="not run"
  ctest_outcome[${preset}]="not run"
  ((setup_status == 0)) || continue

  log="3_${preset}_configure.log"
  step_begin "[3/6] ${preset}: configure from scratch (log: ${log})"
  rm -rf -- "build/${preset}"
  status=0
  run_logged "${log}" cmake --preset "${preset}" || status=$?
  step_outcome "${status}" "${preset} configure" "${log}"
  configure_outcome[${preset}]="${outcome} (${log})"
  if ((status != 0)); then
    build_outcome[${preset}]="not run (configure failed)"
    ctest_outcome[${preset}]="not run (configure failed)"
    continue
  fi

  log="3_${preset}_build.log"
  step_begin "[3/6] ${preset}: build (log: ${log})"
  status=0
  run_logged "${log}" cmake --build --preset "${preset}" || status=$?
  step_outcome "${status}" "${preset} build" "${log}"
  build_outcome[${preset}]="${outcome} (${log})"
  if ((status != 0)); then
    ctest_outcome[${preset}]="not run (build failed)"
    continue
  fi

  log="3_${preset}_ctest.log"
  step_begin "[3/6] ${preset}: ctest (log: ${log})"
  status=0
  run_logged "${log}" ctest --preset "${preset}" || status=$?
  step_outcome "${status}" "${preset} ctest" "${log}"
  ctest_outcome[${preset}]="${outcome} (${log})"
done

# ---------------------------------------------------------------------------------------------
# 4. smoke_simt on every visible GPU
# ---------------------------------------------------------------------------------------------
step_begin "[4/6] ${SMOKE_BIN} (output: 4_smoke_simt.txt)"
smoke_file="${run_dir}/4_smoke_simt.txt"
if [[ -x "${SMOKE_BIN}" ]]; then
  smoke_status=0
  run_logged 4_smoke_simt.txt "${SMOKE_BIN}" || smoke_status=$?
  if ((smoke_status == 77)); then
    no_gpu=1
    smoke_outcome="skipped: no usable GPU (exit status 77)"
    printf '    skipped: no usable GPU (exit status 77)\n'
  elif ((smoke_status == 0)) && grep -qx 'SIMT_VERDICT: PASS' "${smoke_file}"; then
    step_outcome 0 "smoke_simt" "4_smoke_simt.txt"
    smoke_outcome="${outcome}"
  elif ((smoke_status == 0)); then
    step_outcome 1 "smoke_simt" "4_smoke_simt.txt"
    smoke_outcome="FAILED (exit status 0 without SIMT_VERDICT: PASS)"
  else
    step_outcome "${smoke_status}" "smoke_simt" "4_smoke_simt.txt"
    smoke_outcome="${outcome}"
  fi
else
  echo "not run: ${SMOKE_BIN} does not exist (the t4 build failed or did not run)" >"${smoke_file}"
  step_outcome 1 "smoke_simt" "4_smoke_simt.txt"
  smoke_outcome="FAILED (not run: binary missing)"
fi

# ---------------------------------------------------------------------------------------------
# 5. Environment report
# ---------------------------------------------------------------------------------------------
step_begin "[5/6] scripts/env_report.sh --preset t4 (log: 5_env_report.log)"
report_status=0
run_logged 5_env_report.log scripts/env_report.sh --preset t4 || report_status=$?
report_path=""
if ((report_status == 0)); then
  report_path="$(tail -n 1 "${run_dir}/5_env_report.log")"
  if [[ "${report_path}" == "${repo_root}/results/env/"*.txt && -f "${report_path}" ]]; then
    cp -- "${report_path}" "${run_dir}/"
  else
    echo "kaggle_run.sh: the last line of the output is not a report in results/env/" \
      >>"${run_dir}/5_env_report.log"
    report_status=1
  fi
fi
step_outcome "${report_status}" "env_report" "5_env_report.log"
report_outcome="${outcome}"
if ((report_status == 0)); then
  report_name="$(basename -- "${report_path}")"
  report_line="${report_name} (copy of results/env/${report_name} in the clone)"
else
  report_line="${NA} (see 5_env_report.log)"
fi

# ---------------------------------------------------------------------------------------------
# 6. Summary, secret scan, archive
# ---------------------------------------------------------------------------------------------
step_begin "[6/6] summary.txt, secret scan, archive"
toolkit_root="$(presets_field toolkit-root)"
venv="$(presets_field venv)"
nvcc_version="${NA}"
if [[ -n "${toolkit_root}" && -x "${toolkit_root}/bin/nvcc" ]] &&
  nvcc_output="$("${toolkit_root}/bin/nvcc" --version 2>&1)"; then
  nvcc_version="$(grep -m 1 'release' <<<"${nvcc_output}" || true)"
  nvcc_version="${nvcc_version:-${NA}}"
fi
nvjitlink_version="${NA}"
if [[ -n "${venv}" && -x "${venv}/bin/python" ]]; then
  nvjitlink_version="$("${venv}/bin/python" -I -c \
    'import importlib.metadata as m; print(m.version("nvidia-nvjitlink"))' 2>/dev/null)" ||
    nvjitlink_version="${NA}"
fi

smoke_versions="$(grep -E '^(cuda_runtime_version|cuda_driver_version|cublas_version|cublaslt_version|cuda_runtime_compiled|cuda_runtime_loaded|cublas_compiled|cublas_loaded|device_count|device[0-9]+_name|device[0-9]+_compute_capability): ' \
  "${smoke_file}" || true)"
smoke_verdicts="$(grep -E '^(device[0-9]+_verdict|[A-Z][A-Z0-9_]*|skip_reason): ' "${smoke_file}" ||
  true)"
smoke_p2p="$(grep -E '^p2p_access_[0-9]+_to_[0-9]+: ' "${smoke_file}" || true)"
# RPATH and RUNPATH of smoke_simt: the CUDA libraries must come from the RPATH (the lib-shim).
smoke_rpath="${NA} (binary missing)"
smoke_runpath="${NA} (binary missing)"
if [[ -f "${SMOKE_BIN}" ]]; then
  if ! command -v readelf >/dev/null 2>&1; then
    smoke_rpath="${NA} (readelf not found)"
    smoke_runpath="${smoke_rpath}"
  elif dynamic_output="$(readelf -d "${SMOKE_BIN}" 2>&1)"; then
    smoke_rpath="$(dynamic_entry RPATH "${dynamic_output}")"
    smoke_runpath="$(dynamic_entry RUNPATH "${dynamic_output}")"
    smoke_rpath="${smoke_rpath:-none}"
    smoke_runpath="${smoke_runpath:-none}"
  else
    smoke_rpath="${NA} (readelf -d failed)"
    smoke_runpath="${smoke_rpath}"
  fi
fi
smoke_libraries=""
if [[ -x "${SMOKE_BIN}" ]] && ldd_output="$(ldd "${SMOKE_BIN}" 2>&1)"; then
  smoke_libraries="$(grep -E 'lib(cudart|cublas|cublasLt|nvJitLink|cuda)\.so' <<<"${ldd_output}" |
    sed -e 's/^[[:space:]]*//' -e 's/ (0x[0-9a-f]*)$//' || true)"
fi
if [[ -z "${smoke_p2p}" ]]; then
  if grep -q '^device_count: [01]$' "${smoke_file}"; then
    smoke_p2p="${NA} (fewer than 2 devices)"
  elif ((no_gpu)); then
    smoke_p2p="${NA} (no usable GPU)"
  fi
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  topo_p2p="$(nvidia-smi topo -p2p r 2>&1)" || topo_p2p="${NA} (nvidia-smi topo -p2p r failed)"
else
  topo_p2p="${NA} (nvidia-smi not found)"
fi

notes=()
for preset in "${PRESETS[@]}"; do
  log="${run_dir}/3_${preset}_ctest.log"
  if [[ -f "${log}" ]] && grep -qx 'skipped test_sync_from_zip' <<<"$(ctest_results "${log}")"; then
    notes+=("test_sync_from_zip skipped on ${preset}: rsync, unzip or zip missing (not a failure)")
  fi
done
if [[ "${git_status_summary}" == NOT\ CLEAN* ]]; then
  notes+=("the clone was not clean at the start: the commit does not describe all the code built")
fi

summary="${run_dir}/summary.txt"
{
  echo "# sbmc Kaggle run summary"
  echo "run: ${name}"
  echo "ref: ${ref}"
  echo "date_utc: ${date_utc}"
  echo "commit: ${commit}"
  echo "git_status: ${git_status_summary}"
  echo "nproc: ${nproc_value}"
  echo "gpus_visible: ${gpu_count}"
  echo "context: ${context_outcome} (1_context.log)"
  echo
  echo "[toolkit]"
  echo "setup: ${setup_outcome} (2_setup.log)"
  echo "nvcc: ${nvcc_version}"
  echo "nvidia_nvjitlink: ${nvjitlink_version}"
  for preset in "${PRESETS[@]}"; do
    echo
    echo "[preset ${preset}]"
    echo "configure: ${configure_outcome[${preset}]}"
    echo "build: ${build_outcome[${preset}]}"
    echo "ctest: ${ctest_outcome[${preset}]}"
    log="${run_dir}/3_${preset}_ctest.log"
    if [[ -f "${log}" ]]; then
      echo "tests_passed: $(ctest_list "${log}" passed)"
      echo "tests_failed: $(ctest_list "${log}" failed)"
      echo "tests_skipped: $(ctest_list "${log}" skipped)"
    else
      echo "tests_passed: ${NA}"
      echo "tests_failed: ${NA}"
      echo "tests_skipped: ${NA}"
    fi
  done
  echo
  echo "[smoke_simt]"
  echo "binary: ${SMOKE_BIN}"
  echo "rpath (readelf -d): ${smoke_rpath}"
  echo "runpath (readelf -d): ${smoke_runpath}"
  echo "result: ${smoke_outcome}"
  echo "output: 4_smoke_simt.txt"
  echo "versions:"
  indent "${smoke_versions}"
  echo "verdicts:"
  indent "${smoke_verdicts}"
  echo "libraries (ldd):"
  indent "${smoke_libraries}"
  echo
  echo "[p2p]"
  echo "cuda_device_can_access_peer (smoke_simt):"
  indent "${smoke_p2p}"
  echo "nvidia_smi_topo_p2p_r:"
  indent "${topo_p2p}"
  echo
  echo "[env_report]"
  echo "file: ${report_line}"
  echo "result: ${report_outcome} (5_env_report.log)"
  echo
  echo "[notes]"
  if ((${#notes[@]} == 0)); then
    echo "none"
  else
    printf -- '- %s\n' "${notes[@]}"
  fi
} >"${summary}"

secret_scan
if ((${#scan_found[@]} > 0)); then
  failed+=("secret scan")
  scan_line="FOUND the value of ${scan_found[*]} in the run folder: no archive made"
elif ((scan_errors > 0)); then
  failed+=("secret scan")
  scan_line="FAILED (grep could not search the run folder ${scan_errors} times)"
else
  scan_line="clean (${scan_searched} variables searched)"
fi

if ((${#failed[@]} > 0)); then
  exit_code=1
  verdict="FAIL"
elif ((no_gpu)); then
  exit_code=2
  verdict="NO_GPU"
else
  exit_code=0
  verdict="PASS"
fi
if ((${#scan_found[@]} > 0)); then
  archive_line="not made (secret scan)"
else
  archive_line="${archive}"
fi
failed_line="none"
if ((${#failed[@]} > 0)); then
  failed_line="$(printf '%s, ' "${failed[@]}")"
  failed_line="${failed_line%, }"
fi
{
  echo
  echo "[result]"
  echo "secret_scan: ${scan_line}"
  echo "failed: ${failed_line}"
  echo "elapsed_s: ${SECONDS}"
  echo "exit_code: ${exit_code}"
  echo "verdict: ${verdict}"
  echo "archive: ${archive_line}"
} >>"${summary}"

if ((${#scan_found[@]} > 0)); then
  echo "kaggle_run.sh: the value of ${scan_found[*]} was found in ${run_dir}." >&2
  echo "kaggle_run.sh: nothing is printed and no archive is made; inspect and delete the folder." >&2
  exit 1
fi

if ! tar -czf "${archive}" -C "${out}" "${name}"; then
  echo "kaggle_run.sh: error: cannot create ${archive}" >&2
  exit 1
fi

for log in "${failed_logs[@]}"; do
  echo
  echo "---- last 40 lines of ${log} ----"
  tail -n 40 -- "${run_dir}/${log}"
done
echo
cat -- "${summary}"
echo
echo "Run folder: ${run_dir}"
echo "Archive   : ${archive}"
exit "${exit_code}"
