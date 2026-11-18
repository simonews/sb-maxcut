#!/usr/bin/env bash
# Automated test of cmake/sbmc_build_info.cmake, run with cmake -P in a temporary git repository:
# the git hash and the dirty flag of the build information, which ignores the folder results/ of
# the repository root (a results folder elsewhere and the files results_* of the root still
# count), and the header rewritten only when its content changes.
#
# Usage: tests/scripts/test_build_info.sh <cmake> <path/to/cmake/sbmc_build_info.cmake>
# Output: one line "BUILD_INFO <case>: PASS|FAIL (<detail>)" per case, "N/A" for a case that
# cannot apply on this machine, and "BUILD_INFO_VERDICT: PASS|FAIL" at the end.
# Exit codes: 0 every applicable case passed; 1 at least one case failed, or wrong usage;
#             77 git is not installed.
set -euo pipefail

if (($# != 2)); then
  echo "usage: $0 <cmake> <path/to/sbmc_build_info.cmake>" >&2
  exit 1
fi
cmake_bin="$1"
script="$2"

if ! command -v git >/dev/null 2>&1; then
  echo "SKIP: git is not installed"
  exit 77
fi
git_bin="$(command -v git)"
[[ -f "${script}" ]] || {
  echo "error: script not found: ${script}" >&2
  exit 1
}

# Isolate git from the user's configuration (global excludes, hooks, default branch name).
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME="sbmc test" GIT_AUTHOR_EMAIL="sbmc-test@example.invalid"
export GIT_COMMITTER_NAME="sbmc test" GIT_COMMITTER_EMAIL="sbmc-test@example.invalid"

work_dir="$(mktemp -d)"
trap 'rm -rf -- "${work_dir}"' EXIT

repo="${work_dir}/repo"
header="${work_dir}/out/build_info.hpp"
log="${work_dir}/cmake.log"

failures=()
passes=0
not_applicable=0

# Runs the script with SBMC_SOURCE_DIR=$1 and SBMC_GIT=$2 (empty: no git); output in ${log}.
run_script() {
  "${cmake_bin}" "-DSBMC_SOURCE_DIR=$1" "-DSBMC_OUTPUT=${header}" "-DSBMC_GIT=$2" \
    "-DSBMC_COMPILER=GNU 13.3.0" -DSBMC_BUILD_TYPE=Release -P "${script}" >"${log}" 2>&1
}

# dirty and git_hash of the generated header (empty when missing).
header_dirty() {
  [[ -f "${header}" ]] || return 0
  sed -n 's/^inline constexpr int dirty = \(-\{0,1\}[0-9][0-9]*\);$/\1/p' "${header}"
}
header_hash() {
  [[ -f "${header}" ]] || return 0
  sed -n 's/^inline constexpr std::string_view git_hash = "\(.*\)";$/\1/p' "${header}"
}

# Changes to tracked files as git reports them, on one line.
tracked_changes() {
  local lines
  lines="$(git -C "${repo}" status --porcelain --untracked-files=no)"
  if [[ -z "${lines}" ]]; then
    printf 'none'
  else
    printf '%s' "${lines}" | tr '\n' ',' | sed 's/,/, /g'
  fi
}

# Records the outcome of a case: name, pass|fail|na, then the words of the detail.
record() {
  local name="$1" outcome="$2" detail="${*:3}"
  case "${outcome}" in
    pass)
      passes=$((passes + 1))
      echo "BUILD_INFO ${name}: PASS (${detail})"
      ;;
    na)
      not_applicable=$((not_applicable + 1))
      echo "BUILD_INFO ${name}: N/A (${detail})"
      ;;
    *)
      failures+=("${name}")
      echo "BUILD_INFO ${name}: FAIL (${detail})"
      sed 's/^/    | /' "${log}"
      ;;
  esac
}

# Runs the script on the repository with git and checks dirty against $2 and git_hash against
# HEAD.
check_dirty() {
  local name="$1" expected="$2"
  local status=0 dirty hash changes
  changes="$(tracked_changes)"
  run_script "${repo}" "${git_bin}" || status=$?
  dirty="$(header_dirty)"
  hash="$(header_hash)"
  if ((status == 0)) && [[ "${dirty}" == "${expected}" && "${hash}" == "${head}" ]]; then
    record "${name}" pass "dirty ${dirty}; tracked changes: ${changes}"
  else
    record "${name}" fail "expected dirty ${expected} and git_hash ${head}, got dirty '${dirty}'" \
      "and git_hash '${hash}', cmake exit ${status}; tracked changes: ${changes}"
  fi
}

# --- Repository -----------------------------------------------------------------------------

mkdir -p -- "${repo}/src" "${repo}/results" "${repo}/sub/results"
printf 'int main() { return 0; }\n' >"${repo}/src/a.cpp"
printf 'instance,best_cut\nG1,11624\n' >"${repo}/results/run.csv"
printf 'nested results\n' >"${repo}/sub/results/x.txt"
printf 'old results\n' >"${repo}/results_old.csv"
git -C "${repo}" init -q -b main
git -C "${repo}" add -A
git -C "${repo}" commit -q -m "initial"
head="$(git -C "${repo}" rev-parse HEAD)"

# --- Cases ----------------------------------------------------------------------------------

# 1. Clean working tree.
check_dirty clean 0

# 2. A row appended to a tracked file of results/.
printf 'G2,11620\n' >>"${repo}/results/run.csv"
check_dirty results-modified 0

# 3. Untracked files in results/ and in the root, results/run.csv still modified.
printf 'new\n' >"${repo}/results/new.csv"
printf 'untracked\n' >"${repo}/untracked.txt"
check_dirty results-untracked 0

# 4. A tracked file of results/ deleted from the working tree, then restored.
rm -- "${repo}/results/run.csv"
check_dirty results-deleted 0
git -C "${repo}" checkout -q -- results/run.csv

# 5. A tracked file in a results folder that is not the one of the root, then restored.
printf 'changed\n' >>"${repo}/sub/results/x.txt"
check_dirty nested-results 1
git -C "${repo}" checkout -q -- sub/results/x.txt

# 6. A tracked file of the root whose name starts with "results", then restored.
printf 'changed\n' >>"${repo}/results_old.csv"
check_dirty prefix-name 1
git -C "${repo}" checkout -q -- results_old.csv

# 7. A source file modified.
printf '// local change\n' >>"${repo}/src/a.cpp"
check_dirty source-modified 1

# 8. The same change added to the index.
git -C "${repo}" add -- src/a.cpp
check_dirty source-staged 1

# 9. The change removed from the index and from the working tree, and a row appended again to
#    results/run.csv.
git -C "${repo}" checkout -q HEAD -- src/a.cpp
printf 'G3,11622\n' >>"${repo}/results/run.csv"
check_dirty source-restored 0

# 10. Two consecutive runs without changes: the second one writes nothing.
run_status=0
run_script "${repo}" "${git_bin}" || run_status=$?
cp -- "${header}" "${work_dir}/header.before"
run_script "${repo}" "${git_bin}" || run_status=$?
if ((run_status == 0)) && ! grep -qF "build_info.hpp:" "${log}" &&
  cmp -s -- "${header}" "${work_dir}/header.before"; then
  record unchanged-header pass "second run: no message, header identical byte for byte"
else
  record unchanged-header fail "cmake exit ${run_status}; the second run printed the message or" \
    "changed the header"
fi

# 11. No git (SBMC_GIT empty).
run_status=0
run_script "${repo}" "" || run_status=$?
dirty="$(header_dirty)"
hash="$(header_hash)"
if ((run_status == 0)) && [[ "${dirty}" == "-1" && "${hash}" == "unknown" ]]; then
  record no-git pass "dirty -1, git_hash unknown"
else
  record no-git fail "expected dirty -1 and git_hash unknown, got dirty '${dirty}' and git_hash" \
    "'${hash}', cmake exit ${run_status}"
fi

# 12. A folder that is not a git repository.
empty_dir="${work_dir}/not-a-repository"
mkdir -p -- "${empty_dir}"
if git -C "${empty_dir}" rev-parse --git-dir >/dev/null 2>&1; then
  record not-a-repository na "${empty_dir} is inside a git repository on this machine"
else
  run_status=0
  run_script "${empty_dir}" "${git_bin}" || run_status=$?
  dirty="$(header_dirty)"
  hash="$(header_hash)"
  if ((run_status == 0)) && [[ "${dirty}" == "-1" && "${hash}" == "unknown" ]]; then
    record not-a-repository pass "dirty -1, git_hash unknown"
  else
    record not-a-repository fail "expected dirty -1 and git_hash unknown, got dirty '${dirty}'" \
      "and git_hash '${hash}', cmake exit ${run_status}"
  fi
fi

echo
echo "test_build_info: ${passes} passed, ${#failures[@]} failed, ${not_applicable} not applicable"
if ((${#failures[@]} > 0)); then
  echo "BUILD_INFO_VERDICT: FAIL"
  exit 1
fi
echo "BUILD_INFO_VERDICT: PASS"
