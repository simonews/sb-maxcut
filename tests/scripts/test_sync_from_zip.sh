#!/usr/bin/env bash
# Automated test of scripts/sync_from_zip.sh, run in temporary git repositories that use the
# project's own .sync-exclude and .gitignore.
#
# Usage: tests/scripts/test_sync_from_zip.sh [path/to/sync_from_zip.sh]
# Exit codes: 0 every scenario gave the expected result; 1 at least one scenario failed;
#             77 a required tool (rsync, unzip, zip, git) is missing.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd -- "${script_dir}/../.." && pwd -P)"
sync_script="${1:-${project_root}/scripts/sync_from_zip.sh}"

for tool in rsync unzip zip git; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "SKIP: '${tool}' is not installed (on Ubuntu: sudo apt install rsync unzip zip git)"
    exit 77
  fi
done
[[ -f "${sync_script}" ]] || {
  echo "error: sync script not found: ${sync_script}" >&2
  exit 1
}

# Isolate git from the user's configuration (global excludes could change what is ignored).
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME="sbmc test" GIT_AUTHOR_EMAIL="sbmc-test@example.invalid"
export GIT_COMMITTER_NAME="sbmc test" GIT_COMMITTER_EMAIL="sbmc-test@example.invalid"

work_dir="$(mktemp -d)"
trap 'rm -rf -- "${work_dir}"' EXIT

failures=()
passes=0

# --- Fixtures -------------------------------------------------------------------------------

# Creates a repository in $1 with tracked files and files in excluded folders.
make_repo() {
  local repo="$1"
  mkdir -p -- "${repo}/src" "${repo}/results" "${repo}/build"
  cp -- "${project_root}/.sync-exclude" "${project_root}/.gitignore" "${repo}/"
  printf 'cmake_minimum_required(VERSION 3.28)\n' >"${repo}/CMakeLists.txt"
  printf '# Changelog\n' >"${repo}/CHANGELOG.md"
  printf 'keep v1\n' >"${repo}/src/keep.txt"
  printf 'modify v1\n' >"${repo}/src/modify.txt"
  printf 'remove v1\n' >"${repo}/src/remove.txt"
  printf 'user results\n' >"${repo}/results/y"
  printf 'build artifact\n' >"${repo}/build/x"
  printf '{ "local": true }\n' >"${repo}/CMakeUserPresets.json"
  git -C "${repo}" init -q -b main
  git -C "${repo}" add -A
  git -C "${repo}" commit -q -m "initial"
}

# Creates in $1 the content of a valid delivery for the repository $2: src/modify.txt changed,
# src/remove.txt removed, everything else identical, no excluded paths.
make_delivery_tree() {
  local stage="$1" repo="$2"
  local root="${stage}/sb-maxcut"
  mkdir -p -- "${root}/src"
  cp -- "${repo}/.sync-exclude" "${repo}/.gitignore" "${repo}/CMakeLists.txt" \
    "${repo}/CHANGELOG.md" "${root}/"
  cp -- "${repo}/src/keep.txt" "${root}/src/"
  printf 'modify v2\n' >"${root}/src/modify.txt"
}

# Zips the given entries of the folder $2 into $1.
make_zip() {
  local zip_file="$1" stage="$2"
  shift 2
  (cd -- "${stage}" && zip -q -r -X "${zip_file}" "$@")
}

# Runs the sync script in repository $1 with zip $2, answering $3; output in $1.log.
run_sync() {
  local repo="$1" zip_file="$2" answer="$3"
  local status=0
  (cd -- "${repo}" && printf '%s\n' "${answer}" | bash "${sync_script}" "${zip_file}") \
    >"${repo}.log" 2>&1 || status=$?
  printf '%s' "${status}"
}

# Records the outcome of a scenario: name, expected and actual exit code, optional message
# that must appear in the output, optional extra check result (0 = ok).
check() {
  local name="$1" expected="$2" actual="$3" log="$4" pattern="${5:-}" extra="${6:-0}"
  local ok=1
  [[ "${actual}" == "${expected}" ]] || ok=0
  if [[ -n "${pattern}" ]] && ! grep -qF -- "${pattern}" "${log}"; then
    ok=0
  fi
  [[ "${extra}" == "0" ]] || ok=0
  if ((ok)); then
    passes=$((passes + 1))
    echo "PASS ${name} (exit ${actual})"
  else
    local expectation="exit ${expected}"
    [[ -z "${pattern}" ]] || expectation+=" and message \"${pattern}\""
    [[ "${extra}" == "0" ]] || expectation+=" and a passing content check"
    failures+=("${name}")
    echo "FAIL ${name}: expected ${expectation}, got exit ${actual}"
    sed 's/^/    | /' "${log}"
  fi
}

# Prints a fingerprint of a working tree (paths and contents, .git excluded).
tree_fingerprint() {
  (cd -- "$1" && find . -path ./.git -prune -o -type f -print0 | LC_ALL=C sort -z |
    xargs -0 sha256sum)
}

# --- Scenarios ------------------------------------------------------------------------------

# (a) Valid zip: one tracked file removed, one modified; excluded paths untouched.
scenario_a() {
  local base="${work_dir}/a" status extra=0
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" y)"
  [[ ! -e "${base}/repo/src/remove.txt" ]] || extra=1
  [[ "$(cat "${base}/repo/src/modify.txt")" == "modify v2" ]] || extra=1
  [[ "$(cat "${base}/repo/src/keep.txt")" == "keep v1" ]] || extra=1
  [[ "$(cat "${base}/repo/results/y")" == "user results" ]] || extra=1
  [[ "$(cat "${base}/repo/build/x")" == "build artifact" ]] || extra=1
  [[ "$(cat "${base}/repo/CMakeUserPresets.json")" == '{ "local": true }' ]] || extra=1
  check "(a) valid zip applied, excluded paths intact" 0 "${status}" "${base}/repo.log" \
    "Summary: 0 new, 1 modified, 1 deleted file(s)." "${extra}"
}

# (b) Working tree with uncommitted changes: a modified tracked file, then an untracked file.
scenario_b() {
  local base="${work_dir}/b" status
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  printf 'local edit\n' >>"${base}/repo/src/keep.txt"
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" y)"
  check "(b1) modified tracked file" 1 "${status}" "${base}/repo.log" "working tree is not clean"

  make_repo "${base}/repo2"
  printf 'new\n' >"${base}/repo2/src/untracked.txt"
  status="$(run_sync "${base}/repo2" "${base}/delivery.zip" y)"
  check "(b2) untracked file" 1 "${status}" "${base}/repo2.log" "working tree is not clean"
}

# (c) Wrong top level: two entries at the top, then no sb-maxcut folder at all.
scenario_c() {
  local base="${work_dir}/c" status
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  printf 'stray\n' >"${base}/stage/extra.txt"
  make_zip "${base}/two-entries.zip" "${base}/stage" sb-maxcut extra.txt
  status="$(run_sync "${base}/repo" "${base}/two-entries.zip" y)"
  check "(c1) two top-level entries" 1 "${status}" "${base}/repo.log" "exactly one top-level entry"

  mkdir -p -- "${base}/stage2"
  cp -R -- "${base}/stage/sb-maxcut" "${base}/stage2/project"
  make_zip "${base}/no-root.zip" "${base}/stage2" project
  status="$(run_sync "${base}/repo" "${base}/no-root.zip" y)"
  check "(c2) no sb-maxcut folder" 1 "${status}" "${base}/repo.log" "folder sb-maxcut/ is missing"
}

# (d) A file larger than 5 MiB.
scenario_d() {
  local base="${work_dir}/d" status
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  head -c $((6 * 1024 * 1024)) /dev/zero >"${base}/stage/sb-maxcut/src/big.bin"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" y)"
  check "(d) file larger than 5 MiB" 1 "${status}" "${base}/repo.log" "file larger than 5 MiB"
}

# (e) An excluded path inside the zip.
scenario_e() {
  local base="${work_dir}/e" status
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  mkdir -p -- "${base}/stage/sb-maxcut/build"
  printf 'artifact\n' >"${base}/stage/sb-maxcut/build/a"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" y)"
  check "(e) excluded path in the zip" 1 "${status}" "${base}/repo.log" \
    "path matches a .sync-exclude pattern: sb-maxcut/build/a"
}

# (f) The zip's .sync-exclude differs from the repository's.
scenario_f() {
  local base="${work_dir}/f" status
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  printf 'extra/\n' >>"${base}/stage/sb-maxcut/.sync-exclude"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" y)"
  check "(f) different .sync-exclude" 2 "${status}" "${base}/repo.log" "differs from the repository"
}

# (g) The user answers "n": nothing changes.
scenario_g() {
  local base="${work_dir}/g" status before after extra=0
  make_repo "${base}/repo"
  make_delivery_tree "${base}/stage" "${base}/repo"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  before="$(tree_fingerprint "${base}/repo")"
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" n)"
  after="$(tree_fingerprint "${base}/repo")"
  [[ "${before}" == "${after}" ]] || extra=1
  check "(g) answer 'n', nothing changed" 3 "${status}" "${base}/repo.log" "Cancelled" "${extra}"
}

# (h) A .sync-exclude entry that git does not ignore.
scenario_h() {
  local base="${work_dir}/h" status
  make_repo "${base}/repo"
  printf 'scratch/\n' >>"${base}/repo/.sync-exclude"
  git -C "${base}/repo" commit -q -am "add an entry that git does not ignore"
  make_delivery_tree "${base}/stage" "${base}/repo"
  make_zip "${base}/delivery.zip" "${base}/stage" sb-maxcut
  status="$(run_sync "${base}/repo" "${base}/delivery.zip" y)"
  check "(h) .sync-exclude entry not ignored by git" 1 "${status}" "${base}/repo.log" \
    "not ignored by git"
}

scenario_a
scenario_b
scenario_c
scenario_d
scenario_e
scenario_f
scenario_g
scenario_h

echo
if ((${#failures[@]} > 0)); then
  echo "test_sync_from_zip: ${#failures[@]} failed, ${passes} passed. Failed: ${failures[*]}"
  exit 1
fi
echo "test_sync_from_zip: all ${passes} checks passed (scenarios a-h)"
