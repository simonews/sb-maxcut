#!/usr/bin/env bash
# Copies the commit HEAD for the build on Windows (phase 07, part 4, decision W1). Run from WSL,
# inside the repository:
#
#   scripts/windows/stage.sh [DEST]
#
# DEST (default /mnt/c/Users/Simone/sbmc-win) receives DEST/src, replaced entirely by
# "git archive HEAD", with DEST/src/.sbmc_build_info (git_hash=<HEAD>, dirty=0: read by
# cmake/sbmc_build_info.cmake, since Windows has no git) and data/raw/k2000/WK2000_1.rud when the
# repository has it (test_k2000). The builds of scripts/windows/build.ps1 stay outside src, in
# DEST\build-<preset>, and its logs in DEST\logs. A working tree with changes to tracked files
# (results/ excluded, as for dirty) is reported: the copy is HEAD, not the working tree. Last line:
# "STAGE src=<DEST/src> git_hash=<HEAD> files=<n>".
#
# Exit status: 0 success; 1 error (not a repository, DEST not usable, git archive failed).
set -euo pipefail

die() {
  echo "stage.sh: $*" >&2
  exit 1
}

root="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
cd "$root"
dest="${1:-/mnt/c/Users/Simone/sbmc-win}"
dest="${dest%/}"
case "$dest" in
  /*) ;;
  *) die "DEST must be an absolute path, got '$dest'" ;;
esac
src="$dest/src"

hash="$(git rev-parse HEAD)"
[[ "$hash" =~ ^[0-9a-f]{40}$ ]] || die "git rev-parse HEAD gave '$hash'"
if [ -n "$(git --no-optional-locks status --porcelain --untracked-files=no -- . \
  ':(exclude)results')" ]; then
  echo "stage.sh: WARNING: the working tree has changes to tracked files: the copy is HEAD" \
    "($hash), not the working tree"
fi

mkdir -p "$dest" || die "cannot create $dest"
rm -rf -- "$src"
mkdir -- "$src"
git archive --format=tar HEAD | tar -x -C "$src" || die "git archive HEAD | tar failed"
printf 'git_hash=%s\ndirty=0\n' "$hash" >"$src/.sbmc_build_info"
k2000="data/raw/k2000/WK2000_1.rud"
if [ -f "$k2000" ]; then
  mkdir -p "$src/data/raw/k2000"
  cp -- "$k2000" "$src/$k2000"
else
  echo "stage.sh: WARNING: $k2000 missing (scripts/fetch_data.sh --k2000): test_k2000 will be" \
    "skipped on Windows"
fi
echo "STAGE src=$src git_hash=$hash files=$(find "$src" -type f | wc -l)"
