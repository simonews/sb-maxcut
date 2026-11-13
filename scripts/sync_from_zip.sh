#!/usr/bin/env bash
# Applies a delivery zip (sb-maxcut_FNN_RN.zip) to the working tree of this git repository.
#
# Usage: scripts/sync_from_zip.sh <path/to/sb-maxcut_FNN_RN.zip>
#
# The zip must contain a single top-level folder, sb-maxcut/. Paths listed in .sync-exclude
# are never copied from the zip and never deleted from the working tree. The script shows a
# dry run and asks for confirmation (read from stdin) before changing anything.
#
# Exit codes: 0 applied (or nothing to apply); 1 missing prerequisites or failed checks;
#             2 the zip's .sync-exclude differs from the repository's; 3 cancelled by the user.
set -euo pipefail

readonly ROOT_NAME="sb-maxcut"
readonly MAX_FILE_BYTES=$((5 * 1024 * 1024))
readonly SENTINELS=(CMakeLists.txt CHANGELOG.md .sync-exclude)

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  echo "Usage: scripts/sync_from_zip.sh <path/to/${ROOT_NAME}_FNN_RN.zip>"
}

# Prints one "<pattern><TAB><annotation>" line per rule of an exclude file. Whole-line comments
# ('#' or ';') and blank lines are skipped; a trailing "# word" is returned as the annotation.
parse_excludes() {
  local line pattern annotation
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "${line}" || "${line}" == \#* || "${line}" == \;* ]]; then
      continue
    fi
    if [[ "${line}" =~ ^(.*[^[:space:]])[[:space:]]+#(.*)$ ]]; then
      pattern="${BASH_REMATCH[1]}"
      annotation="${BASH_REMATCH[2]}"
      annotation="${annotation#"${annotation%%[![:space:]]*}"}"
      annotation="${annotation%"${annotation##*[![:space:]]}"}"
    else
      pattern="${line}"
      annotation=""
    fi
    printf '%s\t%s\n' "${pattern}" "${annotation}"
  done <"$1"
}

# Turns an rsync pattern into a representative path for git check-ignore: wildcards become a
# plain character and directory patterns get a file inside them.
probe_path() {
  local path="${1#/}"
  path="${path//\*\*/x}"
  path="${path//\*/x}"
  path="${path//\?/x}"
  if [[ "${path}" == */ ]]; then
    path="${path}probe"
  fi
  printf '%s' "${path}"
}

if (($# != 1)); then
  usage >&2
  exit 1
fi
case "$1" in
  -h | --help)
    usage
    exit 0
    ;;
esac
zip_file="$1"

# 1. Prerequisites.
missing_tools=()
for tool in git unzip rsync; do
  command -v "${tool}" >/dev/null 2>&1 || missing_tools+=("${tool}")
done
if ((${#missing_tools[@]} > 0)); then
  die "missing required tools: ${missing_tools[*]} (on Ubuntu: sudo apt install ${missing_tools[*]})"
fi
[[ -f "${zip_file}" ]] || die "zip file not found: ${zip_file}"

# 2. Repository root.
repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" ||
  die "the current directory is not inside a git working tree"
exclude_file="${repo_root}/.sync-exclude"
[[ -f "${exclude_file}" ]] || die "${exclude_file} not found: first sync: use the manual procedure"

# 3. Clean working tree (untracked files that are not ignored count as changes).
tree_status="$(git -C "${repo_root}" status --porcelain)"
if [[ -n "${tree_status}" ]]; then
  {
    echo "error: the working tree is not clean:"
    printf '%s\n' "${tree_status}"
    echo "Code changes enter this repository only through deliveries: commit, stash or discard"
    echo "the local changes listed above, then run the script again."
  } >&2
  exit 1
fi

# 4. Extraction into a temporary folder, removed on exit in any case.
tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "${tmp_dir}"' EXIT
extract_dir="${tmp_dir}/extract"
mkdir -p -- "${extract_dir}" "${tmp_dir}/empty"
unzip -q -n "${zip_file}" -d "${extract_dir}" </dev/null || die "cannot extract ${zip_file}"
src_dir="${extract_dir}/${ROOT_NAME}"

# rsync reads every line of an exclude file as a pattern, trailing comments included, so the
# annotated .sync-exclude is reduced to bare patterns before being passed to rsync.
rsync_excludes="${tmp_dir}/sync-exclude.patterns"
parse_excludes "${exclude_file}" | cut -f1 >"${rsync_excludes}"

# 5. Structure checks: every problem is reported, not only the first one.
problems=()
shopt -s nullglob dotglob
top_entries=("${extract_dir}"/*)
shopt -u nullglob dotglob
if ((${#top_entries[@]} != 1)); then
  names=()
  for entry in "${top_entries[@]}"; do
    names+=("$(basename -- "${entry}")")
  done
  problems+=("the zip must contain exactly one top-level entry, the folder ${ROOT_NAME}/" \
    "(found ${#top_entries[@]}: ${names[*]:-none})")
fi

structure_ok=1
if [[ ! -d "${src_dir}" || -L "${src_dir}" ]]; then
  problems+=("the folder ${ROOT_NAME}/ is missing at the top of the zip")
  structure_ok=0
fi

if ((structure_ok)); then
  for sentinel in "${SENTINELS[@]}"; do
    [[ -f "${src_dir}/${sentinel}" ]] || problems+=("missing sentinel file ${ROOT_NAME}/${sentinel}")
  done

  all_paths="${tmp_dir}/paths.all"
  kept_paths="${tmp_dir}/paths.kept"
  rsync -a --dry-run --out-format='%n' "${src_dir}/" "${tmp_dir}/empty/" | LC_ALL=C sort >"${all_paths}"
  rsync -a --dry-run --out-format='%n' --exclude-from="${rsync_excludes}" \
    "${src_dir}/" "${tmp_dir}/empty/" | LC_ALL=C sort >"${kept_paths}"
  while IFS= read -r excluded; do
    problems+=("path matches a .sync-exclude pattern: ${ROOT_NAME}/${excluded}")
  done < <(LC_ALL=C comm -23 "${all_paths}" "${kept_paths}")
fi

while IFS= read -r -d '' big_file; do
  problems+=("file larger than 5 MiB: ${big_file#"${extract_dir}/"} ($(stat -c %s -- "${big_file}") bytes)")
done < <(find "${extract_dir}" -type f -size +"${MAX_FILE_BYTES}"c -print0)

if ((${#problems[@]} > 0)); then
  {
    echo "error: ${zip_file} failed the structure checks:"
    printf '  - %s\n' "${problems[@]}"
  } >&2
  exit 1
fi

# 6. The zip must carry the same .sync-exclude as the repository.
if ! cmp -s -- "${exclude_file}" "${src_dir}/.sync-exclude"; then
  {
    echo "error: the zip's .sync-exclude differs from the repository's:"
    diff -u --label "repository/.sync-exclude" --label "zip/${ROOT_NAME}/.sync-exclude" \
      -- "${exclude_file}" "${src_dir}/.sync-exclude" || true
    echo "Review the change and apply it by hand (.sync-exclude and .gitignore, then commit);"
    echo "after that, run the script again."
  } >&2
  exit 2
fi

# 7. Consistency with git: every pattern without "# tracked" or "# internal" must be ignored.
not_ignored=()
while IFS=$'\t' read -r pattern annotation; do
  if [[ "${annotation}" == "tracked" || "${annotation}" == "internal" ]]; then
    continue
  fi
  probe="$(probe_path "${pattern}")"
  if ! git -C "${repo_root}" check-ignore -q --no-index -- "${probe}"; then
    not_ignored+=("${pattern} (probe path: ${probe})")
  fi
done < <(parse_excludes "${exclude_file}")
if ((${#not_ignored[@]} > 0)); then
  {
    echo "error: .sync-exclude entries not ignored by git (add them to .gitignore, or annotate"
    echo "them with '# tracked' if git must version them):"
    printf '  - %s\n' "${not_ignored[@]}"
  } >&2
  exit 1
fi

# 8. Dry run. Content is compared with checksums and modification times are not copied:
# changed files get the current time, so incremental builds always see them as modified.
rsync_options=(-a --no-times --checksum --delete --itemize-changes
  "--exclude-from=${rsync_excludes}")
changes="$(rsync "${rsync_options[@]}" --dry-run "${src_dir}/" "${repo_root}/")"
if [[ -z "${changes}" ]]; then
  echo "Nothing to apply: the working tree already matches ${zip_file}."
  exit 0
fi

new_files=0
modified_files=0
deleted_files=0
while IFS= read -r item; do
  if [[ "${item}" == "*deleting "* ]]; then
    [[ "${item}" == */ ]] || deleted_files=$((deleted_files + 1))
  elif [[ "${item}" =~ ^[\>c][fL]\+{9} ]]; then
    new_files=$((new_files + 1))
  elif [[ "${item}" =~ ^[\>c.][fL] ]]; then
    modified_files=$((modified_files + 1))
  fi
done <<<"${changes}"

echo "Dry run (rsync --itemize-changes) for ${zip_file}:"
printf '%s\n' "${changes}"
echo
echo "Summary: ${new_files} new, ${modified_files} modified, ${deleted_files} deleted file(s)."

# 9. Confirmation.
printf 'Apply these changes? [y/N] '
answer=""
read -r answer || true
echo
case "${answer}" in
  y | Y | yes | Yes | YES) ;;
  *)
    echo "Cancelled: nothing was changed."
    exit 3
    ;;
esac

# 10. Apply and show the resulting git status.
rsync "${rsync_options[@]}" "${src_dir}/" "${repo_root}/" >/dev/null
echo "Applied ${zip_file}. git status --short:"
git -C "${repo_root}" status --short
