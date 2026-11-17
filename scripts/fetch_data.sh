#!/usr/bin/env bash
# Checks the committed G-set files and downloads into data/raw/ the data that is not in git: the
# other 58 G-set files (data/raw/gset/) and K2000 (data/raw/k2000/WK2000_1.rud). Every file is
# checked against its SHA-256 before it is kept; data/raw/ is ignored by git and excluded by
# scripts/sync_from_zip.sh.
#
# Usage: scripts/fetch_data.sh [--gset] [--k2000] [--help]   (no option: both parts)
#
# --gset: for each of the 71 names of data/gset/SHA256SUMS, a file present in data/gset/
#   (committed) must have its SHA-256 ("committed ok"); otherwise data/raw/gset/<name> is kept
#   when its SHA-256 is right ("cached"), and downloaded from Stanford when it is missing or
#   wrong ("downloaded").
# --k2000: data/raw/k2000/WK2000_1.rud, with the same logic and the SHA-256 written below.
#
# A download goes to a temporary file in the destination folder and replaces the destination only
# when its SHA-256 is right. A wrong SHA-256 is an error: the temporary file is removed and the
# existing file is left untouched.
#
# Summary on stdout: "gset: <a> committed ok, <b> downloaded, <c> cached", "k2000: downloaded" or
# "k2000: cached" (the parts that ran), then "fetch_data: OK". Exit status 0 on success, 1 at the
# first error, with a message on stderr.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
gset_dir="${repo_root}/data/gset"
gset_sums="${gset_dir}/SHA256SUMS"
raw_gset_dir="${repo_root}/data/raw/gset"
raw_k2000_file="${repo_root}/data/raw/k2000/WK2000_1.rud"

gset_url="https://web.stanford.edu/~yyye/yyye/Gset"
k2000_url="https://raw.githubusercontent.com/haribara/SA-complete-graph/WK2000/WK2000_1.rud"
k2000_sha256="9ed615e5e18726914f12740b7f9bedb6b69676477ed5958fac42c8ab0c252ba7"
gset_files=71

# Temporary file of the download in progress, removed on exit whatever happens.
tmp_file=""
cleanup() {
  if [[ -n "${tmp_file}" ]]; then
    rm -f -- "${tmp_file}"
  fi
}
trap cleanup EXIT

die() {
  printf 'fetch_data: error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: scripts/fetch_data.sh [--gset] [--k2000]

Checks and downloads the data that is not in git, verifying every file against its SHA-256.
Without options, does both parts.

Options:
  --gset      check the committed G-set files (data/gset/) and download the other G-set files
              listed in data/gset/SHA256SUMS into data/raw/gset/ (from ${gset_url}/)
  --k2000     download K2000 into data/raw/k2000/WK2000_1.rud
  -h, --help  show this help
EOF
}

sha256_of() {
  sha256sum -- "$1" | cut -d ' ' -f 1
}

# Downloads <url> into <destination> when its SHA-256 is <expected>, through a temporary file in
# the destination folder; otherwise exits with an error and leaves <destination> untouched.
fetch() {
  local url="$1" destination="$2" expected="$3" folder actual
  folder="$(dirname -- "${destination}")"
  mkdir -p -- "${folder}"
  tmp_file="$(mktemp "${folder}/.$(basename -- "${destination}").download.XXXXXX")"
  printf '==> downloading %s\n' "${url}"
  if ! curl -fsSL --retry 3 -o "${tmp_file}" "${url}"; then
    die "download failed: ${url}"
  fi
  actual="$(sha256_of "${tmp_file}")"
  if [[ "${actual}" != "${expected}" ]]; then
    die "${url}: SHA-256 ${actual}, expected ${expected}: the download was discarded and" \
      "${destination} was not modified"
  fi
  # mktemp creates the file with mode 600: give it the mode of a new file under the umask.
  chmod "$(printf '%o' "$((0666 & ~$(umask)))")" -- "${tmp_file}"
  mv -f -- "${tmp_file}" "${destination}"
  tmp_file=""
}

do_gset=0
do_k2000=0
while (($# > 0)); do
  case "$1" in
    --gset) do_gset=1 ;;
    --k2000) do_k2000=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done
if ((do_gset == 0 && do_k2000 == 0)); then
  do_gset=1
  do_k2000=1
fi

command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found (coreutils)"
command -v curl >/dev/null 2>&1 || die "curl not found (on Ubuntu: sudo apt install curl)"

summary=()

if ((do_gset)); then
  [[ -f "${gset_sums}" ]] || die "${gset_sums} not found"
  # Read and check the whole list before touching any file.
  names=()
  hashes=()
  line_number=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line_number=$((line_number + 1))
    if [[ ! "${line}" =~ ^([0-9a-f]{64})\ \ (G[0-9]+)$ ]]; then
      die "${gset_sums}:${line_number}: not a line '<sha256>  G<number>'"
    fi
    hashes+=("${BASH_REMATCH[1]}")
    names+=("${BASH_REMATCH[2]}")
  done <"${gset_sums}"
  ((${#names[@]} == gset_files)) ||
    die "${gset_sums}: ${#names[@]} lines, ${gset_files} expected"

  committed=0
  downloaded=0
  cached=0
  for index in "${!names[@]}"; do
    name="${names[index]}"
    hash="${hashes[index]}"
    if [[ -e "${gset_dir}/${name}" ]]; then
      [[ "$(sha256_of "${gset_dir}/${name}")" == "${hash}" ]] ||
        die "data/gset/${name} does not match data/gset/SHA256SUMS: the committed G-set files" \
          "must never change (restore them with git)"
      committed=$((committed + 1))
    elif [[ -f "${raw_gset_dir}/${name}" &&
      "$(sha256_of "${raw_gset_dir}/${name}")" == "${hash}" ]]; then
      cached=$((cached + 1))
    else
      fetch "${gset_url}/${name}" "${raw_gset_dir}/${name}" "${hash}"
      downloaded=$((downloaded + 1))
    fi
  done
  summary+=("gset: ${committed} committed ok, ${downloaded} downloaded, ${cached} cached")
fi

if ((do_k2000)); then
  if [[ -f "${raw_k2000_file}" && "$(sha256_of "${raw_k2000_file}")" == "${k2000_sha256}" ]]; then
    summary+=("k2000: cached")
  else
    fetch "${k2000_url}" "${raw_k2000_file}" "${k2000_sha256}"
    summary+=("k2000: downloaded")
  fi
fi

printf '%s\n' "${summary[@]}"
echo "fetch_data: OK"
