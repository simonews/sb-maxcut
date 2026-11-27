#!/usr/bin/env python3
"""python/sbmc_campaign.py with [[series]] (phase 07, part 4, E9; F07-C4, section 5.3), and the
options of sbmc_run that belong to it on Linux.

Run by CTest as:
  <python> -I -B test_sbmc_campaign_series.py --pyroot <root>/python --root <root>
           --sbmc-run <sbmc_run> --sbmc-spectrum <sbmc_spectrum> --workdir <build>/tests/series
Checks:
1. sbmc_run on Linux: --affinity p1 and --print-topology exit with 2 and a message "available
   only on Windows"; --version exits with 0 and writes "sbmc_run git_hash=... dirty=...
   compiler=... build_type=...";
2. validation: 10 wrong configurations with series (no platform, an unknown platform, windows
   without affinity, linux with affinity, an unknown affinity, threads not increasing, a base that
   is not an earlier series, a base without 1 thread, threads in [campaign] together with
   series, a stage with series): exit status 2, a message that names the file, no file written;
3. --dry-run (the repository is a git copy in --workdir with the files of the test; the Windows
   copy <dest>/src of scripts/windows/stage.sh and <dest>/build-win-cpu/apps/sbmc_run.exe, a
   wrapper, are faked; the fake wslpath gives C:\\sbmc-win\\... under <dest> and
   \\\\wsl.localhost\\Fake<path> elsewhere): one line per point of every series (2 + 3 = 5 per
   configuration of the grid), in the order of the grid, then of the series, then of the threads,
   with the ids <id>_<series>_T<threads>; the Windows series with --affinity, the instance and
   --best-known (target best-known) from the copy (C:\\sbmc-win\\src\\...), --out and
   --replica-out in C:\\sbmc-win\\runs\\<campaign>\\..., no argument that starts with \\\\;
   the Linux series without them and with the paths of the configuration; nothing written;
4. a real run (F07-R4e): the Windows series is a wrapper of sbmc_run (it records its arguments
   and its folder, refuses an argument that starts with \\\\, maps the Windows paths back, drops
   --affinity and answers --version with the commit of HEAD), the Linux series is sbmc_run; stale
   work files of the first Windows configuration are deleted before its run: 10 complete
   configurations, the arguments recorded as in the dry run, the wrapper started in
   <dest>/src, no file left in <dest>/runs; a second run skips every configuration; status: 10
   done;
5. the Windows copy (F07-R4e, F07-C4e section 5.2): a run whose second Windows configuration
   exits with 1 stops with exit status 1, its work files stay in <dest>/runs/<campaign>/ and its
   final files are not written, then the next run skips the first and completes the other 9; a
   file missing in the copy, an executable whose --version gives another commit and a <dest>
   that wslpath -w makes a UNC path stop the run with exit status 1 and a message that names
   the file, the two commits or the UNC path, before any configuration runs.
The files go to --workdir only (emptied first). Needs git (otherwise SKIP).

Exit status: 0 passed, 1 failed, 77 when the executables are missing. The last line is
"SERIES_VERDICT: PASS", "SERIES_VERDICT: FAIL" or "SERIES_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77
DRIVE = "C:\\sbmc-win"

FAKE_WSLPATH = r"""#!/usr/bin/env bash
# Fake wslpath of the test: wslpath -w <dest>/x -> C:\sbmc-win\x (a Windows drive), any other
# absolute path -> \\wsl.localhost\Fake\abs\path (a UNC path, as the real one for WSL files).
[ "$1" = "-w" ] && [ -n "$2" ] || { echo "fake wslpath: usage" >&2; exit 1; }
case "$2" in /*) ;; *) echo "fake wslpath: $2 is not absolute" >&2; exit 1 ;; esac
case "$2" in
  "@DEST@") printf 'C:\\sbmc-win\n' ;;
  "@DEST@"/*) printf 'C:\\sbmc-win%s\n' "$(printf '%s' "${2#"@DEST@"}" | tr '/' '\\')" ;;
  *) printf '\\\\wsl.localhost\\Fake%s\n' "$(printf '%s' "$2" | tr '/' '\\')" ;;
esac
"""

WRAPPER = r"""#!/usr/bin/env bash
# Fake sbmc_run.exe of the test: answers --version with the commit of HEAD (SERIES_HASH
# overrides it); records its arguments and its folder; refuses an argument that starts with \\;
# maps C:\sbmc-win back to <dest>, drops --affinity and runs sbmc_run; exits with 1 after the run
# of the configuration SERIES_FAIL_TAG.
if [ "$1" = "--version" ]; then
  echo "sbmc_run git_hash=${SERIES_HASH:-@HASH@} dirty=0 compiler=fake build_type=Release"
  exit 0
fi
printf '%s\n' "$*" >> "@LOG@"
printf '%s\n' "$PWD" >> "@LOG@.cwd"
args=()
skip=0
previous=""
tag=""
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  if [ "$a" = "--affinity" ]; then skip=1; continue; fi
  case "$a" in
    '\\'*) echo "fake sbmc_run.exe: UNC path $a" >&2; exit 3 ;;
    'C:\sbmc-win\'*) a="@DEST@/${a#'C:\sbmc-win\'}"; a="${a//\\//}" ;;
  esac
  [ "$previous" = "--tag" ] && tag="$a"
  previous="$a"
  args+=("$a")
done
"@REAL@" "${args[@]}" || exit $?
if [ -n "${SERIES_FAIL_TAG:-}" ] && [ "$tag" = "$SERIES_FAIL_TAG" ]; then exit 1; fi
"""


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"SERIES {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def campaign_text(output: Path, run: str, spectrum: str, series: str, instance: str,
                  campaign_extra: str = "", extra: str = "", name: str = "series") -> str:
    return (f'[campaign]\nname = "{name}"\noutput = "{output}"\nsbmc_run = "{run}"\n'
            f'sbmc_spectrum = "{spectrum}"\nreplicas = 8\nreps = 2\nwarmup = 0\nseed = 1\n'
            f"{campaign_extra}\n{series}\n{instance}\n"
            '[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\nb = [2.0, 3.0]\n'
            "steps = { s24 = [10] }\n" + extra)


def series_text(wrapper: Path, run: str) -> str:
    return (f'[[series]]\nid = "win-p"\nsbmc_run = "{wrapper}"\nplatform = "windows"\n'
            'affinity = "p"\nthreads = [1, 2]\n\n'
            f'[[series]]\nid = "lin"\nsbmc_run = "{run}"\nplatform = "linux"\n'
            "threads = [1, 2, 3]\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of the series of the runner.")
    parser.add_argument("--pyroot", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sbmc-run", required=True)
    parser.add_argument("--sbmc-spectrum", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    print(f"python: {platform.python_version()} ({sys.executable})")
    root = Path(args.root).resolve()
    run_exe = str(Path(args.sbmc_run).resolve())
    spectrum = str(Path(args.sbmc_spectrum).resolve())
    tool = Path(args.pyroot).resolve() / "sbmc_campaign.py"
    missing = [p for p in (run_exe, spectrum, str(tool)) if not Path(p).exists()]
    if shutil.which("git") is None:
        missing.append("git")
    if missing:
        print(f"missing: {', '.join(missing)}")
        print("SERIES_VERDICT: SKIP")
        return EXIT_SKIP
    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    (work / "bin").mkdir(parents=True)
    report = Report()
    files = ["tests/data/sparse_24_unit.rudy", "data/best_known.csv"]
    # The repository of the runner: a git copy with the files of the test (HEAD is checked).
    repo = work / "repo"
    for name in files:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, repo / name)
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.org"]
    for step in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "test"]):
        subprocess.run([*git, *step], check=True, capture_output=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    # The Windows folder: the copy of stage.sh and the wrapper as the executable of win-cpu.
    dest = work / "win"
    for name in files:
        (dest / "src" / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, dest / "src" / name)
    (dest / "src" / ".sbmc_build_info").write_text(f"git_hash={head}\ndirty=0\n",
                                                   encoding="ascii")
    wslpath = work / "bin" / "wslpath"
    wslpath.write_text(FAKE_WSLPATH.replace("@DEST@", str(dest)), encoding="utf-8")
    wslpath.chmod(0o755)
    log = work / "wrapper.log"
    wrapper = dest / "build-win-cpu" / "apps" / "sbmc_run.exe"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(WRAPPER.replace("@LOG@", str(log)).replace("@REAL@", run_exe)
                       .replace("@HASH@", head).replace("@DEST@", str(dest)), encoding="utf-8")
    wrapper.chmod(0o755)
    env = dict(os.environ, PATH=f"{work / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    rudy = repo / "tests" / "data" / "sparse_24_unit.rudy"

    def runner(*arguments: str, extra_env: dict[str, str] | None = None
               ) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(tool), "--root", str(repo),
                               *arguments], capture_output=True, text=True, check=False,
                              timeout=600, env=dict(env, **(extra_env or {})))

    def calls() -> list[str]:
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    # 1. sbmc_run on Linux.
    complete = ["--instance", str(rudy), "--algo", "sa", "--replicas", "4", "--sweeps", "10",
                "--beta0", "0.1", "--beta1", "1"]
    for option in (["--affinity", "p1", *complete], ["--print-topology"]):
        result = subprocess.run([run_exe, *option], capture_output=True, text=True, check=False)
        report.check(f"sbmc_run {option[0]} on Linux: exit 2, Windows only",
                     result.returncode == 2 and "available only on Windows" in result.stderr,
                     f"exit {result.returncode}: {result.stderr.strip().splitlines()[0]}")
    result = subprocess.run([run_exe, "--version"], capture_output=True, text=True, check=False)
    report.check("sbmc_run --version", result.returncode == 0 and re.fullmatch(
        r"sbmc_run git_hash=\S+ dirty=-?[01] compiler=.+ build_type=\S+\n", result.stdout)
        is not None, result.stdout.strip())

    instance = f'[[instance]]\nid = "s24"\nargs = ["--instance", "{rudy}"]\ntarget = 40\n'
    good_series = series_text(wrapper, run_exe)

    # 2. Validation.
    output = work / "invalid"
    cases = {
        "no platform": good_series.replace('platform = "windows"\n', "", 1),
        "unknown platform": good_series.replace('platform = "windows"', 'platform = "mac"', 1),
        "windows without affinity": good_series.replace('affinity = "p"\n', "", 1),
        "linux with affinity": good_series + 'affinity = "p"\n',
        "unknown affinity": good_series.replace('affinity = "p"', 'affinity = "q"', 1),
        "threads not increasing": good_series.replace("threads = [1, 2, 3]",
                                                      "threads = [1, 3, 3]", 1),
        "base not an earlier series": good_series + 'base = "nope"\n',
        "base without 1 thread": good_series.replace("threads = [1, 2]\n",
                                                     "threads = [2]\n", 1) + 'base = "win-p"\n',
    }
    texts = {name: campaign_text(output, run_exe, spectrum, series, instance)
             for name, series in cases.items()}
    texts["threads with series"] = campaign_text(output, run_exe, spectrum, good_series, instance,
                                                 campaign_extra="threads = 2")
    texts["stage with series"] = campaign_text(
        output, run_exe, spectrum, good_series, instance,
        extra='\n[[stage]]\nname = "t"\ninstance = "s24"\n'
              'runs = [ { method = "sa", a = 1.0, b = 2.0, steps = 10 } ]\n')
    for k, (name, text) in enumerate(texts.items()):
        path = work / f"bad_{k}.toml"
        path.write_text(text, encoding="utf-8")
        result = runner("run", str(path), "--notes", "x", "--dry-run")
        ok = (result.returncode == 2 and result.stderr.startswith("sbmc_campaign: ")
              and str(path) in result.stderr and not output.exists())
        report.check(f"validation, {name}: exit 2, no file", ok,
                     f"exit {result.returncode}: {result.stderr.strip()}")

    # 3. Dry run.
    output = work / "campaign"
    config = work / "series.toml"
    best_known = instance.replace("target = 40", 'target = "best-known"').replace(
        'id = "s24"', 'id = "s24b"')
    text = campaign_text(output, run_exe, spectrum, good_series, instance)
    config.write_text(text, encoding="utf-8")
    result = runner("run", str(config), "--notes", "dry", "--dry-run")
    lines = result.stdout.splitlines()
    commands = [shlex.split(line.split(": ", 1)[1]) for line in lines]
    expected_ids = [f"sa_a1.0_b{b}_S10_{s}_T{t}" for b in ("2.0", "3.0")
                    for s, ts in (("win-p", (1, 2)), ("lin", (1, 2, 3))) for t in ts]
    ids = [c[c.index("--tag") + 1] for c in commands]
    report.check("dry run: 10 lines in the order grid, series, threads",
                 result.returncode == 0 and ids == expected_ids and not output.exists(),
                 f"exit {result.returncode}, {len(lines)} lines {result.stderr.strip()}")
    problems = []
    for c, config_id in zip(commands, ids):
        windows = "_win-p_" in config_id
        threads = config_id.rsplit("_T", 1)[1]
        if c[c.index("--threads") + 1] != threads:
            problems.append(f"{config_id}: --threads")
        paths = [c[c.index(o) + 1] for o in ("--instance", "--out", "--replica-out")]
        if windows:
            if c[0] != str(wrapper) or c[c.index("--affinity") + 1] != "p":
                problems.append(f"{config_id}: executable or --affinity")
            expected = [f"{DRIVE}\\src\\tests\\data\\sparse_24_unit.rudy",
                        f"{DRIVE}\\runs\\series\\s24\\sa\\{config_id}.csv",
                        f"{DRIVE}\\runs\\series\\raw\\s24\\sa\\{config_id}_replicas.csv"]
            if paths != expected or any(a.startswith("\\\\") for a in c):
                problems.append(f"{config_id}: paths {paths}")
        else:
            if c[0] != run_exe or "--affinity" in c:
                problems.append(f"{config_id}: executable or --affinity")
            if paths != [str(rudy), str(output / "s24" / "sa" / f"{config_id}.csv"),
                         str(output / "raw" / "s24" / "sa" / f"{config_id}_replicas.csv")]:
                problems.append(f"{config_id}: paths {paths}")
    report.check("dry run: --affinity, inputs from the copy and outputs in runs/ for win-p only, "
                 "no argument starting with \\\\", not problems, "; ".join(problems[:3]))
    config_bk = work / "series_bk.toml"
    config_bk.write_text(campaign_text(output, run_exe, spectrum, good_series,
                                       best_known).replace("s24 = [10]", "s24b = [10]"),
                         encoding="utf-8")
    result = runner("run", str(config_bk), "--notes", "dry", "--dry-run")
    first = shlex.split(result.stdout.splitlines()[0].split(": ", 1)[1]) if result.stdout else []
    bk = first[first.index("--best-known") + 1] if "--best-known" in first else ""
    report.check("dry run, target best-known: --best-known from the copy",
                 bk == f"{DRIVE}\\src\\data\\best_known.csv", bk or result.stderr.strip())

    # 4. Real run, then a second run. Stale work files of the first Windows configuration.
    stale = [dest / "runs" / "series" / "s24" / "sa" / f"{expected_ids[0]}.csv",
             dest / "runs" / "series" / "raw" / "s24" / "sa" / f"{expected_ids[0]}_replicas.csv"]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n", encoding="utf-8")
    result = runner("run", str(config), "--notes", "real")
    done = [line for line in result.stdout.splitlines() if " done " in line]
    dry = [" ".join(c[1:]).replace("--notes dry", "--notes real") for c in commands
           if "_win-p_" in c[c.index("--tag") + 1]]
    report.check("real run: 10 configurations done", result.returncode == 0 and len(done) == 10,
                 f"exit {result.returncode}, {len(done)} done {result.stderr.strip()[-200:]}")
    report.check("real run: the wrapper got the arguments of the dry run", calls() == dry,
                 f"{len(calls())} calls")
    folders = set((work / "wrapper.log.cwd").read_text(encoding="utf-8").splitlines())
    left = [p for p in (dest / "runs").rglob("*") if p.is_file()]
    finals = [output / "s24" / "sa" / f"{expected_ids[0]}.csv",
              output / "raw" / "s24" / "sa" / f"{expected_ids[0]}_replicas.csv"]
    texts_final = [p.read_text(encoding="utf-8") if p.exists() else "stale" for p in finals]
    report.check("real run: wrapper in <dest>/src, outputs moved, stale work files deleted",
                 folders == {str(dest / "src")} and not left
                 and not any("stale" in x for x in texts_final),
                 f"folders {sorted(folders)}, {len(left)} files left")
    result = runner("run", str(config), "--notes", "real")
    skips = [line for line in result.stdout.splitlines() if " skip " in line]
    report.check("second run: every configuration skip", result.returncode == 0 and
                 len(skips) == 10 and len(calls()) == 4, f"{len(skips)} skip")
    result = runner("status", str(config))
    report.check("status: 10 done", "STATUS_TOTAL configurations=10 done=10 " in result.stdout,
                 result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr)

    # 5. The Windows copy: a failure, then the resumed run; missing file, commit, UNC.
    fail_output = work / "fails"
    fail_config = work / "fails.toml"
    fail_config.write_text(campaign_text(fail_output, run_exe, spectrum, good_series, instance,
                                         name="fails"), encoding="utf-8")
    failing = expected_ids[1]
    before = len(calls())
    result = runner("run", str(fail_config), "--notes", "fail",
                    extra_env={"SERIES_FAIL_TAG": failing})
    kept = [dest / "runs" / "fails" / "s24" / "sa" / f"{failing}.csv",
            dest / "runs" / "fails" / "raw" / "s24" / "sa" / f"{failing}_replicas.csv"]
    final = [fail_output / "s24" / "sa" / f"{failing}.csv",
             fail_output / "raw" / "s24" / "sa" / f"{failing}_replicas.csv"]
    report.check("exit 1 of a Windows configuration: runner exit 1, work files kept, no final "
                 "files", result.returncode == 1 and all(p.is_file() for p in kept)
                 and not any(p.exists() for p in final) and len(calls()) == before + 2
                 and "its files stay in" in result.stdout,
                 f"exit {result.returncode}, calls {len(calls()) - before}")
    result = runner("run", str(fail_config), "--notes", "fail")
    lines = result.stdout.splitlines()
    report.check("resumed run: 1 skip, 9 done, the failed configuration complete",
                 result.returncode == 0 and len([x for x in lines if " skip " in x]) == 1
                 and len([x for x in lines if " done " in x]) == 9
                 and all(p.is_file() for p in final) and not any(p.exists() for p in kept),
                 f"exit {result.returncode}")

    def stopped(name: str, expected: str, extra_env: dict[str, str] | None = None,
                series: str = good_series) -> None:
        path = work / f"{name}.toml"
        out = work / name
        path.write_text(campaign_text(out, run_exe, spectrum, series, instance, name=name),
                        encoding="utf-8")
        start = len(calls())
        result = runner("run", str(path), "--notes", name, extra_env=extra_env)
        message = result.stdout + result.stderr
        report.check(f"{name}: exit 1, nothing run, message", result.returncode == 1
                     and len(calls()) == start and expected in message
                     and not (out / "s24").exists(),
                     f"exit {result.returncode}: {message.strip().splitlines()[-1:]}")

    hidden = dest / "src" / "tests" / "data" / "sparse_24_unit.rudy"
    hidden.rename(hidden.with_name("hidden"))
    stopped("missing", "tests/data/sparse_24_unit.rudy is missing in the Windows copy")
    hidden.with_name("hidden").rename(hidden)
    stopped("stale", f"git_hash={'0' * 40}, HEAD is {head}", {"SERIES_HASH": "0" * 40})
    unc = work / "unc"
    shutil.copytree(dest, unc, symlinks=True)
    unc_series = series_text(unc / "build-win-cpu" / "apps" / "sbmc_run.exe", run_exe)
    stopped("unc_run", "not a path on a Windows drive", series=unc_series)

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"SERIES_VERDICT: {verdict}")
    return EXIT_PASS if verdict == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
