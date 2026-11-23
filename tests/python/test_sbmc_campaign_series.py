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
3. --dry-run with a fake wslpath (\\\\wsl.localhost\\Fake<path>): one line per point of every
   series (2 + 3 = 5 per configuration of the grid), in the order of the grid, then of the
   series, then of the threads, with the ids <id>_<series>_T<threads>; the Windows series with
   --affinity, absolute Windows paths for the instance, --out, --replica-out and --best-known
   (target best-known), the Linux series without them and with the paths of the configuration;
   nothing written;
4. a real run: the Windows series is a wrapper of sbmc_run (it records its arguments, maps the
   Windows paths back and drops --affinity), the Linux series is sbmc_run: 10 complete
   configurations, the arguments recorded as in the dry run; a second run skips every
   configuration; status: 10 done.
The files go to --workdir only (emptied first).

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
FAKE_PREFIX = "\\\\wsl.localhost\\Fake"

FAKE_WSLPATH = r"""#!/usr/bin/env bash
# Fake wslpath of the test: wslpath -w /abs/path -> \\wsl.localhost\Fake\abs\path
[ "$1" = "-w" ] && [ -n "$2" ] || { echo "fake wslpath: usage" >&2; exit 1; }
case "$2" in /*) ;; *) echo "fake wslpath: $2 is not absolute" >&2; exit 1 ;; esac
printf '\\\\wsl.localhost\\Fake%s\n' "$(printf '%s' "$2" | tr '/' '\\')"
"""

WRAPPER = r"""#!/usr/bin/env bash
# Fake sbmc_run.exe of the test: records its arguments, maps the Windows paths back, drops
# --affinity and runs sbmc_run.
printf '%s\n' "$*" >> "@LOG@"
args=()
skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  if [ "$a" = "--affinity" ]; then skip=1; continue; fi
  case "$a" in
    '\\wsl.localhost\Fake'*) a="${a#'\\wsl.localhost\Fake'}"; a="${a//\\//}" ;;
  esac
  args+=("$a")
done
exec "@REAL@" "${args[@]}"
"""


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"SERIES {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def campaign_text(output: Path, run: str, spectrum: str, series: str, instance: str,
                  campaign_extra: str = "", extra: str = "") -> str:
    return (f'[campaign]\nname = "series"\noutput = "{output}"\nsbmc_run = "{run}"\n'
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
    if missing:
        print(f"missing: {', '.join(missing)}")
        print("SERIES_VERDICT: SKIP")
        return EXIT_SKIP
    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    (work / "bin").mkdir(parents=True)
    report = Report()
    wslpath = work / "bin" / "wslpath"
    wslpath.write_text(FAKE_WSLPATH, encoding="utf-8")
    wslpath.chmod(0o755)
    log = work / "wrapper.log"
    wrapper = work / "bin" / "sbmc_run.exe"
    wrapper.write_text(WRAPPER.replace("@LOG@", str(log)).replace("@REAL@", run_exe),
                       encoding="utf-8")
    wrapper.chmod(0o755)
    env = dict(os.environ, PATH=f"{work / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    rudy = root / "tests" / "data" / "sparse_24_unit.rudy"

    def runner(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(tool), "--root", str(root),
                               *arguments], capture_output=True, text=True, check=False,
                              timeout=600, env=env)

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
            if any(not p.startswith(FAKE_PREFIX + "\\") or "/" in p for p in paths):
                problems.append(f"{config_id}: paths {paths}")
            if paths[0] != FAKE_PREFIX + str(rudy).replace("/", "\\"):
                problems.append(f"{config_id}: instance {paths[0]}")
        else:
            if c[0] != run_exe or "--affinity" in c:
                problems.append(f"{config_id}: executable or --affinity")
            if paths != [str(rudy), str(output / "s24" / "sa" / f"{config_id}.csv"),
                         str(output / "raw" / "s24" / "sa" / f"{config_id}_replicas.csv")]:
                problems.append(f"{config_id}: paths {paths}")
    report.check("dry run: --affinity and the Windows paths of the series win-p only",
                 not problems, "; ".join(problems[:3]))
    config_bk = work / "series_bk.toml"
    config_bk.write_text(campaign_text(output, run_exe, spectrum, good_series,
                                       best_known).replace("s24 = [10]", "s24b = [10]"),
                         encoding="utf-8")
    result = runner("run", str(config_bk), "--notes", "dry", "--dry-run")
    first = shlex.split(result.stdout.splitlines()[0].split(": ", 1)[1]) if result.stdout else []
    bk = first[first.index("--best-known") + 1] if "--best-known" in first else ""
    report.check("dry run, target best-known: --best-known with the Windows path of the table",
                 bk == FAKE_PREFIX + str(root / "data" / "best_known.csv").replace("/", "\\"),
                 bk or result.stderr.strip())

    # 4. Real run, then a second run.
    result = runner("run", str(config), "--notes", "real")
    done = [line for line in result.stdout.splitlines() if " done " in line]
    recorded = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    dry = [" ".join(c[1:]).replace("--notes dry", "--notes real") for c in commands
           if "_win-p_" in c[c.index("--tag") + 1]]
    report.check("real run: 10 configurations done", result.returncode == 0 and len(done) == 10,
                 f"exit {result.returncode}, {len(done)} done {result.stderr.strip()[-200:]}")
    report.check("real run: the wrapper got the arguments of the dry run", recorded == dry,
                 f"{len(recorded)} calls")
    result = runner("run", str(config), "--notes", "real")
    skips = [line for line in result.stdout.splitlines() if " skip " in line]
    report.check("second run: every configuration skip", result.returncode == 0 and
                 len(skips) == 10 and len(log.read_text(encoding="utf-8").splitlines()) == 4,
                 f"{len(skips)} skip")
    result = runner("status", str(config))
    report.check("status: 10 done", "STATUS_TOTAL configurations=10 done=10 " in result.stdout,
                 result.stdout.strip().splitlines()[-1] if result.stdout else result.stderr)

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"SERIES_VERDICT: {verdict}")
    return EXIT_PASS if verdict == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
