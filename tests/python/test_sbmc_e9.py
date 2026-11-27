#!/usr/bin/env python3
"""python/sbmc_e9.py: the analysis of E9 and the bitwise comparison (phase 07, part 4; F07-C4,
sections 5.3 and 5.4).

Run by CTest as:
  <python> -I -B test_sbmc_e9.py --pyroot <root>/python --root <root> --sbmc-run <sbmc_run>
           --thresholds <sbmc_sa_thresholds> --workdir <build>/tests/e9
Checks:
1. analyze on a fake campaign (two loads, five series as in bench/configs/f07_e9.toml with fewer
   threads, 4 repetitions, the masks and classes of the i7-13620H): e9_summary.csv with one row
   per configuration whose median, Q1, Q3, IQR, speedup (win-p against win-p1 at 1 thread),
   efficiency and speedup against the fastest Windows series at 1 thread equal those computed
   here independently (sorted lists, linear interpolation; relative tolerance 1e-12); the plots
   scaling_<instance>_<method>, PDF and PNG, not empty; E9_TABLE lines; E9_CONSISTENCY: PASS;
2. the checks fail where they must: a win-e row with the class of the P cores (masks), a WSL row
   with another best_cut (cuts linux), SA different between Windows and WSL (cuts sa
   windows=linux); dSB different between the platforms is only an E9_INFO line (PASS);
3. bitwise with sbmc_run and sbmc_sa_thresholds of this build on both sides, on G1 and on a dense
   instance with the id K2000 (F07-R4e: the repository is a git copy in --workdir with G1; the
   Windows side is a fake <dest> with the copy <dest>/src and wrappers in <dest>/build-win-cpu/
   and <dest>/build-win-cpu-strict/, which answer --version with the commit of HEAD, refuse an
   argument that starts with \\\\ and map C:\\sbmc-win back; the fake wslpath gives
   C:\\sbmc-win\\... under <dest>): BITWISE_SA: PASS and BITWISE_DSB_STRICT: PASS, the instance
   read from the copy, the outputs moved from <dest>/runs/bitwise to --workdir; with the wrapper
   that changes final_cut of replica 3: BITWISE_SA: FAIL with the first different replica and the
   thresholds compared (0 different); with a wrapper that gives another commit: exit 1 before any
   case; compare_thresholds on two texts with one different threshold and one different beta.
Files in --workdir only (emptied first). Needs git (otherwise SKIP).

Exit status: 0 passed, 1 failed, 77 when the executables or NumPy and matplotlib are missing.
The last line is "E9_TEST_VERDICT: PASS", "E9_TEST_VERDICT: FAIL" or "E9_TEST_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77
TOLERANCE = 1e-12
REPS = 4
R = 2
# series -> (platform, affinity, threads, base, cpu_affinity, cpu_classes)
SERIES = {
    "win-p1": ("windows", "p1", [1, 2], None, "0,2,4,6,8,10", "1:0,2,4,6,8,10"),
    "win-p": ("windows", "p", [2, 4], "win-p1", "0-11", "1:0-11"),
    "win-e": ("windows", "e", [1, 2], None, "12-15", "0:12-15"),
    "win-all": ("windows", "all", [1, 2, 4], None, "0-15", "1:0-11 0:12-15"),
    "wsl": ("linux", None, [1, 2, 4], None, "0-15", ""),
}

FAKE_WSLPATH = r"""#!/usr/bin/env bash
[ "$1" = "-w" ] && [ -n "$2" ] || exit 1
case "$2" in
  "@DEST@") printf 'C:\\sbmc-win\n' ;;
  "@DEST@"/*) printf 'C:\\sbmc-win%s\n' "$(printf '%s' "${2#"@DEST@"}" | tr '/' '\\')" ;;
  *) printf '\\\\wsl.localhost\\Fake%s\n' "$(printf '%s' "$2" | tr '/' '\\')" ;;
esac
"""
WRAPPER = r"""#!/usr/bin/env bash
# Fake Windows executable: --version with the commit of HEAD (E9_HASH overrides it); refuses an
# argument that starts with \\; maps C:\sbmc-win back to <dest>, runs the Linux tool; with
# E9_MUTATE set, adds 1 to final_cut of replica 3 of the replica file.
if [ "$1" = "--version" ]; then
  echo "sbmc_run git_hash=${E9_HASH:-@HASH@} dirty=0 compiler=fake build_type=Release"
  exit 0
fi
args=()
replicas=""
previous=""
for a in "$@"; do
  case "$a" in
    '\\'*) echo "fake exe: UNC path $a" >&2; exit 3 ;;
    'C:\sbmc-win\'*) a="@DEST@/${a#'C:\sbmc-win\'}"; a="${a//\\//}" ;;
  esac
  [ "$previous" = "--replica-out" ] && replicas="$a"
  previous="$a"
  args+=("$a")
done
"@REAL@" "${args[@]}" || exit $?
if [ -n "${E9_MUTATE:-}" ] && [ -n "$replicas" ]; then
  awk -F, 'BEGIN { OFS = "," } NR > 1 && $10 == "3" { $13 = $13 + 1 } { print }' "$replicas" \
    > "$replicas.tmp" && mv "$replicas.tmp" "$replicas"
fi
"""


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"E9 {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(abs(a), abs(b))


def median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between the closest ranks, position (n - 1) q."""
    s = sorted(values)
    position = (len(s) - 1) * q
    low = int(position)
    high = min(low + 1, len(s) - 1)
    return s[low] + (s[high] - s[low]) * (position - low)


def campaign_text(output: Path, run: str, wrapper: Path, instance: str) -> str:
    text = (f'[campaign]\nname = "e9"\noutput = "{output}"\nsbmc_run = "{run}"\n'
            f'sbmc_spectrum = "{run}"\nreplicas = {R}\nreps = {REPS}\nwarmup = 0\nseed = 1\n\n')
    for name, (plat, affinity, threads, base, _, _) in SERIES.items():
        exe = wrapper if plat == "windows" else run
        text += (f'[[series]]\nid = "{name}"\nsbmc_run = "{exe}"\nplatform = "{plat}"\n'
                 f"threads = {threads}\n")
        if affinity:
            text += f'affinity = "{affinity}"\n'
        if base:
            text += f'base = "{base}"\n'
        text += "\n"
    return text + instance + (
        '[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\nb = [2.0]\n'
        "steps = { G1 = [100], K2000 = [100] }\n\n"
        '[[method]]\nid = "dsb"\nvariant = "c1"\nf = [1.0]\nc0 = ["sigma"]\n'
        "steps = { G1 = [100], K2000 = [100] }\n")


def t_run(load: int, series: str, threads: int, rep: int) -> float:
    """Times of the fake campaign: 1 thread 1000 ms (E cores 1700), near-linear speedup, a small
    deterministic spread over the repetitions."""
    base = (1000.0 if series != "win-e" else 1700.0) * (1 + 0.5 * load)
    if series == "win-p":
        base *= 0.97
    if series == "wsl":
        base *= 1.03
    return base / (threads ** 0.9) * (1 + 0.01 * ((rep * 7 + threads) % 5))


def write_fake(sc, campaign, root: Path, broken: str = "") -> dict[tuple[str, str], list[float]]:
    """Results and replica files of every configuration; the times per (instance, config id)."""
    times: dict[tuple[str, str], list[float]] = {}
    configs = sc.expand_configs(campaign, root)
    loads = list(dict.fromkeys((c.instance.id, c.method) for c in configs))
    for config in configs:
        load = loads.index((config.instance.id, config.method))
        series = config.series.id
        plat, _, _, _, affinity, classes = SERIES[series]
        rows, replica_rows = [], []
        key = (config.instance.id, config.id)
        times[key] = []
        for rep in range(REPS):
            t = t_run(load, series, config.threads, rep)
            times[key].append(t)
            best, mean = str(500 + 10 * load + rep), f"{480 + load + rep * 0.5:.3f}"
            if broken == "cuts" and series == "wsl" and config.threads == 2 and rep == 1:
                best = "1"
            if broken == "sa" and plat == "linux" and config.method == "sa" and rep == 0:
                best, mean = "2", "2.000"
            if broken == "dsb" and plat == "linux" and config.method == "dsb" and rep == 0:
                best, mean = "3", "3.000"
            row_classes = classes
            if broken == "masks" and series == "win-e":
                row_classes = "1:12-15"
            row = {c: "" for c in sc.RESULTS_COLUMNS_2}
            row.update(schema_version="2", instance=config.instance.id, algo=config.method,
                       variant=config.variant, replicas=str(R), steps=str(config.steps),
                       rep=str(rep), threads=str(config.threads), t_run_ms=repr(t),
                       t_check_ms="0.5", best_cut=best, mean_cut=mean, cpu_affinity=affinity,
                       cpu_classes=row_classes, run_tag=config.id)
            rows.append([row[c] for c in sc.RESULTS_COLUMNS_2])
            for r in range(R):
                replica_rows.append(["1", "t", config.instance.id, config.method, config.variant,
                                     str(R), str(config.steps), "1", str(rep), str(rep * R + r),
                                     best, "1", best])
        sc.write_csv(root / config.results_path(campaign), sc.RESULTS_COLUMNS_2, rows)
        sc.write_csv(root / config.replica_path(campaign), sc.REPLICA_COLUMNS, replica_rows)
    return times


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of python/sbmc_e9.py.")
    parser.add_argument("--pyroot", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sbmc-run", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    print(f"python: {platform.python_version()} ({sys.executable})")
    root = Path(args.root).resolve()
    run_exe = str(Path(args.sbmc_run).resolve())
    thresholds = str(Path(args.thresholds).resolve())
    tool = Path(args.pyroot).resolve() / "sbmc_e9.py"
    missing = [p for p in (run_exe, thresholds, str(tool)) if not Path(p).exists()]
    if shutil.which("git") is None:
        missing.append("git")
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as error:
        missing.append(f"NumPy or matplotlib ({error})")
    if missing:
        print(f"missing: {', '.join(missing)}")
        print("E9_TEST_VERDICT: SKIP")
        return EXIT_SKIP
    sys.path.insert(0, str(Path(args.pyroot).resolve()))
    import sbmc_campaign as sc
    import sbmc_e9

    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    (work / "bin").mkdir(parents=True)
    report = Report()
    # The repository: a git copy with G1 (HEAD is checked); the Windows folder <dest>.
    repo = work / "repo"
    dest = work / "win"
    for folder in (repo, dest / "src"):
        (folder / "data" / "gset").mkdir(parents=True)
        shutil.copyfile(root / "data" / "gset" / "G1", folder / "data" / "gset" / "G1")
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.org"]
    for step in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "test"]):
        subprocess.run([*git, *step], check=True, capture_output=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    (dest / "src" / ".sbmc_build_info").write_text(f"git_hash={head}\ndirty=0\n",
                                                   encoding="ascii")
    root = repo
    wslpath = work / "bin" / "wslpath"
    wslpath.write_text(FAKE_WSLPATH.replace("@DEST@", str(dest)), encoding="utf-8")
    wrappers = {}
    for name, real, place in (("run", run_exe, "build-win-cpu/apps/sbmc_run.exe"),
                              ("strict", run_exe, "build-win-cpu-strict/apps/sbmc_run.exe"),
                              ("thresholds", thresholds,
                               "build-win-cpu/tests/tools/sbmc_sa_thresholds.exe")):
        wrappers[name] = dest / place
        wrappers[name].parent.mkdir(parents=True, exist_ok=True)
        wrappers[name].write_text(WRAPPER.replace("@REAL@", real).replace("@HASH@", head)
                                  .replace("@DEST@", str(dest)), encoding="utf-8")
    for path in (wslpath, *wrappers.values()):
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{work / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
    instance = ('[[instance]]\nid = "G1"\nargs = ["--instance", "' + str(root / "data/gset/G1")
                + '"]\ntarget = "none"\n\n[[instance]]\nid = "K2000"\n'
                'args = ["--dense", "200", "--instance-seed", "1"]\ntarget = "none"\n\n')

    def e9(*arguments: str, extra_env: dict[str, str] | None = None
           ) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(tool), "--root", str(root),
                               *arguments], capture_output=True, text=True, check=False,
                              timeout=900, env=dict(env, **(extra_env or {})))

    # 1. analyze.
    def fake_campaign(name: str, broken: str = ""):
        output = work / name
        config = work / f"{name}.toml"
        config.write_text(campaign_text(output, run_exe, wrappers["run"], instance),
                          encoding="utf-8")
        campaign = sc.load_campaign(config)
        times = write_fake(sc, campaign, root, broken)
        return config, output, campaign, times

    config, output, campaign, times = fake_campaign("good")
    result = e9("analyze", str(config))
    lines = result.stdout.splitlines()
    report.check("analyze: exit 0, E9_DONE", result.returncode == 0 and lines[-1:] == ["E9_DONE"],
                 f"exit {result.returncode}: {result.stderr.strip()}")
    with open(output / "e9_summary.csv", newline="", encoding="utf-8") as handle:
        summary = {(r["instance"], r["config_id"]): r for r in csv.DictReader(handle)}
    configs = sc.expand_configs(campaign, root)
    report.check("e9_summary.csv: one row per configuration (2 x 2 x 12)",
                 len(summary) == len(configs) == 48, f"{len(summary)} rows")
    wrong = []
    for c in configs:
        i = c.instance.id
        row = summary[(i, c.id)]
        t = times[(i, c.id)]
        load = c.id[: -len(f"_{c.series.id}_T{c.threads}")]
        reference = c.series.base or c.series.id
        t1 = median(times[(i, f"{load}_{reference}_T1")])
        win_t1 = min(median(times[(i, f"{load}_{s}_T1")]) for s, v in SERIES.items()
                     if v[0] == "windows" and 1 in v[2])
        expected = {"t_run_median_ms": median(t), "t_run_q1_ms": percentile(t, 0.25),
                    "t_run_q3_ms": percentile(t, 0.75),
                    "t_run_iqr_ms": percentile(t, 0.75) - percentile(t, 0.25),
                    "t1_median_ms": t1, "speedup": t1 / median(t),
                    "efficiency": t1 / median(t) / c.threads}
        if c.series.platform == "windows":
            expected.update(speedup_windows=win_t1 / median(t),
                            efficiency_windows=win_t1 / median(t) / c.threads)
        elif row["speedup_windows"] or row["efficiency_windows"]:
            wrong.append(f"{c.id}: speedup_windows on wsl")
        for key, value in expected.items():
            if not close(float(row[key]), value):
                wrong.append(f"{c.id}: {key} {row[key]} != {value!r}")
        if row["reference_series"] != reference or row["rows"] != str(REPS):
            wrong.append(f"{c.id}: reference {row['reference_series']}, rows {row['rows']}")
    report.check("e9_summary.csv: median, Q1, Q3, IQR, speedups and efficiencies as computed "
                 "here", not wrong, "; ".join(wrong[:3]) or "48 rows x 7-9 values")
    plots = [output / "plots" / f"scaling_{i}_{m}.{e}" for i in ("G1", "K2000")
             for m in ("sa", "dsb") for e in ("pdf", "png")]
    report.check("plots scaling_<instance>_<method>, PDF and PNG, not empty",
                 all(p.exists() and p.stat().st_size > 1000 for p in plots),
                 ", ".join(f"{p.name} {p.stat().st_size if p.exists() else 0}" for p in plots[:2]))
    tables = [line for line in lines if line.startswith("E9_TABLE ")]
    report.check("E9_TABLE: one line per load with the five series", len(tables) == 4 and all(
        line.count(" | ") == 4 for line in tables), tables[0] if tables else "")
    report.check("good campaign: E9_CONSISTENCY: PASS", "E9_CONSISTENCY: PASS" in lines,
                 "; ".join(line for line in lines if line.startswith("E9_CHECK")))

    # 2. The checks that must fail.
    for broken, check in (("masks", "E9_CHECK masks: FAIL"), ("cuts", "E9_CHECK cuts linux: FAIL"),
                          ("sa", "E9_CHECK cuts sa windows=linux: FAIL")):
        config, _, _, _ = fake_campaign(f"broken_{broken}", broken)
        result = e9("analyze", str(config))
        lines = result.stdout.splitlines()
        hit = [line for line in lines if line.startswith(check)]
        report.check(f"broken {broken}: {check}, E9_CONSISTENCY: FAIL, exit 0",
                     result.returncode == 0 and bool(hit) and "E9_CONSISTENCY: FAIL" in lines,
                     hit[0] if hit else result.stderr.strip())
    config, _, _, _ = fake_campaign("broken_dsb", "dsb")
    result = e9("analyze", str(config))
    lines = result.stdout.splitlines()
    info = [line for line in lines if line.startswith("E9_INFO cuts") and " dsb " in line
            and "different" in line]
    report.check("dSB different between the platforms: E9_INFO only, E9_CONSISTENCY: PASS",
                 len(info) == 2 and "E9_CONSISTENCY: PASS" in lines, info[0] if info else "")

    # 3. bitwise.
    bw = ["bitwise", str(config), "--workdir", str(work / "bitwise"), "--win-run",
          str(wrappers["run"]), "--win-strict-run", str(wrappers["strict"]), "--win-thresholds",
          str(wrappers["thresholds"]), "--wsl-run", run_exe, "--wsl-strict-run", run_exe,
          "--wsl-thresholds", thresholds]
    result = e9(*bw)
    lines = result.stdout.splitlines()
    report.check("bitwise, same build on both sides: BITWISE_SA and BITWISE_DSB_STRICT PASS",
                 result.returncode == 0 and "BITWISE_SA: PASS" in lines
                 and "BITWISE_DSB_STRICT: PASS" in lines and len(lines) == 5,
                 "; ".join(lines) or result.stderr.strip())
    moved = sorted(p.name for p in (work / "bitwise").glob("*_windows*.csv"))
    left = [p for p in (dest / "runs").rglob("*") if p.is_file()]
    report.check("bitwise: the Windows outputs moved from <dest>/runs/bitwise to --workdir",
                 len(moved) == 6 and not left, f"{len(moved)} moved, {len(left)} left")
    result = e9(*bw, extra_env={"E9_MUTATE": "1"})
    lines = result.stdout.splitlines()
    first = [line for line in lines
             if line.startswith("BITWISE sa_G1: different: first replica 3:")]
    compared = [line for line in lines if line.startswith("BITWISE sa_G1 thresholds: 1000 sweeps")
                and line.endswith(" 0 different")]
    report.check("bitwise, replica 3 changed: BITWISE_SA FAIL, first replica 3, thresholds equal",
                 result.returncode == 0 and "BITWISE_SA: FAIL" in lines and bool(first)
                 and bool(compared) and "BITWISE_DSB_STRICT: FAIL" in lines,
                 "; ".join(lines[:2]) or result.stderr.strip())
    result = e9(*bw, extra_env={"E9_HASH": "0" * 40})
    report.check("bitwise, a Windows build of another commit: exit 1 before any case",
                 result.returncode == 1 and not result.stdout.strip()
                 and f"HEAD is {head}" in result.stderr, result.stderr.strip()[-160:])
    a = "0 0.5 100,90,0\n1 1 50,0\n2 2 7,0\n"
    b = "0 0.5 100,91,0\r\n1 1.0000000000000002 50,0\r\n2 2 7,0\r\n"
    sweeps, differences = sbmc_e9.compare_thresholds(a, b)
    report.check("compare_thresholds: one threshold and one beta different, CRLF ignored",
                 sweeps == 3 and differences == [
                     "sweep 0, delta_e 2: threshold 90 against 91",
                     "sweep 1: beta 1 against 1.0000000000000002"], "; ".join(differences))

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"E9_TEST_VERDICT: {verdict}")
    return EXIT_PASS if verdict == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
