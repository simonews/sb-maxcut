#!/usr/bin/env python3
"""python/sbmc_dwave.py, the external check with dwave-samplers (F07-C3, sections 5.4 and 5.7).

Run by CTest as:
  <python> -I -B test_sbmc_dwave.py --pyroot <root>/python --root <root> --sbmc-run <sbmc_run>
           --data <root>/tests/data --workdir <build>/tests/dwave
Checks:
1. statistics (values of cldArchitect, verifica/f07/c3_ref.py; relative tolerance 1e-12): ours
   [10, 11, ..., 19], dwave [11, 12, ..., 20, 20, 21]: z -1.3438638879193565, D 0.25, D_crit
   0.8349401176132333, PASS; the schedule of S = 1 and the geometric one of S = 4;
2. with dwave-samplers importable: the check on sparse_24_unit.rudy, S 200, beta 0.5 -> 2 (a 1 and
   b 2: h 2, w 1), 256 reads and R 256 (sbmc_run of the build of the tests, one thread): exit 0,
   "DWAVE sparse_24_unit: PASS", DWAVE_VERDICT: PASS, dwave_check.csv with the verdict PASS and the
   nanoseconds per proposal of both sides > 0, final_cuts_sparse_24_unit.csv with 256 rows, the
   plot; without dwave-samplers: the tool writes "DWAVE_VERDICT: SKIP (dwave-samplers not
   installed)", exit 0 and no file, and that part of the test is marked SKIP: the test passes
   when the statistics pass.

Exit status: 0 passed, 1 failed, 77 when NumPy or matplotlib cannot be imported or sbmc_run is
missing. The last line is "DWAVE_VERDICT: PASS", "DWAVE_VERDICT: FAIL" or "DWAVE_VERDICT: SKIP"
(the verdict of the test, not of the tool).
"""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77
TOLERANCE = 1e-12


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"DWAVE {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(abs(a), abs(b))


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of python/sbmc_dwave.py.")
    parser.add_argument("--pyroot", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sbmc-run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as error:
        print(f"missing: {error} (run scripts/setup_python.sh, then configure again)")
        print("DWAVE_VERDICT: SKIP")
        return EXIT_SKIP
    sbmc_run = Path(args.sbmc_run).resolve()
    if not sbmc_run.exists():
        print(f"missing: {sbmc_run}")
        print("DWAVE_VERDICT: SKIP")
        return EXIT_SKIP
    pyroot = Path(args.pyroot).resolve()
    sys.path.insert(0, str(pyroot))
    import sbmc_dwave

    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report = Report()

    # 1. Statistics.
    ours = list(range(10, 20))
    theirs = list(range(11, 21)) + [20, 21]
    z = sbmc_dwave.z_score(ours, theirs)
    d = sbmc_dwave.ks_statistic(ours, theirs)
    d_crit = sbmc_dwave.ks_critical(len(ours), len(theirs))
    report.check("z, D and D_crit of the two samples, PASS",
                 close(z, -1.3438638879193565) and close(d, 0.25)
                 and close(d_crit, 0.8349401176132333) and sbmc_dwave.verdict(z, d, d_crit),
                 f"z {z!r}, D {d!r}, D_crit {d_crit!r}")
    report.check("a sample far away: FAIL", not sbmc_dwave.verdict(
        sbmc_dwave.z_score(ours, [v + 20 for v in theirs]),
        sbmc_dwave.ks_statistic(ours, [v + 20 for v in theirs]), d_crit), "")
    schedule = sbmc_dwave.schedule(0.5, 2.0, 4)
    expected = [0.5 * 4.0 ** (k / 3) for k in range(4)]
    report.check("schedule: [beta0] for S = 1, geometric for S = 4",
                 sbmc_dwave.schedule(0.5, 2.0, 1) == [0.5] and schedule == expected
                 and close(schedule[-1], 2.0), f"{schedule}")

    # 2. The check on sparse_24_unit.rudy.
    toml = work / "dwave.toml"
    output = work / "out"
    rudy = Path(args.data).resolve() / "sparse_24_unit.rudy"
    toml.write_text(
        f'[campaign]\nname = "test_dwave"\noutput = "{output}"\nsbmc_run = "{sbmc_run}"\n'
        'sbmc_spectrum = "none"\nreplicas = 256\nthreads = 1\nreps = 1\nwarmup = 1\nseed = 1\n'
        f'[dwave]\nseed = 1\n[[instance]]\nid = "sparse_24_unit"\nargs = ["--instance", '
        f'"{rudy}"]\ntarget = "none"\n[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\n'
        "b = [2.0]\nsteps = { sparse_24_unit = [200] }\n", encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-B", str(pyroot / "sbmc_dwave.py"), "--root",
                             str(Path(args.root).resolve()), "check", str(toml), "--notes",
                             "test of the external check"],
                            capture_output=True, text=True, check=False, timeout=600)
    lines = result.stdout.strip().splitlines()
    if lines and lines[-1] == "DWAVE_VERDICT: SKIP (dwave-samplers not installed)":
        print("DWAVE the check on sparse_24_unit.rudy: SKIP (dwave-samplers not installed)")
        report.check("without dwave-samplers: exit 0, no file", result.returncode == 0
                     and not output.exists(), f"exit {result.returncode}")
    else:
        report.check("the check on sparse_24_unit.rudy: exit 0, PASS",
                     result.returncode == 0 and bool(lines)
                     and lines[-1] == "DWAVE_VERDICT: PASS"
                     and any(line.startswith("DWAVE sparse_24_unit: PASS") for line in lines),
                     f"exit {result.returncode}: {lines[-3:]} {result.stderr.strip()[:300]}")
        try:
            with open(output / "dwave_check.csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with open(output / "final_cuts_sparse_24_unit.csv", newline="",
                      encoding="utf-8") as handle:
                cuts = list(csv.DictReader(handle))
        except OSError as error:
            rows, cuts = [], []
            print(f"cannot read the files of the check: {error}")
        row = rows[0] if len(rows) == 1 else {}
        ok = (bool(row) and row["verdict"] == "PASS" and row["steps"] == "200"
              and close(float(row["beta0"]), 0.5) and close(float(row["beta1"]), 2.0)
              and row["reads"] == "256" and float(row["ns_per_proposal_ours"]) > 0
              and float(row["ns_per_proposal_dwave"]) > 0 and len(cuts) == 256
              and set(cuts[0]) == {"ours", "dwave"})
        report.check("dwave_check.csv and final_cuts_sparse_24_unit.csv", ok, f"{row}")
        plots = [output / "plots" / f"final_cuts_sparse_24_unit.{e}" for e in ("pdf", "png")]
        report.check("the plot final_cuts_sparse_24_unit, PDF and PNG",
                     all(p.exists() and p.stat().st_size > 1000 for p in plots), "")

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"DWAVE_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
