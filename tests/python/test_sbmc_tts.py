#!/usr/bin/env python3
"""python/sbmc_tts.py: the time to solution on a fixture (F06-C1, section 5.10).

Run by CTest as:
  <python> -I -B test_sbmc_tts.py --pyroot <root>/python --workdir <build>/tests/tts
The fixture (a CSV of results schema 1 and its replica file) is written to --workdir, with three
groups (fx: dsb c1, R 100, five repetitions; fy: sa c0, R 10, no replica at the target; fz: bsb
c0, R 4, every replica at the target) and one row without a target (a skipped group). The values
expected by the checks were computed by cldArchitect; floats are compared with a relative
tolerance of 1e-9, integers exactly. Checks:
1. the whole fixture with --replicas, --bootstrap 1000, --seed 1 and --out: exit status 0, nine
   lines of summary with the expected values, one SKIP line, for fx at 1000 the bootstrap
   interval around TTS99;
2. the same call again: the same summary byte for byte; --bootstrap 0: the bootstrap columns
   empty, the other columns equal;
3. without --replicas and with --targets 1000: fx from the columns n_success and mean_cut; since
   F06-R2 (decision Q3) also without --replicas and without --targets: exit status 0, the three
   groups at 1000 only, fx with the same values;
4. exit status 1: a replica row removed, a replica row without its row of results, (seed, rep)
   twice in a group, a wrong n_success, a wrong mean_cut, a wrong header, a missing file; added
   in F06-R1, so that every error has a test: a replica row repeated, a row of results repeated,
   a wrong best_cut, the replica file with another header, --out in a missing folder;
5. exit status 2: --targets 995 and (since F06-R2) --targets 1000,995 without --replicas, with
   the message of the relaxed targets; --bootstrap 10, --targets 0, --targets 1001,
   --targets 1000,1000, no --results; added in F06-R1: --seed -1, --bootstrap x,
   --targets 1000,x;
6. (added in F06-R1) a group with the times 1, 2 and 9 ms: t_batch is their median, 2, not their
   mean, and the quartiles are those of numpy.percentile (the times of the fixture are
   symmetric, where the median and the mean coincide);
7. (added in F06-R1) the bootstrap intervals of fx at the three fractions are equal to those of
   the procedure of section 5.9 of F06-C1, written again in this file (the other checks of the
   bootstrap only bound the interval).

Exit status: 0 passed, 1 failed, 77 when NumPy cannot be imported (run scripts/setup_python.sh,
then configure again). The last line is "TTS_VERDICT: PASS", "TTS_VERDICT: FAIL" or
"TTS_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77

RESULTS_COLUMNS = [
    "schema_version", "timestamp", "git_hash", "dirty", "host", "os", "gpu", "driver", "cuda",
    "cublas", "compiler", "algo", "variant", "instance", "n", "nnz", "replicas", "steps", "dt",
    "c0", "beta0", "beta1", "seed", "rep", "t_setup_ms", "t_solve_ms", "t_step_us", "best_cut",
    "mean_cut", "n_success", "p_s", "target", "tts99_ms", "ranks", "t_comm_ms",
    "sm_clock_avg_mhz", "temp_max_c", "power_avg_w", "a0", "eval_every", "first_replica",
    "threads", "target_source", "notes",
]
REPLICA_COLUMNS = [
    "schema_version", "timestamp", "instance", "algo", "variant", "replicas", "steps", "seed",
    "rep", "replica", "best_cut", "best_step", "final_cut",
]
BOOTSTRAP_COLUMNS = ["bootstrap", "bootstrap_seed", "tts99_boot_lo_ms", "tts99_boot_hi_ms"]
OUTPUTS = ["results.csv", "replicas.csv", "summary.csv", "summary_again.csv",
           "summary_no_boot.csv", "summary_no_replicas.csv", "summary_default_no_replicas.csv",
           "bad.csv", "bad_replicas.csv",
           "skewed.csv", "summary_skewed.csv"]


class Report:
    """One line per check, and the count of failures."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"TTS {name}: {'PASS' if ok else 'FAIL'}{suffix}")
        if not ok:
            self.failures += 1


# --- Fixture --------------------------------------------------------------------------------


def results_row(**values: str) -> dict[str, str]:
    row = {c: "" for c in RESULTS_COLUMNS}
    row.update({"schema_version": "1", "git_hash": "unknown", "dirty": "0", "host": "fixture",
                "os": "fixture os", "compiler": "GNU 13.3.0", "ranks": "1", "t_setup_ms": "0.100"})
    row.update(values)
    return row


def fixture() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """The rows of results and of the replica file of section 5.10."""
    results: list[dict[str, str]] = []
    replicas: list[dict[str, str]] = []

    def add(row: dict[str, str], cuts: list[int]) -> None:
        results.append(row)
        first = int(row["first_replica"])
        for r, cut in enumerate(cuts):
            replicas.append({
                "schema_version": "1", **{c: row[c] for c in (
                    "timestamp", "instance", "algo", "variant", "replicas", "steps", "seed",
                    "rep")},
                "replica": str(first + r), "best_cut": str(cut), "best_step": str(7 + r % 5),
                "final_cut": str(cut - r % 3)})

    # fx: dsb c1, R 100, five repetitions.
    abc = [(3, 2, 5), (0, 1, 1), (5, 0, 3), (2, 4, 0), (0, 0, 2)]
    times = ["100.000", "110.000", "90.000", "105.000", "95.000"]
    for k, ((a, b, c), t) in enumerate(zip(abc, times)):
        cuts = [1000] * a + [996] * b + [991] * c + [900] * (100 - a - b - c)
        add(results_row(timestamp=f"2026-11-18T10:00:0{k}Z", algo="dsb", variant="c1",
                        instance="fx", n="10", nnz="20", replicas="100", steps="50", dt="0.5",
                        c0="0.25", a0="1", eval_every="0", threads="2", seed="7", rep=str(k),
                        first_replica=str(100 * k), t_solve_ms=t,
                        t_step_us=f"{float(t) * 20:.3f}", best_cut=str(max(cuts)),
                        mean_cut=format(sum(cuts) / 100, ".3f"), n_success=str(a),
                        p_s=repr(a / 100), target="1000", target_source="user",
                        notes="fixture"), cuts)
    # fy: sa c0, R 10, no replica at the target.
    for k, (t, cuts) in enumerate([("20.000", list(range(49, 39, -1))), ("30.000", [45] * 10)]):
        add(results_row(timestamp=f"2026-11-18T11:00:0{k}Z", algo="sa", variant="c0",
                        instance="fy", n="12", nnz="30", replicas="10", steps="20", beta0="0.1",
                        beta1="3", threads="1", seed="3", rep=str(k), first_replica=str(10 * k),
                        t_solve_ms=t, t_step_us=f"{float(t) * 50:.3f}", best_cut=str(max(cuts)),
                        mean_cut=format(sum(cuts) / 10, ".3f"), n_success="0", p_s="0.0",
                        target="50", target_source="user", notes="fixture"), cuts)
    # fz: bsb c0, R 4, every replica at the target.
    for k, t in enumerate(["5.000", "6.000", "7.000"]):
        add(results_row(timestamp=f"2026-11-18T12:00:0{k}Z", algo="bsb", variant="c0",
                        instance="fz", n="8", nnz="12", replicas="4", steps="10", dt="1.25",
                        c0="0.1", a0="1", eval_every="0", threads="1", seed="1", rep=str(k),
                        first_replica=str(4 * k), t_solve_ms=t, t_step_us=f"{float(t) * 100:.3f}",
                        best_cut="10", mean_cut="10.000", n_success="4", p_s="1.0",
                        target="10", target_source="user", notes="fixture"), [10] * 4)
    # A row without a target: its group is skipped, its replica rows are ignored.
    add(results_row(timestamp="2026-11-18T13:00:00Z", algo="dsb", variant="c0", instance="fw",
                    n="4", nnz="5", replicas="2", steps="10", dt="1.25", c0="0.3", a0="1",
                    eval_every="0", threads="1", seed="2", rep="0", first_replica="0",
                    t_solve_ms="1.000", t_step_us="100.000", best_cut="4", mean_cut="4.000",
                    target_source="none", notes="fixture"), [4, 4])
    return results, replicas


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


# --- Expected values (cldArchitect) ---------------------------------------------------------

COMMON = {
    "fx": {"t_batch_ms": 100.0, "t_batch_q1_ms": 95.0, "t_batch_q3_ms": 105.0, "best_cut": 1000,
           "mean_cut": 905.346, "gap_best_pct": 0.0, "gap_mean_pct": 9.4654, "rows": 5,
           "replicas_total": 500},
    "fy": {"t_batch_ms": 25.0, "t_batch_q1_ms": 22.5, "t_batch_q3_ms": 27.5, "best_cut": 49,
           "mean_cut": 44.75, "gap_best_pct": 2.0, "gap_mean_pct": 10.5, "rows": 2,
           "replicas_total": 20},
    "fz": {"t_batch_ms": 6.0, "t_batch_q1_ms": 5.5, "t_batch_q3_ms": 6.5, "best_cut": 10,
           "mean_cut": 10.0, "gap_best_pct": 0.0, "gap_mean_pct": 0.0, "rows": 3,
           "replicas_total": 12},
}
FY = {"target_value": 50, "successes": 0, "p_s": 0.0, "p_s_lo": 0.0,
      "p_s_hi": 0.1611251580528193, "tts99_ms": None, "tts99_lo_ms": 65.52836899910729,
      "tts99_hi_ms": None}
FZ = {"target_value": 10, "successes": 12, "p_s": 1.0, "p_s_lo": 0.7575059933447593,
      "p_s_hi": 1.0, "tts99_ms": 6.0, "tts99_lo_ms": 6.0, "tts99_hi_ms": 6.0}
EXPECTED = {
    ("fx", 1000): {"target_value": 1000, "successes": 10, "p_s": 0.02,
                   "p_s_lo": 0.010899183596210811, "p_s_hi": 0.03642018324687934,
                   "tts99_ms": 227.94817118369895, "tts99_lo_ms": 124.1287299129367,
                   "tts99_hi_ms": 420.2175336827916},
    ("fx", 995): {"target_value": 995, "successes": 17, "p_s": 0.034,
                  "p_s_lo": 0.021334519808532196, "p_s_hi": 0.0537713655016345,
                  "tts99_ms": 133.1303221550525, "tts99_lo_ms": 100.0,
                  "tts99_hi_ms": 213.54448639860223},
    ("fx", 990): {"target_value": 990, "successes": 28, "p_s": 0.056,
                  "p_s_lo": 0.03902562068315149, "p_s_hi": 0.0797447936467069,
                  "tts99_ms": 100.0, "tts99_lo_ms": 100.0, "tts99_hi_ms": 115.68590396249796},
    **{("fy", f): FY for f in (1000, 995, 990)},
    **{("fz", f): FZ for f in (1000, 995, 990)},
}


def compare(row: dict[str, str], expected: dict[str, object]) -> list[str]:
    """Names of the columns that differ: integers exactly, floats with tolerance 1e-9."""
    wrong = []
    for column, value in expected.items():
        text = row.get(column)
        if value is None:
            ok = text == ""
        elif isinstance(value, int):
            ok = text == str(value)
        else:
            try:
                got = float(text) if text is not None else math.nan
            except ValueError:
                got = math.nan
            ok = math.isclose(got, value, rel_tol=1e-9, abs_tol=0.0 if value else 1e-300)
        if not ok:
            wrong.append(f"{column}={text!r} expected {value!r}")
    return wrong


def bootstrap_by_hand(np, successes: list[int], times: list[float], replicas: int,
                      resamples: int, seed: list[int]) -> tuple[float, float]:
    """The bootstrap of F06-C1 section 5.9, written again here from the specification."""
    rng = np.random.default_rng(seed)
    rows = len(successes)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, rows, size=rows)
        drawn = sum(int(rng.binomial(replicas, successes[j] / replicas)) for j in idx)
        p = drawn / (rows * replicas)
        t_b = float(np.median(np.asarray(times, dtype=float)[idx]))
        if p == 0:
            values.append(math.inf)
        elif p == 1:
            values.append(t_b)
        else:
            values.append(t_b * max(1.0, math.log(0.01) / (replicas * math.log1p(-p))))
    values.sort()
    return values[math.ceil(0.025 * resamples) - 1], values[math.ceil(0.975 * resamples) - 1]


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# --- Checks ---------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of python/sbmc_tts.py.")
    parser.add_argument("--pyroot", required=True, help="folder that contains sbmc_tts.py")
    parser.add_argument("--workdir", required=True, help="build folder for the files of the test")
    args = parser.parse_args()
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import numpy as np
    except ImportError as error:
        print(f"numpy: cannot be imported ({error}); run scripts/setup_python.sh, then configure "
              "again")
        print("TTS_VERDICT: SKIP")
        return EXIT_SKIP
    print(f"numpy: {np.__version__}")

    tool = Path(args.pyroot) / "sbmc_tts.py"
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (workdir / name).unlink(missing_ok=True)
    report = Report()

    def run(arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(tool), *arguments],
                              capture_output=True, text=True, timeout=600, check=False)

    results, replicas = fixture()
    results_csv, replicas_csv = workdir / "results.csv", workdir / "replicas.csv"
    write_csv(results_csv, RESULTS_COLUMNS, results)
    write_csv(replicas_csv, REPLICA_COLUMNS, replicas)
    both = ["--results", str(results_csv), "--replicas", str(replicas_csv)]

    # 1. The whole fixture.
    summary_csv = workdir / "summary.csv"
    result = run(both + ["--bootstrap", "1000", "--seed", "1", "--out", str(summary_csv)])
    rows = read_summary(summary_csv)
    report.check("whole fixture: exit 0 and nine lines", result.returncode == 0 and len(rows) == 9,
                 f"exit {result.returncode}, {len(rows)} lines"
                 + (f": {result.stderr.strip()}" if result.stderr.strip() else ""))
    skip_lines = [line for line in result.stdout.splitlines() if line.startswith("SKIP ")]
    done = [line for line in result.stdout.splitlines() if line.startswith("TTS_DONE ")]
    report.check("one SKIP line and TTS_DONE",
                 len(skip_lines) == 1 and "instance=fw" in skip_lines[0]
                 and done == ["TTS_DONE groups=3 lines=9 skipped=1"], f"{skip_lines} {done}")
    by_key = {(r.get("instance"), int(r.get("target_permille", "0"))): r for r in rows}
    for (instance, permille), expected in EXPECTED.items():
        row = by_key.get((instance, permille), {})
        wrong = compare(row, {**expected, **COMMON[instance], "bootstrap": 1000,
                              "bootstrap_seed": 1})
        report.check(f"{instance} at {permille}", bool(row) and not wrong,
                     "; ".join(wrong) if wrong else "values as expected")
    fx = by_key.get(("fx", 1000), {})
    try:
        lo, tts99, hi = (float(fx[c]) for c in ("tts99_boot_lo_ms", "tts99_ms",
                                                "tts99_boot_hi_ms"))
        ok = lo <= tts99 <= hi
    except (KeyError, ValueError):
        lo = tts99 = hi = math.nan
        ok = False
    report.check("fx at 1000: bootstrap interval around TTS99", ok, f"[{lo}, {hi}], tts {tts99}")
    fy_boot = [(by_key.get(("fy", f), {}).get("tts99_boot_lo_ms"),
                by_key.get(("fy", f), {}).get("tts99_boot_hi_ms")) for f in (1000, 995, 990)]
    report.check("fy: bootstrap inf and inf", fy_boot == [("inf", "inf")] * 3, str(fy_boot))
    fz_boot = [by_key.get(("fz", f), {}).get(c, "") for f in (1000, 995, 990)
               for c in ("tts99_boot_lo_ms", "tts99_boot_hi_ms")]
    try:
        ok = all(5.0 <= float(v) <= 7.0 for v in fz_boot)
    except ValueError:
        ok = False
    report.check("fz: bootstrap inside [5.0, 7.0]", ok, str(fz_boot))
    # fx (group 0) at the three fractions (indices 0, 1, 2): the successes per row are a, a + b
    # and a + b + c of the fixture.
    fx_successes = {1000: [3, 0, 5, 2, 0], 995: [5, 1, 5, 6, 0], 990: [10, 2, 8, 6, 2]}
    fx_times = [100.0, 110.0, 90.0, 105.0, 95.0]
    wrong = []
    for index, (permille, successes) in enumerate(fx_successes.items()):
        expected = bootstrap_by_hand(np, successes, fx_times, 100, 1000, [1, 0, index])
        got = tuple(by_key.get(("fx", permille), {}).get(c, "")
                    for c in ("tts99_boot_lo_ms", "tts99_boot_hi_ms"))
        if got != tuple(repr(v) for v in expected):
            wrong.append(f"{permille}: {got} expected {expected}")
    report.check("fx: bootstrap equal to the procedure of section 5.9", not wrong,
                 "; ".join(wrong) if wrong else "three intervals equal")

    # 2. The same call again, and without bootstrap.
    again = workdir / "summary_again.csv"
    run(both + ["--bootstrap", "1000", "--seed", "1", "--out", str(again)])
    report.check("same call: same summary byte for byte",
                 again.exists() and summary_csv.exists()
                 and again.read_bytes() == summary_csv.read_bytes())
    no_boot = workdir / "summary_no_boot.csv"
    result = run(both + ["--bootstrap", "0", "--out", str(no_boot)])
    rows_no_boot = read_summary(no_boot)
    empty = all(r.get(c) == "" for r in rows_no_boot for c in BOOTSTRAP_COLUMNS)
    others = len(rows_no_boot) == len(rows) and all(
        {c: v for c, v in a.items() if c not in BOOTSTRAP_COLUMNS}
        == {c: v for c, v in b.items() if c not in BOOTSTRAP_COLUMNS}
        for a, b in zip(rows, rows_no_boot))
    report.check("--bootstrap 0: bootstrap columns empty, the others equal",
                 result.returncode == 0 and empty and others,
                 f"exit {result.returncode}, empty {empty}, others equal {others}")

    # 3. Without --replicas, --targets 1000.
    no_replicas = workdir / "summary_no_replicas.csv"
    result = run(["--results", str(results_csv), "--targets", "1000", "--out", str(no_replicas)])
    rows_plain = read_summary(no_replicas)
    fx = next((r for r in rows_plain if r.get("instance") == "fx"), {})
    wrong = compare(fx, {**EXPECTED[("fx", 1000)], **COMMON["fx"]})
    report.check("without --replicas: fx from n_success and mean_cut",
                 result.returncode == 0 and len(rows_plain) == 3 and not wrong,
                 f"exit {result.returncode}, {len(rows_plain)} lines" +
                 (": " + "; ".join(wrong) if wrong else ""))
    # Q3 (F06-R2): without --replicas and without --targets, the target 1000 alone.
    default_plain = workdir / "summary_default_no_replicas.csv"
    result = run(["--results", str(results_csv), "--out", str(default_plain)])
    rows_default = read_summary(default_plain)
    lines = [(r.get("instance"), r.get("target_permille")) for r in rows_default]
    fx = next((r for r in rows_default if r.get("instance") == "fx"), {})
    wrong = compare(fx, {**EXPECTED[("fx", 1000)], **COMMON["fx"]})
    report.check("the default targets without --replicas: 1000 only",
                 result.returncode == 0 and lines == [("fx", "1000"), ("fy", "1000"),
                                                      ("fz", "1000")] and not wrong,
                 f"exit {result.returncode}, lines {lines}" +
                 (": " + "; ".join(wrong) if wrong else ""))

    # 4. Errors in the data: exit status 1.
    bad, bad_replicas = workdir / "bad.csv", workdir / "bad_replicas.csv"

    def data_case(name: str, results_rows: list[dict[str, str]],
                  replica_rows: list[dict[str, str]] | None, extra: list[str],
                  header: list[str] | None = None, missing: bool = False,
                  replica_header: list[str] | None = None) -> None:
        if header is None:
            write_csv(bad, RESULTS_COLUMNS, results_rows)
        else:
            with open(bad, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows([[r[c] for c in RESULTS_COLUMNS] for r in results_rows])
        arguments = ["--results", str(workdir / "missing.csv") if missing else str(bad)]
        if replica_rows is not None:
            write_csv(bad_replicas, REPLICA_COLUMNS, replica_rows)
            if replica_header is not None:
                with open(bad_replicas, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle, lineterminator="\n")
                    writer.writerow(replica_header)
                    writer.writerows([[r[c] for c in REPLICA_COLUMNS] for r in replica_rows])
            arguments += ["--replicas", str(bad_replicas)]
        result = run(arguments + extra + ["--bootstrap", "0"])
        first = result.stderr.strip().splitlines()[0:1]
        report.check(f"exit 1: {name}",
                     result.returncode == 1 and bool(first) and first[0].startswith("sbmc_tts: "),
                     f"exit {result.returncode}: {first}")

    data_case("a replica row removed", results, replicas[:99] + replicas[100:], [])
    orphan = dict(replicas[0], rep="9", replica="900")
    data_case("a replica row without its row of results", results, replicas + [orphan], [])
    twice = dict(results[-2], timestamp="2026-11-18T12:00:09Z")
    data_case("(seed, rep) twice in a group", results + [twice], None, ["--targets", "1000"])
    wrong_success = [dict(r) for r in results]
    wrong_success[1]["n_success"] = "1"
    data_case("a wrong n_success", wrong_success, replicas, [])
    wrong_mean = [dict(r) for r in results]
    wrong_mean[2]["mean_cut"] = "907.731"
    data_case("a wrong mean_cut", wrong_mean, replicas, [])
    data_case("a wrong header", results, replicas, [],
              header=RESULTS_COLUMNS[:-1] + ["note"])
    data_case("a missing file", results, None, ["--targets", "1000"], missing=True)
    data_case("a replica row repeated", results, replicas + [replicas[0]], [])
    data_case("a row of results repeated", results + [results[0]], replicas, [])
    wrong_best = [dict(r) for r in results]
    wrong_best[0]["best_cut"] = "999"
    data_case("a wrong best_cut", wrong_best, replicas, [])
    data_case("the replica file with another header", results, replicas, [],
              replica_header=REPLICA_COLUMNS[:-1] + ["final"])
    data_case("--out in a missing folder", results, replicas,
              ["--out", str(workdir / "no_such_folder" / "summary.csv")])

    # 5. Errors of the command line: exit status 2.
    usage_cases = [
        ("--targets 995 without --replicas", ["--results", str(results_csv), "--targets", "995"]),
        ("--targets 1000,995 without --replicas",
         ["--results", str(results_csv), "--targets", "1000,995"]),
        ("--bootstrap 10", both + ["--bootstrap", "10"]),
        ("--targets 0", both + ["--targets", "0"]),
        ("--targets 1001", both + ["--targets", "1001"]),
        ("--targets 1000,1000", both + ["--targets", "1000,1000"]),
        ("no --results", ["--replicas", str(replicas_csv)]),
        ("--seed -1", both + ["--seed", "-1"]),
        ("--bootstrap x", both + ["--bootstrap", "x"]),
        ("--targets 1000,x", both + ["--targets", "1000,x"]),
    ]
    for name, arguments in usage_cases:
        result = run(arguments)
        first = result.stderr.strip().splitlines()[0:1]
        relaxed = "without --replicas" not in name or (
            bool(first) and "relaxed targets need --replicas" in first[0])
        report.check(f"exit 2: {name}",
                     result.returncode == 2 and bool(first) and first[0].startswith("sbmc_tts: ")
                     and result.stdout == "" and relaxed, f"exit {result.returncode}: {first}")

    # 6. Skewed times: the median, not the mean.
    skewed, skewed_summary = workdir / "skewed.csv", workdir / "summary_skewed.csv"
    fz_rows = [r for r in results if r["instance"] == "fz"]
    rows_skewed = [dict(r, timestamp=f"2026-11-18T14:00:0{k}Z", t_solve_ms=f"{t:.3f}",
                        t_step_us=f"{t * 100:.3f}")
                   for k, (r, t) in enumerate(zip(fz_rows, [1.0, 2.0, 9.0]))]
    write_csv(skewed, RESULTS_COLUMNS, rows_skewed)
    result = run(["--results", str(skewed), "--targets", "1000", "--bootstrap", "0",
                  "--out", str(skewed_summary)])
    rows = read_summary(skewed_summary)
    row = rows[0] if len(rows) == 1 else {}
    wrong = compare(row, {"t_batch_ms": 2.0, "t_batch_q1_ms": 1.5, "t_batch_q3_ms": 5.5,
                          "tts99_ms": 2.0})
    detail = "; ".join(wrong) if wrong else "t_batch 2.0"
    report.check("skewed times: t_batch is the median",
                 result.returncode == 0 and bool(row) and not wrong,
                 f"exit {result.returncode}: {detail}")

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"TTS_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
