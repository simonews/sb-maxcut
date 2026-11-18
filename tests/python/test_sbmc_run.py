#!/usr/bin/env python3
"""sbmc_run (apps/sbmc_run.cpp): command line, exit statuses and the CSV of results schema 1.

Run by CTest from the repository root (the defaults of sbmc_run are relative to it) as:
  <python> -I -B test_sbmc_run.py --sbmc-run <sbmc_run> --trace-tool <sbmc_trace>
                                  --workdir <build>/tests/apps --strict-fp 0|1
Checks, as in F04-C2, section 5.9, and F05-C2, section 5.9:
1. exit status 2 for the errors of the command line (--variant and --threads included);
2. exit status 1 for the errors of the library, of reading and of writing (a CSV with another
   header is left unchanged);
3. tests/data/four_nodes.rudy, two recorded repetitions: the 44 columns and their values;
4. repetition 1 against sbmc_trace with --first-replica 8 (same instance and parameters);
5. a third run on the same CSV: three rows, one header;
6. data/gset/G1 with --target best-known, and --name G1 on another instance;
7. a dense instance with a warm-up run: one row, no target;
8. one line "rep <k>: ..." on stdout per recorded repetition, consistent with the CSV;
9. the variant c1: two repetitions with two threads; G1 with one and with three threads, equal
   rows but the times; --threads 0; with --strict-fp 1 (strict presets) the same results as c0
   row by row, in dSB and in bSB.
The files of the test (CSV files and a trace) are written only in --workdir, a folder of the
build tree; their previous versions are removed at the start.

Exit status: 0 passed, 1 failed, 77 when NumPy cannot be imported (run scripts/setup_python.sh,
then configure again). The last line is "RUN_VERDICT: PASS" or "RUN_VERDICT: FAIL"
("RUN_VERDICT: SKIP" with exit status 77).
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 77

COLUMNS = [
    "schema_version", "timestamp", "git_hash", "dirty", "host", "os", "gpu", "driver", "cuda",
    "cublas", "compiler", "algo", "variant", "instance", "n", "nnz", "replicas", "steps", "dt",
    "c0", "beta0", "beta1", "seed", "rep", "t_setup_ms", "t_solve_ms", "t_step_us", "best_cut",
    "mean_cut", "n_success", "p_s", "target", "tts99_ms", "ranks", "t_comm_ms",
    "sm_clock_avg_mhz", "temp_max_c", "power_avg_w", "a0", "eval_every", "first_replica",
    "threads", "target_source", "notes",
]
ALWAYS_EMPTY = ["gpu", "driver", "cuda", "cublas", "beta0", "beta1", "tts99_ms", "t_comm_ms",
                "sm_clock_avg_mhz", "temp_max_c", "power_avg_w"]
REP_LINE = re.compile(r"^rep (\d+): best_cut (-?\d+) mean_cut (-?\d+\.\d{3})"
                      r"(?: n_success (\d+)/(\d+))? t_solve_ms (\d+\.\d{3})$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
GIT_HASH = re.compile(r"^(?:[0-9a-f]{40}|unknown)$")
THREE_DECIMALS = re.compile(r"^\d+\.\d{3}$")
TRACE_HEADER = struct.Struct("<8sqiiiiiIIIQqq")

FOUR = "tests/data/four_nodes.rudy"
NOTES = 'a,b "c"'
OUTPUTS = ["runs.csv", "g1.csv", "name.csv", "dense.csv", "bad_header.csv", "unused.csv",
           "rep1.trc", "c1_torus.csv", "c1_g1_threads1.csv", "c1_g1_threads3.csv",
           "c1_threads0.csv", "compare_c0_dsb.csv", "compare_c1_dsb.csv", "compare_c0_bsb.csv",
           "compare_c1_bsb.csv"]
# Columns that may differ between two runs of the same configuration.
TIMES = ["timestamp", "t_setup_ms", "t_solve_ms", "t_step_us"]


class Report:
    """One line per check, and the count of failures."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"RUN {name}: {'PASS' if ok else 'FAIL'}{suffix}")
        if not ok:
            self.failures += 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checks of sbmc_run.")
    parser.add_argument("--sbmc-run", required=True, help="path of the sbmc_run executable")
    parser.add_argument("--trace-tool", required=True, help="path of the sbmc_trace executable")
    parser.add_argument("--workdir", required=True, help="build folder for the files of the test")
    parser.add_argument("--strict-fp", required=True, choices=["0", "1"],
                        help="1 in the strict presets (no FMA contraction), 0 otherwise")
    return parser.parse_args()


class Runner:
    """Runs sbmc_run from the repository root."""

    def __init__(self, executable: str, root: Path) -> None:
        self.executable = executable
        self.root = root

    def __call__(self, arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([self.executable, *arguments], cwd=self.root, capture_output=True,
                              text=True, timeout=600, check=False)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]], list[list[str]]]:
    with open(path, newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    header = table[0] if table else []
    return header, [dict(zip(header, row)) for row in table[1:]], table[1:]


def rep_lines(stdout: str) -> list[re.Match]:
    return [m for m in (REP_LINE.match(line) for line in stdout.splitlines()) if m]


def check_stdout(report: Report, name: str, result: subprocess.CompletedProcess,
                 rows: list[dict[str, str]], with_target: bool) -> None:
    """Check 8: one line per recorded repetition, consistent with the rows of the CSV."""
    lines = rep_lines(result.stdout)
    others = [line for line in result.stdout.splitlines() if not REP_LINE.match(line)]
    ok = len(lines) == len(rows) and not others
    for k, (line, row) in enumerate(zip(lines, rows)):
        ok = ok and int(line.group(1)) == k and line.group(2) == row["best_cut"]
        ok = ok and line.group(3) == row["mean_cut"] and line.group(6) == row["t_solve_ms"]
        if with_target:
            ok = ok and line.group(4) == row["n_success"] and line.group(5) == row["replicas"]
        else:
            ok = ok and line.group(4) is None
    report.check(f"stdout {name}", ok, f"{len(lines)} rep lines, {len(others)} other lines")


def check_row_common(row: dict[str, str]) -> list[str]:
    """Columns whose form is the same in every row; returns the names that are wrong."""
    wrong = []
    if row.get("schema_version") != "1":
        wrong.append("schema_version")
    if not TIMESTAMP.match(row.get("timestamp", "")):
        wrong.append("timestamp")
    if not GIT_HASH.match(row.get("git_hash", "")):
        wrong.append("git_hash")
    if row.get("dirty") not in ("0", "1", ""):
        wrong.append("dirty")
    for column in ("t_setup_ms", "t_solve_ms", "t_step_us"):
        value = row.get(column, "")
        if not THREE_DECIMALS.match(value) or float(value) < 0:
            wrong.append(column)
    for column in ALWAYS_EMPTY:
        if row.get(column) != "":
            wrong.append(column)
    for column in ("variant", "ranks", "threads", "host", "os", "compiler"):
        if not row.get(column):
            wrong.append(column)
    threads = row.get("threads", "")
    if (row.get("variant") not in ("c0", "c1") or row.get("ranks") != "1"
            or not re.fullmatch(r"[0-9]+", threads) or int(threads) < 1
            or (row.get("variant") == "c0" and threads != "1")):
        wrong.append("variant/ranks/threads")
    return wrong


def trace_best_cuts(path: Path, np) -> list[int]:
    """best_cut of every replica, from a trace of sbmc_trace (format of F04-C1, section 5.6)."""
    data = path.read_bytes()
    _, n, replicas, steps, record, *_ = TRACE_HEADER.unpack_from(data, 0)
    offset = (TRACE_HEADER.size + 4 * steps + n * n + 4 * replicas * n
              + 8 * record * replicas * n + 8 * replicas)
    return np.frombuffer(data, dtype="<i8", count=replicas, offset=offset).tolist()


def same_rows_but(rows_a: list[dict[str, str]], rows_b: list[dict[str, str]],
                  ignored: list[str]) -> list[str]:
    """Columns that differ between two lists of rows, apart from the ignored ones."""
    if len(rows_a) != len(rows_b):
        return ["number of rows"]
    return sorted({c for a, b in zip(rows_a, rows_b) for c in COLUMNS
                   if c not in ignored and a.get(c) != b.get(c)})


def check_c1(report: Report, run: Runner, workdir: Path, strict: bool) -> None:
    """Check 9: the variant c1 of sbmc_run."""
    torus = ["--instance", "tests/data/torus_4x6_pm1.rudy"]

    # a. Two repetitions with two threads: the common checks, variant c1, threads 2.
    path = workdir / "c1_torus.csv"
    result = run(torus + ["--algo", "dsb", "--replicas", "40", "--steps", "100", "--dt", "1.25",
                          "--seed", "3", "--target", "24", "--reps", "2", "--warmup", "0",
                          "--variant", "c1", "--threads", "2", "--notes", NOTES,
                          "--out", str(path)])
    _, rows, raw = read_rows(path) if path.exists() else ([], [], [])
    wrong = [] if result.returncode == 0 and len(raw) == 2 else ["exit or rows"]
    for k, row in enumerate(rows):
        wrong += [f"rep {k}: {c}" for c in check_row_common(row)]
        expected = {"variant": "c1", "threads": "2", "rep": str(k), "first_replica": str(40 * k),
                    "instance": "torus_4x6_pm1", "replicas": "40", "notes": NOTES}
        wrong += [f"rep {k}: {c}" for c, v in expected.items() if row.get(c) != v]
    report.check("c1 torus_4x6_pm1.rudy, --threads 2", not wrong,
                 "wrong: " + ", ".join(wrong) if wrong else f"exit {result.returncode}, 2 rows")
    check_stdout(report, "c1 torus_4x6_pm1.rudy", result, rows, True)

    # b. G1 with one and with three threads: the same rows except threads and the times.
    g1_rows = []
    for threads in ("1", "3"):
        path = workdir / f"c1_g1_threads{threads}.csv"
        result = run(["--instance", "data/gset/G1", "--algo", "dsb", "--replicas", "96",
                      "--steps", "20", "--dt", "0.5", "--target", "best-known", "--reps", "2",
                      "--warmup", "1", "--variant", "c1", "--threads", threads,
                      "--out", str(path)])
        _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
        ok = (result.returncode == 0 and len(rows) == 2
              and all(r.get("threads") == threads and not check_row_common(r) for r in rows))
        report.check(f"c1 G1 --threads {threads}", ok,
                     f"exit {result.returncode}, {len(rows)} rows, threads "
                     f"{[r.get('threads') for r in rows]}")
        g1_rows.append(rows)
    differ = same_rows_but(g1_rows[0], g1_rows[1], TIMES + ["threads"])
    report.check("c1 G1: same rows with 1 and 3 threads", not differ,
                 "columns that differ: " + ", ".join(differ) if differ else
                 "equal but threads and times")

    # c. --threads 0: the default number of OpenMP threads.
    path = workdir / "c1_threads0.csv"
    result = run(torus + ["--algo", "dsb", "--replicas", "40", "--steps", "10", "--dt", "1.25",
                          "--warmup", "0", "--variant", "c1", "--threads", "0",
                          "--out", str(path)])
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    threads = rows[0].get("threads", "") if len(rows) == 1 else ""
    report.check("c1 --threads 0", result.returncode == 0
                 and re.fullmatch(r"[0-9]+", threads) is not None and int(threads) >= 1,
                 f"exit {result.returncode}, threads '{threads}'")

    # d. Strict presets: c0 and c1 give the same results row by row.
    if not strict:
        print("RUN c1 against c0: not compared in the production presets (FMA contraction may "
              "change the bits)")
        return
    for algo, extra in (("dsb", []), ("bsb", ["--eval-every", "7"])):
        variant_rows = {}
        for variant, more in (("c0", []), ("c1", ["--threads", "2"])):
            path = workdir / f"compare_{variant}_{algo}.csv"
            result = run(torus + ["--algo", algo, "--replicas", "40", "--steps", "100", "--dt",
                                  "1.25", "--seed", "3", "--target", "24", "--reps", "2",
                                  "--warmup", "0", "--variant", variant] + more + extra
                         + ["--out", str(path)])
            _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
            variant_rows[variant] = rows if result.returncode == 0 else []
        c0_rows, c1_rows = variant_rows["c0"], variant_rows["c1"]
        compared = ["best_cut", "mean_cut", "n_success", "p_s", "c0"]
        differ = sorted({c for a, b in zip(c0_rows, c1_rows) for c in compared
                         if a.get(c) != b.get(c)})
        ok = len(c0_rows) == 2 and len(c1_rows) == 2 and not differ
        report.check(f"c1 against c0, {algo}{' --eval-every 7' if extra else ''}", ok,
                     "columns that differ: " + ", ".join(differ) if differ else
                     f"{len(c0_rows)} and {len(c1_rows)} rows, " +
                     "; ".join(f"rep {k}: best_cut {r.get('best_cut')} n_success "
                               f"{r.get('n_success')}" for k, r in enumerate(c1_rows)))


def main() -> int:
    args = parse_arguments()
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import numpy as np
    except ImportError as error:
        print(f"numpy: cannot be imported ({error}); run scripts/setup_python.sh, then configure "
              "again")
        print("RUN_VERDICT: SKIP")
        return EXIT_SKIP
    print(f"numpy: {np.__version__}")

    report = Report()
    root = Path.cwd()
    if not ((root / "data" / "best_known.csv").is_file() and (root / FOUR).is_file()):
        report.check("working directory", False, f"{root} is not the repository root")
        print("RUN_VERDICT: FAIL")
        return EXIT_FAIL
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (workdir / name).unlink(missing_ok=True)
    run = Runner(args.sbmc_run, root)

    # 1. Errors of the command line: exit status 2, a message on stderr, no repetition.
    base = ["--algo", "dsb", "--replicas", "4", "--steps", "10", "--dt", "1.25"]
    four = ["--instance", FOUR]
    usage_cases = [
        ("no arguments", []),
        ("unknown option", four + base + ["--bogus", "1"]),
        ("--instance with --dense", four + ["--dense", "20", "--instance-seed", "20"] + base),
        ("no instance", base),
        ("--dense without --instance-seed", ["--dense", "20"] + base),
        ("--algo missing", four + base[2:]),
        ("--replicas missing", four + base[:2] + base[4:]),
        ("--steps missing", four + base[:4] + base[6:]),
        ("--dt missing", four + base[:6]),
        ("--algo foo", four + ["--algo", "foo"] + base[2:]),
        ("--variant c2", four + base + ["--variant", "c2"]),
        ("--threads 2 with the default variant", four + base + ["--threads", "2"]),
        ("--variant c0 --threads 1", four + base + ["--variant", "c0", "--threads", "1"]),
        ("--variant c1 --threads 1025", four + base + ["--variant", "c1", "--threads", "1025"]),
        ("--variant c1 --threads -1", four + base + ["--variant", "c1", "--threads", "-1"]),
        ("--variant c1 --threads x", four + base + ["--variant", "c1", "--threads", "x"]),
        ("--c0 foo", four + base + ["--c0", "foo"]),
        ("--c0 0", four + base + ["--c0", "0"]),
        ("--target foo", four + base + ["--target", "foo"]),
        ("--notes with a line break", four + base + ["--notes", "a\nb"]),
        ("--instance-seed without --dense", four + base + ["--instance-seed", "3"]),
    ]
    for name, arguments in usage_cases:
        result = run(arguments + ["--out", str(workdir / "unused.csv")] if arguments else [])
        ok = (result.returncode == 2 and result.stderr.strip() != ""
              and not rep_lines(result.stdout) and not (workdir / "unused.csv").exists())
        report.check(f"exit 2: {name}", ok,
                     f"exit {result.returncode}: {result.stderr.strip().splitlines()[0:1]}")

    # 2. Errors of the library, of reading and of writing: exit status 1.
    bad_header = workdir / "bad_header.csv"
    bad_header.write_text("a,b,c\n1,2,3\n", encoding="ascii")
    error_cases = [
        ("--replicas 0", four + ["--algo", "dsb", "--replicas", "0", "--steps", "10", "--dt",
                                 "1.25"]),
        ("--target best-known for a dense instance not in the table",
         ["--dense", "20", "--instance-seed", "20"] + base + ["--target", "best-known"]),
        ("--instance on a missing file", ["--instance", "tests/data/missing.rudy"] + base),
        ("--out with another header", four + base + ["--out", str(bad_header)]),
        ("--variant c1 --replicas 0", four + ["--algo", "dsb", "--replicas", "0", "--steps", "10",
                                              "--dt", "1.25", "--variant", "c1"]),
    ]
    for name, arguments in error_cases:
        result = run(arguments)
        ok = result.returncode == 1 and result.stderr.strip() != "" and not rep_lines(result.stdout)
        report.check(f"exit 1: {name}", ok,
                     f"exit {result.returncode}: {result.stderr.strip().splitlines()[0:1]}")
    report.check("--out with another header: file unchanged",
                 bad_header.read_text(encoding="ascii") == "a,b,c\n1,2,3\n")

    # 3. Two recorded repetitions on the 4-node graph.
    runs = workdir / "runs.csv"
    first_run = ["--instance", FOUR, "--algo", "dsb", "--replicas", "8", "--steps", "100",
                 "--dt", "1.25", "--c0", "0.3", "--seed", "2", "--warmup", "0", "--target", "4",
                 "--notes", NOTES, "--out", str(runs)]
    result = run(first_run + ["--reps", "2"])
    stderr = result.stderr.strip()
    report.check("four_nodes run", result.returncode == 0,
                 f"exit {result.returncode}" + (f": {stderr}" if stderr else ""))
    header, rows, raw = read_rows(runs) if runs.exists() else ([], [], [])
    report.check("header of schema 1", header == COLUMNS, f"{len(header)} columns")
    report.check("two rows of 44 fields", len(raw) == 2 and all(len(r) == 44 for r in raw),
                 f"{len(raw)} rows")
    fixed = {"algo": "dsb", "variant": "c0", "instance": "four_nodes", "n": "4", "nnz": "5",
             "replicas": "8", "steps": "100", "dt": "1.25", "c0": "0.3", "a0": "1", "seed": "2",
             "eval_every": "0", "ranks": "1", "threads": "1", "target": "4",
             "target_source": "user", "notes": NOTES}
    for k, row in enumerate(rows):
        wrong = check_row_common(row)
        wrong += [c for c, v in fixed.items() if row.get(c) != v]
        if row.get("rep") != str(k) or row.get("first_replica") != str(8 * k):
            wrong.append("rep/first_replica")
        report.check(f"four_nodes rep {k} columns", not wrong,
                     "wrong: " + ", ".join(wrong) if wrong else "44 columns as expected")
    if rows:
        rep0 = rows[0]
        report.check("four_nodes rep 0 results",
                     (rep0["best_cut"], rep0["n_success"], rep0["p_s"], rep0["mean_cut"])
                     == ("4", "8", "1", "4.000"),
                     f"best_cut {rep0['best_cut']}, n_success {rep0['n_success']}, "
                     f"p_s {rep0['p_s']}, mean_cut {rep0['mean_cut']}")
    check_stdout(report, "four_nodes", result, rows, True)

    # 4. Repetition 1 against sbmc_trace on the global replicas 8 .. 15.
    trace = workdir / "rep1.trc"
    traced = subprocess.run(
        [args.trace_tool, "--rudy", FOUR, "--algorithm", "dsb", "--replicas", "8", "--steps",
         "100", "--record", "1", "--seed", "2", "--dt", "1.25", "--c0", "0.3",
         "--first-replica", "8", "--out", str(trace)],
        cwd=root, capture_output=True, text=True, timeout=600, check=False)
    if traced.returncode == 0 and len(rows) == 2:
        cuts = trace_best_cuts(trace, np)
        rep1 = rows[1]
        expected = (str(max(cuts)), str(sum(c >= 4 for c in cuts)), f"{sum(cuts) / 8:.3f}")
        got = (rep1["best_cut"], rep1["n_success"], rep1["mean_cut"])
        report.check("rep 1 against sbmc_trace",
                     got == expected and float(rep1["p_s"]) == int(rep1["n_success"]) / 8,
                     f"best_cut, n_success, mean_cut: CSV {got}, trace {expected}")
    else:
        report.check("rep 1 against sbmc_trace", False,
                     f"sbmc_trace exit {traced.returncode} {traced.stderr.strip()}")

    # 5. A third run on the same file: three rows, one header.
    result = run(first_run + ["--reps", "1"])
    _, rows, raw = read_rows(runs) if runs.exists() else ([], [], [])
    header_lines = runs.read_text(encoding="utf-8").splitlines().count(",".join(COLUMNS))
    report.check("third run on the same CSV",
                 result.returncode == 0 and len(raw) == 3 and header_lines == 1,
                 f"{len(raw)} rows, {header_lines} header line(s)")
    check_stdout(report, "third run", result, rows[2:], True)

    # 6. G1 with its best-known value, and --name G1 on another instance.
    g1 = workdir / "g1.csv"
    result = run(["--instance", "data/gset/G1", "--algo", "dsb", "--replicas", "2", "--steps",
                  "10", "--dt", "0.5", "--target", "best-known", "--out", str(g1)])
    _, rows, _ = read_rows(g1) if g1.exists() else ([], [], [])
    row = rows[0] if len(rows) == 1 else {}
    got = {c: row.get(c) for c in ("instance", "n", "nnz", "target", "target_source")}
    report.check("G1 with --target best-known",
                 result.returncode == 0 and len(rows) == 1 and not check_row_common(row)
                 and got == {"instance": "G1", "n": "800", "nnz": "19176", "target": "11624",
                             "target_source": "best_known"}, str(got))
    check_stdout(report, "G1", result, rows, True)
    named = workdir / "name.csv"
    result = run(["--instance", FOUR, "--name", "G1", "--algo", "dsb", "--replicas", "2",
                  "--steps", "10", "--dt", "0.5", "--target", "best-known", "--out", str(named)])
    _, rows, _ = read_rows(named) if named.exists() else ([], [], [])
    row = rows[0] if len(rows) == 1 else {}
    got = {c: row.get(c) for c in ("instance", "target", "n_success", "target_source")}
    report.check("--name G1 on four_nodes.rudy",
                 result.returncode == 0 and got == {"instance": "G1", "target": "11624",
                                                    "n_success": "0",
                                                    "target_source": "best_known"}, str(got))

    # 7. Dense instance, one warm-up run (not recorded), no target.
    dense = workdir / "dense.csv"
    result = run(["--dense", "20", "--instance-seed", "20", "--algo", "bsb", "--replicas", "2",
                  "--steps", "10", "--dt", "1.25", "--warmup", "1", "--out", str(dense)])
    _, rows, _ = read_rows(dense) if dense.exists() else ([], [], [])
    row = rows[0] if len(rows) == 1 else {}
    got = {c: row.get(c) for c in ("instance", "nnz", "algo", "target", "n_success", "p_s",
                                   "target_source")}
    report.check("dense instance with a warm-up run",
                 result.returncode == 0 and len(rows) == 1 and not check_row_common(row)
                 and got == {"instance": "dense_n20_s20", "nnz": "190", "algo": "bsb",
                             "target": "", "n_success": "", "p_s": "",
                             "target_source": "none"}, str(got))
    check_stdout(report, "dense", result, rows, False)

    check_c1(report, run, workdir, args.strict_fp == "1")

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"RUN_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
