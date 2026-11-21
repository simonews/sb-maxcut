#!/usr/bin/env python3
"""sbmc_run (apps/sbmc_run.cpp): command line, exit statuses and the CSV of results schema 2.

Run by CTest from the repository root (the defaults of sbmc_run are relative to it) as:
  <python> -I -B test_sbmc_run.py --sbmc-run <sbmc_run> --trace-tool <sbmc_trace>
                                  --workdir <build>/tests/apps --strict-fp 0|1
Checks, as in F04-C2, section 5.9, F05-C2, section 5.9, F06-C1, section 5.11, F06-C2,
section 5.10, and F07-C1, section 5.12:
1. exit status 2 for the errors of the command line (--variant, --threads, the options of --algo sa
   and --replica-out included; the messages of phase 06 are compared: --variant c2 with dsb and
   bsb, --threads with c0, --replica-out on the same file as --out, written in four ways);
2. exit status 1 for the errors of the library, of reading and of writing (a CSV or a replica file
   with another header is left unchanged), and for paths of --out and --replica-out that cannot be
   compared (a loop of symbolic links);
3. tests/data/four_nodes.rudy, two recorded repetitions: the 56 columns and their values;
4. repetition 1 against sbmc_trace with --first-replica 8 (same instance and parameters);
5. a third run on the same CSV: three rows, one header;
6. data/gset/G1 with --target best-known; --name G1 on another instance (another n and m: exit
   status 1); a table of the test with a row of the right n and m and one with another m;
7. a dense instance with a warm-up run: one row, no target;
8. one line "rep <k>: ..." on stdout per recorded repetition, consistent with the CSV (t_solve_ms,
   t_run_ms and t_check_ms included);
9. the variant c1: two repetitions with two threads; G1 with one and with three threads, equal
   rows but the times; --threads 0; with --strict-fp 1 (strict presets) the same results as c0
   row by row, in dSB and in bSB;
10. simulated annealing (--algo sa) with --replica-out: the rows and the replica file against the
    values of F06-C1, section 5.8;
11. --replica-out with dsb: the replica rows against the rows of results, and a second run that
    appends without a second header;
12. the variant c2 of --algo sa: the run of check 10 with --variant c2 --threads 2 (one block,
    threads 1, the results and the replica file of c0), and data/gset/G1 with R 133 (several
    blocks, the last one partial, for every RB) and --threads 3 against c0;
13. the columns of schema 2 (F07-C1, section 5.12): in every row of every run t_prep_ms, t_run_ms
    and t_check_ms >= 0 with a sum <= t_solve_ms + 0.002, t_step_us from t_run_ms, dt_rule and
    c0_rule ("value" and "sigma" with --c0 auto, "value" and "value" with --c0 0.3, empty for sa),
    lambda_min, lambda_max and cpu_classes empty, cpu_affinity equal to the Cpus_allowed_list of
    this process, omp_places and omp_proc_bind equal to its OMP_PLACES and OMP_PROC_BIND; a run
    under "taskset -c 0" (skipped, and said so, without taskset or without CPU 0); OMP_PLACES=cores
    and OMP_PROC_BIND=close in the environment, then neither; OMP_PLACES with a line break (exit
    status 1, no row, a message that names it); --tag with a comma, read back with the csv module,
    and with a line break (exit status 2); an --out with the header of schema 1 (exit status 1,
    "results schema 2" in the message, the file unchanged).
The files of the test (CSV files and a trace) are written only in --workdir, a folder of the
build tree; their previous versions are removed at the start.

Exit status: 0 passed, 1 failed, 77 when NumPy cannot be imported (run scripts/setup_python.sh,
then configure again). The last line is "RUN_VERDICT: PASS" or "RUN_VERDICT: FAIL"
("RUN_VERDICT: SKIP" with exit status 77).
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 77

# Results schema 1 (until phase 06): an --out with its header is refused.
COLUMNS_1 = [
    "schema_version", "timestamp", "git_hash", "dirty", "host", "os", "gpu", "driver", "cuda",
    "cublas", "compiler", "algo", "variant", "instance", "n", "nnz", "replicas", "steps", "dt",
    "c0", "beta0", "beta1", "seed", "rep", "t_setup_ms", "t_solve_ms", "t_step_us", "best_cut",
    "mean_cut", "n_success", "p_s", "target", "tts99_ms", "ranks", "t_comm_ms",
    "sm_clock_avg_mhz", "temp_max_c", "power_avg_w", "a0", "eval_every", "first_replica",
    "threads", "target_source", "notes",
]
# Results schema 2: the first 43 columns of schema 1, the 12 columns of phase 07, notes.
COLUMNS = COLUMNS_1[:-1] + [
    "t_prep_ms", "t_run_ms", "t_check_ms", "dt_rule", "c0_rule", "lambda_min", "lambda_max",
    "cpu_affinity", "cpu_classes", "omp_places", "omp_proc_bind", "run_tag", "notes",
]
ALWAYS_EMPTY = ["gpu", "driver", "cuda", "cublas", "tts99_ms", "t_comm_ms", "sm_clock_avg_mhz",
                "temp_max_c", "power_avg_w", "lambda_min", "lambda_max", "cpu_classes"]
SB_ONLY = ["dt", "c0", "a0", "eval_every"]  # empty in the rows of SA
SA_ONLY = ["beta0", "beta1"]  # empty in the rows of SB
REPLICA_COLUMNS = ["schema_version", "timestamp", "instance", "algo", "variant", "replicas",
                   "steps", "seed", "rep", "replica", "best_cut", "best_step", "final_cut"]
REP_LINE = re.compile(r"^rep (\d+): best_cut (-?\d+) mean_cut (-?\d+\.\d{3})"
                      r"(?: n_success (\d+)/(\d+))? t_solve_ms (\d+\.\d{3})"
                      r" t_run_ms (\d+\.\d{3}) t_check_ms (\d+\.\d{3})$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
GIT_HASH = re.compile(r"^(?:[0-9a-f]{40}|unknown)$")
THREE_DECIMALS = re.compile(r"^\d+\.\d{3}$")
TRACE_HEADER = struct.Struct("<8sqiiiiiIIIQqq")

FOUR = "tests/data/four_nodes.rudy"
NOTES = 'a,b "c"'
OUTPUTS = ["runs.csv", "g1.csv", "name.csv", "dense.csv", "bad_header.csv", "unused.csv",
           "unused_replicas.csv", "rep1.trc", "c1_torus.csv", "c1_g1_threads1.csv",
           "c1_g1_threads3.csv", "c1_threads0.csv", "compare_c0_dsb.csv", "compare_c1_dsb.csv",
           "compare_c0_bsb.csv", "compare_c1_bsb.csv", "sa.csv", "sa_replicas.csv",
           "dsb_replicas_runs.csv", "dsb_replicas.csv", "bad_replicas.csv", "toy_runs.csv",
           "toy_best_known.csv", "c2.csv", "c2_replicas.csv", "c2_g1.csv", "c0_g1.csv",
           "loop", "q4.csv", "taskset.csv", "omp_env.csv", "omp_none.csv", "omp_newline.csv",
           "tag.csv", "schema1.csv"]
# F06-C1, section 5.8: sbmc_run on four_nodes.rudy, --algo sa --replicas 8 --sweeps 10
# --beta0 0.1 --beta1 3 --seed 2 --target 4 --reps 2 (rep 0: SA-1; rep 1: global replicas 8-15).
SA_BEST_SWEEP = [6, 0, 2, 1, 0, 3, 0, 1, 1, 3, 5, 7, 6, 1, 1, 1]
# Columns that may differ between two runs of the same configuration.
TIMES = ["timestamp", "t_setup_ms", "t_solve_ms", "t_step_us", "t_prep_ms", "t_run_ms",
         "t_check_ms"]
TIME_COLUMNS = ["t_setup_ms", "t_solve_ms", "t_step_us", "t_prep_ms", "t_run_ms", "t_check_ms"]


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

    def __call__(self, arguments: list[str], env: dict[str, str] | None = None,
                 prefix: list[str] | None = None) -> subprocess.CompletedProcess:
        return subprocess.run([*(prefix or []), self.executable, *arguments], cwd=self.root,
                              capture_output=True, text=True, timeout=600, check=False, env=env)


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
        ok = ok and line.group(7) == row["t_run_ms"] and line.group(8) == row["t_check_ms"]
        if with_target:
            ok = ok and line.group(4) == row["n_success"] and line.group(5) == row["replicas"]
        else:
            ok = ok and line.group(4) is None
    report.check(f"stdout {name}", ok, f"{len(lines)} rep lines, {len(others)} other lines")


def check_row_common(row: dict[str, str]) -> list[str]:
    """Columns whose form is the same in every row; returns the names that are wrong."""
    wrong = []
    if row.get("schema_version") != "2":
        wrong.append("schema_version")
    if not TIMESTAMP.match(row.get("timestamp", "")):
        wrong.append("timestamp")
    if not GIT_HASH.match(row.get("git_hash", "")):
        wrong.append("git_hash")
    if row.get("dirty") not in ("0", "1", ""):
        wrong.append("dirty")
    for column in TIME_COLUMNS:
        value = row.get(column, "")
        if not THREE_DECIMALS.match(value) or float(value) < 0:
            wrong.append(column)
    for column in ALWAYS_EMPTY:
        if row.get(column) != "":
            wrong.append(column)
    empty, filled = (SB_ONLY, SA_ONLY) if row.get("algo") == "sa" else (SA_ONLY, SB_ONLY[:2])
    wrong += [c for c in empty if row.get(c) != ""]
    wrong += [c for c in filled if row.get(c) == ""]
    for column in ("variant", "ranks", "threads", "host", "os", "compiler"):
        if not row.get(column):
            wrong.append(column)
    threads = row.get("threads", "")
    if (row.get("variant") not in ("c0", "c1", "c2") or row.get("ranks") != "1"
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


def check_toy_table(report: Report, run: Runner, workdir: Path) -> None:
    """Check 6: a table of the test, one row with the n and m of four_nodes.rudy, one without."""
    table = workdir / "toy_best_known.csv"
    header = Path("data/best_known.csv").read_text(encoding="utf-8").splitlines()[0]
    table.write_text(f"{header}\ntoy,4,5,4,best_known,test,,2026-11-18,\n"
                     "toy_bad,4,6,4,best_known,test,,2026-11-18,\n", encoding="utf-8")
    path = workdir / "toy_runs.csv"
    common = ["--instance", FOUR, "--algo", "dsb", "--replicas", "2", "--steps", "10", "--dt",
              "1.25", "--target", "best-known", "--best-known", str(table), "--out", str(path)]
    result = run(common + ["--name", "toy"])
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    row = rows[0] if len(rows) == 1 else {}
    got = {c: row.get(c) for c in ("instance", "target", "target_source")}
    report.check("--name toy (n and m of the instance)",
                 result.returncode == 0 and got == {"instance": "toy", "target": "4",
                                                    "target_source": "best_known"},
                 f"exit {result.returncode}, {got}")
    result = run(common + ["--name", "toy_bad"])
    first = result.stderr.strip().splitlines()[0:1]
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    report.check("--name toy_bad (another m)",
                 result.returncode == 1 and len(rows) == 1 and not rep_lines(result.stdout)
                 and first == [f"sbmc_run: --target best-known: toy_bad in {table} has n 4 and "
                               "m 6, the instance has n 4 and 5 edges"],
                 f"exit {result.returncode}: {first}")


def check_replica_rows(rows: list[dict[str, str]], replicas: list[dict[str, str]],
                       r: int, target: int) -> list[str]:
    """The replica rows of every repetition against its row of results; returns what is wrong."""
    wrong = []
    if len(replicas) != r * len(rows):
        return [f"{len(replicas)} replica rows for {len(rows)} rows of {r} replicas"]
    for k, row in enumerate(rows):
        block = replicas[k * r:(k + 1) * r]
        first = int(row["first_replica"])
        keys = ("timestamp", "instance", "algo", "variant", "replicas", "steps", "seed", "rep")
        if any(rr.get(c) != row.get(c) for rr in block for c in keys):
            wrong.append(f"rep {k}: key columns")
        if [int(rr["replica"]) for rr in block] != list(range(first, first + r)):
            wrong.append(f"rep {k}: replica indices")
        if any(rr.get("schema_version") != "1" for rr in block):
            wrong.append(f"rep {k}: schema_version")
        cuts = [int(rr["best_cut"]) for rr in block]
        if str(max(cuts)) != row["best_cut"]:
            wrong.append(f"rep {k}: maximum")
        if str(sum(c >= target for c in cuts)) != row["n_success"]:
            wrong.append(f"rep {k}: successes")
        if f"{sum(cuts) / r:.3f}" != row["mean_cut"]:
            wrong.append(f"rep {k}: mean")
        steps = int(row["steps"])
        if any(not 0 <= int(rr["best_step"]) <= steps for rr in block):
            wrong.append(f"rep {k}: best_step outside [0, steps]")
        if any(int(rr["final_cut"]) > int(rr["best_cut"]) for rr in block):
            wrong.append(f"rep {k}: final_cut > best_cut")
    return wrong


def check_sa(report: Report, run: Runner, workdir: Path) -> None:
    """Check 10: --algo sa with --replica-out, against F06-C1, section 5.8."""
    path, replica_path = workdir / "sa.csv", workdir / "sa_replicas.csv"
    result = run(["--instance", FOUR, "--algo", "sa", "--replicas", "8", "--sweeps", "10",
                  "--beta0", "0.1", "--beta1", "3", "--seed", "2", "--target", "4", "--reps", "2",
                  "--notes", NOTES, "--out", str(path), "--replica-out", str(replica_path)])
    _, rows, raw = read_rows(path) if path.exists() else ([], [], [])
    wrong = [] if result.returncode == 0 and len(raw) == 2 else ["exit or rows"]
    for k, row in enumerate(rows):
        wrong += [f"rep {k}: {c}" for c in check_row_common(row)]
        expected = {"algo": "sa", "variant": "c0", "instance": "four_nodes", "n": "4", "nnz": "5",
                    "replicas": "8", "steps": "10", "dt": "", "c0": "", "a0": "",
                    "eval_every": "", "beta0": "0.1", "beta1": "3", "seed": "2", "rep": str(k),
                    "first_replica": str(8 * k), "threads": "1", "best_cut": "4",
                    "mean_cut": "4.000", "n_success": "8", "p_s": "1", "target": "4",
                    "target_source": "user", "notes": NOTES}
        wrong += [f"rep {k}: {c}" for c, v in expected.items() if row.get(c) != v]
        # t_step_us = t_run_ms * 1000 / S from the unrounded time: t_run_ms has 3 decimals.
        if abs(float(row["t_step_us"]) - float(row["t_run_ms"]) * 100.0) > 0.0005 + 0.05:
            wrong.append(f"rep {k}: t_step_us")
    report.check("sa four_nodes.rudy, rows", not wrong,
                 "wrong: " + ", ".join(wrong) if wrong else "2 rows with the values of 5.8")
    check_stdout(report, "sa", result, rows, True)
    header, replicas, _ = read_rows(replica_path) if replica_path.exists() else ([], [], [])
    wrong = [] if header == REPLICA_COLUMNS else ["header"]
    wrong += check_replica_rows(rows, replicas, 8, 4) if len(rows) == 2 else ["rows"]
    got = [(int(r["replica"]), int(r["best_step"]), int(r["final_cut"])) for r in replicas]
    if got != [(g, SA_BEST_SWEEP[g], 4) for g in range(16)]:
        wrong.append("replica, best_step, final_cut")
    report.check("sa four_nodes.rudy, replica file", not wrong,
                 "wrong: " + ", ".join(wrong) if wrong else
                 "16 rows, best_step " + " ".join(str(b) for _, b, _ in got))


def check_replica_out(report: Report, run: Runner, workdir: Path) -> None:
    """Check 11: --replica-out with dsb, and a second run on the same two files."""
    path, replica_path = workdir / "dsb_replicas_runs.csv", workdir / "dsb_replicas.csv"
    arguments = ["--instance", FOUR, "--algo", "dsb", "--replicas", "8", "--steps", "100", "--dt",
                 "1.25", "--seed", "5", "--target", "4", "--reps", "2", "--warmup", "0",
                 "--out", str(path), "--replica-out", str(replica_path)]
    result = run(arguments)
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    header, replicas, _ = read_rows(replica_path) if replica_path.exists() else ([], [], [])
    wrong = [] if result.returncode == 0 and len(rows) == 2 else ["exit or rows"]
    wrong += [] if header == REPLICA_COLUMNS else ["header"]
    wrong += check_replica_rows(rows, replicas, 8, 4)
    report.check("--replica-out with dsb", not wrong,
                 "wrong: " + ", ".join(wrong) if wrong else f"{len(replicas)} replica rows")
    result = run(arguments)
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    _, replicas, _ = read_rows(replica_path) if replica_path.exists() else ([], [], [])
    header_lines = replica_path.read_text(encoding="utf-8").splitlines().count(
        ",".join(REPLICA_COLUMNS)) if replica_path.exists() else 0
    wrong = [] if result.returncode == 0 and len(rows) == 4 and header_lines == 1 else [
        f"exit {result.returncode}, {len(rows)} rows, {header_lines} header lines"]
    wrong += check_replica_rows(rows, replicas, 8, 4)
    report.check("--replica-out, second run on the same files", not wrong,
                 "wrong: " + ", ".join(wrong) if wrong else
                 f"{len(replicas)} replica rows, {header_lines} header line")


def check_c2(report: Report, run: Runner, workdir: Path) -> None:
    """Check 12: the variant c2 of --algo sa, against c0 (C2 equals the reference in every preset).
    """
    # a. The run of check 10 with --variant c2 --threads 2: one block of replicas, threads 1.
    path, replica_path = workdir / "c2.csv", workdir / "c2_replicas.csv"
    result = run(["--instance", FOUR, "--algo", "sa", "--variant", "c2", "--threads", "2",
                  "--replicas", "8", "--sweeps", "10", "--beta0", "0.1", "--beta1", "3", "--seed",
                  "2", "--target", "4", "--reps", "2", "--notes", NOTES, "--out", str(path),
                  "--replica-out", str(replica_path)])
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    sa_path, sa_replica_path = workdir / "sa.csv", workdir / "sa_replicas.csv"
    _, c0_rows, _ = read_rows(sa_path) if sa_path.exists() else ([], [], [])
    wrong = [] if result.returncode == 0 and len(rows) == 2 and len(c0_rows) == 2 else [
        f"exit {result.returncode}, {len(rows)} rows of c2, {len(c0_rows)} rows of c0"]
    for k, (row, c0_row) in enumerate(zip(rows, c0_rows)):
        wrong += [f"rep {k}: {c}" for c in check_row_common(row)]
        if row.get("variant") != "c2" or row.get("threads") != "1":
            wrong.append(f"rep {k}: variant {row.get('variant')} threads {row.get('threads')}")
        wrong += [f"rep {k}: {c}" for c in ("best_cut", "mean_cut", "n_success")
                  if row.get(c) != c0_row.get(c)]
    report.check("c2 four_nodes.rudy, rows against c0", not wrong,
                 "wrong: " + ", ".join(wrong) if wrong else "2 rows, variant c2, threads 1")
    check_stdout(report, "c2", result, rows, True)
    _, replicas, _ = read_rows(replica_path) if replica_path.exists() else ([], [], [])
    _, c0_replicas, _ = read_rows(sa_replica_path) if sa_replica_path.exists() else ([], [], [])
    differ = sorted({c for a, b in zip(replicas, c0_replicas) for c in REPLICA_COLUMNS
                     if c not in ("timestamp", "variant") and a.get(c) != b.get(c)})
    ok = (len(replicas) == 16 and len(c0_replicas) == 16 and not differ
          and all(r.get("variant") == "c2" for r in replicas))
    report.check("c2 four_nodes.rudy, replica file against c0", ok,
                 "columns that differ: " + ", ".join(differ) if differ else
                 f"{len(replicas)} and {len(c0_replicas)} rows, equal but timestamp and variant")

    # b. G1 with R 133 (several blocks, the last one partial, for every RB) and three threads.
    g1 = ["--instance", "data/gset/G1", "--algo", "sa", "--replicas", "133", "--sweeps", "20",
          "--beta0", "0.1", "--beta1", "3", "--seed", "4", "--reps", "2", "--target",
          "best-known"]
    variant_rows = {}
    for variant, more in (("c0", []), ("c2", ["--threads", "3"])):
        path = workdir / f"{variant}_g1.csv"
        result = run(g1 + ["--variant", variant] + more + ["--out", str(path)])
        _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
        variant_rows[variant] = rows if result.returncode == 0 else []
    c0_rows, c2_rows = variant_rows["c0"], variant_rows["c2"]
    differ = same_rows_but(c0_rows, c2_rows,
                           TIMES + ["git_hash", "dirty", "variant", "threads"])
    ok = (len(c0_rows) == 2 and len(c2_rows) == 2 and not differ
          and all(r.get("threads") == "3" and r.get("variant") == "c2" for r in c2_rows)
          and not any(check_row_common(r) for r in c2_rows))
    report.check("c2 G1, R 133, --threads 3, against c0", ok,
                 "columns that differ: " + ", ".join(differ) if differ else
                 f"{len(c0_rows)} and {len(c2_rows)} rows, threads of c2 "
                 f"{[r.get('threads') for r in c2_rows]}")


def allowed_cpus() -> str:
    """The text after "Cpus_allowed_list:" in /proc/self/status, without leading and trailing
    spaces and tabs, as sbmc_run reads it; empty when the file or the line is missing."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Cpus_allowed_list:"):
                    return line.rstrip("\n")[len("Cpus_allowed_list:"):].strip(" \t")
    except OSError:
        return ""
    return ""


def cpulist_has(cpulist: str, cpu: int) -> bool:
    """Whether a cpulist such as "0-3,8" contains cpu."""
    for part in filter(None, cpulist.split(",")):
        first, _, last = part.partition("-")
        try:
            if int(first) <= cpu <= int(last or first):
                return True
        except ValueError:
            return False
    return False


def without_omp(environment: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in environment.items() if k not in ("OMP_PLACES", "OMP_PROC_BIND")}


def check_schema2(report: Report, run: Runner, workdir: Path) -> None:
    """Check 13, the runs: taskset, OMP_PLACES and OMP_PROC_BIND, --tag, an --out of schema 1."""
    small = ["--instance", FOUR, "--algo", "sa", "--replicas", "4", "--sweeps", "10",
             "--beta0", "0.1", "--beta1", "3", "--warmup", "0"]

    # cpu_affinity under "taskset -c 0".
    taskset = shutil.which("taskset")
    if taskset is None or not cpulist_has(allowed_cpus(), 0):
        print("RUN taskset -c 0: skipped (" + ("taskset not found" if taskset is None else
              f"CPU 0 not allowed to this process: {allowed_cpus()!r}") + ")")
    else:
        path = workdir / "taskset.csv"
        result = run(small + ["--out", str(path)], prefix=[taskset, "-c", "0"])
        _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
        got = [r.get("cpu_affinity") for r in rows]
        report.check("taskset -c 0: cpu_affinity 0", result.returncode == 0 and got == ["0"],
                     f"exit {result.returncode}, cpu_affinity {got}")

    # OMP_PLACES and OMP_PROC_BIND in the environment, then neither of them.
    for name, extra, expected in (
            ("OMP_PLACES=cores OMP_PROC_BIND=close", {"OMP_PLACES": "cores",
                                                      "OMP_PROC_BIND": "close"},
             ("cores", "close")),
            ("neither OMP_PLACES nor OMP_PROC_BIND", {}, ("", ""))):
        path = workdir / ("omp_env.csv" if extra else "omp_none.csv")
        result = run(small + ["--out", str(path)], env={**without_omp(dict(os.environ)), **extra})
        _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
        got = [(r.get("omp_places"), r.get("omp_proc_bind")) for r in rows]
        report.check(f"{name}: omp_places and omp_proc_bind",
                     result.returncode == 0 and got == [expected],
                     f"exit {result.returncode}, {got}")
    path = workdir / "omp_newline.csv"
    result = run(small + ["--out", str(path)],
                 env={**without_omp(dict(os.environ)), "OMP_PLACES": "cores\nthreads"})
    ours = [line for line in result.stderr.splitlines() if line.startswith("sbmc_run: ")]
    report.check("OMP_PLACES with a line break: exit 1, no row, a message that names it",
                 result.returncode == 1 and not path.exists() and not rep_lines(result.stdout)
                 and bool(ours) and "OMP_PLACES" in ours[0],
                 f"exit {result.returncode}, file {path.exists()}: {ours[0:1]}")

    # --tag with a comma and a space, read back with the csv module.
    path = workdir / "tag.csv"
    result = run(small + ["--tag", "E2 cfg,1", "--out", str(path)])
    _, rows, _ = read_rows(path) if path.exists() else ([], [], [])
    got = [r.get("run_tag") for r in rows]
    report.check("--tag 'E2 cfg,1': run_tag", result.returncode == 0 and got == ["E2 cfg,1"],
                 f"exit {result.returncode}, run_tag {got}")

    # An --out with the header of schema 1: exit status 1, the file unchanged.
    path = workdir / "schema1.csv"
    old = ",".join(COLUMNS_1) + "\n" + ",".join(["x"] * len(COLUMNS_1)) + "\n"
    path.write_text(old, encoding="utf-8")
    result = run(small + ["--out", str(path)])
    first = result.stderr.strip().splitlines()[0:1]
    report.check("--out with the header of schema 1: exit 1, file unchanged",
                 result.returncode == 1 and bool(first) and "results schema 2" in first[0]
                 and not rep_lines(result.stdout) and path.read_text(encoding="utf-8") == old,
                 f"exit {result.returncode}: {first}")


def check_every_row(report: Report, workdir: Path) -> None:
    """Check 13, every row of every CSV of results schema 2 written by the test."""
    affinity = allowed_cpus()
    environment = (os.environ.get("OMP_PLACES", ""), os.environ.get("OMP_PROC_BIND", ""))
    special = {"taskset.csv", "omp_env.csv", "omp_none.csv"}  # checked by check_schema2
    wrong: list[str] = []
    files = rows_total = 0
    for path in sorted(workdir.glob("*.csv")):
        header, rows, _ = read_rows(path)
        if header != COLUMNS:
            continue
        files += 1
        for k, row in enumerate(rows, start=2):
            rows_total += 1
            where = f"{path.name}:{k}"
            try:
                prep, run_ms, check, solve, step = (float(row[c]) for c in (
                    "t_prep_ms", "t_run_ms", "t_check_ms", "t_solve_ms", "t_step_us"))
                steps = int(row["steps"])
            except (KeyError, ValueError):
                wrong.append(f"{where}: times")
                continue
            if min(prep, run_ms, check) < 0 or prep + run_ms + check > solve + 0.002:
                wrong.append(f"{where}: t_prep_ms + t_run_ms + t_check_ms > t_solve_ms + 0.002")
            if abs(step - run_ms * 1000 / steps) > 0.0005 + 0.0005 * 1000 / steps:
                wrong.append(f"{where}: t_step_us is not t_run_ms * 1000 / steps")
            if row["algo"] == "sa":
                if row["dt_rule"] != "" or row["c0_rule"] != "":
                    wrong.append(f"{where}: dt_rule or c0_rule not empty for sa")
            elif row["dt_rule"] != "value" or row["c0_rule"] not in ("sigma", "value"):
                wrong.append(f"{where}: dt_rule {row['dt_rule']!r}, c0_rule {row['c0_rule']!r}")
            if any(row[c] != "" for c in ("lambda_min", "lambda_max", "cpu_classes")):
                wrong.append(f"{where}: lambda_min, lambda_max or cpu_classes not empty")
            if path.name not in special:
                if row["cpu_affinity"] != affinity:
                    wrong.append(f"{where}: cpu_affinity {row['cpu_affinity']!r}, "
                                 f"expected {affinity!r}")
                if (row["omp_places"], row["omp_proc_bind"]) != environment:
                    wrong.append(f"{where}: omp_places and omp_proc_bind")
    report.check("every row of schema 2: times, rules, empty columns, affinity, OpenMP variables",
                 files > 0 and not wrong,
                 "; ".join(wrong[:10]) if wrong else f"{rows_total} rows in {files} files")


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
        ("--variant c3", four + base + ["--variant", "c3"]),
        ("--variant c1 --threads 1025", four + base + ["--variant", "c1", "--threads", "1025"]),
        ("--variant c1 --threads -1", four + base + ["--variant", "c1", "--threads", "-1"]),
        ("--variant c1 --threads x", four + base + ["--variant", "c1", "--threads", "x"]),
        ("--c0 foo", four + base + ["--c0", "foo"]),
        ("--c0 0", four + base + ["--c0", "0"]),
        ("--target foo", four + base + ["--target", "foo"]),
        ("--notes with a line break", four + base + ["--notes", "a\nb"]),
        ("--tag with a line break", four + base + ["--tag", "a\nb"]),
        ("--instance-seed without --dense", four + base + ["--instance-seed", "3"]),
    ]
    for name, arguments in usage_cases:
        result = run(arguments + ["--out", str(workdir / "unused.csv")] if arguments else [])
        ok = (result.returncode == 2 and result.stderr.strip() != ""
              and not rep_lines(result.stdout) and not (workdir / "unused.csv").exists())
        report.check(f"exit 2: {name}", ok,
                     f"exit {result.returncode}: {result.stderr.strip().splitlines()[0:1]}")

    # The messages of F06-C1, section 5.6, each compared with the first line of stderr.
    sa = ["--algo", "sa", "--replicas", "4", "--sweeps", "10", "--beta0", "0.1", "--beta1", "3"]
    out = ["--out", str(workdir / "unused.csv")]
    # The same file relative to the repository root, where sbmc_run runs.
    out_relative = ["--out", os.path.relpath(workdir / "unused.csv", root)]
    replica_out = ["--replica-out", str(workdir / "unused_replicas.csv")]
    sa_usage_cases = [
        ("--sweeps missing with sa", four + sa[:4] + sa[6:], "--sweeps is required with --algo sa"),
        ("--beta0 missing with sa", four + sa[:6] + sa[8:], "--beta0 is required with --algo sa"),
        ("--beta1 missing with sa", four + sa[:8], "--beta1 is required with --algo sa"),
        ("--steps with sa", four + sa + ["--steps", "10"], "--steps is not used with --algo sa"),
        ("--dt with sa", four + sa + ["--dt", "1.25"], "--dt is not used with --algo sa"),
        ("--c0 with sa", four + sa + ["--c0", "auto"], "--c0 is not used with --algo sa"),
        ("--a0 with sa", four + sa + ["--a0", "1"], "--a0 is not used with --algo sa"),
        ("--eval-every with sa", four + sa + ["--eval-every", "0"],
         "--eval-every is not used with --algo sa"),
        ("--sweeps with dsb", four + base + ["--sweeps", "10"], "--sweeps requires --algo sa"),
        ("--beta0 with dsb", four + base + ["--beta0", "0.1"], "--beta0 requires --algo sa"),
        ("--beta1 with dsb", four + base + ["--beta1", "3"], "--beta1 requires --algo sa"),
        ("--variant c1 with sa", four + sa + ["--variant", "c1"],
         "--variant c1 is not available with --algo sa"),
        ("--beta1 < --beta0", four + sa[:6] + ["--beta0", "3", "--beta1", "0.1"],
         "--beta1 must be >= --beta0"),
        ("--beta0 0", four + sa[:6] + ["--beta0", "0", "--beta1", "3"],
         "--beta0: expected a finite number > 0, got '0'"),
        ("--replica-out without --out", four + base + replica_out, "--replica-out requires --out"),
        # F06-C2, section 5.5.
        ("--variant c2 with dsb", four + base + ["--variant", "c2"],
         "--variant c2 requires --algo sa"),
        ("--variant c2 with bsb", four + ["--algo", "bsb"] + base[2:] + ["--variant", "c2"],
         "--variant c2 requires --algo sa"),
        ("--threads 2 with the default variant", four + base + ["--threads", "2"],
         "--threads requires --variant c1 or c2"),
        ("--variant c0 --threads 1", four + base + ["--variant", "c0", "--threads", "1"],
         "--threads requires --variant c1 or c2"),
        ("--threads 2 with sa and the default variant", four + sa + ["--threads", "2"],
         "--threads requires --variant c1 or c2"),
        ("--variant c3 with sa", four + sa + ["--variant", "c3"],
         "--variant: c3 not in {c0,c1,c2}"),
        ("--replica-out on the --out file", four + sa + out + ["--replica-out", out[1]],
         "--replica-out must differ from --out"),
        ("--replica-out on the --out file, with ./", four + sa + out_relative
         + ["--replica-out", "./" + out_relative[1]], "--replica-out must differ from --out"),
        ("--replica-out on the --out file, absolute", four + sa + out_relative
         + ["--replica-out", out[1]], "--replica-out must differ from --out"),
    ]
    for name, arguments, message in sa_usage_cases:
        result = run(arguments + (out if "--replica-out" not in arguments else []))
        first = result.stderr.strip().splitlines()[0:1]
        ok = (result.returncode == 2 and first == [f"sbmc_run: {message}"]
              and not rep_lines(result.stdout) and not (workdir / "unused.csv").exists()
              and not (workdir / "unused_replicas.csv").exists())
        report.check(f"exit 2: {name}", ok, f"exit {result.returncode}: {first}")

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
        ("--replica-out with another header",
         four + base + ["--out", str(workdir / "unused.csv"),
                        "--replica-out", str(workdir / "bad_replicas.csv")]),
    ]
    (workdir / "bad_replicas.csv").write_text("a,b,c\n1,2,3\n", encoding="ascii")
    for name, arguments in error_cases:
        result = run(arguments)
        ok = result.returncode == 1 and result.stderr.strip() != "" and not rep_lines(result.stdout)
        report.check(f"exit 1: {name}", ok,
                     f"exit {result.returncode}: {result.stderr.strip().splitlines()[0:1]}")
    # The same new file named "q4.csv" and "./q4.csv" from the folder of the test: the first
    # element of "q4.csv" does not exist, so the paths are compared only once made absolute.
    q4 = subprocess.run([args.sbmc_run, "--instance", str(root / FOUR)] + sa
                        + ["--out", "q4.csv", "--replica-out", "./q4.csv"], cwd=workdir,
                        capture_output=True, text=True, timeout=600, check=False)
    first = q4.stderr.strip().splitlines()[0:1]
    report.check("exit 2: --replica-out on the --out file, a new file in the current folder",
                 q4.returncode == 2 and first == ["sbmc_run: --replica-out must differ from --out"]
                 and not (workdir / "q4.csv").exists(), f"exit {q4.returncode}: {first}")
    loop = workdir / "loop"
    os.symlink("loop", loop)  # a symbolic link to itself: the path cannot be resolved
    result = run(four + sa + out + ["--replica-out", str(loop / "replicas.csv")])
    first = result.stderr.strip().splitlines()[0:1]
    report.check("exit 1: --replica-out in a loop of symbolic links",
                 result.returncode == 1 and bool(first) and "--out" in first[0]
                 and "--replica-out" in first[0] and "cannot compare" in first[0]
                 and not rep_lines(result.stdout) and not (workdir / "unused.csv").exists(),
                 f"exit {result.returncode}: {first}")
    report.check("--out with another header: file unchanged",
                 bad_header.read_text(encoding="ascii") == "a,b,c\n1,2,3\n")
    report.check("--replica-out with another header: file unchanged, no CSV",
                 (workdir / "bad_replicas.csv").read_text(encoding="ascii") == "a,b,c\n1,2,3\n"
                 and not (workdir / "unused.csv").exists())

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
    report.check("header of schema 2", header == COLUMNS, f"{len(header)} columns")
    report.check("two rows of 56 fields", len(raw) == 2 and all(len(r) == 56 for r in raw),
                 f"{len(raw)} rows")
    fixed = {"algo": "dsb", "variant": "c0", "instance": "four_nodes", "n": "4", "nnz": "5",
             "replicas": "8", "steps": "100", "dt": "1.25", "c0": "0.3", "a0": "1", "seed": "2",
             "eval_every": "0", "ranks": "1", "threads": "1", "target": "4",
             "target_source": "user", "notes": NOTES, "dt_rule": "value", "c0_rule": "value",
             "run_tag": ""}
    for k, row in enumerate(rows):
        wrong = check_row_common(row)
        wrong += [c for c, v in fixed.items() if row.get(c) != v]
        if row.get("rep") != str(k) or row.get("first_replica") != str(8 * k):
            wrong.append("rep/first_replica")
        report.check(f"four_nodes rep {k} columns", not wrong,
                     "wrong: " + ", ".join(wrong) if wrong else "56 columns as expected")
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
    got = {c: row.get(c) for c in ("instance", "n", "nnz", "target", "target_source", "dt_rule",
                                   "c0_rule")}
    report.check("G1 with --target best-known (--c0 auto: c0_rule sigma)",
                 result.returncode == 0 and len(rows) == 1 and not check_row_common(row)
                 and got == {"instance": "G1", "n": "800", "nnz": "19176", "target": "11624",
                             "target_source": "best_known", "dt_rule": "value",
                             "c0_rule": "sigma"}, str(got))
    check_stdout(report, "G1", result, rows, True)
    named = workdir / "name.csv"
    result = run(["--instance", FOUR, "--name", "G1", "--algo", "dsb", "--replicas", "2",
                  "--steps", "10", "--dt", "0.5", "--target", "best-known", "--out", str(named)])
    first = result.stderr.strip().splitlines()[0:1]
    expected = ["sbmc_run: --target best-known: G1 in data/best_known.csv has n 800 and m 19176, "
                "the instance has n 4 and 5 edges"]
    report.check("--name G1 on four_nodes.rudy",
                 result.returncode == 1 and first == expected and not named.exists()
                 and not rep_lines(result.stdout), f"exit {result.returncode}: {first}")
    check_toy_table(report, run, workdir)

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
    check_sa(report, run, workdir)
    check_replica_out(report, run, workdir)
    check_c2(report, run, workdir)
    check_schema2(report, run, workdir)
    check_every_row(report, workdir)

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"RUN_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
