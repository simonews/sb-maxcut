#!/usr/bin/env python3
"""sbmc_bruteforce (apps/sbmc_bruteforce.cpp): command line, exit statuses and the output line.

Run by CTest as:
  <python> -I -B test_sbmc_bruteforce.py --tool <sbmc_bruteforce> --data <tests/data>
                                         --workdir <build>/tests/bruteforce
Checks, as in F05-C1, section 5.9, and F05-C2, section 5.10:
1. exit status 2 for the errors of the command line, with an empty stdout and a message on
   stderr that starts with "sbmc_bruteforce: " (--threads above 1024 and a default name that is
   not valid among them);
2. exit status 1 for the errors of the library and of reading, with an empty stdout;
3. four_nodes.rudy with one thread: one line on stdout, empty stderr, the word BRUTEFORCE and the
   11 fields key=value in order, the values, the times in fixed notation with three decimals;
4. weighted_isolated_13.rudy with two threads (weights up to 2^31 - 1, two isolated nodes);
5. a dense instance with 16 spins and three threads;
6. a dense instance with one spin;
7. a dense instance with 20 spins and two optimal partitions, with one and with four threads;
8. --name.
The tool writes nothing; the test writes its input files only in --workdir, a folder of the
build tree (created when missing), and runs the tool from there.

Standard library only. Exit status: 0 passed, 1 failed. The last line is "BF_VERDICT: PASS" or
"BF_VERDICT: FAIL".
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
PREFIX = "sbmc_bruteforce: "
KEYS = ["instance", "n", "m", "W", "max_cut", "min_energy", "optimal_count", "threads",
        "t_setup_ms", "t_solve_ms", "config"]
INTEGER_KEYS = ["n", "m", "W", "max_cut", "min_energy", "optimal_count", "threads"]
THREE_DECIMALS = re.compile(r"^\d+\.\d{3}$")
INTEGER = re.compile(r"^-?\d+$")
CONFIG = re.compile(r"^[+-]+$")


class Report:
    """One line per check, and the count of failures."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"BF {name}: {'PASS' if ok else 'FAIL'}{suffix}")
        if not ok:
            self.failures += 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checks of sbmc_bruteforce.")
    parser.add_argument("--tool", required=True, help="path of the sbmc_bruteforce executable")
    parser.add_argument("--data", required=True, help="the folder tests/data")
    parser.add_argument("--workdir", required=True, help="build folder for the files of the test")
    return parser.parse_args()


class Runner:
    """Runs sbmc_bruteforce from the work folder."""

    def __init__(self, executable: str, workdir: Path) -> None:
        self.executable = executable
        self.workdir = workdir

    def __call__(self, arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([self.executable, *arguments], cwd=self.workdir,
                              capture_output=True, text=True, timeout=600, check=False)


def first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0] if lines else ""


def parse_line(stdout: str) -> tuple[dict[str, str] | None, str]:
    """The fields of the single output line, or None with the reason."""
    if not stdout.endswith("\n") or stdout.count("\n") != 1:
        return None, f"stdout is not exactly one line: {stdout!r}"
    tokens = stdout[:-1].split(" ")
    if len(tokens) != 1 + len(KEYS) or tokens[0] != "BRUTEFORCE":
        return None, f"expected BRUTEFORCE and {len(KEYS)} fields separated by single spaces"
    fields: dict[str, str] = {}
    for key, token in zip(KEYS, tokens[1:]):
        name, sep, value = token.partition("=")
        if name != key or sep != "=" or value == "":
            return None, f"expected the field {key}=<value>, got {token!r}"
        fields[key] = value
    return fields, ""


def form_problems(fields: dict[str, str]) -> list[str]:
    """Fields whose form is wrong: integers, times with three decimals, config of n signs."""
    wrong = [key for key in INTEGER_KEYS if not INTEGER.match(fields[key])]
    wrong += [key for key in ("t_setup_ms", "t_solve_ms") if not THREE_DECIMALS.match(fields[key])]
    if not CONFIG.match(fields["config"]) or (
            INTEGER.match(fields["n"]) and len(fields["config"]) != int(fields["n"])):
        wrong.append("config")
    return wrong


def check_run(report: Report, name: str, result: subprocess.CompletedProcess,
              expected: dict[str, str]) -> dict[str, str] | None:
    """Exit 0, one well-formed line and the expected values; returns the fields."""
    if result.returncode != 0:
        report.check(name, False, f"exit {result.returncode}: {first_line(result.stderr)}")
        return None
    fields, reason = parse_line(result.stdout)
    if fields is None:
        report.check(name, False, reason)
        return None
    wrong = form_problems(fields)
    wrong += [key for key, value in expected.items() if fields.get(key) != value]
    if result.stderr != "":
        wrong.append("stderr not empty")
    detail = ("wrong: " + ", ".join(wrong) + "; " if wrong else "") + result.stdout.strip()
    report.check(name, not wrong, detail)
    return fields


def main() -> int:
    args = parse_arguments()
    print(f"python: {platform.python_version()} ({sys.executable})")
    report = Report()
    # Absolute paths: the tool runs from the work folder.
    tool = str(Path(args.tool).resolve())
    data = Path(args.data).resolve()
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    four = str(data / "four_nodes.rudy")
    weighted = str(data / "weighted_isolated_13.rudy")
    missing = workdir / "missing.rudy"
    missing.unlink(missing_ok=True)
    self_loop = workdir / "self_loop.rudy"
    self_loop.write_text("3 1\n1 1 1\n", encoding="ascii")
    # A copy of four_nodes.rudy whose file name, the default --name, contains a space.
    spaced = workdir / "a b.rudy"
    spaced.write_bytes((data / "four_nodes.rudy").read_bytes())
    run = Runner(tool, workdir)

    # 1. Errors of the command line: exit status 2, empty stdout, prefixed message on stderr.
    usage_cases = [
        ("no arguments", []),
        ("--bogus 1", ["--instance", four, "--bogus", "1"]),
        ("--instance with --dense", ["--instance", four, "--dense", "16", "--instance-seed", "16"]),
        ("--dense 16 without --instance-seed", ["--dense", "16"]),
        ("--instance-seed 1 without --dense", ["--instance", four, "--instance-seed", "1"]),
        ("--threads -1", ["--instance", four, "--threads", "-1"]),
        ("--threads x", ["--instance", four, "--threads", "x"]),
        ("--dense 0x10", ["--dense", "0x10", "--instance-seed", "1"]),
        ("--name 'a b'", ["--instance", four, "--name", "a b"]),
        ("--name ''", ["--instance", four, "--name", ""]),
        ("--threads 1025", ["--instance", four, "--threads", "1025"]),
        ("default name not valid ('a b.rudy' without --name)", ["--instance", str(spaced)]),
    ]
    for name, arguments in usage_cases:
        result = run(arguments)
        ok = (result.returncode == 2 and result.stdout == ""
              and result.stderr.startswith(PREFIX))
        report.check(f"exit 2: {name}", ok,
                     f"exit {result.returncode}, stdout {len(result.stdout)} characters: "
                     f"{first_line(result.stderr)}")

    # 2. Errors of the library and of reading: exit status 1, empty stdout.
    error_cases = [
        ("--instance on a missing file", ["--instance", str(missing)]),
        ("--dense 41", ["--dense", "41", "--instance-seed", "1"]),
        ("--instance on a rudy file with a self-loop", ["--instance", str(self_loop)]),
    ]
    for name, arguments in error_cases:
        result = run(arguments)
        ok = (result.returncode == 1 and result.stdout == ""
              and result.stderr.startswith(PREFIX))
        report.check(f"exit 1: {name}", ok,
                     f"exit {result.returncode}, stdout {len(result.stdout)} characters: "
                     f"{first_line(result.stderr)}")

    # 3. The 4-node example with one thread.
    check_run(report, "four_nodes.rudy --threads 1", run(["--instance", four, "--threads", "1"]),
              {"instance": "four_nodes", "n": "4", "m": "5", "W": "5", "max_cut": "4",
               "min_energy": "-3", "optimal_count": "1", "threads": "1", "config": "+-+-"})

    # 4. Large weights of both signs and two isolated nodes, two threads.
    check_run(report, "weighted_isolated_13.rudy --threads 2",
              run(["--instance", weighted, "--threads", "2"]),
              {"instance": "weighted_isolated_13", "n": "13", "m": "19", "W": "2147483669",
               "max_cut": "6442452014", "min_energy": "-10737420359", "optimal_count": "4",
               "threads": "2", "config": "+-+--+-++-+++"})

    # 5. A dense instance, three threads.
    check_run(report, "--dense 16 --instance-seed 16 --threads 3",
              run(["--dense", "16", "--instance-seed", "16", "--threads", "3"]),
              {"instance": "dense_n16_s16", "n": "16", "m": "120", "W": "12", "max_cut": "29",
               "min_energy": "-46", "optimal_count": "1", "threads": "3",
               "config": "++--+-+++---+-++"})

    # 6. One spin, default number of threads.
    fields = check_run(report, "--dense 1 --instance-seed 7",
                       run(["--dense", "1", "--instance-seed", "7"]),
                       {"instance": "dense_n1_s7", "n": "1", "m": "0", "W": "0", "max_cut": "0",
                        "min_energy": "0", "optimal_count": "1", "config": "+"})
    if fields is not None:
        report.check("--dense 1: threads >= 1", int(fields["threads"]) >= 1,
                     f"threads {fields['threads']}")

    # 7. The same dense instance with one and with four threads: same values but threads and
    # times.
    expected_20 = {"instance": "dense_n20_s20", "n": "20", "m": "190", "W": "10",
                   "max_cut": "35", "min_energy": "-60", "optimal_count": "2",
                   "config": "++--++++-------+-+-+"}
    runs_20 = []
    for threads in ("1", "4"):
        runs_20.append(check_run(report, f"--dense 20 --instance-seed 20 --threads {threads}",
                                 run(["--dense", "20", "--instance-seed", "20", "--threads",
                                      threads]), expected_20))
    if all(r is not None for r in runs_20):
        one, four_threads = runs_20
        same = all(one[k] == four_threads[k] for k in KEYS
                   if k not in ("threads", "t_setup_ms", "t_solve_ms"))
        report.check("--dense 20: same values with 1 and 4 threads",
                     same and one["threads"] == "1" and 1 <= int(four_threads["threads"]) <= 4,
                     f"threads {one['threads']} and {four_threads['threads']}")
    else:
        report.check("--dense 20: same values with 1 and 4 threads", False, "a run failed")

    # 8. --name.
    check_run(report, "four_nodes.rudy --name custom_name",
              run(["--instance", four, "--name", "custom_name"]),
              {"instance": "custom_name", "n": "4", "max_cut": "4", "config": "+-+-"})

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"BF_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
