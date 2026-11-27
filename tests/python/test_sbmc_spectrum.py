#!/usr/bin/env python3
"""sbmc_spectrum (apps/sbmc_spectrum.cpp): command line, exit statuses and the output line.

Run by CTest as:
  <python> -I -B test_sbmc_spectrum.py --tool <sbmc_spectrum> --data <tests/data>
                                       --gset <data/gset> --workdir <build>/tests/spectrum
Checks, as in F07-C2, section 5.7:
1. four_nodes.rudy, data/gset/G1 and --dense 200 --instance-seed 7: exit status 0, one line with
   the word SPECTRUM and the 11 fields key=value in the order of section 5.4; lambda_min,
   lambda_max, dt_auto and c0_lambda as in the table of the section (|lambda - reference| <= 1e-9
   max(|lambda_min|, |lambda_max|), dt and c0 relative 1e-6); n and m; the names;
2. --iterations 10 on G1: the values of the table of a few iterations;
3. a rudy file "5 0" (written by the test in its folder): exit status 0, dt_auto=none,
   c0_lambda=none, c0_sigma=none;
4. exit status 2 for the errors of the command line (no instance, --instance with --dense,
   --iterations 0, 100001 and x, --name with a space), exit status 1 for a missing file, with an
   empty stdout and a message on stderr that starts with "sbmc_spectrum: ";
5. in the cases of success, one line on stdout and an empty stderr.
The tool writes nothing; the test writes its input files only in --workdir, a folder of the
build tree (created when missing), and runs the tool from there.

Standard library only. Exit status: 0 passed, 1 failed, 77 when the tool or a data file is
missing. The last line is "SPECTRUM_VERDICT: PASS", "SPECTRUM_VERDICT: FAIL" or
"SPECTRUM_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 77
PREFIX = "sbmc_spectrum: "
KEYS = ["instance", "n", "m", "lambda_min", "lambda_max", "iterations", "restarts", "dt_auto",
        "c0_lambda", "c0_sigma", "t_ms"]
THREE_DECIMALS = re.compile(r"^\d+\.\d{3}$")

# The reference values of F07-C2, section 5.7: (lambda_min, lambda_max, dt_auto, c0_lambda).
REFERENCE = {
    "four_nodes": (-2.56155281280883, 1.5615528128088298, 1.0879054, 0.6403882),
    "G1": (-48.78749417418736, 13.27415171569161, 0.8175546, 0.07533438),
    "dense_n200_s7": (-26.77769686885779, 27.717738844425185, 1.2607352, 0.03607798),
}
G1_TEN = (-48.7874941741805, 12.7715488120668)  # G1 with 10 iterations


class Report:
    """One line per check, and the count of failures."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"SPECTRUM {name}: {'PASS' if ok else 'FAIL'}{suffix}")
        if not ok:
            self.failures += 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checks of sbmc_spectrum.")
    parser.add_argument("--tool", required=True, help="path of the sbmc_spectrum executable")
    parser.add_argument("--data", required=True, help="the folder tests/data")
    parser.add_argument("--gset", required=True, help="the folder data/gset")
    parser.add_argument("--workdir", required=True, help="build folder for the files of the test")
    return parser.parse_args()


class Runner:
    """Runs sbmc_spectrum from the work folder."""

    def __init__(self, executable: str, workdir: Path) -> None:
        self.executable = executable
        self.workdir = workdir

    def __call__(self, arguments: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run([self.executable, *arguments], cwd=self.workdir,
                              capture_output=True, text=True, timeout=600, check=False)


def parse_line(stdout: str) -> dict[str, str] | None:
    """The fields of the only line on stdout, in the order of KEYS, or None."""
    lines = stdout.splitlines()
    if len(lines) != 1:
        return None
    words = lines[0].split(" ")
    if not words or words[0] != "SPECTRUM" or len(words) != len(KEYS) + 1:
        return None
    fields = {}
    for key, word in zip(KEYS, words[1:]):
        name, sep, value = word.partition("=")
        if name != key or sep != "=":
            return None
        fields[key] = value
    return fields


def close_lambdas(fields: dict[str, str], lambda_min: float, lambda_max: float) -> bool:
    try:
        got_min, got_max = float(fields["lambda_min"]), float(fields["lambda_max"])
    except ValueError:
        return False
    tolerance = 1e-9 * max(abs(lambda_min), abs(lambda_max))
    return abs(got_min - lambda_min) <= tolerance and abs(got_max - lambda_max) <= tolerance


def close_relative(text: str, reference: float) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return abs(value - reference) <= 1e-6 * abs(reference)


def positive_float(text: str) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return math.isfinite(value) and value > 0


def check_success(report: Report, name: str, result: subprocess.CompletedProcess,
                  expected: dict[str, str]) -> dict[str, str] | None:
    """Checks 5 and the fields of check 1 common to every case; returns the fields."""
    fields = parse_line(result.stdout)
    report.check(f"{name}: exit 0, one line with the 11 fields in order, empty stderr",
                 result.returncode == 0 and fields is not None and result.stderr == "",
                 f"exit {result.returncode}, stdout {result.stdout.strip()!r}, "
                 f"stderr {result.stderr.strip()!r}")
    if fields is None:
        return None
    wrong = [f"{k}={fields.get(k)!r}" for k, v in expected.items() if fields.get(k) != v]
    if not THREE_DECIMALS.match(fields["t_ms"]):
        wrong.append(f"t_ms={fields['t_ms']!r}")
    report.check(f"{name}: names, sizes, iterations and t_ms", not wrong, ", ".join(wrong))
    return fields


def main() -> int:
    args = parse_arguments()
    print(f"python: {platform.python_version()} ({sys.executable})")
    tool = Path(args.tool).resolve()
    data = Path(args.data).resolve()
    gset = Path(args.gset).resolve()
    workdir = Path(args.workdir).resolve()
    four = data / "four_nodes.rudy"
    g1 = gset / "G1"
    missing_inputs = [str(p) for p in (tool, four, g1) if not p.exists()]
    if missing_inputs:
        print(f"missing: {', '.join(missing_inputs)}")
        print("SPECTRUM_VERDICT: SKIP")
        return EXIT_SKIP
    workdir.mkdir(parents=True, exist_ok=True)
    empty = workdir / "empty_5.rudy"
    empty.write_text("5 0\n", encoding="ascii")
    absent = workdir / "missing.rudy"
    absent.unlink(missing_ok=True)
    run = Runner(str(tool), workdir)
    report = Report()

    # 1 and 5. The three instances of the table.
    cases = [
        ("four_nodes.rudy", ["--instance", str(four)],
         {"instance": "four_nodes", "n": "4", "m": "5", "iterations": "4", "restarts": "0"}),
        ("G1", ["--instance", str(g1)],
         {"instance": "G1", "n": "800", "m": "19176", "iterations": "100", "restarts": "0"}),
        ("--dense 200 --instance-seed 7", ["--dense", "200", "--instance-seed", "7"],
         {"instance": "dense_n200_s7", "n": "200", "m": "19900", "iterations": "100",
          "restarts": "0"}),
    ]
    for name, arguments, expected in cases:
        fields = check_success(report, name, run(arguments), expected)
        if fields is None:
            continue
        lambda_min, lambda_max, dt, c0 = REFERENCE[expected["instance"]]
        report.check(f"{name}: lambda_min and lambda_max",
                     close_lambdas(fields, lambda_min, lambda_max),
                     f"{fields['lambda_min']} {fields['lambda_max']}")
        report.check(f"{name}: dt_auto, c0_lambda and c0_sigma",
                     close_relative(fields["dt_auto"], dt)
                     and close_relative(fields["c0_lambda"], c0)
                     and positive_float(fields["c0_sigma"]),
                     f"dt_auto {fields['dt_auto']}, c0_lambda {fields['c0_lambda']}, "
                     f"c0_sigma {fields['c0_sigma']}")

    # 2. G1 with 10 iterations.
    fields = check_success(report, "G1 --iterations 10",
                           run(["--instance", str(g1), "--iterations", "10"]),
                           {"instance": "G1", "n": "800", "m": "19176", "iterations": "10"})
    if fields is not None:
        report.check("G1 --iterations 10: lambda_min and lambda_max",
                     close_lambdas(fields, *G1_TEN),
                     f"{fields['lambda_min']} {fields['lambda_max']}")

    # 3. A graph without edges.
    fields = check_success(report, "a graph without edges", run(["--instance", str(empty)]),
                           {"instance": "empty_5", "n": "5", "m": "0", "lambda_min": "0",
                            "lambda_max": "0", "dt_auto": "none", "c0_lambda": "none",
                            "c0_sigma": "none"})

    # 4. Errors: exit status 2 for the command line, 1 for reading; empty stdout.
    errors = [
        ("no instance", [], 2),
        ("--instance with --dense", ["--instance", str(four), "--dense", "20",
                                     "--instance-seed", "1"], 2),
        ("--iterations 0", ["--instance", str(four), "--iterations", "0"], 2),
        ("--iterations 100001", ["--instance", str(four), "--iterations", "100001"], 2),
        ("--iterations x", ["--instance", str(four), "--iterations", "x"], 2),
        ("--name with a space", ["--instance", str(four), "--name", "a b"], 2),
        ("a missing file", ["--instance", str(absent)], 1),
    ]
    for name, arguments, status in errors:
        result = run(arguments)
        first = result.stderr.strip().splitlines()[0:1]
        report.check(f"exit {status}: {name}",
                     result.returncode == status and result.stdout == "" and bool(first)
                     and first[0].startswith(PREFIX),
                     f"exit {result.returncode}: {first}")

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"SPECTRUM_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
