#!/usr/bin/env python3
"""T8: the sequential reference C0 (C++) against the Python reference P0, through traces.

Run by CTest as: <python> -I -B test_t8.py --pyroot <root>/python --data <root>/tests/data
--tool <sbmc_trace> --workdir <build>/tests/python. For every case the tool runs C0 and writes a
trace (format: tests/tools/sbmc_trace.cpp) into --workdir; this script reads it, with a check of
its exact size, and compares it with P0 (float64):
  A. coherence: header equal to the arguments; J symmetric with a zero diagonal, equal to the J
     of P0 read from the rudy file (or +-1 off the diagonal for the dense instance); W equal to
     P0's; c0 within 1e-6 (relative) of P0's auto_c0, or the bits of 0.3f; |p - p_P0| <= 1e-7;
     x0 in [-0.1, 0.1);
  B. one step at a time: P0 starts from the C++ state k (k = 0: x0 and y = 0), does one step
     with its own p[k] and the c0 of the trace and is compared with the C++ state k + 1:
     |C++ - P0| <= 1e-5 (1 + |P0|) on every element of x and y;
  C. free run (K < M): P0 runs the first K steps of the M-step run from x0 alone:
     |C++ - P0| <= 1e-4 max(|P0|, 0.01) on every element, and the signs of x are identical;
  D. best cut (K = M): P0's best-cut rule on the recorded states gives exactly the best_energy,
     best_step and final_cut of the trace.
One line per case, "max" being the largest ratio between error and tolerance; then one line for
the exit statuses of sbmc_trace on invalid command lines (2) and on errors of the library, of
the instance and of the output file (1). Exit status: 0
passed, 1 failed, 77 when NumPy cannot be imported. The last line is "T8_VERDICT: PASS" or
"T8_VERDICT: FAIL" ("T8_VERDICT: SKIP" with exit status 77).
"""

from __future__ import annotations

import argparse
import platform
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 77

DT = 1.25
A0 = 1.0
DENSE_N = 64
DENSE_SEED = 1
HEADER = struct.Struct("<8sqiiiiiIIIQqq")  # 72 bytes; dt, a0, c0 read as raw float32 bits


@dataclass(frozen=True)
class Case:
    name: str
    instance: str  # a file of tests/data, or "dense" (n = 64, instance seed 1)
    algorithm: str  # "dsb" or "bsb"
    replicas: int
    steps: int
    record: int
    seed: int
    c0: str  # "0.3" or "auto"
    eval_every: int
    checks: str  # "ABD" or "ABC"


CASES = (
    Case("t8_four_dsb", "four_nodes.rudy", "dsb", 4, 20, 20, 7, "0.3", 0, "ABD"),
    Case("t8_four_bsb", "four_nodes.rudy", "bsb", 4, 20, 20, 7, "0.3", 5, "ABD"),
    Case("t8_torus_dsb", "torus_4x6_pm1.rudy", "dsb", 4, 30, 30, 20, "auto", 0, "ABD"),
    Case("t8_torus_bsb", "torus_4x6_pm1.rudy", "bsb", 4, 30, 30, 20, "auto", 3, "ABD"),
    Case("t8_dense_dsb", "dense", "dsb", 3, 20, 20, 9, "auto", 0, "ABD"),
    Case("t8_dense_bsb", "dense", "bsb", 3, 20, 20, 9, "auto", 1, "ABD"),
    Case("t8_torus_dsb_early", "torus_4x6_pm1.rudy", "dsb", 4, 1000, 20, 11, "auto", 0, "ABC"),
    Case("t8_torus_bsb_early", "torus_4x6_pm1.rudy", "bsb", 4, 1000, 20, 11, "auto", 0, "ABC"),
    Case("t8_dense_dsb_early", "dense", "dsb", 3, 1000, 20, 10, "auto", 0, "ABC"),
    Case("t8_dense_bsb_early", "dense", "bsb", 3, 1000, 20, 10, "auto", 0, "ABC"),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T8: C0 against P0 through sbmc_trace.")
    parser.add_argument("--pyroot", required=True, help="folder that contains sbmc_ref")
    parser.add_argument("--data", required=True, help="tests/data")
    parser.add_argument("--tool", required=True, help="the sbmc_trace executable")
    parser.add_argument("--workdir", required=True, help="build folder for the traces")
    return parser.parse_args()


def read_trace(path: Path, np):
    """Fields of a trace file as a dict; ValueError if the magic or the size is wrong."""
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: {len(data)} bytes, shorter than the header")
    (magic, n, replicas, steps, record, algorithm, eval_every, dt_bits, a0_bits, c0_bits, seed,
     first_replica, total_weight) = HEADER.unpack_from(data, 0)
    if magic != b"SBMCTRC1":
        raise ValueError(f"{path}: magic {magic!r} is not SBMCTRC1")
    expected = (HEADER.size + 4 * steps + n * n + 4 * replicas * n
                + 8 * record * replicas * n + 28 * replicas)
    if len(data) != expected:
        raise ValueError(f"{path}: {len(data)} bytes, the format needs {expected}")
    fields = {
        "n": n, "replicas": replicas, "steps": steps, "record": record, "algorithm": algorithm,
        "eval_every": eval_every, "dt_bits": dt_bits, "a0_bits": a0_bits, "c0_bits": c0_bits,
        "seed": seed, "first_replica": first_replica, "total_weight": total_weight,
    }
    offset = HEADER.size

    def take(dtype: str, count: int, shape=None):
        nonlocal offset
        array = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += array.nbytes
        return array.reshape(shape) if shape is not None else array

    fields["p"] = take("<f4", steps)
    fields["j"] = take("<i1", n * n, (n, n)).astype(np.int64)
    fields["x0"] = take("<f4", replicas * n, (replicas, n))
    fields["trace_x"] = take("<f4", record * replicas * n, (record, replicas, n))
    fields["trace_y"] = take("<f4", record * replicas * n, (record, replicas, n))
    fields["best_energy"] = take("<i8", replicas)
    fields["best_cut"] = take("<i8", replicas)
    fields["best_step"] = take("<i4", replicas)
    fields["final_cut"] = take("<i8", replicas)
    return fields


def float32_bits(value: float, np) -> int:
    return int(np.array(value, dtype=np.float32).view(np.uint32))


def run_case(case: Case, args: argparse.Namespace, np, p0) -> bool:
    """Runs the tool, reads the trace and prints the line of the case; True if it passes."""
    data = Path(args.data)
    trace_path = Path(args.workdir) / f"{case.name}.trc"
    command = [args.tool]
    if case.instance == "dense":
        command += ["--dense", str(DENSE_N), "--instance-seed", str(DENSE_SEED)]
    else:
        command += ["--rudy", str(data / case.instance)]
    command += ["--algorithm", case.algorithm, "--replicas", str(case.replicas),
                "--steps", str(case.steps), "--record", str(case.record),
                "--seed", str(case.seed), "--dt", str(DT), "--a0", str(A0), "--c0", case.c0,
                "--eval-every", str(case.eval_every), "--out", str(trace_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(f"  sbmc_trace exit status {completed.returncode}: {completed.stderr.strip()}")
        print(f"T8 {case.name}: tool error FAIL")
        return False
    try:
        t = read_trace(trace_path, np)
    except ValueError as error:
        print(f"  {error}")
        print(f"T8 {case.name}: trace error FAIL")
        return False

    problems = []
    discrete = case.algorithm == "dsb"
    j = t["j"]
    n = t["n"]

    # A. Coherence.
    if case.instance == "dense":
        expected_n = DENSE_N
        off_diagonal = j[~np.eye(n, dtype=bool)]
        if not np.all(np.abs(off_diagonal) == 1):
            problems.append("A: J of the dense instance is not +-1 off the diagonal")
    else:
        reference_j = p0.coupling_matrix(p0.read_rudy(data / case.instance))
        expected_n = reference_j.shape[0]
        if j.shape != reference_j.shape or not np.array_equal(j, reference_j):
            problems.append("A: J differs from P0's coupling_matrix")
    header = (t["n"], t["replicas"], t["steps"], t["record"], t["algorithm"], t["eval_every"],
              t["dt_bits"], t["a0_bits"], t["seed"], t["first_replica"])
    expected_header = (expected_n, case.replicas, case.steps, case.record, 1 if discrete else 0,
                       case.eval_every, float32_bits(DT, np), float32_bits(A0, np), case.seed, 0)
    if header != expected_header:
        problems.append(f"A: header {header} differs from the arguments {expected_header}")
    if not np.array_equal(j, j.T) or np.any(np.diag(j) != 0):
        problems.append("A: J is not symmetric with a zero diagonal")
    w = p0.total_weight(j)
    if t["total_weight"] != w:
        problems.append(f"A: W {t['total_weight']} differs from P0's {w}")
    c0 = float(np.array(t["c0_bits"], dtype=np.uint32).view(np.float32))
    if case.c0 == "auto":
        reference_c0 = p0.auto_c0(j)
        if abs(c0 - reference_c0) > 1e-6 * reference_c0:
            problems.append(f"A: c0 {c0!r} differs from P0's auto_c0 {reference_c0!r}")
    elif t["c0_bits"] != float32_bits(float(case.c0), np):
        problems.append(f"A: c0 bits {t['c0_bits']:#010x} are not those of {case.c0}f")
    p_reference = p0.pressure_schedule(A0, case.steps)
    p_error = float(np.max(np.abs(t["p"].astype(np.float64) - p_reference)))
    if p_error > 1e-7:
        problems.append(f"A: |p - p_P0| reaches {p_error:.3g}")
    x0 = t["x0"].astype(np.float64)
    if np.any(x0 < -0.1) or np.any(x0 >= 0.1):
        problems.append("A: x0 outside [-0.1, 0.1)")

    trace_x = t["trace_x"].astype(np.float64)
    trace_y = t["trace_y"].astype(np.float64)

    # B. One step at a time, from the C++ states.
    one_step = 0.0
    for k in range(case.record):
        x = x0 if k == 0 else trace_x[k - 1]
        y = np.zeros_like(x0) if k == 0 else trace_y[k - 1]
        x_p0, y_p0 = p0.sb_step(j, x, y, p_reference[k], DT, A0, c0, discrete)
        for cpp, ref in ((trace_x[k], x_p0), (trace_y[k], y_p0)):
            ratio = np.abs(cpp - ref) / (1e-5 * (1.0 + np.abs(ref)))
            one_step = max(one_step, float(np.max(ratio)))
    if one_step > 1.0:
        problems.append(f"B: one-step error {one_step:.3g} times the tolerance")
    line = f"T8 {case.name}: one-step max={one_step:.3g}"

    # C. Free run of the first K steps of the M-step run.
    if "C" in case.checks:
        free_run = 0.0
        same_signs = True
        x, y = x0, np.zeros_like(x0)
        for k in range(case.record):
            x, y = p0.sb_step(j, x, y, p_reference[k], DT, A0, c0, discrete)
            for cpp, ref in ((trace_x[k], x), (trace_y[k], y)):
                ratio = np.abs(cpp - ref) / (1e-4 * np.maximum(np.abs(ref), 0.01))
                free_run = max(free_run, float(np.max(ratio)))
            same_signs = same_signs and np.array_equal(p0.sign(trace_x[k]), p0.sign(x))
        if free_run > 1.0:
            problems.append(f"C: free-run error {free_run:.3g} times the tolerance")
        if not same_signs:
            problems.append("C: the signs of x differ")
        line += (f" free-run max={free_run:.3g} "
                 f"signs={'identical' if same_signs else 'different'}")

    # D. Best cut from the recorded states.
    if "D" in case.checks:
        best_energy, best_step, final_energy = p0.best_from_states(
            j, x0, trace_x, discrete, case.eval_every)
        equal = (np.array_equal(best_energy, t["best_energy"])
                 and np.array_equal(best_step, t["best_step"])
                 and np.array_equal((w - final_energy) // 2, t["final_cut"])
                 and np.array_equal((w - t["best_energy"]) // 2, t["best_cut"]))
        if not equal:
            problems.append(
                f"D: P0 best_energy {best_energy.tolist()}, best_step {best_step.tolist()}, "
                f"final_cut {((w - final_energy) // 2).tolist()}; trace best_energy "
                f"{t['best_energy'].tolist()}, best_step {t['best_step'].tolist()}, final_cut "
                f"{t['final_cut'].tolist()}, best_cut {t['best_cut'].tolist()}")
        line += f" best={'equal' if equal else 'different'}"

    for problem in problems:
        print(f"  {problem}")
    print(f"{line} {'PASS' if not problems else 'FAIL'}")
    return not problems


def check_tool_errors(args: argparse.Namespace) -> bool:
    """Exit statuses of sbmc_trace: 2 for an invalid command line, 1 for the errors of the
    library, of the instance and of the output file, 0 for a valid command."""
    data = Path(args.data)
    work = Path(args.workdir)
    heavy = work / "t8_heavy_weight.rudy"  # |w| = 200 does not fit the INT8 J of the trace
    heavy.write_text("3 1\n1 2 200\n", encoding="ascii")
    base = {"--rudy": str(data / "four_nodes.rudy"), "--algorithm": "dsb", "--replicas": "2",
            "--steps": "5", "--record": "5", "--seed": "1",
            "--out": str(work / "t8_errors.trc")}

    def command(change=None, remove=(), extra=()):
        options = dict(base)
        options.update(change or {})
        for key in remove:
            del options[key]
        line = [args.tool]
        for key, value in options.items():
            line += [key, value]
        return line + list(extra)

    dense = {"--dense": "4", "--instance-seed": "1"}
    cases = (
        ("valid command", command(), 0),
        ("no arguments", [args.tool], 2),
        ("unknown option", command(extra=["--bogus", "1"]), 2),
        ("repeated option", command(extra=["--steps", "5"]), 2),
        ("option without value", command(extra=["--dt"]), 2),
        ("integer not decimal", command({"--replicas": "x"}), 2),
        ("integer out of range", command({"--replicas": "2147483648"}), 2),
        ("negative seed", command({"--seed": "-1"}), 2),
        ("float not finite", command({"--dt": "inf"}), 2),
        ("float not a number", command({"--c0": "abc"}), 2),
        ("unknown algorithm", command({"--algorithm": "sa"}), 2),
        ("record 0", command({"--record": "0"}), 2),
        ("record above steps", command({"--record": "6"}), 2),
        ("missing required option", command(remove=["--seed"]), 2),
        ("no instance", command(remove=["--rudy"]), 2),
        ("two instances", command(dense), 2),
        ("dense without instance seed", command({"--dense": "4"}, remove=["--rudy"]), 2),
        ("instance seed with rudy", command({"--instance-seed": "1"}), 2),
        ("library: replicas 0", command({"--replicas": "0"}), 1),
        ("library: missing rudy file", command({"--rudy": str(data / "missing.rudy")}), 1),
        ("instance: n * n above 2^24",
         command({"--dense": "4097", "--instance-seed": "1"}, remove=["--rudy"]), 1),
        ("instance: |w| above 127", command({"--rudy": str(heavy)}), 1),
        ("library: c0 auto undefined",
         command({"--dense": "2", "--instance-seed": "0"}, remove=["--rudy"]), 1),
        ("output folder missing", command({"--out": str(work / "missing" / "x.trc")}), 1),
    )
    failures = []
    for name, line, expected in cases:
        status = subprocess.run(line, capture_output=True, text=True, check=False).returncode
        if status != expected:
            failures.append(f"{name}: exit status {status}, expected {expected}")
    for failure in failures:
        print(f"  {failure}")
    passed = len(cases) - len(failures)
    print(f"T8 sbmc_trace errors: {passed}/{len(cases)} exit statuses as expected "
          f"{'PASS' if not failures else 'FAIL'}")
    return not failures


def main() -> int:
    args = parse_arguments()
    sys.path.insert(0, str(Path(args.pyroot).resolve()))
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import numpy as np
    except ImportError as error:
        print(f"numpy: cannot be imported ({error}); run scripts/setup_python.sh, then configure "
              "again")
        print("T8_VERDICT: SKIP")
        return EXIT_SKIP
    print(f"numpy: {np.__version__}")

    import sbmc_ref as p0

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    passed = [run_case(case, args, np, p0) for case in CASES]
    passed.append(check_tool_errors(args))
    verdict = "PASS" if all(passed) else "FAIL"
    print(f"T8_VERDICT: {verdict}")
    return EXIT_PASS if all(passed) else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
