#!/usr/bin/env python3
"""P0 alone: the Python reference (python/sbmc_ref) against exact values.

Run by CTest as: <python> -I -B test_p0.py --pyroot <root>/python --data <root>/tests/data.
Checks: T1 on the 4-node example, the two 4 x 6 tori (size of the graph, W, SHA-256 of the
files), the pressure schedule, the automatic c0, the sign, three SB runs on the 4-node graph
with a given x0, and the errors of the rudy reader. No random generator is used.

Exit status: 0 passed, 1 failed, 77 when NumPy cannot be imported (run scripts/setup_python.sh,
then configure again). The last line is "P0_VERDICT: PASS" or "P0_VERDICT: FAIL"
("P0_VERDICT: SKIP" with exit status 77).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import tempfile
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 77

TORUS_SHA256 = {
    "torus_4x6_unit.rudy": "625afa89daf0279edc893e515eb351759661e2d30a91d16a41c4d99d45d7be39",
    "torus_4x6_pm1.rudy": "bf266e3c9d5d46b07c170935a473ad532e9fd28f1b6fc99756d55cfe49482c0f",
}


class Report:
    """One line per check, and the count of failures."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"P0 {name}: {'PASS' if ok else 'FAIL'}{suffix}")
        if not ok:
            self.failures += 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 checks (python/sbmc_ref).")
    parser.add_argument("--pyroot", required=True, help="folder that contains sbmc_ref")
    parser.add_argument("--data", required=True, help="tests/data")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    sys.path.insert(0, str(Path(args.pyroot).resolve()))
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import numpy as np
    except ImportError as error:
        print(f"numpy: cannot be imported ({error}); run scripts/setup_python.sh, then configure "
              "again")
        print("P0_VERDICT: SKIP")
        return EXIT_SKIP
    print(f"numpy: {np.__version__}")

    import sbmc_ref as p0

    data = Path(args.data)
    report = Report()

    # T1: the 4-node example.
    graph = p0.read_rudy(data / "four_nodes.rudy")
    j4 = p0.coupling_matrix(graph)
    edges = list(zip(graph.u.tolist(), graph.v.tolist(), graph.w.tolist()))
    report.check("T1 edges", graph.n == 4 and edges == [(0, 1, 1), (0, 2, 1), (0, 3, 1),
                                                        (1, 2, 1), (2, 3, 1)], str(edges))
    report.check("T1 W", p0.total_weight(j4) == 5)
    spins = np.array([[1, -1, 1, -1], [1, 1, -1, -1], [1, 1, 1, 1]])
    energies = [p0.ising_energy(j4, s) for s in spins]
    cuts = [p0.cut_direct(j4, s) for s in spins]
    report.check("T1 E and C", energies == [-3, -1, 5] and cuts == [4, 3, 0],
                 f"E {energies}, C {cuts}")
    report.check("T1 per replica",
                 p0.ising_energy(j4, spins).tolist() == [-3, -1, 5]
                 and p0.cut_direct(j4, spins).tolist() == [4, 3, 0])

    # The two tori.
    for name, expected_w in (("torus_4x6_unit.rudy", 48), ("torus_4x6_pm1.rudy", 16)):
        torus = p0.read_rudy(data / name)
        w = p0.total_weight(p0.coupling_matrix(torus))
        digest = hashlib.sha256((data / name).read_bytes()).hexdigest()
        report.check(f"torus {name}",
                     torus.n == 24 and len(torus.u) == 48 and w == expected_w
                     and digest == TORUS_SHA256[name],
                     f"n {torus.n}, edges {len(torus.u)}, W {w}, sha256 {digest[:12]}...")

    # Pressure schedule and sign.
    report.check("pressure_schedule",
                 p0.pressure_schedule(1.0, 4).tolist() == [0.75, 0.5, 0.25, 0.0]
                 and p0.pressure_schedule(1.0, 1).tolist() == [0.0])
    signs = p0.sign(np.array([0.0, -0.0, 1e-300, -1e-300]))
    report.check("sign", signs.tolist() == [1, 1, 1, -1] and signs.dtype == np.int64,
                 str(signs.tolist()))

    # Automatic c0.
    for name, expected in (("four_nodes.rudy", 0.6708203932499369),
                           ("torus_4x6_unit.rudy", 0.26926842082973707),
                           ("torus_4x6_pm1.rudy", 0.24713568499516533)):
        c0 = p0.auto_c0(p0.coupling_matrix(p0.read_rudy(data / name)))
        report.check(f"auto_c0 {name}", abs(c0 - expected) <= 1e-12, f"{c0!r}")

    # SB on the 4-node graph with a given x0 (one row per replica).
    x0 = np.array([[0.01, -0.02, 0.03, -0.04], [-0.05, 0.06, 0.07, -0.08],
                   [0.09, 0.01, -0.03, 0.05], [0.0, 0.02, -0.06, 0.04]])
    runs = (
        ("dSB", True, 0, [4, 3, 4, 4], [0, 0, 4, 4], [4, 3, 4, 4]),
        ("bSB eval_every 0", False, 0, [4, 4, 4, 4], [100, 100, 100, 100], None),
        ("bSB eval_every 10", False, 10, [4, 4, 4, 4], [0, 70, 70, 10], None),
    )
    for name, discrete, eval_every, best_cut, best_step, final_cut in runs:
        run = p0.run_sb(j4, x0, steps=100, dt=1.25, a0=1.0, c0=0.3, discrete=discrete,
                        eval_every=eval_every)
        ok = run.best_cut.tolist() == best_cut and run.best_step.tolist() == best_step
        if final_cut is not None:
            ok = ok and run.final_cut.tolist() == final_cut
        report.check(f"SB {name}", ok,
                     f"best_cut {run.best_cut.tolist()}, best_step {run.best_step.tolist()}, "
                     f"final_cut {run.final_cut.tolist()}")

    # Errors of the reader, from temporary files outside the source tree.
    bad_inputs = (
        ("self-loop", "4 1\n1 1 1\n"),
        ("repeated edge", "4 2\n1 2 1\n2 1 1\n"),
        ("zero weight", "4 1\n1 2 0\n"),
        ("missing edge line", "4 2\n1 2 1\n"),
        ("node out of range", "4 1\n1 5 1\n"),
        ("non-integer weight", "4 1\n1 2 1.5\n"),
        # Further rules of the reader.
        ("no header", "\n  \n"),
        ("header without M", "4\n"),
        ("M too large", "4 7\n"),
        ("edge line with two tokens", "4 1\n1 2\n"),
        ("extra edge line", "4 1\n1 2 1\n2 3 1\n"),
    )
    with tempfile.TemporaryDirectory(prefix="sbmc_p0_") as folder:
        for name, text in bad_inputs:
            path = os.path.join(folder, "case.rudy")
            with open(path, "w", encoding="ascii", newline="\n") as handle:
                handle.write(text)
            try:
                p0.read_rudy(path)
                report.check(f"reader error {name}", False, "no ValueError")
            except ValueError as error:
                report.check(f"reader error {name}", True, str(error).replace(folder, "<tmp>"))

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"P0_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
