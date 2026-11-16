#!/usr/bin/env python3
"""The data of data/: G-set fingerprints, the 13 committed G-set files read by P0, best_known.csv.

Run by CTest as: <python> -I -B test_data.py --pyroot <root>/python --data-root <root>/data.
Checks:
- data/gset/SHA256SUMS: 71 lines "<sha256>  G<number>" sorted by instance number, SHA-256 of the
  file f37f6fdd...; the 13 files of data/gset/ match their lines;
- P0 (sbmc_ref.read_rudy) on the 13 files: n, m and W as in T6 (W from the edge list: the dense
  coupling matrix of G81 alone would take 3.2 GB);
- data/best_known.csv read with the csv module: the header and the 14 rows of F04-C2, section 5.2;
  for each instance of data/gset/, best_known <= the sum of the positive weights.
Nothing is written.

Exit status: 0 passed, 1 failed, 77 when NumPy cannot be imported (run scripts/setup_python.sh,
then configure again). The last line is "DATA_VERDICT: PASS" or "DATA_VERDICT: FAIL"
("DATA_VERDICT: SKIP" with exit status 77).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import re
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SKIP = 77

SHA256SUMS_SHA256 = "f37f6fdd1d07de7c3344bcdb75010ef48e04bbbc314496f9b3cc930c848ddfe5"
GSET_FILES = 71

# name: (n, m, W, sum of the positive weights)
GSET = {
    "G1": (800, 19176, 19176, 19176),
    "G2": (800, 19176, 19176, 19176),
    "G11": (800, 1600, 34, 817),
    "G12": (800, 1600, -4, 798),
    "G14": (800, 4694, 4694, 4694),
    "G15": (800, 4661, 4661, 4661),
    "G22": (2000, 19990, 19990, 19990),
    "G43": (1000, 9990, 9990, 9990),
    "G48": (3000, 6000, 6000, 6000),
    "G55": (5000, 12498, 12498, 12498),
    "G63": (7000, 41459, 41459, 41459),
    "G70": (10000, 9999, 9999, 9999),
    "G81": (20000, 40000, 34, 20017),
}

HEADER = ["instance", "n", "m", "best_known", "status", "source_1", "source_2", "checked",
          "notes"]
MA = "F. Ma, J.-K. Hao 2017, arXiv:1510.09156, Table 5"
SHEN = "Z.-S. Shen et al. 2024, arXiv:2412.09285, Table S1"
DATE = "2026-11-16"
ROWS = [
    ["G1", "800", "19176", "11624", "best_known", MA, SHEN, DATE, ""],
    ["G2", "800", "19176", "11620", "best_known", MA, SHEN, DATE, ""],
    ["G11", "800", "1600", "564", "best_known", MA, SHEN, DATE, ""],
    ["G12", "800", "1600", "556", "best_known", MA, SHEN, DATE, ""],
    ["G14", "800", "4694", "3064", "best_known", MA, SHEN, DATE, ""],
    ["G15", "800", "4661", "3050", "best_known", MA, SHEN, DATE, ""],
    ["G22", "2000", "19990", "13359", "best_known", MA, SHEN, DATE, ""],
    ["G43", "1000", "9990", "6660", "best_known", MA, SHEN, DATE, ""],
    ["G48", "3000", "6000", "6000", "optimal", MA, SHEN, DATE,
     "equals the number of edges, all of weight +1: every edge is cut"],
    ["G55", "5000", "12498", "10299", "best_known", MA, "", DATE,
     "single source found; U. Benlic, J.-K. Hao 2013 (BLS) reported 10294"],
    ["G63", "7000", "41459", "27052", "best_known",
     "R. Ota et al. 2026, doi:10.5281/zenodo.19189885", "", DATE,
     "public certificate not yet checked here; previous best 27047 "
     "(N. Khan, N. Shukla 2025, arXiv:2510.21105)"],
    ["G70", "10000", "9999", "9595", "best_known",
     "K. M. Zick 2023, arXiv:2311.09275, Table II", "", DATE,
     "attributed by Zick to Chen et al. 2023 (arXiv:2307.00783) and reached by Cosm; "
     "F. Ma, J.-K. Hao 2017 gave 9591"],
    ["G81", "20000", "40000", "14060", "reported_optimal", "K. M. Zick 2025, arXiv:2505.18508",
     "arXiv:2606.25313, Table S2", DATE,
     "reported as proven optimal with an exact solver in arXiv:2606.25313"],
    ["K2000", "2000", "1999000", "33337", "best_known",
     "H. Goto et al. 2021, Sci. Adv. 7, eabe7953",
     "H. Goto, R. Hidaka, K. Tatsumura 2026, arXiv:2508.17655v5", DATE,
     "file data/raw/k2000/WK2000_1.rud; both sources from the same group"],
]

SUMS_LINE = re.compile(r"^([0-9a-f]{64})  G([0-9]+)$")


class Report:
    """One line per check, and the count of failures."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        print(f"DATA {name}: {'PASS' if ok else 'FAIL'}{suffix}")
        if not ok:
            self.failures += 1


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checks of the data of data/.")
    parser.add_argument("--pyroot", required=True, help="folder that contains sbmc_ref")
    parser.add_argument("--data-root", required=True, help="data/ of the repository")
    return parser.parse_args()


def check_sha256sums(data: Path, report: Report) -> None:
    sums = data / "gset" / "SHA256SUMS"
    lines = sums.read_bytes().decode("ascii").split("\n")
    ends_with_newline = lines[-1] == ""
    if ends_with_newline:
        lines.pop()
    parsed = [SUMS_LINE.match(line) for line in lines]
    well_formed = ends_with_newline and all(parsed)
    numbers = [int(m.group(2)) for m in parsed if m]
    report.check("SHA256SUMS format", well_formed and len(lines) == GSET_FILES,
                 f"{len(lines)} lines")
    increasing = all(a < b for a, b in zip(numbers, numbers[1:]))
    report.check("SHA256SUMS order", increasing,
                 f"G{numbers[0]} ... G{numbers[-1]}" if numbers else "empty")
    actual = sha256_of(sums)
    report.check("SHA256SUMS SHA-256", actual == SHA256SUMS_SHA256, actual[:12] + "...")
    listed = {f"G{m.group(2)}": m.group(1) for m in parsed if m}
    committed = sorted(p.name for p in (data / "gset").iterdir() if p.name != "SHA256SUMS")
    report.check("committed files", committed == sorted(GSET), ", ".join(committed))
    for name in GSET:
        actual = sha256_of(data / "gset" / name)
        report.check(f"{name} SHA-256", listed.get(name) == actual, actual[:12] + "...")


def check_p0_reader(data: Path, report: Report, p0) -> dict[str, int]:
    positive = {}
    for name, (n, m, w, plus) in GSET.items():
        graph = p0.read_rudy(data / "gset" / name)
        got = (graph.n, len(graph.w), int(graph.w.sum()), int(graph.w[graph.w > 0].sum()))
        positive[name] = got[3]
        report.check(f"P0 read_rudy {name}", got == (n, m, w, plus),
                     f"n {got[0]}, m {got[1]}, W {got[2]}, positive {got[3]}")
    return positive


def check_best_known(data: Path, report: Report, positive: dict[str, int]) -> None:
    with open(data / "best_known.csv", newline="", encoding="ascii") as handle:
        rows = list(csv.reader(handle))
    report.check("best_known.csv header", rows[:1] == [HEADER], ",".join(rows[0]) if rows else "")
    body = rows[1:]
    report.check("best_known.csv rows", len(body) == len(ROWS), f"{len(body)} rows")
    for expected, actual in zip(ROWS, body):
        report.check(f"best_known.csv {expected[0]}", actual == expected,
                     "" if actual == expected else f"got {actual}")
    by_name = {row[0]: row for row in body if row}
    for name, plus in positive.items():
        value = int(by_name[name][3]) if name in by_name else None
        report.check(f"best_known {name} <= positive weights", value is not None and value <= plus,
                     f"{value} <= {plus}")


def main() -> int:
    args = parse_arguments()
    sys.path.insert(0, str(Path(args.pyroot).resolve()))
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import numpy as np
    except ImportError as error:
        print(f"numpy: cannot be imported ({error}); run scripts/setup_python.sh, then configure "
              "again")
        print("DATA_VERDICT: SKIP")
        return EXIT_SKIP
    print(f"numpy: {np.__version__}")

    import sbmc_ref as p0

    data = Path(args.data_root)
    report = Report()
    check_sha256sums(data, report)
    positive = check_p0_reader(data, report, p0)
    check_best_known(data, report, positive)

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"DATA_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
