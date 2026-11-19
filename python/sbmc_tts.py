#!/usr/bin/env python3
"""sbmc_tts: the time to solution TTS99 of the runs of sbmc_run (reference document, section 8.5).

Usage:
  python3 python/sbmc_tts.py --results FILE [--replicas FILE] [--targets 1000,995,990]
                             [--bootstrap B] [--seed S] [--out FILE]

--results is a CSV of results schema 1 (sbmc/results.hpp); --replicas is the replica file of the
same runs (sbmc_run --replica-out), needed for the relaxed targets (995 and 990 thousandths of the
target) and used by the bootstrap. The rows of results are grouped by the 21 columns of GROUP_KEY,
in the order of first appearance; groups without a target are skipped. For every group and every
target fraction f (in thousandths): the successes over all repetitions, P_s with its Wilson
interval at 95%, the batch time (median of t_solve_ms), TTS99 and its interval from the Wilson
bounds, a bootstrap interval of TTS99 (B resamples with a fixed seed), the best and the mean cut,
and their gaps from the target.

Until results schema 2 (phase 07) t_solve_ms includes the preparation and the final check of the
solver: the TTS of these runs is slightly pessimistic.

Exit status: 0 success; 1 error in the data (missing or unreadable file, header, join of the two
files, cross checks, duplicates); 2 error of the command line. Messages on stderr start with
"sbmc_tts: ". Standard library and NumPy only; importable as a module (pure functions).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
# Columns that define a group of rows of results (runs of the same configuration).
GROUP_KEY = [
    "git_hash", "dirty", "host", "compiler", "algo", "variant", "instance", "n", "nnz", "replicas",
    "steps", "dt", "c0", "beta0", "beta1", "a0", "eval_every", "threads", "target",
    "target_source", "notes",
]
# Columns that join a row of results with its rows in the replica file.
JOIN_KEY = ["timestamp", "instance", "algo", "variant", "replicas", "steps", "seed", "rep"]
SUMMARY_COLUMNS = GROUP_KEY + [
    "target_permille", "target_value", "rows", "replicas_total", "successes", "p_s", "p_s_lo",
    "p_s_hi", "t_batch_ms", "t_batch_q1_ms", "t_batch_q3_ms", "tts99_ms", "tts99_lo_ms",
    "tts99_hi_ms", "bootstrap", "bootstrap_seed", "tts99_boot_lo_ms", "tts99_boot_hi_ms",
    "best_cut", "mean_cut", "gap_best_pct", "gap_mean_pct",
]
DEFAULT_TARGETS = (1000, 995, 990)
EXIT_OK, EXIT_DATA, EXIT_USAGE = 0, 1, 2


class DataError(Exception):
    """An error in the data: exit status 1."""


class UsageError(Exception):
    """An error of the command line: exit status 2."""


# --- Pure functions -------------------------------------------------------------------------


def target_value(permille: int, target: int) -> int:
    """ceil(permille * target / 1000), in integer arithmetic."""
    return -((-permille * target) // 1000)


def is_success(best_cut: int, permille: int, target: int) -> bool:
    """A replica reaches f thousandths of the target: 1000 * best_cut >= f * target."""
    return 1000 * best_cut >= permille * target


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson interval at 95% of successes/total; exactly 0 for 0 successes, 1 for total."""
    z = statistics.NormalDist().inv_cdf(0.975)
    p = successes / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    lo = 0.0 if successes == 0 else center - half
    hi = 1.0 if successes == total else center + half
    return lo, hi


def tts(p: float, t_batch: float, replicas: int) -> float | None:
    """TTS99 of a batch of R replicas with success probability p per replica and time t_batch.

    t_batch when p = 1; undefined (None) when p = 0; otherwise
    t_batch * max(1, ln(0.01) / (R * log1p(-p))): when one batch already reaches the 99%, the TTS
    is the time of the batch.
    """
    if p == 1:
        return t_batch
    if p == 0:
        return None
    return t_batch * max(1.0, math.log(0.01) / (replicas * math.log1p(-p)))


def bootstrap_interval(successes: Sequence[int], times: Sequence[float], replicas: int,
                       resamples: int, seed: Sequence[int]) -> tuple[float, float]:
    """Bootstrap interval of TTS99: rows resampled with replacement, successes redrawn.

    rng = numpy.random.default_rng(seed); for every resample, idx = rng.integers(0, rows,
    size=rows), then rng.binomial(R, s_j / R) for every drawn row j, in the order of idx;
    p_b = sum / (rows * R), t_b = numpy.median of the drawn times, tts_b = inf when p_b = 0.
    The interval is made of the sorted values at the ranks (from 1) ceil(0.025 B) and
    ceil(0.975 B), computed in integer arithmetic.
    """
    import numpy as np

    rows = len(successes)
    rng = np.random.default_rng(list(seed))
    time_array = np.asarray(times, dtype=float)
    values = []
    for _ in range(resamples):
        idx = rng.integers(0, rows, size=rows)
        drawn = 0
        for j in idx:
            drawn += int(rng.binomial(replicas, successes[j] / replicas))
        p_b = drawn / (rows * replicas)
        t_b = float(np.median(time_array[idx]))
        value = tts(p_b, t_b, replicas)
        values.append(math.inf if value is None else value)
    values.sort()
    lo_rank = -((-25 * resamples) // 1000)
    hi_rank = -((-975 * resamples) // 1000)
    return values[lo_rank - 1], values[hi_rank - 1]


def number(value: float | None) -> str:
    """A float of the summary: repr(); empty when undefined; "inf" for infinity."""
    if value is None:
        return ""
    return repr(float(value))


# --- Reading and checking the files ---------------------------------------------------------


@dataclass
class ResultRow:
    line: int
    values: dict[str, str]
    replicas: int
    rep: int
    first_replica: int
    t_solve_ms: float
    target: int | None
    best_cut: int
    mean_cut: str
    n_success: int | None
    replica_cuts: list[int] = field(default_factory=list)  # with --replicas, by replica index


def read_csv(path: Path, columns: list[str], what: str) -> list[tuple[int, dict[str, str]]]:
    """Rows of a CSV with an exact header, with their line numbers."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DataError(f"cannot read {path}: {error}") from error
    if not table or table[0] != columns:
        raise DataError(f"{path}: the first line is not the header of {what}")
    rows = []
    for number_, row in enumerate(table[1:], start=2):
        if len(row) != len(columns):
            raise DataError(f"{path}:{number_}: {len(row)} fields, {len(columns)} expected")
        rows.append((number_, dict(zip(columns, row))))
    return rows


def to_int(path: Path, line: int, column: str, text: str) -> int:
    try:
        if text.strip() != text or not text:
            raise ValueError
        return int(text)
    except ValueError:
        raise DataError(f"{path}:{line}: {column} = '{text}' is not an integer") from None


def to_float(path: Path, line: int, column: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise DataError(f"{path}:{line}: {column} = '{text}' is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise DataError(f"{path}:{line}: {column} = '{text}' is not a finite number >= 0")
    return value


def read_results(path: Path) -> list[ResultRow]:
    rows = []
    for line, values in read_csv(path, RESULTS_COLUMNS, "results schema 1"):
        replicas = to_int(path, line, "replicas", values["replicas"])
        if replicas < 1:
            raise DataError(f"{path}:{line}: replicas = {replicas} must be at least 1")
        target = None
        n_success = None
        if values["target"] != "":
            target = to_int(path, line, "target", values["target"])
            n_success = to_int(path, line, "n_success", values["n_success"])
        rows.append(ResultRow(
            line=line, values=values, replicas=replicas,
            rep=to_int(path, line, "rep", values["rep"]),
            first_replica=to_int(path, line, "first_replica", values["first_replica"]),
            t_solve_ms=to_float(path, line, "t_solve_ms", values["t_solve_ms"]),
            target=target, best_cut=to_int(path, line, "best_cut", values["best_cut"]),
            mean_cut=values["mean_cut"], n_success=n_success))
    return rows


def join_replicas(path: Path, results: list[ResultRow]) -> None:
    """Fills replica_cuts of the considered rows of results from the replica file."""
    by_key: dict[tuple[str, ...], ResultRow] = {}
    for row in results:
        key = tuple(row.values[c] for c in JOIN_KEY)
        if key in by_key:
            raise DataError(f"line {row.line} of the results repeats the key of line "
                            f"{by_key[key].line} ({', '.join(JOIN_KEY)})")
        by_key[key] = row
    found: dict[int, dict[int, int]] = {}  # id(row) -> replica index -> best_cut
    for line, values in read_csv(path, REPLICA_COLUMNS, "the replica file (schema 1)"):
        key = tuple(values[c] for c in JOIN_KEY)
        row = by_key.get(key)
        if row is None:
            raise DataError(f"{path}:{line}: no row of results with this key "
                            f"({', '.join(f'{c}={v}' for c, v in zip(JOIN_KEY, key))})")
        if row.target is None:
            continue  # the replicas of a skipped row are ignored
        replica = to_int(path, line, "replica", values["replica"])
        best_cut = to_int(path, line, "best_cut", values["best_cut"])
        cuts = found.setdefault(id(row), {})
        if replica in cuts:
            raise DataError(f"{path}:{line}: replica {replica} repeated for the row of results at "
                            f"line {row.line}")
        cuts[replica] = best_cut
    for row in results:
        if row.target is None:
            continue
        cuts = found.get(id(row), {})
        expected = range(row.first_replica, row.first_replica + row.replicas)
        if sorted(cuts) != list(expected):
            raise DataError(f"the row of results at line {row.line} needs the replicas "
                            f"{expected.start}..{expected.stop - 1} in {path}, found "
                            f"{len(cuts)} rows")
        row.replica_cuts = [cuts[r] for r in expected]


def cross_check(row: ResultRow, target: int) -> None:
    """A row of results with a target against its replicas: maximum, successes and mean."""
    cuts = row.replica_cuts
    problems = []
    if max(cuts) != row.best_cut:
        problems.append(f"best_cut {row.best_cut}, maximum of the replicas {max(cuts)}")
    successes = sum(c >= target for c in cuts)
    if successes != row.n_success:
        problems.append(f"n_success {row.n_success}, replicas at the target {successes}")
    mean = format(sum(cuts) / row.replicas, ".3f")
    if mean != row.mean_cut:
        problems.append(f"mean_cut {row.mean_cut}, mean of the replicas {mean}")
    if problems:
        raise DataError(f"the row of results at line {row.line} does not match its replicas: "
                        + "; ".join(problems))


# --- Command line ---------------------------------------------------------------------------


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


def parse_targets(text: str) -> list[int]:
    try:
        values = [int(part) for part in text.split(",")]
    except ValueError:
        raise UsageError(f"--targets: '{text}' is not a list of integers") from None
    if any(not 1 <= v <= 1000 for v in values):
        raise UsageError(f"--targets: every value must be in [1, 1000], got '{text}'")
    if len(set(values)) != len(values):
        raise UsageError(f"--targets: repeated value in '{text}'")
    return values


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = Parser(prog="sbmc_tts", description="Time to solution TTS99 of sbmc_run results.")
    parser.add_argument("--results", required=True, help="CSV of results schema 1")
    parser.add_argument("--replicas", help="replica file of the same runs (sbmc_run --replica-out)")
    parser.add_argument("--targets", default=",".join(map(str, DEFAULT_TARGETS)),
                        help="target fractions in thousandths, distinct integers in [1, 1000] "
                             "(default 1000,995,990; without --replicas only 1000)")
    parser.add_argument("--bootstrap", default="1000",
                        help="bootstrap resamples: 0 (none) or at least 1000 (default 1000)")
    parser.add_argument("--seed", default="1", help="seed of the bootstrap, >= 0 (default 1)")
    parser.add_argument("--out", help="CSV of the summary (overwritten)")
    args = parser.parse_args(list(argv))
    args.targets = parse_targets(args.targets)
    if args.replicas is None and args.targets != [1000]:
        raise UsageError("relaxed targets need --replicas (without it use --targets 1000)")
    try:
        args.bootstrap = int(args.bootstrap)
        args.seed = int(args.seed)
    except ValueError:
        raise UsageError("--bootstrap and --seed take integers") from None
    if args.bootstrap != 0 and args.bootstrap < 1000:
        raise UsageError(f"--bootstrap {args.bootstrap}: 0 (no bootstrap) or at least 1000")
    if args.seed < 0:
        raise UsageError(f"--seed {args.seed} must be >= 0")
    return args


def summarize(args: argparse.Namespace) -> int:
    results = read_results(Path(args.results))
    if args.replicas is not None:
        join_replicas(Path(args.replicas), results)

    groups: dict[tuple[str, ...], list[ResultRow]] = {}
    for row in results:
        groups.setdefault(tuple(row.values[c] for c in GROUP_KEY), []).append(row)

    import numpy as np

    summary = []
    group_index = skipped = 0
    for key, rows in groups.items():
        first = rows[0]
        label = (f"instance={first.values['instance']} algo={first.values['algo']} "
                 f"variant={first.values['variant']} R={first.replicas}")
        if first.target is None:
            print(f"SKIP {label}: no target")
            skipped += 1
            continue
        seen: dict[tuple[str, int], int] = {}
        for row in rows:
            pair = (row.values["seed"], row.rep)
            if pair in seen:
                raise DataError(f"lines {seen[pair]} and {row.line} of the results: seed "
                                f"{pair[0]} and rep {pair[1]} twice in the same group (the same "
                                "replicas counted twice)")
            seen[pair] = row.line
            if args.replicas is not None:
                cross_check(row, first.target)

        target, replicas = first.target, first.replicas
        total = len(rows) * replicas
        times = [row.t_solve_ms for row in rows]
        t_batch = statistics.median(times)
        q1, q3 = (float(v) for v in np.percentile(times, [25, 75]))
        if args.replicas is not None:
            best = max(max(row.replica_cuts) for row in rows)
            mean = sum(sum(row.replica_cuts) for row in rows) / total
        else:
            best = max(row.best_cut for row in rows)
            mean = statistics.fmean(float(row.mean_cut) for row in rows)
        gap_best = None if target == 0 else 100 * (target - best) / target
        gap_mean = None if target == 0 else 100 * (target - mean) / target

        for permille_index, permille in enumerate(args.targets):
            if args.replicas is not None:
                per_row = [sum(is_success(c, permille, target) for c in row.replica_cuts)
                           for row in rows]
            else:
                per_row = [row.n_success for row in rows]
            successes = sum(per_row)
            p_s = successes / total
            lo, hi = wilson(successes, total)
            boot_lo = boot_hi = None
            if args.bootstrap > 0:
                boot_lo, boot_hi = bootstrap_interval(per_row, times, replicas, args.bootstrap,
                                                      [args.seed, group_index, permille_index])
            values = [first.values[c] for c in GROUP_KEY] + [
                str(permille), str(target_value(permille, target)), str(len(rows)), str(total),
                str(successes), number(p_s), number(lo), number(hi), number(t_batch), number(q1),
                number(q3), number(tts(p_s, t_batch, replicas)),
                number(tts(hi, t_batch, replicas)),
                number(tts(lo, t_batch, replicas) if lo > 0 else None),
                str(args.bootstrap) if args.bootstrap > 0 else "",
                str(args.seed) if args.bootstrap > 0 else "", number(boot_lo), number(boot_hi),
                str(best), number(mean), number(gap_best), number(gap_mean)]
            summary.append(values)
            print(f"TTS {label} permille={permille} target_value={target_value(permille, target)} "
                  f"p_s={number(p_s)} tts99_ms={number(tts(p_s, t_batch, replicas)) or 'none'}")
        group_index += 1

    if args.out is not None:
        try:
            with open(args.out, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(SUMMARY_COLUMNS)
                writer.writerows(summary)
        except OSError as error:
            raise DataError(f"cannot write {args.out}: {error}") from error
    print(f"TTS_DONE groups={group_index} lines={len(summary)} skipped={skipped}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(sys.argv[1:] if argv is None else argv)
    except UsageError as error:
        print(f"sbmc_tts: {error}", file=sys.stderr)
        return EXIT_USAGE
    try:
        import numpy  # noqa: F401  (needed by the percentiles and the bootstrap)
    except ImportError as error:
        print(f"sbmc_tts: NumPy cannot be imported ({error}); run scripts/setup_python.sh",
              file=sys.stderr)
        return EXIT_DATA
    try:
        return summarize(args)
    except DataError as error:
        print(f"sbmc_tts: {error}", file=sys.stderr)
        return EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
