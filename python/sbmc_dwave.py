#!/usr/bin/env python3
"""sbmc_dwave: the external check of our SA against dwave-samplers (reference document, section 7,
rule 5; decision L7 of phase 07).

Usage (from the repository root, or with --root):
  python3 python/sbmc_dwave.py check CONFIG --notes TEXT

CONFIG is a campaign (bench/configs/f07_dwave.toml) with the method sa and exactly one
configuration per instance (a, b and S of a selection, or lists of one value each), target none.
Without dwave-samplers importable (scripts/setup_python.sh --dwave) the tool writes
"DWAVE_VERDICT: SKIP (dwave-samplers not installed)", writes no file and exits with 0.

For every instance:
- our side: the configuration run by python/sbmc_campaign.py (resumable: a complete one is not run
  again), sbmc_run --algo sa --variant c2 with the threads, seed, reps and warm-up of the campaign
  (f07_dwave.toml: one thread, seed 1, one repetition after one warm-up run), with --out in
  <output>/<instance>/sa/ and --replica-out in <output>/raw/<instance>/sa/; the final cuts from the
  column final_cut of the replica file; ns per proposal = t_run_ms * 10^6 / (R * S * n);
- dwave: the rudy file read here (weights of a repeated pair added), the model
  dimod.BinaryQuadraticModel.from_ising with h = 0 on the n nodes (in their order) and
  J(u, v) = w_uv, W = sum of w; the schedule [beta0 * (beta1 / beta0)^(k / (S - 1)) for k = 0, ...,
  S - 1] in double ([beta0] for S = 1), the same as sa_beta; SimulatedAnnealingSampler().sample(
  bqm, beta_schedule_type="custom", beta_schedule=schedule, num_reads=R, seed=<[dwave] seed>,
  randomize_order=False, proposal_acceptance_criteria="Metropolis"); final cuts (W - energy) / 2;
  ns per proposal = info["timing"]["sampling_ns"] / (R * S * n);
- statistics: z = (mean ours - mean dwave) / sqrt(var_ours / n_ours + var_dwave / n_dwave) with
  sample variances (ddof = 1); the two-sample Kolmogorov-Smirnov D = max over the values of the
  union of |F_ours(v) - F_dwave(v)| (empirical distribution functions, <= v); D_crit = 1.95 *
  sqrt((n_ours + n_dwave) / (n_ours * n_dwave)) (alpha = 0.001; conservative with repeated
  values). PASS when |z| <= 3 and D <= D_crit.

Files: <output>/dwave_check.csv (one row per instance), <output>/final_cuts_<instance>.csv (columns
ours and dwave, R rows), <output>/plots/final_cuts_<instance>.pdf and .png (the two histograms),
and the manifest of <output>/raw/ (as python/sbmc_campaign.py). Lines on stdout: "DWAVE <instance>:
PASS|FAIL (z=..., D=..., D_crit=..., ns ours=..., dwave=...)", then "DWAVE_VERDICT: PASS|FAIL". A
FAIL is a result, not an error: the exit status is 0. Exit status 1 for an error in the data or
in sbmc_run, 2 for an error of the command line or of the configuration; messages on stderr start
with "sbmc_dwave: ". Standard library, NumPy and matplotlib; dwave-samplers only here and only when
installed; importable as a module (pure functions: z_score, ks_statistic, ks_critical, schedule).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import signal
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # the sibling modules, also under python -I
    sys.path.insert(0, _HERE)

import sbmc_campaign as sc  # noqa: E402

PROG = "sbmc_dwave"
CHECK_COLUMNS = [
    "instance", "n", "m", "steps", "beta0", "beta1", "reads", "seed", "ours_mean", "ours_sd",
    "ours_median", "ours_min", "ours_max", "dwave_mean", "dwave_sd", "dwave_median", "dwave_min",
    "dwave_max", "z", "ks_d", "ks_d_crit", "ns_per_proposal_ours", "ns_per_proposal_dwave",
    "ratio", "verdict",
]
Z_LIMIT = 3.0
KS_FACTOR = 1.95  # alpha = 0.001


# --- Pure functions -------------------------------------------------------------------------


def schedule(beta0: float, beta1: float, steps: int) -> list[float]:
    """beta0 * (beta1 / beta0)^(k / (S - 1)), k = 0, ..., S - 1, in double; [beta0] for S = 1."""
    if steps == 1:
        return [beta0]
    return [beta0 * (beta1 / beta0) ** (k / (steps - 1)) for k in range(steps)]


def z_score(ours: Sequence[float], theirs: Sequence[float]) -> float:
    """(mean ours - mean theirs) / sqrt(var ours / n ours + var theirs / n theirs), ddof = 1."""
    mean_a, mean_b = statistics.fmean(ours), statistics.fmean(theirs)
    se = math.sqrt(statistics.variance(ours) / len(ours)
                   + statistics.variance(theirs) / len(theirs))
    if se == 0:
        return 0.0 if mean_a == mean_b else math.copysign(math.inf, mean_a - mean_b)
    return (mean_a - mean_b) / se


def ks_statistic(ours: Sequence[float], theirs: Sequence[float]) -> float:
    """max over the values of the union of |F_ours(v) - F_theirs(v)| (distribution functions
    <= v)."""
    a, b = sorted(ours), sorted(theirs)
    return max(abs(bisect.bisect_right(a, v) / len(a) - bisect.bisect_right(b, v) / len(b))
               for v in set(a) | set(b))


def ks_critical(n_ours: int, n_theirs: int) -> float:
    return KS_FACTOR * math.sqrt((n_ours + n_theirs) / (n_ours * n_theirs))


def verdict(z: float, d: float, d_crit: float) -> bool:
    return abs(z) <= Z_LIMIT and d <= d_crit


# --- The two sides --------------------------------------------------------------------------


def read_graph(path: Path) -> tuple[int, int, dict[tuple[int, int], int]]:
    """n, edge lines and the weights per pair (0-based, u < v, repeated pairs added)."""
    try:
        tokens = path.read_text(encoding="ascii").split()
        n, m = int(tokens[0]), int(tokens[1])
        values = [int(t) for t in tokens[2:]]
    except (OSError, UnicodeDecodeError, IndexError, ValueError) as error:
        raise sc.DataError(f"cannot read the rudy file {path}: {error}") from error
    if len(values) != 3 * m:
        raise sc.DataError(f"{path}: the header does not match the edges")
    pairs: dict[tuple[int, int], int] = {}
    for k in range(m):
        u, v, w = values[3 * k] - 1, values[3 * k + 1] - 1, values[3 * k + 2]
        if not (0 <= u < n and 0 <= v < n) or u == v:
            raise sc.DataError(f"{path}: edge {k + 1} is not a pair of distinct nodes")
        key = (min(u, v), max(u, v))
        pairs[key] = pairs.get(key, 0) + w
    return n, m, pairs


def dwave_cuts(n: int, pairs: dict[tuple[int, int], int], beta_schedule: list[float], reads: int,
               seed: int) -> tuple[list[int], float]:
    """The final cuts of SimulatedAnnealingSampler and its sampling time in ns."""
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = dimod.BinaryQuadraticModel.from_ising({i: 0.0 for i in range(n)},
                                                {k: float(w) for k, w in pairs.items()})
    result = SimulatedAnnealingSampler().sample(
        bqm, beta_schedule_type="custom", beta_schedule=beta_schedule, num_reads=reads, seed=seed,
        randomize_order=False, proposal_acceptance_criteria="Metropolis")
    total = sum(pairs.values())
    cuts = []
    for energy in result.record.energy:
        value = (total - float(energy)) / 2
        if abs(value - round(value)) > 1e-6:
            raise sc.DataError(f"dwave: the cut {value!r} from the energy {energy!r} is not an "
                               "integer")
        cuts.append(int(round(value)))
    return cuts, float(result.info["timing"]["sampling_ns"])


def our_cuts(config: sc.Config, campaign: sc.Campaign, root: Path) -> tuple[list[int], float]:
    """The final cuts of the replica file of our run (rep 0) and its t_run_ms."""
    rows = sc.read_rows(root / config.results_path(campaign))
    first = [r for r in rows if r["rep"] == "0"]
    if len(first) != 1:
        raise sc.DataError(f"{config.results_path(campaign)}: no single row of rep 0")
    with open(root / config.replica_path(campaign), newline="", encoding="utf-8") as handle:
        replicas = [r for r in csv.DictReader(handle) if r["rep"] == "0"]
    replicas.sort(key=lambda r: int(r["replica"]))
    return [int(r["final_cut"]) for r in replicas], float(first[0]["t_run_ms"])


def plot(stem: Path, instance: str, ours: list[int], theirs: list[int]) -> None:
    import sbmc_e2

    plt = sbmc_e2.pyplot()
    fig, ax = plt.subplots(figsize=sbmc_e2.FIGURE_SIZE)
    lo, hi = min(ours + theirs), max(ours + theirs)
    bins = [v - 0.5 for v in range(lo, hi + 2)] if hi - lo <= 200 else 60
    ax.hist(ours, bins=bins, alpha=0.55, label="ours: SA (C2), one thread")
    ax.hist(theirs, bins=bins, alpha=0.55, label="dwave-samplers SimulatedAnnealingSampler")
    ax.set_xlabel("final cut of a replica")
    ax.set_ylabel("replicas")
    ax.set_title(f"External check of SA: final cuts, {instance}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sbmc_e2.save_figure(fig, stem)


def summary(values: list[int]) -> list[str]:
    num = sc.number
    return [num(statistics.fmean(values)), num(statistics.stdev(values)),
            num(float(statistics.median(values))), str(min(values)), str(max(values))]


def check(campaign: sc.Campaign, root: Path, notes: str) -> int:
    try:
        import dimod  # noqa: F401
        from dwave.samplers import SimulatedAnnealingSampler  # noqa: F401
    except ImportError:
        print("DWAVE_VERDICT: SKIP (dwave-samplers not installed)")
        return sc.EXIT_OK
    if campaign.dwave_seed is None:
        raise sc.UsageError(f"{campaign.path}: missing table [dwave] (key 'seed')")
    configs = sc.expand_configs(campaign, root)
    grid = [c for c in configs if c.stage == sc.GRID]
    if any(c.stage != sc.GRID for c in configs) or any(c.method != "sa" for c in grid):
        raise sc.UsageError(f"{campaign.path}: the external check takes only the method sa and "
                            "no stage")
    for instance in campaign.instances:
        count = sum(c.instance.id == instance.id for c in grid)
        if count != 1 or instance.dense or instance.target != "none":
            raise sc.UsageError(f"{campaign.path}: instance {instance.id}: the check needs a rudy "
                                f"file, the target none and one configuration ({count} found)")
    if not (root / campaign.sbmc_run).is_file():
        raise sc.DataError(f"{campaign.sbmc_run} not found (build the preset first)")
    output = root / campaign.output
    log = sc.Log(output / "campaign.log")
    previous = {s: signal.signal(s, sc._signal_handler) for s in (signal.SIGINT, signal.SIGTERM)}
    rows, passed = [], True
    started = time.monotonic()
    log.event("start", note=f"campaign={campaign.name} check=dwave notes={notes}")
    try:
        for config in grid:
            instance = config.instance
            scales = {instance.id: sc.instance_scales(instance, root)}
            cmd = sc.command(config, campaign, scales, "none", notes)
            if sc._run_one(config, campaign, root, cmd, log) != sc.EXIT_OK:
                raise sc.DataError(f"sbmc_run failed on {instance.id}: see "
                                   f"{campaign.output}/campaign.log")
            ours, t_run = our_cuts(config, campaign, root)
            n, m, pairs = read_graph(root / instance.file)
            h, w = scales[instance.id]
            assert config.a is not None and config.b is not None and w is not None
            beta0, beta1 = config.a / h, config.b / w
            theirs, sampling_ns = dwave_cuts(n, pairs, schedule(beta0, beta1, config.steps),
                                             campaign.replicas, campaign.dwave_seed)
            proposals = campaign.replicas * config.steps * n
            ns_ours, ns_dwave = t_run * 1e6 / proposals, sampling_ns / proposals
            z, d = z_score(ours, theirs), ks_statistic(ours, theirs)
            d_crit = ks_critical(len(ours), len(theirs))
            ok = verdict(z, d, d_crit)
            passed = passed and ok
            num = sc.number
            rows.append([instance.id, str(n), str(m), str(config.steps), num(beta0), num(beta1),
                         str(campaign.replicas), str(campaign.dwave_seed), *summary(ours),
                         *summary(theirs), num(z), num(d), num(d_crit), num(ns_ours),
                         num(ns_dwave), num(ns_dwave / ns_ours), "PASS" if ok else "FAIL"])
            sc.write_csv(output / f"final_cuts_{instance.id}.csv", ["ours", "dwave"],
                         [[str(a), str(b)] for a, b in zip(ours, theirs)])
            plot(output / "plots" / f"final_cuts_{instance.id}", instance.id, ours, theirs)
            print(f"DWAVE {instance.id}: {'PASS' if ok else 'FAIL'} (z={z:.3f}, D={d:.4f}, "
                  f"D_crit={d_crit:.4f}, ns ours={ns_ours:.2f}, dwave={ns_dwave:.2f})", flush=True)
        sc.write_csv(output / "dwave_check.csv", CHECK_COLUMNS, rows)
        log.event("stop", seconds=time.monotonic() - started, status=sc.EXIT_OK,
                  note=f"check {'PASS' if passed else 'FAIL'}")
    except sc.Interrupted as interrupted:
        log.event("stop", seconds=time.monotonic() - started, status=sc.EXIT_DATA,
                  note=f"interrupted by signal {interrupted.args[0]}")
        return sc.EXIT_DATA
    except sc.DataError as error:
        log.event("fail", seconds=time.monotonic() - started, status=sc.EXIT_DATA, note=str(error))
        raise
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        sc.write_manifest(campaign, root)
        log.close()
    print(f"DWAVE_VERDICT: {'PASS' if passed else 'FAIL'}")
    return sc.EXIT_OK


# --- Command line ---------------------------------------------------------------------------


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise sc.UsageError(message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = Parser(prog=PROG, description="External check of SA with dwave-samplers.")
    parser.add_argument("--root", default=".", help="repository root (default: the current "
                                                    "directory)")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    check_parser = commands.add_parser("check", help="run both sides and compare them")
    check_parser.add_argument("config", help="TOML file of the check")
    check_parser.add_argument("--notes", required=True, help="text of the column notes")
    try:
        args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
        if not args.notes or "\n" in args.notes or "\r" in args.notes:
            raise sc.UsageError("--notes must be one non-empty line")
        root = Path(args.root).resolve()
        path = Path(args.config)
        campaign = sc.load_campaign(path if path.is_absolute() else root / path)
        return check(campaign, root, args.notes)
    except sc.UsageError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return sc.EXIT_USAGE
    except (sc.DataError, OSError, KeyError, ValueError) as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return sc.EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
