#!/usr/bin/env python3
"""sbmc_e2: the analysis of the tuning E2 of phase 07 (reference document, section 14.2).

Usage (from the repository root, or with --root):
  python3 python/sbmc_e2.py analyze CONFIG [--bootstrap B] [--seed S]

CONFIG is the campaign of the tuning (bench/configs/f07_e2.toml, run by python/sbmc_campaign.py).
For every configuration of the grid (the stages are excluded) and every level f in {1000, 995,
990} (thousandths of the target): the successes from the replica file, P_s with its Wilson
interval, T_batch (median of t_run_ms) and its quartiles, TTS99 and its interval from the Wilson
bounds, a bootstrap interval of TTS99 (B resamples, default 1000, seed [S, index of the
configuration in the order of the campaign counted from 0 over the whole grid, index of the
level]), the best and the mean cut and their gaps; the definitions are the functions of
python/sbmc_tts.py. Targets: "best-known", the value of data/best_known.csv; an integer, the
value given; "best-found", the largest best_cut of all the runs of the campaign on the instance
(stages and grid): when the grid beats the stages the analysis says so on stdout and writes the
new value in <output>/instances.csv.

Choice, for every instance and method: the level is the most severe one (1000, then 995, then 990)
where at least one configuration of the method has P_s > 0; among the configurations of the method,
the one with the smallest TTS99 at that level; on an exact tie, the smaller T_batch, then the first
in the order of the campaign. No success even at 990: the row says "none".

Files, in the output folder of the campaign: e2_summary.csv (one row per configuration and level,
with the relative parameters a, b or f, c0_rule and the absolute ones beta0, beta1 or dt, c0),
selected.csv (one row per instance and method), plots/sens_<instance>_<method>.pdf and .png
(TTS99 against the steps at the chosen level, one line per (a, b) or (f, c0), and the mean cut
against the steps) and plots/tts_e2.pdf and .png (the TTS99 of the two chosen configurations of
every instance). Lines on stdout: "E2 <instance> <method> level=<f> config=<id> tts99_ms=<x>",
then "E2_DONE".

Exit status: 0 success; 1 error in the data (a configuration not complete, a file that does not
match, matplotlib missing); 2 error of the command line or of the configuration. Messages on
stderr start with "sbmc_e2: ". Standard library, NumPy and matplotlib (Agg backend); importable
as a module: the summary of a configuration, the choice and the plotting helpers are shared with
python/sbmc_e3.py.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # the sibling modules, also under python -I
    sys.path.insert(0, _HERE)

import sbmc_campaign as sc  # noqa: E402
import sbmc_tts  # noqa: E402

PROG = "sbmc_e2"
LEVELS = (1000, 995, 990)
SUMMARY_COLUMNS = [
    "instance", "family", "method", "variant", "config_index", "config_id", "a", "b", "beta0",
    "beta1", "f", "c0_rule", "dt", "c0", "steps", "target", "target_source", "level_permille",
    "level_value", "rows", "replicas_total", "successes", "p_s", "p_s_lo", "p_s_hi",
    "t_batch_ms", "t_batch_q1_ms", "t_batch_q3_ms", "tts99_ms", "tts99_lo_ms", "tts99_hi_ms",
    "bootstrap", "bootstrap_seed", "tts99_boot_lo_ms", "tts99_boot_hi_ms", "best_cut",
    "mean_cut", "gap_best_pct", "gap_mean_pct",
]
SELECTED_COLUMNS = [
    "tuning_instance", "evaluation_instance", "method", "level_permille", "config_id", "a", "b",
    "f", "c0_rule", "steps", "p_s", "p_s_lo", "p_s_hi", "t_batch_ms", "tts99_ms", "tts99_lo_ms",
    "tts99_hi_ms", "tts99_boot_lo_ms", "tts99_boot_hi_ms", "target", "target_source",
]
METHOD_LABELS = {"sa": "SA (C2)", "dsb": "dSB (C1)"}
FIGURE_SIZE = (6.4, 4.8)
WIDE_FIGURE_SIZE = (11.0, 4.4)
PNG_DPI = 150


@dataclass(frozen=True)
class Target:
    value: int
    source: str  # best_known, best_found or given
    note: str


@dataclass
class Level:
    """The statistics of one configuration at one level."""

    permille: int
    value: int
    rows: int
    total: int
    successes: int
    p_s: float
    p_s_lo: float
    p_s_hi: float
    t_batch: float
    q1: float
    q3: float
    tts99: float | None
    tts99_lo: float | None
    tts99_hi: float | None
    boot_lo: float | None
    boot_hi: float | None
    bootstrap: int
    seed: int
    best: int
    mean: float
    gap_best: float | None
    gap_mean: float | None


@dataclass
class Measured:
    """A complete configuration: its rows (with the replicas) and its statistics per level."""

    config: sc.Config
    rows: list[sbmc_tts.ResultRow]
    levels: dict[int, Level]

    def first(self, column: str) -> str:
        return self.rows[0].values[column]

    def median_t_run(self) -> float:
        return statistics.median(r.t_batch for r in self.rows)


# --- Reading and statistics -----------------------------------------------------------------


def load_rows(config: sc.Config, campaign: sc.Campaign, root: Path) -> list[sbmc_tts.ResultRow]:
    """The rows of a complete configuration, joined with their replicas and cross-checked."""
    if sc.config_state(config, campaign, root) != "done":
        raise sc.DataError(f"configuration {config.id} of {config.instance.id} is not complete "
                           f"({config.results_path(campaign)}): run the campaign first")
    try:
        schema, rows = sbmc_tts.read_results(root / config.results_path(campaign))
        if schema.version != 2:
            raise sc.DataError(f"{config.results_path(campaign)}: not results schema 2")
        sbmc_tts.join_replicas(root / config.replica_path(campaign), rows)
        for row in rows:
            if row.target is not None:
                sbmc_tts.cross_check(row, row.target)
    except sbmc_tts.DataError as error:
        raise sc.DataError(str(error)) from error
    return rows


def level_statistics(rows: list[sbmc_tts.ResultRow], target: int, permille: int,
                     bootstrap: int, seed: Sequence[int]) -> Level:
    """P_s, Wilson, T_batch, TTS99 and its intervals at one level (python/sbmc_tts.py)."""
    import numpy as np

    replicas = rows[0].replicas
    per_row = [sum(sbmc_tts.is_success(c, permille, target) for c in row.replica_cuts)
               for row in rows]
    successes = sum(per_row)
    total = len(rows) * replicas
    p_s = successes / total
    lo, hi = sbmc_tts.wilson(successes, total)
    times = [row.t_batch for row in rows]
    t_batch = statistics.median(times)
    q1, q3 = (float(v) for v in np.percentile(times, [25, 75]))
    boot_lo = boot_hi = None
    if bootstrap > 0:
        boot_lo, boot_hi = sbmc_tts.bootstrap_interval(per_row, times, replicas, bootstrap,
                                                       list(seed))
    best = max(max(row.replica_cuts) for row in rows)
    mean = sum(sum(row.replica_cuts) for row in rows) / total
    return Level(
        permille=permille, value=sbmc_tts.target_value(permille, target), rows=len(rows),
        total=total, successes=successes, p_s=p_s, p_s_lo=lo, p_s_hi=hi, t_batch=t_batch, q1=q1,
        q3=q3, tts99=sbmc_tts.tts(p_s, t_batch, replicas),
        tts99_lo=sbmc_tts.tts(hi, t_batch, replicas),
        tts99_hi=sbmc_tts.tts(lo, t_batch, replicas) if lo > 0 else None,
        boot_lo=boot_lo, boot_hi=boot_hi, bootstrap=bootstrap, seed=seed[0], best=best,
        mean=mean, gap_best=100 * (target - best) / target,
        gap_mean=100 * (target - mean) / target)


def campaign_targets(campaign: sc.Campaign, configs: list[sc.Config], root: Path,
                     announce: str) -> dict[str, Target]:
    """The target of every instance for the analysis (see the docstring)."""
    table: dict[str, int] | None = None
    targets = {}
    for instance in campaign.instances:
        if instance.target == "best-known":
            table = sc.best_known_table(root) if table is None else table
            if instance.id not in table:
                raise sc.DataError(f"instance {instance.id}: not in {sc.BEST_KNOWN_FILE}")
            targets[instance.id] = Target(table[instance.id], "best_known", "")
        elif instance.target == "best-found":
            stage = grid = None
            for config in configs:
                if config.instance.id != instance.id:
                    continue
                if sc.config_state(config, campaign, root) != "done":
                    raise sc.DataError(f"configuration {config.id} of {instance.id} is not "
                                       "complete: run the campaign first")
                best = max(int(r["best_cut"]) for r in
                           sc.read_rows(root / config.results_path(campaign)))
                if config.stage == sc.GRID:
                    grid = best if grid is None else max(grid, best)
                else:
                    stage = best if stage is None else max(stage, best)
            assert stage is not None
            value = stage if grid is None else max(stage, grid)
            note = "largest best_cut of the stages"
            if grid is not None and grid > stage:
                note = f"largest best_cut of all the runs (the stages gave {stage})"
                print(f"{announce} {instance.id} best-found target raised from {stage} (stages) "
                      f"to {grid} (grid)")
            targets[instance.id] = Target(value, "best_found", note)
        elif instance.target == "none":
            raise sc.DataError(f"instance {instance.id}: the analysis needs a target, not none")
        else:
            targets[instance.id] = Target(int(instance.target), "given", "")
    return targets


def update_instances(campaign: sc.Campaign, root: Path, targets: dict[str, Target]) -> None:
    """Writes the targets of best-found in <output>/instances.csv, when it exists."""
    path = root / campaign.output / "instances.csv"
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    changed = False
    for row in rows:
        target = targets.get(row["instance"])
        if target is not None and target.source == "best_found":
            if row["target"] != str(target.value) or row["target_note"] != target.note:
                row["target"], row["target_note"] = str(target.value), target.note
                changed = True
    if changed:
        sc.write_csv(path, sc.INSTANCE_COLUMNS, [[r[c] for c in sc.INSTANCE_COLUMNS]
                                                 for r in rows])


def measure(configs: list[sc.Config], campaign: sc.Campaign, root: Path,
            targets: dict[str, Target], bootstrap: int, seed: int) -> list[Measured]:
    """The statistics of every configuration of the grid at the three levels."""
    measured = []
    for config in configs:
        if config.stage != sc.GRID:
            continue
        rows = load_rows(config, campaign, root)
        target = targets[config.instance.id].value
        levels = {permille: level_statistics(rows, target, permille, bootstrap,
                                             [seed, config.index, k])
                  for k, permille in enumerate(LEVELS)}
        measured.append(Measured(config, rows, levels))
    return measured


def summary_rows(measured: list[Measured], targets: dict[str, Target]) -> list[list[str]]:
    out = []
    num = sc.number
    for m in measured:
        c = m.config
        target = targets[c.instance.id]
        for permille in LEVELS:
            s = m.levels[permille]
            out.append([
                c.instance.id, c.instance.family or "", c.method, c.variant, str(c.index), c.id,
                num(c.a), num(c.b), m.first("beta0"), m.first("beta1"), num(c.f), c.c0 or "",
                m.first("dt"), m.first("c0"), str(c.steps), str(target.value), target.source,
                str(permille), str(s.value), str(s.rows), str(s.total), str(s.successes),
                num(s.p_s), num(s.p_s_lo), num(s.p_s_hi), num(s.t_batch), num(s.q1), num(s.q3),
                num(s.tts99), num(s.tts99_lo), num(s.tts99_hi),
                str(s.bootstrap) if s.bootstrap else "", str(s.seed) if s.bootstrap else "",
                num(s.boot_lo), num(s.boot_hi), str(s.best), num(s.mean), num(s.gap_best),
                num(s.gap_mean)])
    return out


def choose(candidates: list[Measured]) -> tuple[int, Measured] | None:
    """The level and the configuration of section 5.3 of F07-C3 (candidates in campaign order)."""
    for permille in LEVELS:
        defined = [m for m in candidates if m.levels[permille].p_s > 0]
        if defined:
            best = min(defined, key=lambda m: (m.levels[permille].tts99,
                                               m.levels[permille].t_batch, m.config.index))
            return permille, best
    return None


# --- Plots ----------------------------------------------------------------------------------


def pyplot():
    """matplotlib.pyplot with the Agg backend (no display)."""
    try:
        import matplotlib
    except ImportError as error:
        raise sc.DataError(f"matplotlib cannot be imported ({error}): run "
                           "scripts/setup_python.sh") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, stem: Path) -> None:
    """stem.pdf and stem.png (150 dpi), without dates in the metadata."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI)
    import matplotlib.pyplot as plt

    plt.close(fig)


def group_label(config: sc.Config) -> str:
    if config.method == "sa":
        return f"a={config.a!r}, b={config.b!r}"
    return f"f={config.f!r}, c0 {config.c0}"


def plot_sensitivity(stem: Path, instance: str, method: str, candidates: list[Measured],
                     permille: int, chosen: Measured | None) -> None:
    """TTS99 against the steps at one level (log-log), and the mean cut against the steps."""
    plt = pyplot()
    fig, (ax_tts, ax_cut) = plt.subplots(1, 2, figsize=WIDE_FIGURE_SIZE)
    groups: dict[str, list[Measured]] = {}
    for m in candidates:
        groups.setdefault(group_label(m.config), []).append(m)
    for g, (label, members) in enumerate(groups.items()):
        members = sorted(members, key=lambda m: m.config.steps)
        shift = 1 + 0.025 * (g - (len(groups) - 1) / 2)  # side by side on the log axis
        xs, ys, lower, upper = [], [], [], []
        missing, open_upper = [], []
        for m in members:
            s = m.levels[permille]
            if s.tts99 is None:
                missing.append(str(m.config.steps))
                continue
            xs.append(m.config.steps * shift)
            ys.append(s.tts99)
            lower.append(s.tts99 - (s.tts99_lo if s.tts99_lo is not None else s.tts99))
            if s.tts99_hi is None:
                open_upper.append(str(m.config.steps))
            upper.append((s.tts99_hi - s.tts99) if s.tts99_hi is not None else 0.0)
        text = label
        if missing:
            text += f"; undefined at {', '.join(missing)}"
        if open_upper:
            text += f"; upper bound infinite at {', '.join(open_upper)}"
        line = ax_tts.errorbar(xs, ys, yerr=[lower, upper], marker="o", capsize=3, label=text)
        cut_x = [m.config.steps for m in members]
        cut_y = [m.levels[permille].mean for m in members]
        ax_cut.plot(cut_x, cut_y, marker="o", color=line[0].get_color(), label=label)
    if chosen is not None and chosen.levels[permille].tts99 is not None:
        keys = list(groups)
        g = keys.index(group_label(chosen.config))
        shift = 1 + 0.025 * (g - (len(keys) - 1) / 2)
        ax_tts.plot([chosen.config.steps * shift], [chosen.levels[permille].tts99], marker="*",
                    markersize=16, color="black", linestyle="none",
                    label=f"chosen: {chosen.config.id}")
    steps_name = "sweeps S" if method == "sa" else "steps M"
    ticks = sorted({m.config.steps for m in candidates})
    for ax in (ax_tts, ax_cut):
        ax.set_xscale("log")
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.minorticks_off()
        ax.set_xlim(ticks[0] / 1.3, ticks[-1] * 1.3)
    if any(m.levels[permille].tts99 is not None for m in candidates):
        ax_tts.set_yscale("log")
    else:
        ax_tts.text(0.5, 0.5, "no success at this level", ha="center", va="center",
                    transform=ax_tts.transAxes)
    ax_tts.set_xlabel(steps_name)
    ax_tts.set_ylabel(f"TTS99 at {permille}/1000 of the target (ms)")
    ax_tts.set_title(f"{instance}, {METHOD_LABELS[method]}: TTS99")
    ax_tts.legend(fontsize=7)
    ax_tts.grid(True, which="both", alpha=0.3)
    ax_cut.set_xlabel(steps_name)
    ax_cut.set_ylabel("mean cut of the replicas")
    ax_cut.set_title(f"{instance}, {METHOD_LABELS[method]}: mean cut")
    ax_cut.legend(fontsize=7)
    ax_cut.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, stem)


def plot_selected(stem: Path, instances: list[str],
                  chosen: dict[tuple[str, str], tuple[int, Measured] | None]) -> None:
    """The TTS99 of the chosen configuration of every instance and method (bars, log y)."""
    plt = pyplot()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    width = 0.38
    labeled: set[str] = set()
    for j, method in enumerate(("dsb", "sa")):
        for k, instance in enumerate(instances):
            x = k + (j - 0.5) * width
            entry = chosen.get((instance, method))
            if entry is None or entry[1].levels[entry[0]].tts99 is None:
                ax.text(x, 0.02, f"{method}: none", ha="center", va="bottom", fontsize=7,
                        rotation=90, transform=ax.get_xaxis_transform())
                continue
            permille, m = entry
            s = m.levels[permille]
            assert s.tts99 is not None
            low = s.tts99 - (s.tts99_lo if s.tts99_lo is not None else s.tts99)
            high = (s.tts99_hi - s.tts99) if s.tts99_hi is not None else 0.0
            ax.bar(x, s.tts99, width, color=f"C{j}", yerr=[[low], [high]], capsize=3,
                   label=None if method in labeled else METHOD_LABELS[method])
            labeled.add(method)
            ax.text(x, s.tts99, str(permille), ha="center", va="bottom", fontsize=7)
    if labeled:
        ax.legend()
        ax.set_yscale("log")
    ax.set_xticks(range(len(instances)))
    ax.set_xticklabels(instances)
    ax.set_ylabel("TTS99 of the chosen configuration (ms)")
    ax.set_title("E2: chosen configurations (label: level in thousandths)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, stem)


# --- Analysis -------------------------------------------------------------------------------


def analyze(campaign: sc.Campaign, root: Path, bootstrap: int, seed: int) -> int:
    configs = sc.expand_configs(campaign, root)
    targets = campaign_targets(campaign, configs, root, "E2")
    update_instances(campaign, root, targets)
    measured = measure(configs, campaign, root, targets, bootstrap, seed)
    output = root / campaign.output
    sc.write_csv(output / "e2_summary.csv", SUMMARY_COLUMNS, summary_rows(measured, targets))

    selected_rows = []
    chosen: dict[tuple[str, str], tuple[int, Measured] | None] = {}
    num = sc.number
    for instance in campaign.instances:
        for method in campaign.methods:
            candidates = [m for m in measured if m.config.instance.id == instance.id
                          and m.config.method == method.id]
            if not candidates:
                continue
            entry = choose(candidates)
            chosen[(instance.id, method.id)] = entry
            target = targets[instance.id]
            if entry is None:
                selected_rows.append([instance.id, instance.family or "", method.id, "", "none"]
                                     + [""] * 14 + [str(target.value), target.source])
                print(f"E2 {instance.id} {method.id} level=none config=none tts99_ms=none")
                plot_level = LEVELS[-1]
                chosen_measured = None
            else:
                permille, m = entry
                s = m.levels[permille]
                c = m.config
                selected_rows.append([
                    instance.id, instance.family or "", method.id, str(permille), c.id,
                    num(c.a), num(c.b), num(c.f), c.c0 or "", str(c.steps), num(s.p_s),
                    num(s.p_s_lo), num(s.p_s_hi), num(s.t_batch), num(s.tts99), num(s.tts99_lo),
                    num(s.tts99_hi), num(s.boot_lo), num(s.boot_hi), str(target.value),
                    target.source])
                print(f"E2 {instance.id} {method.id} level={permille} config={c.id} "
                      f"tts99_ms={num(s.tts99)}")
                plot_level, chosen_measured = permille, m
            plot_sensitivity(output / "plots" / f"sens_{instance.id}_{method.id}", instance.id,
                             method.id, candidates, plot_level, chosen_measured)
    sc.write_csv(output / "selected.csv", SELECTED_COLUMNS, selected_rows)
    plot_selected(output / "plots" / "tts_e2", [i.id for i in campaign.instances], chosen)
    print("E2_DONE")
    return sc.EXIT_OK


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise sc.UsageError(message)


def parse_arguments(argv: Sequence[str], prog: str, what: str) -> argparse.Namespace:
    parser = Parser(prog=prog, description=f"Analysis of {what}.")
    parser.add_argument("--root", default=".", help="repository root (default: the current "
                                                    "directory)")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    analyze_parser = commands.add_parser("analyze", help="summaries, tables and plots")
    analyze_parser.add_argument("config", help="TOML file of the campaign")
    analyze_parser.add_argument("--bootstrap", default="1000",
                                help="bootstrap resamples: 0 (none) or at least 1000 (default "
                                     "1000)")
    analyze_parser.add_argument("--seed", default="1", help="seed of the bootstrap, >= 0 "
                                                            "(default 1)")
    args = parser.parse_args(list(argv))
    try:
        args.bootstrap, args.seed = int(args.bootstrap), int(args.seed)
    except ValueError:
        raise sc.UsageError("--bootstrap and --seed take integers") from None
    if args.bootstrap != 0 and args.bootstrap < 1000:
        raise sc.UsageError(f"--bootstrap {args.bootstrap}: 0 (no bootstrap) or at least 1000")
    if args.seed < 0:
        raise sc.UsageError(f"--seed {args.seed} must be >= 0")
    return args


def run_tool(argv: Sequence[str] | None, prog: str, what: str, analysis) -> int:
    """Common main of sbmc_e2 and sbmc_e3."""
    try:
        args = parse_arguments(sys.argv[1:] if argv is None else argv, prog, what)
        root = Path(args.root).resolve()
        path = Path(args.config)
        campaign = sc.load_campaign(path if path.is_absolute() else root / path)
        try:
            import numpy  # noqa: F401
        except ImportError as error:
            raise sc.DataError(f"NumPy cannot be imported ({error}): run "
                               "scripts/setup_python.sh") from error
        return analysis(campaign, root, args.bootstrap, args.seed)
    except sc.UsageError as error:
        print(f"{prog}: {error}", file=sys.stderr)
        return sc.EXIT_USAGE
    except (sc.DataError, OSError, KeyError, ValueError) as error:
        print(f"{prog}: {error}", file=sys.stderr)
        return sc.EXIT_DATA


def main(argv: Sequence[str] | None = None) -> int:
    return run_tool(argv, PROG, "the tuning E2", analyze)


if __name__ == "__main__":
    sys.exit(main())
