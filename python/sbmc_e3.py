#!/usr/bin/env python3
"""sbmc_e3: the analysis of E3 on the CPU, dSB (C1) against SA (C2) (reference document, 14.2).

Usage (from the repository root, or with --root):
  python3 python/sbmc_e3.py analyze CONFIG [--bootstrap B] [--seed S]

CONFIG is the campaign of E3 (bench/configs/f07_e3.toml): on every evaluation instance, each
method with the relative parameters chosen by E2 on the tuning instance of its family (the
selection of the campaign, results/f07/e2/selected.csv) and the four steps of the grid of E2.
The statistics of a configuration are those of python/sbmc_e2.py (the functions of
python/sbmc_tts.py), with the same seeds of the bootstrap ([S, index of the configuration in the
campaign, index of the level]). Files, in the output folder of the campaign:

- e3_summary.csv: as e2_summary.csv, one row per instance, method, steps and level;
- e3_tts.csv, the main table, one row per instance and level f in {1000, 995, 990}: for each
  method (prefix sb_ for dSB, sa_ for SA) the TTS99 at the configuration with the steps chosen in
  E2 (transfer, without tuning on the evaluation instance), its Wilson and bootstrap intervals,
  and the minimum over the four steps with the steps that give it; the ratio SA/SB of the TTS99
  at the transferred configuration with the conservative interval [lo_SA / hi_SB, hi_SA / lo_SB]
  from the Wilson bounds of TTS99; common_level = 1 in the row of the most severe level where
  both methods, at the transferred configuration, have P_s > 0; note: the note of the instance in
  the campaign (for K2000 "preliminary target (one source)") and the methods without a selection;
- quality_equal_time.csv: per instance and method the points of the four steps ("point": median
  of t_run_ms over the repetitions, median of the mean cuts, median of the best cuts); the
  budgets B of the intersection of the time ranges of the two methods, [max of the minima, min of
  the maxima], its ends and their geometric mean ("budget": the mean cut and the best cut of each
  method interpolated linearly in ln t between the two consecutive points that contain B, no
  extrapolation); an empty intersection gives one row "no_overlap" for the instance;
- plots/tts_bars_<f>.pdf and .png, for every level (plot 1 of section 14.5: bars per instance and
  method, log y, Wilson error bars, undefined values marked), and plots/quality_time_<instance>
  (plot 2: mean cut and best cut against the median of t_run_ms, log x, the two methods, the
  target as a horizontal line).

Lines on stdout: "E3 <instance> level=<f> tts_sb_ms=<x> tts_sa_ms=<x> ratio_sa_sb=<x>" for every
row of e3_tts.csv ("none" when undefined), then "E3_DONE". Exit status: 0 success; 1 error in the
data; 2 error of the command line or of the configuration. Messages on stderr start with
"sbmc_e3: ". Standard library, NumPy and matplotlib (Agg backend); importable as a module (pure
functions: interpolate, budgets, ratio_interval, common_level).
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Sequence

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # the sibling modules, also under python -I
    sys.path.insert(0, _HERE)

import sbmc_campaign as sc  # noqa: E402
import sbmc_e2  # noqa: E402

PROG = "sbmc_e3"
LEVELS = sbmc_e2.LEVELS
METHOD_PREFIX = {"dsb": "sb", "sa": "sa"}
TTS_COLUMNS = ["instance", "tuning_instance", "level_permille", "level_value", "target",
               "target_source"]
for _prefix in ("sb", "sa"):
    TTS_COLUMNS += [f"{_prefix}_{c}" for c in (
        "config_id", "steps", "p_s", "p_s_lo", "p_s_hi", "t_batch_ms", "tts99_ms", "tts99_lo_ms",
        "tts99_hi_ms", "tts99_boot_lo_ms", "tts99_boot_hi_ms", "min_tts99_ms", "min_steps")]
TTS_COLUMNS += ["ratio_sa_sb", "ratio_sa_sb_lo", "ratio_sa_sb_hi", "common_level", "note"]
QUALITY_COLUMNS = ["instance", "kind", "method", "steps", "budget", "t_ms", "mean_cut",
                   "best_cut", "note"]


# --- Pure functions -------------------------------------------------------------------------


def interpolate(points: Sequence[tuple[float, float]], t: float) -> float | None:
    """The value at t, linear in ln t between the two consecutive points (sorted by time) that
    contain t; None outside [first time, last time] (no extrapolation)."""
    pts = sorted(points)
    if not pts or t < pts[0][0] or t > pts[-1][0]:
        return None
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            x = (math.log(t) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return v0 + x * (v1 - v0)
    return pts[0][1]  # a single point, t equal to its time


def budgets(times_a: Sequence[float], times_b: Sequence[float]) -> tuple[float, float,
                                                                         float] | None:
    """The ends of [max of the minima, min of the maxima] and their geometric mean; None when the
    intersection of the two time ranges is empty."""
    if not times_a or not times_b:
        return None
    lo, hi = max(min(times_a), min(times_b)), min(max(times_a), max(times_b))
    if lo > hi:
        return None
    return lo, math.sqrt(lo * hi), hi


def ratio_interval(sa: tuple[float | None, float | None, float | None],
                   sb: tuple[float | None, float | None, float | None]
                   ) -> tuple[float | None, float | None, float | None]:
    """The ratio SA/SB of two TTS99 and its conservative interval [lo_SA / hi_SB, hi_SA / lo_SB].

    Each argument is (TTS99, lower bound, upper bound), None for an undefined TTS99 or an
    infinite upper bound; the ratio is undefined when one of the TTS99 is.
    """
    tts_sa, lo_sa, hi_sa = sa
    tts_sb, lo_sb, hi_sb = sb
    if tts_sa is None or tts_sb is None:
        return None, None, None
    assert lo_sa is not None and lo_sb is not None
    low = 0.0 if hi_sb is None else lo_sa / hi_sb
    high = math.inf if hi_sa is None else hi_sa / lo_sb
    return tts_sa / tts_sb, low, high


def common_level(p_s: dict[int, tuple[float | None, float | None]]) -> int | None:
    """The most severe level where both methods have P_s > 0 (p_s: level -> (P_s of SB, P_s of
    SA), None for a method without configuration)."""
    for permille in LEVELS:
        sb, sa = p_s.get(permille, (None, None))
        if sb is not None and sa is not None and sb > 0 and sa > 0:
            return permille
    return None


# --- Analysis -------------------------------------------------------------------------------


Transfer = dict[tuple[str, str], tuple[str, int]]
Points = dict[str, dict[str, list[tuple[float, float, float]]]]


def transferred_steps(campaign: sc.Campaign, root: Path) -> Transfer:
    """(instance, method) -> (tuning instance, steps of the configuration chosen in E2)."""
    out = {}
    for method in campaign.methods:
        if method.selection is None:
            continue
        path = root / method.selection
        rows = sc.read_selection(path)
        for instance in campaign.instances:
            row = sc._selected_row(rows, path, method.id, instance.id)
            if row["config_id"] != "none":
                out[(instance.id, method.id)] = (row["tuning_instance"], int(row["steps"]))
    return out


def tts_rows(campaign: sc.Campaign, measured: list[sbmc_e2.Measured],
             targets: dict[str, sbmc_e2.Target], transfer: Transfer,
             skipped: list[sc.Skipped]) -> list[dict[str, str]]:
    num = sc.number
    rows = []
    for instance in campaign.instances:
        target = targets[instance.id]
        per_method = {m: [x for x in measured if x.config.instance.id == instance.id
                          and x.config.method == m] for m in METHOD_PREFIX}
        chosen: dict[str, sbmc_e2.Measured | None] = {}
        tuning = ""
        for method in METHOD_PREFIX:
            entry = transfer.get((instance.id, method))
            chosen[method] = None
            if entry is not None:
                tuning = entry[0]
                matches = [x for x in per_method[method] if x.config.steps == entry[1]]
                if len(matches) != 1:
                    raise sc.DataError(f"instance {instance.id}, method {method}: the steps "
                                       f"{entry[1]} chosen in E2 are not among the steps of the "
                                       "campaign")
                chosen[method] = matches[0]
        notes = [instance.note] if instance.note else []
        notes += [f"{s.method}: {s.reason}" for s in skipped if s.instance == instance.id]
        p_s = {}
        for permille in LEVELS:
            p_s[permille] = tuple(None if chosen[m] is None else chosen[m].levels[permille].p_s
                                  for m in ("dsb", "sa"))
        common = common_level(p_s)  # type: ignore[arg-type]
        for permille in LEVELS:
            row = {"instance": instance.id, "tuning_instance": tuning,
                   "level_permille": str(permille),
                   "level_value": str(sc_target_value(permille, target.value)),
                   "target": str(target.value), "target_source": target.source}
            bounds = {}
            for method, prefix in METHOD_PREFIX.items():
                m = chosen[method]
                fields = {c: "" for c in TTS_COLUMNS if c.startswith(prefix + "_")}
                if m is not None:
                    s = m.levels[permille]
                    values = {
                        "config_id": m.config.id, "steps": str(m.config.steps), "p_s": num(s.p_s),
                        "p_s_lo": num(s.p_s_lo), "p_s_hi": num(s.p_s_hi),
                        "t_batch_ms": num(s.t_batch), "tts99_ms": num(s.tts99),
                        "tts99_lo_ms": num(s.tts99_lo), "tts99_hi_ms": num(s.tts99_hi),
                        "tts99_boot_lo_ms": num(s.boot_lo), "tts99_boot_hi_ms": num(s.boot_hi)}
                    fields.update({f"{prefix}_{k}": v for k, v in values.items()})
                    defined = [x for x in per_method[method]
                               if x.levels[permille].tts99 is not None]
                    if defined:
                        best = min(defined, key=lambda x: (x.levels[permille].tts99,
                                                           x.config.index))
                        fields[f"{prefix}_min_tts99_ms"] = num(best.levels[permille].tts99)
                        fields[f"{prefix}_min_steps"] = str(best.config.steps)
                    bounds[method] = (s.tts99, s.tts99_lo, s.tts99_hi)
                else:
                    bounds[method] = (None, None, None)
                row.update(fields)
            ratio, low, high = ratio_interval(bounds["sa"], bounds["dsb"])
            row.update({"ratio_sa_sb": num(ratio), "ratio_sa_sb_lo": num(low),
                        "ratio_sa_sb_hi": num(high),
                        "common_level": "1" if common == permille else "0",
                        "note": "; ".join(notes)})
            rows.append(row)
    return rows


def sc_target_value(permille: int, target: int) -> int:
    return sbmc_e2.sbmc_tts.target_value(permille, target)


def quality_rows(campaign: sc.Campaign,
                 measured: list[sbmc_e2.Measured]) -> tuple[list[list[str]], Points]:
    """The rows of quality_equal_time.csv and the points per instance and method."""
    num = sc.number
    rows: list[list[str]] = []
    points: Points = {}
    for instance in campaign.instances:
        points[instance.id] = {}
        for method in METHOD_PREFIX:
            members = sorted((x for x in measured if x.config.instance.id == instance.id
                              and x.config.method == method), key=lambda x: x.config.steps)
            series = []
            for x in members:
                t = x.median_t_run()
                mean = statistics.median(float(r.mean_cut) for r in x.rows)
                best = statistics.median(r.best_cut for r in x.rows)
                series.append((t, mean, float(best)))
                rows.append([instance.id, "point", method, str(x.config.steps), "", num(t),
                             num(mean), num(best), ""])
            if series:
                points[instance.id][method] = series
        both = points[instance.id]
        span = None
        if len(both) == 2:
            span = budgets([p[0] for p in both["dsb"]], [p[0] for p in both["sa"]])
        if span is None:
            reason = "no overlap" if len(both) == 2 else "no overlap (one method has no points)"
            rows.append([instance.id, "no_overlap", "", "", "", "", "", "", reason])
            continue
        for label, budget in zip(("min", "geomean", "max"), span):
            for method in METHOD_PREFIX:
                series = both[method]
                mean = interpolate([(p[0], p[1]) for p in series], budget)
                best = interpolate([(p[0], p[2]) for p in series], budget)
                rows.append([instance.id, "budget", method, "", label, num(budget), num(mean),
                             num(best), ""])
    return rows, points


def plot_tts_bars(stem: Path, permille: int, rows: list[dict[str, str]]) -> None:
    """Plot 1 of section 14.5: TTS99 per instance and method at one level."""
    plt = sbmc_e2.pyplot()
    fig, ax = plt.subplots(figsize=sbmc_e2.FIGURE_SIZE)
    level_rows = [r for r in rows if r["level_permille"] == str(permille)]
    width = 0.38
    labeled: set[str] = set()
    for j, (method, prefix) in enumerate(METHOD_PREFIX.items()):
        for k, row in enumerate(level_rows):
            x = k + (j - 0.5) * width
            value = row[f"{prefix}_tts99_ms"]
            if value == "":
                ax.text(x, 0.02, f"{method}: undefined", ha="center", va="bottom", fontsize=7,
                        rotation=90, transform=ax.get_xaxis_transform())
                continue
            tts = float(value)
            low = tts - float(row[f"{prefix}_tts99_lo_ms"])
            high_text = row[f"{prefix}_tts99_hi_ms"]
            high = float(high_text) - tts if high_text not in ("", "inf") else 0.0
            ax.bar(x, tts, width, color=f"C{j}", yerr=[[low], [high]], capsize=3,
                   label=None if method in labeled else sbmc_e2.METHOD_LABELS[method])
            labeled.add(method)
            if high_text in ("", "inf"):
                ax.text(x, tts, "upper bound inf", ha="center", va="bottom", fontsize=6)
    if labeled:
        ax.legend()
        ax.set_yscale("log")
    ax.set_xticks(range(len(level_rows)))
    ax.set_xticklabels([r["instance"] for r in level_rows])
    ax.set_ylabel("TTS99 (ms)")
    ax.set_title(f"E3 on the CPU: TTS99 at {permille}/1000 of the target")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    sbmc_e2.save_figure(fig, stem)


def plot_quality(stem: Path, instance: str, series: dict[str, list[tuple[float, float, float]]],
                 target: int) -> None:
    """Plot 2 of section 14.5: mean and best cut against the median of t_run_ms."""
    plt = sbmc_e2.pyplot()
    fig, ax = plt.subplots(figsize=sbmc_e2.FIGURE_SIZE)
    for j, method in enumerate(METHOD_PREFIX):
        if method not in series:
            continue
        pts = series[method]
        label = sbmc_e2.METHOD_LABELS[method]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=f"C{j}",
                label=f"{label}: mean cut")
        ax.plot([p[0] for p in pts], [p[2] for p in pts], marker="s", linestyle="--",
                color=f"C{j}", label=f"{label}: best cut")
    ax.axhline(target, color="black", linewidth=1, linestyle=":", label=f"target {target}")
    ax.set_xscale("log")
    ax.set_xlabel("median of t_run_ms over the repetitions (ms)")
    ax.set_ylabel("cut")
    ax.set_title(f"E3 on the CPU: quality against time, {instance}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    sbmc_e2.save_figure(fig, stem)


def analyze(campaign: sc.Campaign, root: Path, bootstrap: int, seed: int) -> int:
    skipped: list[sc.Skipped] = []
    configs = sc.expand_configs(campaign, root, skipped)
    targets = sbmc_e2.campaign_targets(campaign, configs, root, "E3")
    measured = sbmc_e2.measure(configs, campaign, root, targets, bootstrap, seed)
    output = root / campaign.output
    sc.write_csv(output / "e3_summary.csv", sbmc_e2.SUMMARY_COLUMNS,
                 sbmc_e2.summary_rows(measured, targets))
    transfer = transferred_steps(campaign, root)
    rows = tts_rows(campaign, measured, targets, transfer, skipped)
    sc.write_csv(output / "e3_tts.csv", TTS_COLUMNS, [[r[c] for c in TTS_COLUMNS] for r in rows])
    quality, points = quality_rows(campaign, measured)
    sc.write_csv(output / "quality_equal_time.csv", QUALITY_COLUMNS, quality)
    for permille in LEVELS:
        plot_tts_bars(output / "plots" / f"tts_bars_{permille}", permille, rows)
    for instance in campaign.instances:
        plot_quality(output / "plots" / f"quality_time_{instance.id}", instance.id,
                     points[instance.id], targets[instance.id].value)
    for r in rows:
        print(f"E3 {r['instance']} level={r['level_permille']} "
              f"tts_sb_ms={r['sb_tts99_ms'] or 'none'} tts_sa_ms={r['sa_tts99_ms'] or 'none'} "
              f"ratio_sa_sb={r['ratio_sa_sb'] or 'none'}")
    print("E3_DONE")
    return sc.EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    return sbmc_e2.run_tool(argv, PROG, "E3 on the CPU", analyze)


if __name__ == "__main__":
    sys.exit(main())
