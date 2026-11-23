#!/usr/bin/env python3
"""sbmc_e9: E9 of phase 07, the OpenMP scaling of C1 (dSB) and C2 (SA) on Windows native and in
WSL2 (reference document, section 14.2; decisions D12, L9, W1-W4), and the bitwise comparison of
the two builds (section 5.4 of F07-C4).

Usage (from the repository root, or with --root):
  python3 python/sbmc_e9.py analyze CONFIG
  python3 python/sbmc_e9.py bitwise CONFIG --workdir DIR --win-run EXE --win-strict-run EXE
          --win-thresholds EXE --wsl-run EXE --wsl-strict-run EXE --wsl-thresholds EXE

CONFIG is the campaign with [[series]] (bench/configs/f07_e9.toml, run by
python/sbmc_campaign.py). A load is an instance, a method and its parameters; every load is run
by every series (platform, executable, --affinity) at every thread count of the series.

analyze, for every configuration (all must be complete): the median, Q1 and Q3 of t_run_ms over
the repetitions (statistics.median and numpy.percentile, as sbmc_e2.py); the speedup against the
median at 1 thread of the same series (for a series without 1 thread, of its base series: win-p
uses win-p1), the efficiency speedup / threads; for the Windows series also the speedup and the
efficiency against the smallest median at 1 thread of the Windows series of the load (D12: the
Windows series are compared with themselves; WSL is a control series). Files in the output folder
of the campaign: e9_summary.csv (one row per configuration) and plots/scaling_<instance>_<method>
.pdf and .png (speedup and efficiency against the threads, one line per series, the bars from Q1
and Q3). Checks of consistency:
  masks: every row of a Windows series has its classes (cpu_classes) and its mask (cpu_affinity)
    equal to the union of the classes; all: every class of the Windows rows; p and p1: only the
    highest class, p its whole set, p1 a subset; e: only the lowest class, its whole set; the
    same mask on every row of a series; threads of the rows (the team) equal to the threads
    asked and not above the size of the mask. WSL rows: cpu_classes empty, one mask.
  cuts: best_cut and mean_cut of every repetition equal over the threads and the series of a
    platform (H3: results independent of the number of threads).
  platforms: SA equal between Windows and WSL; dSB only reported (E9_INFO).
Lines on stdout: one "E9 ..." per configuration, one "E9_TABLE ..." per load (the speedup at the
largest thread count of every series), "E9_CHECK <check>: PASS|FAIL (<detail>)", "E9_INFO ...",
"E9_CONSISTENCY: PASS|FAIL", then "E9_DONE". A FAIL of the checks is a result: exit status 0.

bitwise (section 5.4): the Windows build against the WSL build, 4 threads, seed 1, one
repetition, no warm-up, files in --workdir: SA (win-cpu against cpu) on G1 with R 256 and S 1000
and on K2000 with R 32 and S 200, beta0 = a / h and beta1 = b / w from the SA configuration of the
instance in CONFIG; dSB (win-cpu-strict against cpu-strict) on G1 with R 64, M 500, --dt 0.5,
--c0 auto. The replica files must be equal in every column but timestamp. On a difference the
first different replica is written and, for SA, the thresholds tool (tests/tools/
sbmc_sa_thresholds, built with the library on both platforms) is run on both sides with the
schedule of the case and its lines are compared: the sweeps where beta or a threshold differs.
The Windows executables get absolute Windows paths (wslpath -w, as the runner). Lines:
"BITWISE <case>: equal|different ...", "BITWISE_SA: PASS|FAIL", "BITWISE_DSB_STRICT: PASS|FAIL".

Exit status: 0 success (FAIL verdicts included); 1 error in the data or in a child process; 2
error of the command line or of the configuration. Messages on stderr start with "sbmc_e9: ".
Standard library, NumPy and matplotlib (Agg backend, analyze only).
"""

from __future__ import annotations

import argparse
import csv
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # the sibling modules, also under python -I
    sys.path.insert(0, _HERE)

import sbmc_campaign as sc  # noqa: E402

PROG = "sbmc_e9"
SUMMARY_COLUMNS = [
    "instance", "method", "load", "series", "platform", "affinity", "threads", "team", "rows",
    "t_run_median_ms", "t_run_q1_ms", "t_run_q3_ms", "t_run_iqr_ms", "reference_series",
    "t1_median_ms", "speedup", "efficiency", "t1_windows_min_ms", "speedup_windows",
    "efficiency_windows", "best_cut", "mean_cut", "cpu_affinity", "cpu_classes", "config_id",
]
WINDOWS, LINUX = "windows", "linux"


# --- Text forms ------------------------------------------------------------------------------


def parse_cpulist(text: str) -> frozenset[int]:
    """"0-3,8" -> {0, 1, 2, 3, 8}; empty text -> empty set."""
    cpus: set[int] = set()
    for part in filter(None, text.split(",")):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return frozenset(cpus)


def parse_classes(text: str) -> dict[int, frozenset[int]]:
    """"1:0-11 0:12-15" -> {1: {0..11}, 0: {12..15}}."""
    classes: dict[int, frozenset[int]] = {}
    for group in text.split():
        name, _, cpus = group.partition(":")
        classes[int(name)] = parse_cpulist(cpus)
    return classes


def number(value: float | None) -> str:
    return sc.number(value)


# --- Measures --------------------------------------------------------------------------------


@dataclass
class Point:
    """One configuration: a series at a thread count on a load."""

    config: sc.Config
    rows: list[dict[str, str]]
    median: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    reference: str = ""
    t1: float | None = None
    speedup: float | None = None
    efficiency: float | None = None
    t1_windows: float | None = None
    speedup_windows: float | None = None
    efficiency_windows: float | None = None

    @property
    def series(self) -> sc.Series:
        assert self.config.series is not None
        return self.config.series

    @property
    def threads(self) -> int:
        assert self.config.threads is not None
        return self.config.threads


@dataclass
class Load:
    instance: str
    method: str
    name: str  # the id of the configuration without the suffix of the series
    points: list[Point] = field(default_factory=list)


def load_name(config: sc.Config) -> str:
    assert config.series is not None and config.threads is not None
    suffix = f"_{config.series.id}_T{config.threads}"
    assert config.id.endswith(suffix)
    return config.id[: -len(suffix)]


def quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    import numpy as np

    q1, q3 = (float(v) for v in np.percentile(list(values), [25, 75]))
    return float(statistics.median(values)), q1, q3


def measure(campaign: sc.Campaign, root: Path) -> list[Load]:
    if not campaign.series:
        raise sc.DataError(f"{campaign.path}: no [[series]]: not a campaign of E9")
    configs = sc.expand_configs(campaign, root)
    missing = [c.id for c in configs if sc.config_state(c, campaign, root) != "done"]
    if missing:
        raise sc.DataError(f"{len(missing)} of {len(configs)} configurations not complete "
                           f"(first: {missing[0]}): run the campaign first")
    loads: dict[tuple[str, str, str], Load] = {}
    for config in configs:
        name = load_name(config)
        key = (config.instance.id, config.method, name)
        load = loads.setdefault(key, Load(config.instance.id, config.method, name))
        rows = sc.read_rows(root / config.results_path(campaign))
        point = Point(config, rows)
        point.median, point.q1, point.q3 = quartiles([float(r["t_run_ms"]) for r in rows])
        load.points.append(point)
    for load in loads.values():
        one_thread = {p.series.id: p.median for p in load.points if p.threads == 1}
        windows_t1 = [p.median for p in load.points
                      if p.threads == 1 and p.series.platform == WINDOWS]
        for p in load.points:
            p.reference = p.series.base or p.series.id
            p.t1 = one_thread.get(p.reference)
            if p.t1 is not None:
                p.speedup = p.t1 / p.median
                p.efficiency = p.speedup / p.threads
            if p.series.platform == WINDOWS and windows_t1:
                p.t1_windows = min(windows_t1)
                p.speedup_windows = p.t1_windows / p.median
                p.efficiency_windows = p.speedup_windows / p.threads
    return list(loads.values())


def summary_rows(loads: list[Load]) -> list[list[str]]:
    out = []
    for load in loads:
        for p in load.points:
            first = p.rows[0]
            teams = sorted({r["threads"] for r in p.rows})
            best = max(int(r["best_cut"]) for r in p.rows)
            mean = statistics.fmean(float(r["mean_cut"]) for r in p.rows)
            out.append([
                load.instance, load.method, load.name, p.series.id, p.series.platform,
                p.series.affinity or "", str(p.threads), ";".join(teams), str(len(p.rows)),
                number(p.median), number(p.q1), number(p.q3), number(p.q3 - p.q1), p.reference,
                number(p.t1), number(p.speedup), number(p.efficiency), number(p.t1_windows),
                number(p.speedup_windows), number(p.efficiency_windows), str(best), number(mean),
                first["cpu_affinity"], first["cpu_classes"], p.config.id,
            ])
    return out


# --- Checks of consistency ---------------------------------------------------------------------


@dataclass
class Check:
    name: str
    problems: list[str] = field(default_factory=list)
    detail: str = ""

    def line(self) -> str:
        verdict = "FAIL" if self.problems else "PASS"
        problems = list(dict.fromkeys(self.problems))  # one per configuration, not per row
        text = "; ".join(problems[:3]) + (f" (+{len(problems) - 3} more)"
                                          if len(problems) > 3 else "")
        return f"E9_CHECK {self.name}: {verdict} ({text or self.detail})"


def check_masks(loads: list[Load]) -> Check:
    check = Check("masks")
    points = [p for load in loads for p in load.points]
    reference: dict[int, set[int]] = {}
    for p in points:
        if p.series.platform == WINDOWS:
            for r in p.rows:
                for cls, cpus in parse_classes(r["cpu_classes"]).items():
                    reference.setdefault(cls, set()).update(cpus)
    masks: dict[str, set[str]] = {}
    rows = 0
    for p in points:
        where = p.config.id
        for r in p.rows:
            rows += 1
            masks.setdefault(p.series.id, set()).add(r["cpu_affinity"])
            mask = parse_cpulist(r["cpu_affinity"])
            if r["threads"] != str(p.threads):
                check.problems.append(f"{where}: team {r['threads']}, asked {p.threads}")
            if p.series.platform == LINUX:
                if r["cpu_classes"]:
                    check.problems.append(f"{where}: cpu_classes not empty on Linux")
                continue
            classes = parse_classes(r["cpu_classes"])
            if not classes:
                check.problems.append(f"{where}: cpu_classes empty")
                continue
            if frozenset().union(*classes.values()) != mask:
                check.problems.append(f"{where}: classes {r['cpu_classes']} against mask "
                                      f"{r['cpu_affinity']}")
            if p.threads > len(mask):
                check.problems.append(f"{where}: {p.threads} threads on {len(mask)} processors")
            high, low = max(reference), min(reference)  # not empty: classes of this row
            rule = p.series.affinity
            if rule == "all":
                expected = {c: frozenset(s) for c, s in reference.items()}
                if classes != expected:
                    check.problems.append(f"{where}: all is {r['cpu_classes']}")
            elif rule in ("p", "p1") or (rule == "e" and low != high):
                wanted = high if rule in ("p", "p1") else low
                if set(classes) != {wanted}:
                    check.problems.append(f"{where}: {rule} has the classes {sorted(classes)}")
                elif rule == "p1":
                    if not mask <= reference[wanted] or not mask:
                        check.problems.append(f"{where}: p1 {r['cpu_affinity']} not in the class "
                                              f"{wanted}")
                elif mask != reference[wanted]:
                    check.problems.append(f"{where}: {rule} {r['cpu_affinity']} is not the class "
                                          f"{wanted}")
            else:
                check.problems.append(f"{where}: e with one class")
    for series, values in masks.items():
        if len(values) != 1:
            check.problems.append(f"series {series}: {len(values)} different masks")
    check.detail = f"{rows} rows; classes " + " ".join(
        f"{c}:{len(s)}" for c, s in sorted(reference.items(), reverse=True))
    return check


def cuts_by_rep(points: list[Point]) -> dict[str, set[tuple[str, str]]]:
    out: dict[str, set[tuple[str, str]]] = {}
    for p in points:
        for r in p.rows:
            out.setdefault(r["rep"], set()).add((r["best_cut"], r["mean_cut"]))
    return out


def check_cuts(loads: list[Load], platform: str) -> Check:
    check = Check(f"cuts {platform}")
    compared = 0
    for load in loads:
        points = [p for p in load.points if p.series.platform == platform]
        for rep, values in sorted(cuts_by_rep(points).items()):
            compared += 1
            if len(values) != 1:
                check.problems.append(f"{load.instance} {load.method} rep {rep}: "
                                      f"{len(values)} values {sorted(values)[:2]}")
    check.detail = f"{compared} (load, rep) pairs"
    return check


def check_platforms(loads: list[Load]) -> tuple[Check, list[str]]:
    check = Check("cuts sa windows=linux")
    info = []
    for load in loads:
        win = cuts_by_rep([p for p in load.points if p.series.platform == WINDOWS])
        lin = cuts_by_rep([p for p in load.points if p.series.platform == LINUX])
        if not win or not lin:
            continue
        differences = [rep for rep in sorted(set(win) | set(lin)) if win.get(rep) != lin.get(rep)]
        if load.method == "sa":
            for rep in differences:
                check.problems.append(f"{load.instance} rep {rep}: windows {sorted(win[rep])} "
                                      f"linux {sorted(lin[rep])}")
        else:
            if differences:
                rep = differences[0]
                info.append(f"E9_INFO cuts {load.instance} {load.method} windows vs linux: "
                            f"different in {len(differences)} of {len(win)} repetitions "
                            f"(rep {rep}: windows {sorted(win.get(rep, set()))}, linux "
                            f"{sorted(lin.get(rep, set()))})")
            else:
                info.append(f"E9_INFO cuts {load.instance} {load.method} windows vs linux: equal "
                            f"in {len(win)} repetitions")
    check.detail = "SA loads equal"
    return check, info


# --- Plots -----------------------------------------------------------------------------------


def plot_load(load: Load, folder: Path) -> None:
    import sbmc_e2

    plt = sbmc_e2.pyplot()
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=sbmc_e2.WIDE_FIGURE_SIZE)
    top = 1
    for series in dict.fromkeys(p.series.id for p in load.points):
        points = sorted((p for p in load.points if p.series.id == series), key=lambda p: p.threads)
        points = [p for p in points if p.t1 is not None]
        if not points:
            continue
        xs = [p.threads for p in points]
        top = max(top, *xs)
        speed = [p.speedup for p in points]
        assert points[0].t1 is not None
        t1 = points[0].t1
        lower = [s - t1 / p.q3 for s, p in zip(speed, points)]  # type: ignore[operator]
        upper = [t1 / p.q1 - s for s, p in zip(speed, points)]  # type: ignore[operator]
        ax_s.errorbar(xs, speed, yerr=[lower, upper], marker="o", markersize=4, capsize=2,
                      label=series)
        ax_e.plot(xs, [p.efficiency for p in points], marker="o", markersize=4, label=series)
    ax_s.plot([1, top], [1, top], linestyle="--", color="0.6", linewidth=1, label="ideal")
    ax_s.set_xlabel("threads")
    ax_s.set_ylabel("speedup (median t_run at 1 thread / median t_run)")
    ax_s.set_title(f"{load.instance} {load.method}: speedup", fontsize=10)
    ax_e.axhline(1.0, linestyle="--", color="0.6", linewidth=1)
    ax_e.set_xlabel("threads")
    ax_e.set_ylabel("efficiency (speedup / threads)")
    ax_e.set_ylim(0, 1.15)
    ax_e.set_title(f"{load.instance} {load.method}: efficiency", fontsize=10)
    for ax in (ax_s, ax_e):
        ax.set_xticks(sorted({p.threads for p in load.points}))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"E9 {load.name} (win-p: reference win-p1 at 1 thread)", fontsize=9)
    fig.tight_layout()
    sbmc_e2.save_figure(fig, folder / f"scaling_{load.instance}_{load.method}")


# --- analyze -----------------------------------------------------------------------------------


def table_line(load: Load) -> str:
    parts = []
    for series in dict.fromkeys(p.series.id for p in load.points):
        last = max((p for p in load.points if p.series.id == series), key=lambda p: p.threads)
        speed = "-" if last.speedup is None else f"{last.speedup:.2f}"
        parts.append(f"{series} T{last.threads} {last.median:.1f} ms S={speed}")
    return f"E9_TABLE {load.instance} {load.method}: " + " | ".join(parts)


def analyze(campaign: sc.Campaign, root: Path) -> int:
    loads = measure(campaign, root)
    output = root / campaign.output
    sc.write_csv(output / "e9_summary.csv", SUMMARY_COLUMNS, summary_rows(loads))
    for load in loads:
        for p in load.points:
            print(f"E9 {load.instance} {load.method} {p.series.id} threads={p.threads} "
                  f"median_ms={p.median!r} q1_ms={p.q1!r} q3_ms={p.q3!r} "
                  f"speedup={number(p.speedup) or '-'} efficiency={number(p.efficiency) or '-'}")
    for load in loads:
        plot_load(load, output / "plots")
        print(table_line(load))
    checks = [check_masks(loads), check_cuts(loads, WINDOWS), check_cuts(loads, LINUX)]
    platform_check, info = check_platforms(loads)
    checks.append(platform_check)
    for check in checks:
        print(check.line())
    for line in info:
        print(line)
    verdict = "FAIL" if any(c.problems for c in checks) else "PASS"
    print(f"E9_CONSISTENCY: {verdict}")
    print("E9_DONE")
    return sc.EXIT_OK


# --- bitwise -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    name: str
    instance: str
    algo: str
    replicas: int
    steps: int
    strict: bool


CASES = (Case("sa_G1", "G1", "sa", 256, 1000, False),
         Case("sa_K2000", "K2000", "sa", 32, 200, False),
         Case("dsb_G1_strict", "G1", "dsb", 64, 500, True))
REPLICA_KEYS = [c for c in sc.REPLICA_COLUMNS if c != "timestamp"]


def sa_betas(campaign: sc.Campaign, root: Path, instance_id: str) -> tuple[float, float]:
    """beta0 = a / h and beta1 = b / w of the SA configuration of the instance in the campaign."""
    instance = campaign.instance(instance_id)
    config = next(c for c in sc.expand_configs(campaign, root)
                  if c.instance.id == instance_id and c.method == "sa")
    h, w = sc.instance_scales(instance, root)
    assert config.a is not None and config.b is not None and w is not None
    return config.a / h, config.b / w


def run_checked(cmd: list[str], what: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False,
                                env=sc.child_environment())
    except OSError as error:
        raise sc.DataError(f"{what}: cannot run {cmd[0]}: {error}") from error
    if result.returncode != 0:
        raise sc.DataError(f"{what}: {shlex.join(cmd)} exited with {result.returncode}: "
                           f"{result.stderr.strip()}")
    return result


def replica_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_difference(a: list[dict[str, str]], b: list[dict[str, str]]) -> str | None:
    if len(a) != len(b):
        return f"{len(a)} rows against {len(b)}"
    for ra, rb in zip(a, b):
        if any(ra[k] != rb[k] for k in REPLICA_KEYS):
            return (f"replica {ra['replica']}: windows best_cut={ra['best_cut']} "
                    f"best_step={ra['best_step']} final_cut={ra['final_cut']}, wsl "
                    f"best_cut={rb['best_cut']} best_step={rb['best_step']} "
                    f"final_cut={rb['final_cut']}")
    return None


def max_delta(instance: sc.Instance, root: Path) -> int:
    """2 max_i sum_j |w_ij|: the largest delta_e of a flip."""
    if instance.dense:
        return 2 * (instance.dense_n - 1)
    tokens = (root / instance.file).read_text(encoding="ascii").split()
    n, m = int(tokens[0]), int(tokens[1])
    degree = [0] * (n + 1)
    for k in range(m):
        u, v, w = (int(tokens[2 + 3 * k + j]) for j in range(3))
        degree[u] += abs(w)
        degree[v] += abs(w)
    return 2 * max(degree)


def compare_thresholds(win: str, wsl: str) -> tuple[int, list[str]]:
    """Lines "<k> <beta> <t1>,<t2>,..." of the thresholds tool: (sweeps compared, differences)."""
    a = [line.rstrip("\r") for line in win.splitlines() if line.strip()]
    b = [line.rstrip("\r") for line in wsl.splitlines() if line.strip()]
    differences = []
    if len(a) != len(b):
        differences.append(f"{len(a)} lines against {len(b)}")
    for la, lb in zip(a, b):
        if la == lb:
            continue
        ka, beta_a, ta = (la.split(" ") + ["", ""])[:3]
        _, beta_b, tb = (lb.split(" ") + ["", ""])[:3]
        if beta_a != beta_b:
            differences.append(f"sweep {ka}: beta {beta_a} against {beta_b}")
            continue
        va, vb = ta.split(","), tb.split(",")
        for d, (x, y) in enumerate(zip(va, vb), start=1):
            if x != y:
                differences.append(f"sweep {ka}, delta_e {d}: threshold {x} against {y}")
                break
        else:
            differences.append(f"sweep {ka}: {len(va)} thresholds against {len(vb)}")
    return len(a), differences


def bitwise(campaign: sc.Campaign, root: Path, args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    windows_paths = sc.WindowsPaths(root)
    verdicts: dict[str, list[bool]] = {"sa": [], "dsb": []}
    for case in CASES:
        instance = campaign.instance(case.instance)
        if case.algo == "sa":
            beta0, beta1 = sa_betas(campaign, root, case.instance)
            params = ["--algo", "sa", "--variant", "c2", "--sweeps", str(case.steps), "--beta0",
                      repr(beta0), "--beta1", repr(beta1)]
            exes = (args.win_run, args.wsl_run)
        else:
            params = ["--algo", "dsb", "--variant", "c1", "--steps", str(case.steps), "--dt",
                      "0.5", "--c0", "auto"]
            exes = (args.win_strict_run, args.wsl_strict_run)
        files = {}
        for side, exe in zip(("windows", "wsl"), exes):
            out = workdir / f"{case.name}_{side}.csv"
            replicas = workdir / f"{case.name}_{side}_replicas.csv"
            for path in (out, replicas):
                path.unlink(missing_ok=True)
            instance_args = list(instance.args)
            out_arg, replicas_arg = str(out), str(replicas)
            if side == "windows":
                if not instance.dense:
                    instance_args[1] = windows_paths(instance_args[1])
                out_arg, replicas_arg = windows_paths(out), windows_paths(replicas)
            cmd = [exe, *instance_args, "--name", case.instance, *params, "--replicas",
                   str(case.replicas), "--threads", "4", "--seed", "1", "--reps", "1", "--warmup",
                   "0", "--target", "none", "--notes", "F07-C4 5.4", "--tag", case.name, "--out",
                   out_arg, "--replica-out", replicas_arg]
            run_checked(cmd, f"bitwise {case.name} ({side})")
            files[side] = replica_rows(replicas)
        difference = first_difference(files["windows"], files["wsl"])
        verdicts[case.algo].append(difference is None)
        if difference is None:
            print(f"BITWISE {case.name}: equal ({len(files['wsl'])} replicas, R {case.replicas}, "
                  f"steps {case.steps})")
            continue
        print(f"BITWISE {case.name}: different: first {difference}")
        if case.algo == "sa":
            limit = max_delta(instance, root)
            tool_args = ["--beta0", repr(beta0), "--beta1", repr(beta1), "--sweeps",
                         str(case.steps), "--max-delta", str(limit)]
            win = run_checked([args.win_thresholds, *tool_args], "thresholds (windows)").stdout
            wsl = run_checked([args.wsl_thresholds, *tool_args], "thresholds (wsl)").stdout
            sweeps, differences = compare_thresholds(win, wsl)
            listed = ": " + "; ".join(differences[:5]) if differences else ""
            print(f"BITWISE {case.name} thresholds: {sweeps} sweeps, delta_e up to {limit}, "
                  f"{len(differences)} different{listed}")
    print(f"BITWISE_SA: {'PASS' if all(verdicts['sa']) else 'FAIL'}")
    print(f"BITWISE_DSB_STRICT: {'PASS' if all(verdicts['dsb']) else 'FAIL'}")
    return sc.EXIT_OK


# --- Command line ----------------------------------------------------------------------------


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = sc.Parser(prog=PROG, description="E9: analysis of the scaling, bitwise comparison.")
    parser.add_argument("--root", default=".", help="repository root (default: the current "
                                                    "directory)")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=sc.Parser)
    an = commands.add_parser("analyze", help="summary, plots and checks of the campaign")
    an.add_argument("config", help="TOML file of the campaign")
    bw = commands.add_parser("bitwise", help="the Windows build against the WSL build")
    bw.add_argument("config", help="TOML file of the campaign (instances and SA parameters)")
    bw.add_argument("--workdir", required=True, help="folder of the files of the comparison")
    for side in ("win", "wsl"):
        bw.add_argument(f"--{side}-run", required=True, help="sbmc_run of the preset cpu")
        bw.add_argument(f"--{side}-strict-run", required=True,
                        help="sbmc_run of the strict preset")
        bw.add_argument(f"--{side}-thresholds", required=True,
                        help="sbmc_sa_thresholds of the preset cpu")
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(sys.argv[1:] if argv is None else argv)
        root = Path(args.root).resolve()
        path = Path(args.config)
        campaign = sc.load_campaign(path if path.is_absolute() else root / path)
        if args.command == "analyze":
            try:
                import numpy  # noqa: F401
            except ImportError as error:
                raise sc.DataError(f"NumPy cannot be imported ({error}): run "
                                   "scripts/setup_python.sh") from error
            return analyze(campaign, root)
        return bitwise(campaign, root, args)
    except sc.UsageError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return sc.EXIT_USAGE
    except sc.DataError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return sc.EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
