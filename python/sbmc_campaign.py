#!/usr/bin/env python3
"""sbmc_campaign: the resumable runner of the measurement campaigns of sbmc_run (phase 07).

Usage (from the repository root, or with --root):
  python3 python/sbmc_campaign.py run CONFIG --notes TEXT [--dry-run] [--stage NAME]
  python3 python/sbmc_campaign.py status CONFIG
  python3 python/sbmc_campaign.py manifest CONFIG

A campaign is a TOML file (tomllib), validated in full before anything runs or is written:

  [campaign]
  name = "f07_e2"                          # [a-z0-9_]+
  output = "results/f07/e2"                # folder of the campaign
  sbmc_run = "build/cpu/apps/sbmc_run"
  sbmc_spectrum = "build/cpu/apps/sbmc_spectrum"
  replicas = 1024                          # R, >= 1
  threads = 16                             # >= 1; not with [[series]]
  reps = 5                                 # repetitions, >= 1
  warmup = 1                               # warm-up runs, >= 0
  seed = 1                                 # >= 0

  [dwave]                                  # optional, read by python/sbmc_dwave.py
  seed = 1                                 # seed of the sampler of dwave-samplers, >= 0

  [[series]]                               # optional (phase 07, part 4, E9): instead of threads
  id = "win-p1"                            # [a-z0-9-]+
  sbmc_run = "/mnt/c/Users/Simone/sbmc-win/build-win-cpu/apps/sbmc_run.exe"
  platform = "windows"                     # "windows" (sbmc_run.exe through WSL interop) | "linux"
  affinity = "p1"                          # windows only: p1 | p | e | all (--affinity)
  threads = [1, 2, 3, 4, 5, 6]             # strictly increasing, >= 1
  base = "win-p1"                          # optional, for the analysis: the series whose 1 thread
                                           # is the reference of a series without 1 thread

  [[instance]]                             # one or more
  id = "G2"                                # passed as --name; [A-Za-z0-9_]+
  args = ["--instance", "data/gset/G2"]    # or ["--dense", "2000", "--instance-seed", "1"]
  target = "best-known"                    # "best-known" | "best-found" | "none" | integer > 0
  family = "G1"                            # optional: the evaluation instance of its family
  note = "..."                             # optional: a note for the tables of the analysis

  [[method]]                               # one or more; id "sa" (variant "c2") or "dsb" ("c1")
  id = "sa"
  variant = "c2"
  a = [0.5, 1.0, 2.0]                      # sa: beta0 = a / h
  b = [2.0, 3.0]                           # sa: beta1 = b / w
  steps = { G2 = [1000, 2000, 4000, 8000] }  # one list per instance: S (sa) or M (dsb)

  [[method]]
  id = "dsb"
  variant = "c1"
  f = [0.25, 0.5, 1.0]                     # dsb: --dt auto --dt-factor f
  c0 = ["sigma", "lambda"]                 # dsb: --c0 auto (rule sigma) | --c0 lambda
  steps = { G2 = [1000, 2000, 4000, 8000] }

  # Instead of a, b (or f, c0), the relative parameters of an earlier tuning:
  #   selection = "results/f07/e2/selected.csv"   # the row of the method whose tuning_instance or
  #                                               # evaluation_instance is the id of the instance
  #   steps = "selected"                          # the steps of that row (or a table, as above)
  # A row with config_id "none" (no selection) gives no configuration: written in the log.

  [[stage]]                                # optional, run before the grid, in the file order
  name = "target"                          # [a-z0-9_]+, not "grid"
  instance = "dense_n2000_s1"
  target = "none"                          # optional, default "none"
  runs = [ { method = "sa", a = 1.0, b = 2.0, steps = 16000, reps = 1, warmup = 0 },
           { method = "dsb", f = 1.0, c0 = "sigma", steps = 8000, reps = 1, warmup = 0 } ]
  # reps and warmup of a run are optional (default: those of [campaign]).

An instance with target "best-found" needs at least one stage on it: the target of its grid is
the largest best_cut of the runs of its stages, which must be complete before the grid starts.

Scales of an instance (in <output>/instances.csv with the line SPECTRUM of sbmc_spectrum):
h = sqrt((1/n) sum_i sum_j w_ij^2) and w = min |w_ij| over the pairs with w_ij != 0, from the
rudy file read here (the weights of a repeated pair are added; exact integer sums, then one square
root in double); for --dense N, h = sqrt(N - 1) and w = 1 (couplings +-1).

Configurations, in this order: the stages (file order, runs in order), then the grid: instances
in the file order, for each the methods in the file order, for each the product (a, b, steps) or
(f, c0, steps) with steps innermost. Ids: sa_a<a>_b<b>_S<S> and dsb_f<f>_c<sigma|lambda>_M<M>
(numbers with repr()); a run of a stage has the id <stage>_<id>. With [[series]] (no stages
then) every point of the grid is run by every series in the file order, with its threads in
increasing order, and the id ends with _<series>_T<threads> (sa_a1.0_b2.0_S1000_win-p1_T4); the
executable is the sbmc_run of the series. Each configuration is one call:

  sbmc_run <args of the instance> --name <instance> <parameters> --replicas R --threads T
           --seed seed --reps reps --warmup warmup --target <target> --notes <NOTES>
           --tag <configuration> --out <output>/<instance>/<method>/<configuration>.csv
           --replica-out <output>/raw/<instance>/<method>/<configuration>_replicas.csv

with the parameters "--algo sa --variant c2 --sweeps S --beta0 repr(a/h) --beta1 repr(b/w)" or
"--algo dsb --variant c1 --steps M --dt auto --dt-factor repr(f) --c0 auto|lambda", from the
repository root, in the environment of the runner without the OMP_* variables; the stdout of
sbmc_run goes to the log of the campaign, its stderr to the log and to stderr. A series of
platform windows adds "--affinity <rule>" (and "--best-known <file>" with the target best-known),
and no UNC path reaches sbmc_run.exe (F07-R4e; see WindowsCopy): it reads its inputs from the copy
of the repository on Windows and writes its outputs in a folder on the same drive, then the
runner moves them into the repository. The executable lies in <dest>/build-<preset>/ (the
binaryDir of the presets win-*), and <dest>/src is the copy of scripts/windows/stage.sh (with its
.sbmc_build_info); <dest> is translated once with "wslpath -w" and must be on a Windows drive.
The rudy file and the best-known table, paths inside the repository root, become the same
relative paths in <dest>\\src (an error before the run when the file is missing there); --out and
--replica-out become <dest>\\runs\\<campaign>\\<path relative to the output folder>, both deleted
before the run and moved to their place in the repository after exit status 0 (left where they
are otherwise; the configuration is then not complete). sbmc_run.exe runs in <dest>\\src. Before
the first configuration of a Windows executable that runs, "sbmc_run.exe --version" and the
.sbmc_build_info of the copy must give the commit of HEAD (git rev-parse HEAD in the root).

Resuming: a configuration is complete when its results file has exactly reps rows with run_tag
equal to its id and its replica file exactly reps * R rows; it is skipped. An incomplete one (one
of the two files missing, truncated or with another number of rows) is removed (both files) and
run again. When sbmc_run exits with another status than 0, the runner writes "fail" in the log
and stops with exit status 1 without starting the following configurations. SIGINT and SIGTERM
stop the running sbmc_run and the runner (exit status 1); the next run repeats the interrupted
configuration. Every run ends with a line "stop" in the log. A lock
(<output>/raw/.campaign.lock) refuses a second runner on the same campaign; after SIGKILL the
running sbmc_run goes on: wait for it before the next run.

Log: <output>/campaign.log, appended, one line per event: "<UTC time> <event> instance=<id>
config=<id> seconds=<s> exit=<status>" (fields not defined are "-", a note may follow), with the
events start, skip, run, done, fail, stop; the lines of sbmc_run between "run" and "done".

status: done, partial or todo for every configuration, the totals and an estimate of the time
left. Rule: a complete configuration costs the sum over its rows of t_run_ms + t_check_ms, times
(reps + warmup) / reps; a configuration not complete is estimated as steps * q * (reps + warmup),
where q is the mean over the complete configurations of the same instance and method of
(t_run_ms + t_check_ms) / steps per row; "unknown" without such configurations. Times outside the
solver (start of the process, reading, spectrum) are not counted.

manifest: <output>/raw/MANIFEST.sha256 in the format of sha256sum (paths relative to raw/,
sorted), and its copy <output>/raw_manifest.sha256 (versioned; raw/ is ignored by git). The
runner writes them at the end of every run, also after a failure.

--dry-run prints one line per configuration, "run: <command>" or "skip: <command>" (the command
with shlex.join), and runs nothing, writes nothing; a target "best-found" not yet known is shown
as "best-found:pending". --stage NAME runs only that stage ("grid" for the grid).

Exit status: 0 success; 1 error in the data or in a child process; 2 error of the command line or
of the configuration. Messages on stderr start with "sbmc_campaign: ". Standard library only (the
analysis tools add NumPy and matplotlib); importable as a module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

PROG = "sbmc_campaign"
EXIT_OK, EXIT_DATA, EXIT_USAGE = 0, 1, 2
NAME_RE = re.compile(r"[a-z0-9_]+\Z")
ID_RE = re.compile(r"[A-Za-z0-9_]+\Z")
DECIMAL_RE = re.compile(r"[0-9]+\Z")
METHODS = {"sa": "c2", "dsb": "c1"}  # method -> its variant
C0_RULES = {"sigma": "auto", "lambda": "lambda"}  # rule -> value of --c0
GRID = "grid"
BEST_KNOWN_FILE = "data/best_known.csv"
# Results schema 2 and the replica file (sbmc/results.hpp; the same lists as python/sbmc_tts.py).
RESULTS_COLUMNS_2 = [
    "schema_version", "timestamp", "git_hash", "dirty", "host", "os", "gpu", "driver", "cuda",
    "cublas", "compiler", "algo", "variant", "instance", "n", "nnz", "replicas", "steps", "dt",
    "c0", "beta0", "beta1", "seed", "rep", "t_setup_ms", "t_solve_ms", "t_step_us", "best_cut",
    "mean_cut", "n_success", "p_s", "target", "tts99_ms", "ranks", "t_comm_ms",
    "sm_clock_avg_mhz", "temp_max_c", "power_avg_w", "a0", "eval_every", "first_replica",
    "threads", "target_source", "t_prep_ms", "t_run_ms", "t_check_ms", "dt_rule", "c0_rule",
    "lambda_min", "lambda_max", "cpu_affinity", "cpu_classes", "omp_places", "omp_proc_bind",
    "run_tag", "notes",
]
REPLICA_COLUMNS = [
    "schema_version", "timestamp", "instance", "algo", "variant", "replicas", "steps", "seed",
    "rep", "replica", "best_cut", "best_step", "final_cut",
]
INSTANCE_COLUMNS = [
    "instance", "n", "m", "h", "w", "lambda_min", "lambda_max", "dt_auto", "c0_sigma",
    "c0_lambda", "target", "target_source", "family", "target_note",
]
SPECTRUM_KEYS = ["instance", "n", "m", "lambda_min", "lambda_max", "iterations", "restarts",
                 "dt_auto", "c0_lambda", "c0_sigma", "t_ms"]


class UsageError(Exception):
    """An error of the command line or of the configuration: exit status 2."""


class DataError(Exception):
    """An error in the data or in a child process: exit status 1."""


class Interrupted(Exception):
    """SIGINT or SIGTERM during a run."""


# --- Configuration --------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    id: str
    args: tuple[str, ...]
    target: str | int  # "best-known", "best-found", "none" or an integer > 0
    family: str | None
    note: str | None

    @property
    def dense(self) -> bool:
        return self.args[0] == "--dense"

    @property
    def dense_n(self) -> int:
        return int(self.args[1])

    @property
    def file(self) -> str:
        return self.args[1]


@dataclass(frozen=True)
class Method:
    id: str
    variant: str
    a: tuple[float, ...] = ()
    b: tuple[float, ...] = ()
    f: tuple[float, ...] = ()
    c0: tuple[str, ...] = ()
    steps: tuple[tuple[str, tuple[int, ...]], ...] | None = None  # None: "selected"
    selection: str | None = None

    def steps_of(self, instance: str) -> tuple[int, ...] | None:
        return None if self.steps is None else dict(self.steps).get(instance)


@dataclass(frozen=True)
class StageRun:
    method: str
    steps: int
    a: float | None
    b: float | None
    f: float | None
    c0: str | None
    reps: int
    warmup: int


@dataclass(frozen=True)
class Stage:
    name: str
    instance: str
    target: str | int
    runs: tuple[StageRun, ...]


@dataclass(frozen=True)
class Series:
    """A platform, an executable, an --affinity and a list of threads (E9)."""

    id: str
    sbmc_run: str
    platform: str  # "windows" or "linux"
    affinity: str | None
    threads: tuple[int, ...]
    base: str | None


@dataclass(frozen=True)
class Campaign:
    path: Path
    name: str
    output: str
    sbmc_run: str
    sbmc_spectrum: str
    replicas: int
    threads: int | None  # None with [[series]]
    reps: int
    warmup: int
    seed: int
    dwave_seed: int | None
    instances: tuple[Instance, ...]
    methods: tuple[Method, ...]
    stages: tuple[Stage, ...]
    series: tuple[Series, ...] = ()

    def instance(self, instance_id: str) -> Instance:
        return next(i for i in self.instances if i.id == instance_id)


@dataclass(frozen=True)
class Config:
    """One call of sbmc_run."""

    stage: str  # GRID or the name of a stage
    index: int  # position in the order of the campaign: grid from 0; stages -1, -2, ...
    instance: Instance
    method: str
    variant: str
    steps: int
    a: float | None
    b: float | None
    f: float | None
    c0: str | None
    reps: int
    warmup: int
    target: str | int
    id: str
    series: Series | None = None
    threads: int | None = None  # None: the threads of [campaign]

    def team(self, campaign: Campaign) -> int:
        """The value of --threads."""
        threads = self.threads if self.threads is not None else campaign.threads
        assert threads is not None
        return threads

    def executable(self, campaign: Campaign) -> str:
        return self.series.sbmc_run if self.series is not None else campaign.sbmc_run

    @property
    def windows(self) -> bool:
        return self.series is not None and self.series.platform == "windows"

    def results_path(self, campaign: Campaign) -> Path:
        return Path(campaign.output) / self.instance.id / self.method / f"{self.id}.csv"

    def replica_path(self, campaign: Campaign) -> Path:
        return (Path(campaign.output) / "raw" / self.instance.id / self.method
                / f"{self.id}_replicas.csv")


def config_id(method: str, steps: int, a: float | None = None, b: float | None = None,
              f: float | None = None, c0: str | None = None) -> str:
    if method == "sa":
        return f"sa_a{a!r}_b{b!r}_S{steps}"
    return f"dsb_f{f!r}_c{c0}_M{steps}"


class _Checker:
    """Typed access to a table of the configuration, with messages that name the key and file."""

    def __init__(self, path: Path, where: str, table: object) -> None:
        self.path, self.where = path, where
        if not isinstance(table, dict):
            self.fail(f"expected a table, got {type(table).__name__}")
        self.table: dict = table

    def fail(self, message: str) -> None:
        raise UsageError(f"{self.path}: {self.where}: {message}")

    def keys(self, required: Sequence[str], optional: Sequence[str] = ()) -> None:
        unknown = sorted(set(self.table) - set(required) - set(optional))
        if unknown:
            self.fail(f"unknown key '{unknown[0]}'")
        missing = [k for k in required if k not in self.table]
        if missing:
            self.fail(f"missing key '{missing[0]}'")

    def string(self, key: str, pattern: re.Pattern[str] | None = None) -> str:
        value = self.table[key]
        if not isinstance(value, str) or not value:
            self.fail(f"key '{key}': expected a non-empty string, got {value!r}")
        if pattern is not None and not pattern.match(value):
            self.fail(f"key '{key}': '{value}' does not match {pattern.pattern[:-2]}")
        if "\n" in value or "\r" in value:
            self.fail(f"key '{key}': line breaks are not allowed")
        return value

    def integer(self, key: str, minimum: int) -> int:
        value = self.table[key]
        if isinstance(value, bool) or not isinstance(value, int):
            self.fail(f"key '{key}': expected an integer, got {value!r}")
        if value < minimum:
            self.fail(f"key '{key}': {value} must be >= {minimum}")
        return value

    def number(self, key: str) -> float:
        value = self.table[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.fail(f"key '{key}': expected a number, got {value!r}")
        if not math.isfinite(value) or value <= 0:
            self.fail(f"key '{key}': {value!r} must be a finite number > 0")
        return float(value)

    def numbers(self, key: str) -> tuple[float, ...]:
        values = self.table[key]
        if not isinstance(values, list) or not values:
            self.fail(f"key '{key}': expected a non-empty array of numbers, got {values!r}")
        out = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self.fail(f"key '{key}': expected numbers, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                self.fail(f"key '{key}': {value!r} must be a finite number > 0")
            out.append(float(value))
        return tuple(out)

    def rules(self, key: str) -> tuple[str, ...]:
        values = self.table[key]
        if not isinstance(values, list) or not values:
            self.fail(f"key '{key}': expected a non-empty array of rules, got {values!r}")
        for value in values:
            if value not in C0_RULES:
                self.fail(f"key '{key}': unknown rule {value!r} (sigma or lambda)")
        return tuple(values)

    def target(self, key: str) -> str | int:
        value = self.table[key]
        if isinstance(value, str) and value in ("best-known", "best-found", "none"):
            return value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        self.fail(f"key '{key}': unknown target {value!r} (best-known, best-found, none or an "
                  "integer > 0)")
        raise AssertionError  # unreachable


def _instance_args(check: _Checker) -> tuple[str, ...]:
    args = check.table["args"]
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        check.fail(f"key 'args': expected an array of strings, got {args!r}")
    if len(args) == 2 and args[0] == "--instance" and args[1]:
        return tuple(args)
    if (len(args) == 4 and args[0] == "--dense" and args[2] == "--instance-seed"
            and DECIMAL_RE.match(args[1]) and DECIMAL_RE.match(args[3]) and int(args[1]) >= 2):
        return tuple(args)
    check.fail(f"key 'args': expected [\"--instance\", FILE] or [\"--dense\", N, "
               f"\"--instance-seed\", S] with N >= 2, got {args!r}")
    raise AssertionError  # unreachable


def _method(path: Path, k: int, table: object, instance_ids: list[str]) -> Method:
    check = _Checker(path, f"[[method]] {k + 1}", table)
    if "id" not in check.table:
        check.fail("missing key 'id'")
    method = check.table["id"]
    if method not in METHODS:
        check.fail(f"key 'id': unknown method {method!r} (sa or dsb)")
    relative = ("a", "b") if method == "sa" else ("f", "c0")
    selection = "selection" in check.table
    check.keys(["id", "variant", "steps"] + ([] if selection else list(relative)),
               ["selection"] if selection else [])
    variant = check.string("variant")
    if variant != METHODS[method]:
        check.fail(f"key 'variant': '{variant}' is not the variant of {method} "
                   f"({METHODS[method]})")
    steps_value = check.table["steps"]
    steps: tuple[tuple[str, tuple[int, ...]], ...] | None
    if selection and steps_value == "selected":
        steps = None
    else:
        if not isinstance(steps_value, dict):
            check.fail("key 'steps': expected a table of arrays (one per instance)"
                       + (" or \"selected\"" if selection else ""))
        unknown = sorted(set(steps_value) - set(instance_ids))
        if unknown:
            check.fail(f"key 'steps': '{unknown[0]}' is not the id of an instance")
        lists = []
        for instance_id in instance_ids:
            values = steps_value.get(instance_id)
            if values is None:
                check.fail(f"key 'steps': no steps for the instance '{instance_id}'")
            if not isinstance(values, list) or not values:
                check.fail(f"key 'steps.{instance_id}': expected a non-empty array of integers")
            for value in values:
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    check.fail(f"key 'steps.{instance_id}': {value!r} is not an integer >= 1")
            lists.append((instance_id, tuple(values)))
        steps = tuple(lists)
    if selection:
        return Method(method, variant, steps=steps, selection=check.string("selection"))
    if method == "sa":
        return Method(method, variant, a=check.numbers("a"), b=check.numbers("b"), steps=steps)
    return Method(method, variant, f=check.numbers("f"), c0=check.rules("c0"), steps=steps)


def _stage(path: Path, k: int, table: object, campaign: dict, instance_ids: list[str],
           methods: dict[str, Method]) -> Stage:
    check = _Checker(path, f"[[stage]] {k + 1}", table)
    check.keys(["name", "instance", "runs"], ["target"])
    name = check.string("name", NAME_RE)
    if name == GRID:
        check.fail(f"key 'name': '{GRID}' is reserved for the grid")
    instance = check.string("instance")
    if instance not in instance_ids:
        check.fail(f"key 'instance': '{instance}' is not the id of an instance")
    target = check.target("target") if "target" in check.table else "none"
    if target == "best-found":
        check.fail("key 'target': a stage cannot have the target best-found")
    runs_value = check.table["runs"]
    if not isinstance(runs_value, list) or not runs_value:
        check.fail("key 'runs': expected a non-empty array of tables")
    runs = []
    for j, run_table in enumerate(runs_value):
        rc = _Checker(path, f"[[stage]] {k + 1} ({name}), run {j + 1}", run_table)
        if "method" not in rc.table:
            rc.fail("missing key 'method'")
        method = rc.table["method"]
        if method not in methods:
            rc.fail(f"key 'method': unknown method {method!r} (not among the [[method]] tables)")
        relative = ["a", "b"] if method == "sa" else ["f", "c0"]
        rc.keys(["method", "steps"] + relative, ["reps", "warmup"])
        steps = rc.integer("steps", 1)
        reps = rc.integer("reps", 1) if "reps" in rc.table else campaign["reps"]
        warmup = rc.integer("warmup", 0) if "warmup" in rc.table else campaign["warmup"]
        if method == "sa":
            runs.append(StageRun(method, steps, rc.number("a"), rc.number("b"), None, None, reps,
                                 warmup))
        else:
            c0 = rc.table["c0"]
            if c0 not in C0_RULES:
                rc.fail(f"key 'c0': unknown rule {c0!r} (sigma or lambda)")
            runs.append(StageRun(method, steps, None, None, rc.number("f"), c0, reps, warmup))
    return Stage(name, instance, target, tuple(runs))


SERIES_RE = re.compile(r"[a-z0-9-]+\Z")
PLATFORMS = ("windows", "linux")
AFFINITIES = ("p1", "p", "e", "all")


def _series(path: Path, k: int, table: object, earlier: list[Series]) -> Series:
    check = _Checker(path, f"[[series]] {k + 1}", table)
    if "platform" not in check.table:
        check.fail("missing key 'platform'")
    platform = check.table["platform"]
    if platform not in PLATFORMS:
        check.fail(f"key 'platform': unknown platform {platform!r} (windows or linux)")
    required = ["id", "sbmc_run", "platform", "threads"]
    check.keys(required + (["affinity"] if platform == "windows" else []), ["base"])
    series_id = check.string("id", SERIES_RE)
    if any(s.id == series_id for s in earlier):
        check.fail(f"key 'id': repeated series '{series_id}'")
    affinity = None
    if platform == "windows":
        affinity = check.string("affinity")
        if affinity not in AFFINITIES:
            check.fail(f"key 'affinity': unknown rule {affinity!r} (p1, p, e or all)")
    threads = check.table["threads"]
    if not isinstance(threads, list) or not threads:
        check.fail("key 'threads': expected a non-empty array of integers")
    for value in threads:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            check.fail(f"key 'threads': {value!r} is not an integer >= 1")
    if any(b <= a for a, b in zip(threads, threads[1:])):
        check.fail(f"key 'threads': {threads} is not strictly increasing")
    base = None
    if "base" in check.table:
        base = check.string("base", SERIES_RE)
        reference = next((s for s in earlier if s.id == base), None)
        if reference is None:
            check.fail(f"key 'base': '{base}' is not an earlier series")
        assert reference is not None
        if reference.platform != platform or 1 not in reference.threads:
            check.fail(f"key 'base': the series '{base}' must have the platform {platform} and "
                       "1 thread")
    return Series(series_id, check.string("sbmc_run"), platform, affinity, tuple(threads), base)


def load_campaign(path: Path) -> Campaign:
    """Reads and validates a campaign; UsageError names the key and the file."""
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except OSError as error:
        raise UsageError(f"cannot read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise UsageError(f"{path}: not valid TOML: {error}") from error
    top = _Checker(path, "top level", data)
    top.keys(["campaign", "instance", "method"], ["stage", "dwave", "series"])
    with_series = "series" in data
    c = _Checker(path, "[campaign]", data["campaign"])
    c.keys(["name", "output", "sbmc_run", "sbmc_spectrum", "replicas", "reps", "warmup", "seed"]
           + ([] if with_series else ["threads"]))
    values = {"name": c.string("name", NAME_RE), "output": c.string("output"),
              "sbmc_run": c.string("sbmc_run"), "sbmc_spectrum": c.string("sbmc_spectrum"),
              "replicas": c.integer("replicas", 1),
              "threads": None if with_series else c.integer("threads", 1),
              "reps": c.integer("reps", 1), "warmup": c.integer("warmup", 0),
              "seed": c.integer("seed", 0)}
    series: list[Series] = []
    if with_series:
        if not isinstance(data["series"], list) or not data["series"]:
            top.fail("key 'series': expected an array of tables ([[series]])")
        for k, table in enumerate(data["series"]):
            series.append(_series(path, k, table, series))
        if "stage" in data:
            top.fail("key 'stage': stages are not available with [[series]]")
    dwave_seed = None
    if "dwave" in data:
        d = _Checker(path, "[dwave]", data["dwave"])
        d.keys(["seed"])
        dwave_seed = d.integer("seed", 0)

    if not isinstance(data["instance"], list) or not data["instance"]:
        top.fail("key 'instance': expected an array of tables ([[instance]])")
    instances = []
    for k, table in enumerate(data["instance"]):
        ic = _Checker(path, f"[[instance]] {k + 1}", table)
        ic.keys(["id", "args", "target"], ["family", "note"])
        instance_id = ic.string("id", ID_RE)
        if any(i.id == instance_id for i in instances):
            ic.fail(f"key 'id': repeated instance id '{instance_id}'")
        instances.append(Instance(
            instance_id, _instance_args(ic), ic.target("target"),
            ic.string("family", ID_RE) if "family" in ic.table else None,
            ic.string("note") if "note" in ic.table else None))
    instance_ids = [i.id for i in instances]

    if not isinstance(data["method"], list) or not data["method"]:
        top.fail("key 'method': expected an array of tables ([[method]])")
    methods: dict[str, Method] = {}
    for k, table in enumerate(data["method"]):
        method = _method(path, k, table, instance_ids)
        if method.id in methods:
            raise UsageError(f"{path}: [[method]] {k + 1}: key 'id': repeated method "
                             f"'{method.id}'")
        methods[method.id] = method

    stages = []
    stage_values = data.get("stage", [])
    if not isinstance(stage_values, list):
        top.fail("key 'stage': expected an array of tables ([[stage]])")
    for k, table in enumerate(stage_values):
        stage = _stage(path, k, table, values, instance_ids, methods)
        if any(s.name == stage.name for s in stages):
            raise UsageError(f"{path}: [[stage]] {k + 1}: key 'name': repeated stage "
                             f"'{stage.name}'")
        stages.append(stage)
    for instance in instances:
        if instance.target == "best-found" and not any(s.instance == instance.id for s in stages):
            raise UsageError(f"{path}: [[instance]] '{instance.id}': key 'target': best-found "
                             "needs a [[stage]] on this instance")

    campaign = Campaign(path=path, dwave_seed=dwave_seed, instances=tuple(instances),
                        methods=tuple(methods.values()), stages=tuple(stages),
                        series=tuple(series), **values)
    # Repeated configuration ids (for example a = [1.0, 1.0]) are an error of the configuration.
    seen: set[tuple[str, str, str]] = set()
    for config in _expand(campaign, None):
        key = (config.instance.id, config.method, config.id)
        if key in seen:
            raise UsageError(f"{path}: repeated configuration id '{config.id}' (instance "
                             f"'{config.instance.id}', method '{config.method}')")
        seen.add(key)
    return campaign


# --- Expansion of the configurations --------------------------------------------------------


@dataclass(frozen=True)
class Skipped:
    """A method without configurations on an instance (no selection in the tuning)."""

    instance: str
    method: str
    reason: str


def read_selection(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise DataError(f"cannot read the selection {path}: {error}") from error
    needed = {"tuning_instance", "evaluation_instance", "method", "config_id", "a", "b", "f",
              "c0_rule", "steps"}
    if not rows or not needed <= set(rows[0]):
        raise DataError(f"{path}: not a selection (selected.csv of python/sbmc_e2.py)")
    return rows


def _selected_row(rows: list[dict[str, str]], path: Path, method: str,
                  instance: str) -> dict[str, str]:
    found = [r for r in rows if r["method"] == method
             and instance in (r["tuning_instance"], r["evaluation_instance"])]
    if len(found) != 1:
        raise DataError(f"{path}: {len(found)} rows for the method {method} and the instance "
                        f"{instance} (one expected)")
    return found[0]


def _parse_float(text: str, what: str, path: Path) -> float:
    try:
        value = float(text)
    except ValueError:
        raise DataError(f"{path}: {what} = '{text}' is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise DataError(f"{path}: {what} = '{text}' must be a finite number > 0")
    return value


def _expand(campaign: Campaign, root: Path | None,
            skipped: list[Skipped] | None = None) -> list[Config]:
    """The configurations in the order of the campaign. root None: methods with a selection
    give no configuration (for the check of the ids at loading time)."""
    configs: list[Config] = []
    for s_index, stage in enumerate(campaign.stages):
        instance = campaign.instance(stage.instance)
        variant_of = {m.id: m.variant for m in campaign.methods}
        for run in stage.runs:
            base = config_id(run.method, run.steps, run.a, run.b, run.f, run.c0)
            configs.append(Config(stage.name, -(s_index + 1), instance, run.method,
                                  variant_of[run.method], run.steps, run.a, run.b, run.f, run.c0,
                                  run.reps, run.warmup, stage.target, f"{stage.name}_{base}"))
    index = 0
    selections: dict[str, list[dict[str, str]]] = {}
    for instance in campaign.instances:
        for method in campaign.methods:
            if method.selection is not None:
                if root is None:
                    continue
                path = root / method.selection
                if method.selection not in selections:
                    selections[method.selection] = read_selection(path)
                row = _selected_row(selections[method.selection], path, method.id, instance.id)
                if row["config_id"] == "none":
                    if skipped is not None:
                        skipped.append(Skipped(instance.id, method.id,
                                               f"no selection in {method.selection}"))
                    continue
                if method.id == "sa":
                    product = [(_parse_float(row["a"], "a", path),
                                _parse_float(row["b"], "b", path), None, None)]
                else:
                    if row["c0_rule"] not in C0_RULES:
                        raise DataError(f"{path}: c0_rule = '{row['c0_rule']}' is not a rule")
                    product = [(None, None, _parse_float(row["f"], "f", path), row["c0_rule"])]
                steps_list = method.steps_of(instance.id)
                if steps_list is None:
                    if not DECIMAL_RE.match(row["steps"]) or int(row["steps"]) < 1:
                        raise DataError(f"{path}: steps = '{row['steps']}' is not an integer >= 1")
                    steps_list = (int(row["steps"]),)
            else:
                if method.id == "sa":
                    product = [(a, b, None, None) for a in method.a for b in method.b]
                else:
                    product = [(None, None, f, c0) for f in method.f for c0 in method.c0]
                steps_list = method.steps_of(instance.id)
                assert steps_list is not None
            points = [(s, t) for s in campaign.series for t in s.threads] or [(None, None)]
            for a, b, f, c0 in product:
                for steps in steps_list:
                    base = config_id(method.id, steps, a, b, f, c0)
                    for series, threads in points:
                        name = base if series is None else f"{base}_{series.id}_T{threads}"
                        configs.append(Config(GRID, index, instance, method.id, method.variant,
                                              steps, a, b, f, c0, campaign.reps, campaign.warmup,
                                              instance.target, name, series, threads))
                        index += 1
    return configs


def expand_configs(campaign: Campaign, root: Path,
                   skipped: list[Skipped] | None = None) -> list[Config]:
    """The stages, then the grid, in the order of the campaign (selections read from root)."""
    return _expand(campaign, root, skipped)


# --- Scales, spectrum and targets of the instances ------------------------------------------


def rudy_scales(path: Path) -> tuple[int, int, float, int | None]:
    """n, edge lines, h and w of a rudy file; repeated pairs are added, sums are exact."""
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise DataError(f"cannot read {path}: {error}") from error
    tokens = text.split()
    try:
        n, m = int(tokens[0]), int(tokens[1])
        values = [int(t) for t in tokens[2:]]
    except (IndexError, ValueError):
        raise DataError(f"{path}: not a rudy file (integers expected)") from None
    if n < 1 or m < 0 or len(values) != 3 * m:
        raise DataError(f"{path}: the header '{n} {m}' does not match the {len(values)} numbers "
                        "of the edges")
    pairs: dict[int, int] = {}
    for k in range(m):
        u, v, w = values[3 * k], values[3 * k + 1], values[3 * k + 2]
        if not (1 <= u <= n and 1 <= v <= n) or u == v:
            raise DataError(f"{path}: edge {k + 1} ({u}, {v}) is not a pair of distinct nodes in "
                            f"1..{n}")
        key = min(u, v) * (n + 1) + max(u, v)
        pairs[key] = pairs.get(key, 0) + w
    squares = 2 * sum(w * w for w in pairs.values())  # sum over i and j: every pair twice
    nonzero = [abs(w) for w in pairs.values() if w != 0]
    return n, m, math.sqrt(squares / n), (min(nonzero) if nonzero else None)


def instance_scales(instance: Instance, root: Path) -> tuple[float, int | None]:
    """h and w of an instance (section 5.2 of F07-C3)."""
    if instance.dense:
        return math.sqrt(instance.dense_n - 1), 1
    _, _, h, w = rudy_scales(root / instance.file)
    return h, w


def run_spectrum(campaign: Campaign, instance: Instance, root: Path) -> dict[str, str]:
    """The fields of the line SPECTRUM of sbmc_spectrum on the instance."""
    command = [str(root / campaign.sbmc_spectrum), *instance.args, "--name", instance.id]
    try:
        result = subprocess.run(command, cwd=root, env=child_environment(), capture_output=True,
                                text=True, check=False)
    except OSError as error:
        raise DataError(f"cannot run {campaign.sbmc_spectrum}: {error}") from error
    if result.returncode != 0:
        raise DataError(f"sbmc_spectrum exited with {result.returncode} on {instance.id}: "
                        f"{result.stderr.strip()}")
    words = result.stdout.split()
    fields = dict(w.split("=", 1) for w in words[1:] if "=" in w)
    if not words or words[0] != "SPECTRUM" or list(fields) != SPECTRUM_KEYS:
        raise DataError(f"sbmc_spectrum on {instance.id}: unexpected output "
                        f"'{result.stdout.strip()}'")
    return fields


def best_known_table(root: Path) -> dict[str, int]:
    path = root / BEST_KNOWN_FILE
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return {r["instance"]: int(r["best_known"]) for r in csv.DictReader(handle)}
    except (OSError, KeyError, ValueError) as error:
        raise DataError(f"cannot read the best-known values of {path}: {error}") from error


def stage_best(campaign: Campaign, configs: list[Config], instance: Instance,
               root: Path) -> int | None:
    """The largest best_cut of the complete runs of the stages on the instance; None when one of
    them is not complete."""
    best = None
    for config in configs:
        if config.stage == GRID or config.instance.id != instance.id:
            continue
        if config_state(config, campaign, root) != "done":
            return None
        for row in read_rows(root / config.results_path(campaign)):
            best = max(best or 0, int(row["best_cut"]))
    return best


def resolve_targets(campaign: Campaign, configs: list[Config],
                    root: Path) -> dict[str, tuple[str, str, str]]:
    """Per instance: (value of --target, target, target_source); for best-found the value is
    empty while the stages are not complete."""
    table: dict[str, int] | None = None
    out = {}
    for instance in campaign.instances:
        if instance.target == "best-known":
            table = best_known_table(root) if table is None else table
            value = table.get(instance.id)
            out[instance.id] = ("best-known", "" if value is None else str(value), "best_known")
        elif instance.target == "best-found":
            best = stage_best(campaign, configs, instance, root)
            text = "" if best is None else str(best)
            out[instance.id] = (text, text, "best_found")
        elif instance.target == "none":
            out[instance.id] = ("none", "", "none")
        else:
            out[instance.id] = (str(instance.target), str(instance.target), "given")
    return out


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """A CSV of RFC 4180 with '\\n', written through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
        temporary.replace(path)
    except OSError as error:
        raise DataError(f"cannot write {path}: {error}") from error


def number(value: float | None) -> str:
    """A float of a CSV: repr(); empty when undefined; "inf" for infinity."""
    return "" if value is None else repr(float(value))


def write_instances(campaign: Campaign, root: Path, configs: list[Config]) -> None:
    """<output>/instances.csv: scales, spectrum and target of every instance."""
    targets = resolve_targets(campaign, configs, root)
    rows = []
    for instance in campaign.instances:
        h, w = instance_scales(instance, root)
        spectrum = run_spectrum(campaign, instance, root)
        _, target, source = targets[instance.id]
        note = ""
        if instance.target == "best-found":
            note = ("largest best_cut of the stages" if target else
                    "pending: the stages are not complete")
        rows.append([instance.id, spectrum["n"], spectrum["m"], number(h),
                     "" if w is None else str(w), spectrum["lambda_min"], spectrum["lambda_max"],
                     spectrum["dt_auto"], spectrum["c0_sigma"], spectrum["c0_lambda"], target,
                     source, instance.family or "", note])
    write_csv(root / campaign.output / "instances.csv", INSTANCE_COLUMNS, rows)


# --- Commands, states and files -------------------------------------------------------------


def parameters(config: Config, scales: dict[str, tuple[float, int | None]]) -> list[str]:
    if config.method == "sa":
        h, w = scales[config.instance.id]
        if h == 0 or w is None:
            raise DataError(f"instance {config.instance.id}: h = {h!r}, w = {w!r}: no beta "
                            "relative to the scales (an instance without edges)")
        assert config.a is not None and config.b is not None
        return ["--algo", "sa", "--variant", config.variant, "--sweeps", str(config.steps),
                "--beta0", repr(config.a / h), "--beta1", repr(config.b / w)]
    assert config.c0 is not None
    return ["--algo", "dsb", "--variant", config.variant, "--steps", str(config.steps), "--dt",
            "auto", "--dt-factor", repr(config.f), "--c0", C0_RULES[config.c0]]


class WindowsCopy:
    """The files of a Windows executable started from WSL, without UNC paths (F07-R4e).

    With the libstdc++ of MinGW, sbmc_run.exe fails on a path such as
    \\\\wsl.localhost\\<distro>\\... (seen on the laptop: std::filesystem cannot resolve the root of
    the share). The executable lies
    in <dest>/build-<preset>/...; <dest>/src is the copy of scripts/windows/stage.sh. Inputs inside
    the repository root are read from the same relative path in the copy, outputs are written in
    <dest>/runs/<work>/ and moved into the repository by move(). <dest> is translated once with
    "wslpath -w"; a translation that is not on a Windows drive (a UNC path) is an error."""

    def __init__(self, root: Path, executable: str | Path, work: str) -> None:
        self.root = root
        self.executable = Path(executable) if Path(executable).is_absolute() else root / executable
        build = next((p for p in self.executable.parents if p.name.startswith("build-")), None)
        if build is None:
            raise DataError(f"{executable}: a Windows executable must lie in "
                            "<dest>/build-<preset>/ (the presets win-*), next to the copy "
                            "<dest>/src of scripts/windows/stage.sh")
        self.dest = build.parent
        self.copy = self.dest / "src"
        self.info = self.copy / ".sbmc_build_info"
        if not self.info.is_file():
            raise DataError(f"{executable}: {self.info} not found: {self.copy} is not a copy "
                            "made by scripts/windows/stage.sh")
        self.runs = self.dest / "runs" / work
        self.work = work
        self.dest_windows: str | None = None
        self.checked = False

    def windows(self, path: Path) -> str:
        """The Windows path of a path under <dest>."""
        if self.dest_windows is None:
            text = wslpath(self.dest)
            if text.startswith("\\") or not re.match(r"[A-Za-z]:\\", text + "\\"):
                raise DataError(f"wslpath -w {self.dest} gives {text}, not a path on a Windows "
                                "drive: the Windows executables get no UNC path; put the copy "
                                "on a drive (scripts/windows/stage.sh DEST, default "
                                "/mnt/c/Users/Simone/sbmc-win)")
            self.dest_windows = text
        return "\\".join([self.dest_windows, *path.relative_to(self.dest).parts])

    def input(self, path: str | Path) -> str:
        """An input of the repository: the same relative path in the copy (DataError when it is
        outside the root or missing in the copy)."""
        candidate = Path(path)
        absolute = candidate if candidate.is_absolute() else self.root / candidate
        try:
            relative = absolute.relative_to(self.root)
        except ValueError:
            raise DataError(f"{path}: an input of a Windows executable must lie in the "
                            f"repository ({self.root}), to be read from the copy {self.copy}"
                            ) from None
        copied = self.copy / relative
        if not copied.is_file():
            raise DataError(f"{relative.as_posix()} is missing in the Windows copy {self.copy} "
                            "(run scripts/windows/stage.sh again)")
        return self.windows(copied)

    def output(self, relative: Path) -> tuple[Path, str]:
        """An output: its file in <dest>/runs/<work>/ (WSL path) and the Windows path of it."""
        work = self.runs / relative
        return work, self.windows(work)

    def check_current(self, root: Path) -> None:
        """sbmc_run.exe --version and the .sbmc_build_info of the copy give the commit of HEAD
        (checked once)."""
        if self.checked:
            return
        head = git_head(root)
        match = re.search(r"^git_hash=([0-9a-f]{40})$", self.info.read_text(encoding="ascii"),
                          re.MULTILINE)
        staged = match.group(1) if match else "unreadable"
        if staged != head:
            raise DataError(f"the Windows copy {self.copy} is of commit {staged}, HEAD is {head}: "
                            "run scripts/windows/stage.sh and scripts/windows/build.ps1 again")
        try:
            result = subprocess.run([str(self.executable), "--version"], capture_output=True,
                                    text=True, check=False, cwd=self.copy,
                                    env=child_environment(), timeout=120)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DataError(f"cannot run {self.executable} --version: {error}") from error
        found = re.search(r"git_hash=(\S+)", result.stdout)
        built = found.group(1) if found else "unreadable"
        if result.returncode != 0 or built != head:
            raise DataError(f"{self.executable} --version (exit {result.returncode}) gives "
                            f"git_hash={built}, HEAD is {head}: build the copy again "
                            "(scripts/windows/build.ps1)")
        self.checked = True

    @staticmethod
    def move(work: Path, final: Path) -> None:
        """An output written on Windows into its place in the repository (a copy, then a rename
        in the folder of the final file, then the removal of the work file)."""
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(final.name + ".tmp")
            shutil.copyfile(work, temporary)
            os.replace(temporary, final)
            work.unlink()
        except OSError as error:
            raise DataError(f"cannot move {work} to {final}: {error}") from error


def wslpath(path: Path) -> str:
    try:
        result = subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True,
                                check=False)
    except OSError as error:
        raise DataError(f"wslpath: {error} (the Windows executables run from WSL)") from error
    text = result.stdout.strip()
    if result.returncode != 0 or not text:
        raise DataError(f"wslpath -w {path} exited with {result.returncode}: "
                        f"{result.stderr.strip()}")
    return text.rstrip("\\")


def git_head(root: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=False)
    except OSError as error:
        raise DataError(f"git: {error} (the Windows executables are checked against HEAD)"
                        ) from error
    head = result.stdout.strip()
    if result.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise DataError(f"git -C {root} rev-parse HEAD exited with {result.returncode}: "
                        f"{result.stderr.strip()} (the Windows executables are checked "
                        "against HEAD)")
    return head


class WindowsCopies:
    """One WindowsCopy per Windows executable of a campaign, with the work folder
    runs/<campaign>."""

    def __init__(self, root: Path, campaign: Campaign) -> None:
        self.root = root
        self.campaign = campaign
        self.copies: dict[str, WindowsCopy] = {}

    def __call__(self, config: Config) -> WindowsCopy:
        executable = config.executable(self.campaign)
        if executable not in self.copies:
            self.copies[executable] = WindowsCopy(self.root, executable, self.campaign.name)
        return self.copies[executable]

    def outputs(self, config: Config) -> list[tuple[Path, Path]]:
        """(work file, final file) of the two outputs of a Windows configuration."""
        copy = self(config)
        return [(copy.output(relative_output(path, self.campaign))[0], self.root / path)
                for path in (config.results_path(self.campaign),
                             config.replica_path(self.campaign))]


def relative_output(path: Path, campaign: Campaign) -> Path:
    return path.relative_to(Path(campaign.output))


def command(config: Config, campaign: Campaign, scales: dict[str, tuple[float, int | None]],
            target: str, notes: str, executable: str | None = None,
            copies: WindowsCopies | None = None) -> list[str]:
    """The call of sbmc_run of a configuration (paths relative to the repository root; for a
    series of platform windows, which needs copies, the Windows paths of WindowsCopy)."""
    args = list(config.instance.args)
    out, replica_out = str(config.results_path(campaign)), str(config.replica_path(campaign))
    extra: list[str] = []
    if config.windows:
        if copies is None:
            raise DataError(f"configuration {config.id}: a series of platform windows needs the "
                            "Windows copy")
        assert config.series is not None and config.series.affinity is not None
        copy = copies(config)
        if not config.instance.dense:
            args[1] = copy.input(args[1])
        out = copy.output(relative_output(config.results_path(campaign), campaign))[1]
        replica_out = copy.output(relative_output(config.replica_path(campaign), campaign))[1]
        extra = ["--affinity", config.series.affinity]
        if target == "best-known":
            extra += ["--best-known", copy.input(BEST_KNOWN_FILE)]
    return [executable or config.executable(campaign), *args, "--name", config.instance.id,
            *parameters(config, scales), "--replicas", str(campaign.replicas), "--threads",
            str(config.team(campaign)), *extra, "--seed", str(campaign.seed), "--reps",
            str(config.reps), "--warmup", str(config.warmup), "--target", target, "--notes", notes,
            "--tag", config.id, "--out", out, "--replica-out", replica_out]


def read_rows(path: Path) -> list[dict[str, str]]:
    """The rows of a results file of schema 2 (DataError when it is not one)."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DataError(f"cannot read {path}: {error}") from error
    if not table or table[0] != RESULTS_COLUMNS_2:
        raise DataError(f"{path}: not a results file of schema 2")
    if any(len(r) != len(RESULTS_COLUMNS_2) for r in table[1:]):
        raise DataError(f"{path}: a row with another number of fields")
    return [dict(zip(RESULTS_COLUMNS_2, r)) for r in table[1:]]


def _complete_file(path: Path, columns: list[str], rows: int, tag: str | None) -> bool:
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            text = handle.read()
        if not text.endswith("\n"):
            return False
        table = list(csv.reader(text.splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False
    if not table or table[0] != columns or len(table) - 1 != rows:
        return False
    if any(len(r) != len(columns) for r in table[1:]):
        return False
    if tag is not None:
        index = columns.index("run_tag")
        return all(r[index] == tag for r in table[1:])
    return True


def config_state(config: Config, campaign: Campaign, root: Path) -> str:
    """"done", "partial" or "todo" (section 5.2 of F07-C3)."""
    results = root / config.results_path(campaign)
    replicas = root / config.replica_path(campaign)
    if not results.exists() and not replicas.exists():
        return "todo"
    if (_complete_file(results, RESULTS_COLUMNS_2, config.reps, config.id)
            and _complete_file(replicas, REPLICA_COLUMNS, config.reps * campaign.replicas, None)):
        return "done"
    return "partial"


def child_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("OMP_")}


def write_manifest(campaign: Campaign, root: Path) -> int:
    """<output>/raw/MANIFEST.sha256 and its copy <output>/raw_manifest.sha256; files listed."""
    raw = root / campaign.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(p for p in raw.rglob("*") if p.is_file()):
        relative = path.relative_to(raw).as_posix()
        if relative in ("MANIFEST.sha256", ".campaign.lock") or relative.endswith(".tmp"):
            continue
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {relative}\n")
    lines.sort(key=lambda line: line.split("  ", 1)[1])
    text = "".join(lines)
    for path in (raw / "MANIFEST.sha256", root / campaign.output / "raw_manifest.sha256"):
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    return len(lines)


class Log:
    """<output>/campaign.log, one line per event, also printed on stdout."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8")

    def event(self, event: str, instance: str = "-", config: str = "-",
              seconds: float | None = None, status: int | None = None, note: str = "") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = (f"{stamp} {event} instance={instance} config={config} "
                f"seconds={'-' if seconds is None else format(seconds, '.3f')} "
                f"exit={'-' if status is None else status}" + (f" {note}" if note else ""))
        self.handle.write(line + "\n")
        self.handle.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def _signal_handler(signum: int, _frame: object) -> None:
    raise Interrupted(signum)


def _run_one(config: Config, campaign: Campaign, root: Path, cmd: list[str], log: Log,
             copies: WindowsCopies | None = None) -> int:
    state = config_state(config, campaign, root)
    if state == "done":
        log.event("skip", config.instance.id, config.id, note="complete")
        return EXIT_OK
    cwd = root
    outputs: list[tuple[Path, Path]] = []
    if config.windows:
        assert copies is not None
        copies(config).check_current(root)
        cwd = copies(config).copy
    note = ""
    if state == "partial":
        for path in (config.results_path(campaign), config.replica_path(campaign)):
            (root / path).unlink(missing_ok=True)
        note = "incomplete files removed, repeated"
    for path in (config.results_path(campaign), config.replica_path(campaign)):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
    if config.windows:
        assert copies is not None
        outputs = copies.outputs(config)
        for work, _ in outputs:
            work.unlink(missing_ok=True)
            work.parent.mkdir(parents=True, exist_ok=True)
    log.event("run", config.instance.id, config.id, note=note)
    start = time.monotonic()
    try:
        process = subprocess.Popen([str(root / cmd[0]), *cmd[1:]], cwd=cwd,
                                   env=child_environment(), stdout=log.handle,
                                   stderr=subprocess.PIPE, text=True)
    except OSError as error:
        log.event("fail", config.instance.id, config.id, time.monotonic() - start,
                  note=f"cannot start {cmd[0]}: {error}")
        return EXIT_DATA
    try:
        _, stderr = process.communicate()
    except Interrupted:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    seconds = time.monotonic() - start
    if stderr:
        log.handle.write(stderr if stderr.endswith("\n") else stderr + "\n")
        log.handle.flush()
        sys.stderr.write(stderr)
    if process.returncode != 0:
        log.event("fail", config.instance.id, config.id, seconds, process.returncode,
                  note="sbmc_run failed; the following configurations are not started"
                  + ("; its files stay in " + str(copies(config).runs)
                     if config.windows and copies is not None else ""))
        return EXIT_DATA
    for work, final in outputs:
        WindowsCopy.move(work, final)
    if config_state(config, campaign, root) != "done":
        log.event("fail", config.instance.id, config.id, seconds, process.returncode,
                  note="the files are not complete after sbmc_run")
        return EXIT_DATA
    log.event("done", config.instance.id, config.id, seconds, process.returncode)
    return EXIT_OK


def run_campaign(campaign: Campaign, root: Path, notes: str, stage: str | None,
                 dry_run: bool) -> int:
    skipped: list[Skipped] = []
    configs = expand_configs(campaign, root, skipped)
    if stage is not None and stage != GRID and all(s.name != stage for s in campaign.stages):
        raise UsageError(f"--stage {stage}: no such stage in {campaign.path} (stages: "
                         f"{', '.join([s.name for s in campaign.stages] + [GRID])})")
    selected = [c for c in configs if stage is None or c.stage == stage]
    needs_scales = {c.instance.id for c in selected if c.method == "sa"}
    for instance in campaign.instances:
        if not instance.dense and not (root / instance.file).is_file():
            raise DataError(f"instance {instance.id}: {instance.file} not found")
    scales = {i.id: instance_scales(i, root) for i in campaign.instances
              if i.id in needs_scales or not dry_run}
    copies = WindowsCopies(root, campaign) if any(c.windows for c in selected) else None

    if dry_run:
        targets = resolve_targets(campaign, configs, root)
        for config in selected:
            target = (targets[config.instance.id][0] if config.stage == GRID
                      else _stage_target(config))
            state = config_state(config, campaign, root)
            mark = "skip" if state == "done" else "run"
            print(f"{mark}: " + shlex.join(command(config, campaign, scales,
                                                   target or "best-found:pending", notes,
                                                   copies=copies)))
        for entry in skipped:
            print(f"skip: instance {entry.instance}, method {entry.method}: {entry.reason}")
        return EXIT_OK

    for executable in dict.fromkeys(c.executable(campaign) for c in selected):
        if not (root / executable).is_file():
            raise DataError(f"{executable} not found (build the preset first)")
    output = root / campaign.output
    (output / "raw").mkdir(parents=True, exist_ok=True)
    lock = open(output / "raw" / ".campaign.lock", "w", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise DataError(f"another runner is working on {campaign.output}") from None
    log = Log(output / "campaign.log")
    previous = {s: signal.signal(s, _signal_handler) for s in (signal.SIGINT, signal.SIGTERM)}
    status = EXIT_OK
    interrupted_by: int | None = None
    started = time.monotonic()
    log.event("start", note=f"campaign={campaign.name} stage={stage or 'all'} "
                            f"configurations={len(selected)} notes={notes}")
    try:
        write_instances(campaign, root, configs)
        for entry in skipped:
            if stage in (None, GRID):
                log.event("skip", entry.instance, entry.method, note=entry.reason)
        for config in selected:
            if config.stage == GRID and config.target == "best-found":
                target = resolve_targets(campaign, configs, root)[config.instance.id][0]
                if not target:
                    log.event("fail", config.instance.id, config.id,
                              note="best-found target unknown: run the stages first")
                    status = EXIT_DATA
                    break
            elif config.stage == GRID:
                target = resolve_targets(campaign, configs, root)[config.instance.id][0]
            else:
                target = _stage_target(config)
            cmd = command(config, campaign, scales, target, notes, copies=copies)
            status = _run_one(config, campaign, root, cmd, log, copies)
            if status != EXIT_OK:
                break
        if status == EXIT_OK and any(i.target == "best-found" for i in campaign.instances):
            write_instances(campaign, root, configs)
    except Interrupted as interrupted:
        interrupted_by = interrupted.args[0]
        status = EXIT_DATA
    except DataError as error:
        log.event("fail", seconds=time.monotonic() - started, status=EXIT_DATA, note=str(error))
        status = EXIT_DATA
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        files = write_manifest(campaign, root)
        if interrupted_by is not None:
            note = f"interrupted by signal {interrupted_by}"
        elif status == EXIT_OK:
            note = "complete"
        else:
            note = "stopped after a failure"
        log.event("stop", seconds=time.monotonic() - started, status=status,
                  note=f"{note}; manifest of {files} files")
        log.close()
        lock.close()
    return status


def _stage_target(config: Config) -> str:
    return str(config.target)


# --- status ---------------------------------------------------------------------------------


def config_cost_ms(config: Config, campaign: Campaign, root: Path) -> tuple[float, float]:
    """Of a complete configuration: its cost (ms) and q, (t_run_ms + t_check_ms) / steps per row."""
    rows = read_rows(root / config.results_path(campaign))
    times = [float(r["t_run_ms"]) + float(r["t_check_ms"]) for r in rows]
    cost = sum(times) * (config.reps + config.warmup) / config.reps
    return cost, sum(t / config.steps for t in times) / len(times)


def status(campaign: Campaign, root: Path) -> int:
    skipped: list[Skipped] = []
    configs = expand_configs(campaign, root, skipped)
    states = [config_state(c, campaign, root) for c in configs]
    rates: dict[tuple[str, str], list[float]] = {}
    costs: dict[int, float] = {}
    for k, (config, state) in enumerate(zip(configs, states)):
        if state == "done":
            costs[k], q = config_cost_ms(config, campaign, root)
            rates.setdefault((config.instance.id, config.method), []).append(q)
    remaining = 0.0
    unknown = 0
    for k, (config, state) in enumerate(zip(configs, states)):
        detail = ""
        if state == "done":
            detail = f" ({costs[k] / 1000:.1f} s)"
        else:
            q = rates.get((config.instance.id, config.method))
            if q:
                estimate = config.steps * (sum(q) / len(q)) * (config.reps + config.warmup)
                remaining += estimate
                detail = f" (estimate {estimate / 1000:.1f} s)"
            else:
                unknown += 1
                detail = " (estimate unknown)"
        print(f"STATUS {config.stage} {config.instance.id} {config.id}: {state}{detail}")
    for entry in skipped:
        print(f"STATUS {GRID} {entry.instance} {entry.method}: skipped ({entry.reason})")
    print(f"STATUS_TOTAL configurations={len(configs)} done={states.count('done')} "
          f"partial={states.count('partial')} todo={states.count('todo')} "
          f"estimated_remaining_s={remaining / 1000:.1f} unknown={unknown}")
    return EXIT_OK


# --- Command line ---------------------------------------------------------------------------


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = Parser(prog=PROG, description="Resumable runner of the campaigns of sbmc_run.")
    parser.add_argument("--root", default=".", help="repository root (default: the current "
                                                    "directory)")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    run = commands.add_parser("run", help="run the configurations that are not complete")
    run.add_argument("config", help="TOML file of the campaign")
    run.add_argument("--notes", required=True, help="text of the column notes (one line)")
    run.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    run.add_argument("--stage", help='only this stage ("grid" for the grid)')
    st = commands.add_parser("status", help="state of every configuration and time left")
    st.add_argument("config", help="TOML file of the campaign")
    man = commands.add_parser("manifest", help="write the SHA-256 manifest of raw/")
    man.add_argument("config", help="TOML file of the campaign")
    args = parser.parse_args(list(argv))
    if args.command == "run":
        if not args.notes or "\n" in args.notes or "\r" in args.notes:
            raise UsageError("--notes must be one non-empty line")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_arguments(sys.argv[1:] if argv is None else argv)
        root = Path(args.root).resolve()
        config_path = Path(args.config)
        campaign = load_campaign(config_path if config_path.is_absolute() else root / config_path)
        if args.command == "run":
            return run_campaign(campaign, root, args.notes, args.stage, args.dry_run)
        if args.command == "status":
            return status(campaign, root)
        files = write_manifest(campaign, root)
        print(f"MANIFEST {campaign.output}/raw/MANIFEST.sha256 files={files}")
        return EXIT_OK
    except UsageError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return EXIT_USAGE
    except DataError as error:
        print(f"{PROG}: {error}", file=sys.stderr)
        return EXIT_DATA


if __name__ == "__main__":
    sys.exit(main())
