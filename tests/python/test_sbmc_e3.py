#!/usr/bin/env python3
"""python/sbmc_e3.py, the analysis of E3 on the CPU (F07-C3, sections 5.5 and 5.7).

Run by CTest as:
  <python> -I -B test_sbmc_e3.py --pyroot <root>/python --root <root> --workdir <build>/tests/e3
Checks (values of cldArchitect, verifica/f07/c3_ref.py; floats with a relative tolerance of
1e-12):
1. interpolation in ln t: SA at (10 ms, 90.0) and (40 ms, 94.0), SB at (20 ms, 92.0) and (80 ms,
   95.0): budgets 20.0, 28.284271247461902 and 40.0; SA 92.0, 93.0, 94.0; SB 92.0, 92.75, 93.5;
   no extrapolation (None outside the range of the points); disjoint ranges: no budget;
2. ratio and level on values written here: the ratio SA/SB of two TTS99 with the interval
   [lo_SA / hi_SB, hi_SA / lo_SB], an infinite upper bound, an undefined TTS99; the common level,
   the most severe one where both P_s are > 0;
3. a fake campaign written by the test (a selection of E2 with the family tx -> ex, two methods
   with two steps each, R 4, 2 repetitions), analyzed: e3_tts.csv with three rows, the
   transferred configurations, the ratio and its interval from the columns of the row, the common
   level 995 (SB has no success at 1000), the note of the instance; quality_equal_time.csv with the
   points and the three budgets; the plots tts_bars_<f> and quality_time_ex, PDF and PNG, not
   empty; the lines "E3 ..." and E3_DONE.

Exit status: 0 passed, 1 failed, 77 when NumPy or matplotlib cannot be imported. The last line is
"E3_VERDICT: PASS", "E3_VERDICT: FAIL" or "E3_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import csv
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77
TOLERANCE = 1e-12


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"E3 {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def close(a: float | None, b: float) -> bool:
    return a is not None and abs(a - b) <= TOLERANCE * max(abs(a), abs(b))


def write_config(sc, folder: Path, instance: str, method: str, config: str, steps: int,
                 times: tuple[float, ...], cuts: tuple[list[int], ...], target: int,
                 minute: int) -> None:
    replicas = len(cuts[0])
    rows, replica_rows = [], []
    variant = "c2" if method == "sa" else "c1"
    for rep, (t_run, values) in enumerate(zip(times, cuts)):
        stamp = f"2026-11-22T11:{minute:02d}:{rep:02d}Z"
        row = {c: "" for c in sc.RESULTS_COLUMNS_2}
        row.update({
            "schema_version": "2", "timestamp": stamp, "git_hash": "test", "dirty": "0",
            "host": "test", "os": "test", "compiler": "test", "algo": method,
            "variant": variant, "instance": instance, "n": "24", "nnz": "96",
            "replicas": str(replicas), "steps": str(steps), "seed": "1", "rep": str(rep),
            "t_setup_ms": "0.1", "t_solve_ms": repr(t_run + 1.0), "t_step_us": "1",
            "best_cut": str(max(values)), "mean_cut": format(sum(values) / replicas, ".3f"),
            "first_replica": str(rep * replicas), "threads": "2", "t_prep_ms": "0.5",
            "t_run_ms": repr(t_run), "t_check_ms": "0.5", "cpu_affinity": "0-1",
            "run_tag": config, "notes": "fake", "target": str(target), "target_source": "user",
            "n_success": str(sum(v >= target for v in values)),
            "p_s": repr(sum(v >= target for v in values) / replicas)})
        if method == "sa":
            row.update({"beta0": "0.5", "beta1": "2"})
        else:
            row.update({"dt": "1.1", "c0": "0.3", "dt_rule": "auto", "c0_rule": "sigma"})
        rows.append([row[c] for c in sc.RESULTS_COLUMNS_2])
        for k, value in enumerate(values):
            replica_rows.append(["1", stamp, instance, method, variant, str(replicas), str(steps),
                                 "1", str(rep), str(rep * replicas + k), str(value),
                                 str(steps - 1), str(value)])
    sc.write_csv(folder / instance / method / f"{config}.csv", sc.RESULTS_COLUMNS_2, rows)
    sc.write_csv(folder / "raw" / instance / method / f"{config}_replicas.csv",
                 sc.REPLICA_COLUMNS, replica_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of python/sbmc_e3.py.")
    parser.add_argument("--pyroot", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    print(f"python: {platform.python_version()} ({sys.executable})")
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as error:
        print(f"missing: {error} (run scripts/setup_python.sh, then configure again)")
        print("E3_VERDICT: SKIP")
        return EXIT_SKIP
    pyroot = Path(args.pyroot).resolve()
    sys.path.insert(0, str(pyroot))
    import sbmc_campaign as sc
    import sbmc_e3

    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report = Report()

    # 1. Interpolation and budgets.
    sa = [(10.0, 90.0), (40.0, 94.0)]
    sb = [(20.0, 92.0), (80.0, 95.0)]
    span = sbmc_e3.budgets([p[0] for p in sa], [p[0] for p in sb])
    expected = (20.0, 28.284271247461902, 40.0)
    report.check("budgets of the intersection [20, 40] and its geometric mean",
                 span is not None and all(close(a, b) for a, b in zip(span, expected)), f"{span}")
    if span is not None:
        got_sa = [sbmc_e3.interpolate(sa, b) for b in span]
        got_sb = [sbmc_e3.interpolate(sb, b) for b in span]
        report.check("SA at the budgets: 92.0, 93.0, 94.0",
                     all(close(g, e) for g, e in zip(got_sa, (92.0, 93.0, 94.0))), f"{got_sa}")
        report.check("SB at the budgets: 92.0, 92.75, 93.5",
                     all(close(g, e) for g, e in zip(got_sb, (92.0, 92.75, 93.5))), f"{got_sb}")
    outside = [sbmc_e3.interpolate(sa, 9.999), sbmc_e3.interpolate(sa, 40.001),
               sbmc_e3.interpolate(sb, 10.0)]
    report.check("no extrapolation", outside == [None, None, None], f"{outside}")
    report.check("disjoint ranges: no overlap",
                 sbmc_e3.budgets([1.0, 2.0], [3.0, 4.0]) is None
                 and sbmc_e3.budgets([], [3.0]) is None, "")

    # 2. Ratio, its interval and the common level.
    ratio = sbmc_e3.ratio_interval((30.0, 20.0, 60.0), (10.0, 5.0, 40.0))
    report.check("ratio 3.0 in [0.5, 12.0]", close(ratio[0], 3.0) and close(ratio[1], 0.5)
                 and close(ratio[2], 12.0), f"{ratio}")
    ratio = sbmc_e3.ratio_interval((30.0, 20.0, None), (10.0, 5.0, None))
    report.check("infinite upper bounds: [0, inf]", close(ratio[0], 3.0) and ratio[1] == 0.0
                 and ratio[2] == math.inf, f"{ratio}")
    report.check("undefined TTS99: no ratio",
                 sbmc_e3.ratio_interval((None, None, None), (10.0, 5.0, 40.0))
                 == (None, None, None), "")
    level = sbmc_e3.common_level({1000: (0.0, 0.2), 995: (0.1, 0.3), 990: (0.2, 0.4)})
    report.check("common level 995 (SB has no success at 1000)", level == 995, f"{level}")
    level = sbmc_e3.common_level({1000: (0.0, 0.2), 995: (0.0, 0.3), 990: (None, 0.4)})
    report.check("no common level", level is None, f"{level}")

    # 3. A fake campaign.
    folder = work / "camp"
    selection = work / "selected.csv"
    sc.write_csv(selection, ["tuning_instance", "evaluation_instance", "method", "level_permille",
                             "config_id", "a", "b", "f", "c0_rule", "steps"],
                 [["tx", "ex", "sa", "1000", "sa_a1.0_b2.0_S100", "1.0", "2.0", "", "", "100"],
                  ["tx", "ex", "dsb", "995", "dsb_f0.5_csigma_M200", "", "", "0.5", "sigma",
                   "200"]])
    toml = work / "e3.toml"
    toml.write_text(
        f'[campaign]\nname = "fake_e3"\noutput = "{folder}"\nsbmc_run = "none"\n'
        'sbmc_spectrum = "none"\nreplicas = 4\nthreads = 2\nreps = 2\nwarmup = 0\nseed = 1\n'
        '[[instance]]\nid = "ex"\nargs = ["--instance", "ex.rudy"]\ntarget = 200\n'
        'note = "preliminary target (one source)"\n'
        f'[[method]]\nid = "sa"\nvariant = "c2"\nselection = "{selection}"\n'
        "steps = { ex = [100, 200] }\n"
        f'[[method]]\nid = "dsb"\nvariant = "c1"\nselection = "{selection}"\n'
        "steps = { ex = [100, 200] }\n", encoding="utf-8")
    write_config(sc, folder, "ex", "sa", "sa_a1.0_b2.0_S100", 100, (10.0, 12.0),
                 ([200, 199, 198, 200], [197, 200, 199, 199]), 200, 0)
    write_config(sc, folder, "ex", "sa", "sa_a1.0_b2.0_S200", 200, (30.0, 34.0),
                 ([200, 200, 199, 200], [200, 200, 200, 199]), 200, 1)
    write_config(sc, folder, "ex", "dsb", "dsb_f0.5_csigma_M100", 100, (5.0, 6.0),
                 ([198, 199, 197, 199], [199, 196, 199, 198]), 200, 2)
    write_config(sc, folder, "ex", "dsb", "dsb_f0.5_csigma_M200", 200, (11.0, 13.0),
                 ([199, 199, 198, 199], [199, 199, 197, 199]), 200, 3)
    result = subprocess.run([sys.executable, "-I", "-B", str(pyroot / "sbmc_e3.py"), "--root",
                             str(Path(args.root).resolve()), "analyze", str(toml)],
                            capture_output=True, text=True, check=False, timeout=600)
    report.check("analyze: exit 0, three lines E3 and E3_DONE", result.returncode == 0
                 and result.stdout.strip().endswith("E3_DONE")
                 and sum(line.startswith("E3 ex level=") for line in result.stdout.splitlines())
                 == 3, f"exit {result.returncode}: {result.stderr.strip()[:300]}")
    if result.returncode == 0:
        with open(folder / "e3_tts.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        levels = [r["level_permille"] for r in rows]
        report.check("e3_tts.csv: three rows, the transferred configurations",
                     levels == ["1000", "995", "990"]
                     and all(r["sa_config_id"] == "sa_a1.0_b2.0_S100"
                             and r["sb_config_id"] == "dsb_f0.5_csigma_M200"
                             and r["tuning_instance"] == "tx" for r in rows), f"{levels}")
        common = [r["common_level"] for r in rows]
        report.check("common level 995", common == ["0", "1", "0"]
                     and rows[0]["sb_tts99_ms"] == "" and rows[0]["ratio_sa_sb"] == "", f"{common}")
        r = rows[1]
        ratio = float(r["sa_tts99_ms"]) / float(r["sb_tts99_ms"])
        lo = float(r["sa_tts99_lo_ms"]) / float(r["sb_tts99_hi_ms"])
        hi_text = r["sa_tts99_hi_ms"]
        hi = math.inf if hi_text in ("", "inf") else float(hi_text) / float(r["sb_tts99_lo_ms"])
        report.check("ratio SA/SB at 995 and its conservative interval",
                     close(float(r["ratio_sa_sb"]), ratio)
                     and close(float(r["ratio_sa_sb_lo"]), lo)
                     and (float(r["ratio_sa_sb_hi"]) == hi
                          or close(float(r["ratio_sa_sb_hi"]), hi)),
                     f"{r['ratio_sa_sb']} [{r['ratio_sa_sb_lo']}, {r['ratio_sa_sb_hi']}]")
        report.check("the note of the instance and the minimum over the steps",
                     all(x["note"] == "preliminary target (one source)" for x in rows)
                     and rows[2]["sa_min_steps"] in ("100", "200"), f"{rows[2]['note']}")
        with open(folder / "quality_equal_time.csv", newline="", encoding="utf-8") as handle:
            quality = list(csv.DictReader(handle))
        points = [q for q in quality if q["kind"] == "point"]
        budget_rows = [q for q in quality if q["kind"] == "budget"]
        report.check("quality_equal_time.csv: 4 points and 3 budgets per method",
                     len(points) == 4 and len(budget_rows) == 6
                     and {q["budget"] for q in budget_rows} == {"min", "geomean", "max"},
                     f"{len(points)} points, {len(budget_rows)} budget rows")
        plots = folder / "plots"
        names = [plots / f"{stem}.{ext}" for stem in ("tts_bars_1000", "tts_bars_995",
                                                      "tts_bars_990", "quality_time_ex")
                 for ext in ("pdf", "png")]
        report.check("plots: tts_bars_<f> and quality_time_ex, PDF and PNG, not empty",
                     all(p.exists() and p.stat().st_size > 1000 for p in names),
                     ", ".join(f"{p.name} {p.stat().st_size if p.exists() else 'missing'}"
                               for p in names))

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"E3_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
