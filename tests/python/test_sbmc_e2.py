#!/usr/bin/env python3
"""python/sbmc_e2.py, the analysis of the tuning E2 (F07-C3, sections 5.3 and 5.7).

Run by CTest as:
  <python> -I -B test_sbmc_e2.py --pyroot <root>/python --root <root> --workdir <build>/tests/e2
The test writes fake campaigns to --workdir: a TOML file and the files of results schema 2 and
the replica files of three configurations of SA (R 4, 2 repetitions): A (S 1000, t_run 10 and
20 ms, best cuts per replica [200, 199, 198, 197] and [196, 199, 200, 194]), B (S 2000, 40 and
60 ms, [200, 200, 199, 199] and [200, 199, 199, 198]) and C (S 4000, 100 and 120 ms,
[200, 200, 200, 199] and [200, 200, 200, 200]). Checks (values of cldArchitect, verifica/f07/
c3_ref.py; floats with a relative tolerance of 1e-12):
1. target 200: successes A 2, B 3, C 7 at 1000, TTS99 60.0294208473832, 122.47698473569484 and
   110.0, the interval of A [19.330677796359478, 232.85873270020147]; at 995 (199) successes 4, 7,
   8 and TTS99 24.914460711655217, 50.0, 110.0; at 990 (198) 5, 8, 8 and 17.60692612340843,
   50.0, 110.0; the choice A at the level 1000, in selected.csv and on stdout;
2. target 201: no success at 1000 (TTS99 empty), the choice at 995 (value 200): A with
   60.0294208473832;
3. every replica far from the target, even at 990: the choice "none";
4. best-found: the target is the largest best_cut of the stage and of the grid, also when the
   grid beats the stage (here 200 against 199 of the stage), and the line on stdout says so;
5. the plots exist, PDF and PNG, not empty; exit status 0 and the last line E2_DONE.

Exit status: 0 passed, 1 failed, 77 when NumPy or matplotlib cannot be imported. The last line is
"E2_VERDICT: PASS", "E2_VERDICT: FAIL" or "E2_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77
TOLERANCE = 1e-12
CONFIGS = {  # steps -> (t_run of the two repetitions, best cuts of the replicas per repetition)
    1000: ((10.0, 20.0), ([200, 199, 198, 197], [196, 199, 200, 194])),
    2000: ((40.0, 60.0), ([200, 200, 199, 199], [200, 199, 199, 198])),
    4000: ((100.0, 120.0), ([200, 200, 200, 199], [200, 200, 200, 200])),
}
EXPECTED_200 = {  # level -> successes and TTS99 of A, B, C
    1000: ([2, 3, 7], [60.0294208473832, 122.47698473569484, 110.0]),
    995: ([4, 7, 8], [24.914460711655217, 50.0, 110.0]),
    990: ([5, 8, 8], [17.60692612340843, 50.0, 110.0]),
}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"E2 {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def close(text: str, value: float) -> bool:
    try:
        return abs(float(text) - value) <= TOLERANCE * abs(value)
    except ValueError:
        return False


def write_config(sc, folder: Path, instance: str, config: str, steps: int, times, cuts,
                 target: int | None, seed_time: int) -> None:
    """The results file (schema 2) and the replica file of one configuration."""
    replicas = len(cuts[0])
    rows, replica_rows = [], []
    for rep, (t_run, values) in enumerate(zip(times, cuts)):
        stamp = f"2026-11-22T10:{seed_time:02d}:{rep:02d}Z"
        row = {c: "" for c in sc.RESULTS_COLUMNS_2}
        row.update({
            "schema_version": "2", "timestamp": stamp, "git_hash": "test", "dirty": "0",
            "host": "test", "os": "test", "compiler": "test", "algo": "sa", "variant": "c2",
            "instance": instance, "n": "24", "nnz": "96", "replicas": str(replicas),
            "steps": str(steps), "beta0": "0.5", "beta1": "2", "seed": "1", "rep": str(rep),
            "t_setup_ms": "0.1", "t_solve_ms": repr(t_run + 1.0), "t_step_us": "1",
            "best_cut": str(max(values)), "mean_cut": format(sum(values) / replicas, ".3f"),
            "first_replica": str(rep * replicas), "threads": "2", "t_prep_ms": "0.5",
            "t_run_ms": repr(t_run), "t_check_ms": "0.5", "cpu_affinity": "0-1",
            "run_tag": config, "notes": "fake"})
        if target is None:
            row["target_source"] = "none"
        else:
            row.update({"target": str(target), "target_source": "user",
                        "n_success": str(sum(v >= target for v in values)),
                        "p_s": repr(sum(v >= target for v in values) / replicas)})
        rows.append([row[c] for c in sc.RESULTS_COLUMNS_2])
        for k, value in enumerate(values):
            replica_rows.append(["1", stamp, instance, "sa", "c2", str(replicas), str(steps), "1",
                                 str(rep), str(rep * replicas + k), str(value), str(steps - 1),
                                 str(value)])
    sc.write_csv(folder / instance / "sa" / f"{config}.csv", sc.RESULTS_COLUMNS_2, rows)
    sc.write_csv(folder / "raw" / instance / "sa" / f"{config}_replicas.csv", sc.REPLICA_COLUMNS,
                 replica_rows)


def fake_campaign(sc, work: Path, name: str, target: str, file_target: int, cuts_shift: int = 0,
                  stage_best: int | None = None) -> Path:
    """A campaign with the configurations A, B, C (and a stage when stage_best is given)."""
    folder = work / name
    toml = work / f"{name}.toml"
    stage = ""
    if stage_best is not None:
        stage = ('[[stage]]\nname = "target"\ninstance = "fx"\n'
                 'runs = [ { method = "sa", a = 1.0, b = 2.0, steps = 5000, reps = 1, '
                 "warmup = 0 } ]\n")
    toml.write_text(
        f'[campaign]\nname = "{name}"\noutput = "{folder}"\nsbmc_run = "none"\n'
        'sbmc_spectrum = "none"\nreplicas = 4\nthreads = 2\nreps = 2\nwarmup = 0\nseed = 1\n'
        f'[[instance]]\nid = "fx"\nargs = ["--instance", "fx.rudy"]\ntarget = {target}\n'
        'family = "fy"\n[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\nb = [2.0]\n'
        "steps = { fx = [1000, 2000, 4000] }\n" + stage, encoding="utf-8")
    for k, (steps, (times, cuts)) in enumerate(CONFIGS.items()):
        shifted = [[c - cuts_shift for c in values] for values in cuts]
        write_config(sc, folder, "fx", f"sa_a1.0_b2.0_S{steps}", steps, times, shifted,
                     file_target, k)
    if stage_best is not None:
        write_config(sc, folder, "fx", "target_sa_a1.0_b2.0_S5000", 5000, (300.0,),
                     ([stage_best, stage_best - 3, stage_best - 5, stage_best - 1],), None, 30)
    return toml


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of python/sbmc_e2.py.")
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
        print("E2_VERDICT: SKIP")
        return EXIT_SKIP
    pyroot = Path(args.pyroot).resolve()
    sys.path.insert(0, str(pyroot))
    import sbmc_campaign as sc

    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report = Report()

    def analyze(toml: Path) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(pyroot / "sbmc_e2.py"), "--root",
                               str(Path(args.root).resolve()), "analyze", str(toml)],
                              capture_output=True, text=True, check=False, timeout=600)

    # 1. Target 200.
    toml = fake_campaign(sc, work, "t200", "200", 200)
    result = analyze(toml)
    report.check("target 200: exit 0, last line E2_DONE", result.returncode == 0
                 and result.stdout.strip().endswith("E2_DONE"),
                 f"exit {result.returncode}: {result.stderr.strip()[:300]}")
    if result.returncode == 0:
        summary = read(work / "t200" / "e2_summary.csv")
        for level, (successes, tts) in EXPECTED_200.items():
            rows = [r for r in summary if r["level_permille"] == str(level)]
            got_s = [int(r["successes"]) for r in rows]
            got_t = [r["tts99_ms"] for r in rows]
            report.check(f"target 200, level {level}: successes and TTS99 of A, B, C",
                         got_s == successes and all(close(g, e) for g, e in zip(got_t, tts))
                         and len(rows) == 3, f"successes {got_s}, TTS99 {got_t}")
        a = next(r for r in summary if r["level_permille"] == "1000"
                 and r["config_id"] == "sa_a1.0_b2.0_S1000")
        report.check("target 200: interval of TTS99 of A at 1000",
                     close(a["tts99_lo_ms"], 19.330677796359478)
                     and close(a["tts99_hi_ms"], 232.85873270020147),
                     f"[{a['tts99_lo_ms']}, {a['tts99_hi_ms']}]")
        selected = read(work / "t200" / "selected.csv")
        report.check("target 200: the choice A at the level 1000",
                     len(selected) == 1 and selected[0]["config_id"] == "sa_a1.0_b2.0_S1000"
                     and selected[0]["level_permille"] == "1000"
                     and selected[0]["evaluation_instance"] == "fy"
                     and "E2 fx sa level=1000 config=sa_a1.0_b2.0_S1000 tts99_ms=60.0294208473832"
                     in result.stdout, f"{selected}")
        plots = work / "t200" / "plots"
        names = [plots / f"{stem}.{ext}" for stem in ("sens_fx_sa", "tts_e2")
                 for ext in ("pdf", "png")]
        report.check("plots: sens_fx_sa and tts_e2, PDF and PNG, not empty",
                     all(p.exists() and p.stat().st_size > 1000 for p in names),
                     ", ".join(f"{p.name} {p.stat().st_size if p.exists() else 'missing'}"
                               for p in names))

    # 2. Target 201: nothing at 1000, the choice at 995.
    result = analyze(fake_campaign(sc, work, "t201", "201", 200))
    ok = result.returncode == 0
    if ok:
        summary = read(work / "t201" / "e2_summary.csv")
        at_1000 = [r["tts99_ms"] for r in summary if r["level_permille"] == "1000"]
        selected = read(work / "t201" / "selected.csv")
        ok = (at_1000 == ["", "", ""] and selected[0]["level_permille"] == "995"
              and selected[0]["config_id"] == "sa_a1.0_b2.0_S1000"
              and close(selected[0]["tts99_ms"], 60.0294208473832))
    report.check("target 201: no success at 1000, the choice A at 995", ok,
                 f"exit {result.returncode}: {result.stderr.strip()[:300]}")

    # 3. No success at all: "none".
    result = analyze(fake_campaign(sc, work, "tnone", "200", 200, cuts_shift=100))
    ok = result.returncode == 0
    if ok:
        selected = read(work / "tnone" / "selected.csv")
        ok = (selected[0]["config_id"] == "none" and selected[0]["level_permille"] == ""
              and "E2 fx sa level=none config=none tts99_ms=none" in result.stdout)
    report.check("no success even at 990: the choice none", ok,
                 f"exit {result.returncode}: {result.stderr.strip()[:300]}")

    # 4. best-found: the grid (200) beats the stage (199); then the stage (205) beats the grid.
    for name, stage_best, expected in (("found_grid", 199, 200), ("found_stage", 205, 205)):
        result = analyze(fake_campaign(sc, work, name, '"best-found"', 200,
                                       stage_best=stage_best))
        ok = result.returncode == 0
        if ok:
            selected = read(work / name / "selected.csv")
            raised = f"best-found target raised from {stage_best} (stages) to 200 (grid)"
            ok = (selected[0]["target"] == str(expected)
                  and selected[0]["target_source"] == "best_found"
                  and ((raised in result.stdout) == (expected == 200)))
        report.check(f"best-found with the stage at {stage_best}: target {expected}", ok,
                     f"exit {result.returncode}: {result.stdout.strip()[:200]} "
                     f"{result.stderr.strip()[:200]}")

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"E2_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
