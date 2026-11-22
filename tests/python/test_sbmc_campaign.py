#!/usr/bin/env python3
"""python/sbmc_campaign.py, the runner of the campaigns (F07-C3, sections 5.2 and 5.7).

Run by CTest from the repository root as:
  <python> -I -B test_sbmc_campaign.py --pyroot <root>/python --root <root>
           --sbmc-run <sbmc_run> --sbmc-spectrum <sbmc_spectrum> --workdir <build>/tests/campaign
Checks (reference values of cldArchitect, verifica/f07/c3_ref.py; floats with a relative
tolerance of 1e-12):
1. scales: h and w of four_nodes.rudy, torus_4x6_pm1.rudy, sparse_24_unit.rudy,
   weighted_isolated_13.rudy, data/gset/G2, G12, G15 and of --dense 200; beta0 of G12 with a 0.5
   (0.25) and of G2 with a 1 (0.1444278629379538) in the commands of --dry-run;
2. validation: 12 wrong configurations (unknown key, missing key, wrong type, a <= 0, empty
   steps, repeated configuration id, unknown method, variant of another method, unknown target,
   a stage called grid, best-found without a stage, repeated instance id): exit status 2, a
   message that starts with "sbmc_campaign: " and names the file, no file written;
3. --dry-run on torus_4x6_pm1.rudy with sa 2 x 2 x 2 and dsb 2 x 2 x 2: 16 lines "run: ..." in
   the order of section 5.2 with the expected parameters, nothing written;
4. real runs with sbmc_run (R 8, 2 threads, 2 repetitions, no warm-up, S and M 10 and 20) on
   sparse_24_unit.rudy: 4 complete configurations, instances.csv with the expected row; a second
   run: every configuration "skip"; after truncating the last row of a results file and removing
   a replica file, a third run repeats only those two configurations; MANIFEST.sha256 equal to
   the SHA-256 of the files of raw/ (and its copy raw_manifest.sha256); status: 4 done;
5. failure: an instance id that is not in data/best_known.csv with the target best-known: exit
   status 1, the second configuration not started, one line "fail" in the log.
The files go to --workdir only (a folder of the build tree, emptied first).

Exit status: 0 passed, 1 failed, 77 when the executables are missing. The last line is
"CAMPAIGN_VERDICT: PASS", "CAMPAIGN_VERDICT: FAIL" or "CAMPAIGN_VERDICT: SKIP".
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_SKIP = 0, 1, 77
TOLERANCE = 1e-12
SCALES = {  # file (relative to the root) -> (h, w)
    "tests/data/four_nodes.rudy": (1.5811388300841898, 1),
    "tests/data/torus_4x6_pm1.rudy": (2.0, 1),
    "tests/data/sparse_24_unit.rudy": (2.0, 1),
    "tests/data/weighted_isolated_13.rudy": (1883467753.9321375, 1),
    "data/gset/G2": (6.923871749245504, 1),
    "data/gset/G12": (2.0, 1),
    "data/gset/G15": (3.413575837739657, 1),
}
DENSE_200 = (14.106735979665885, 1)
# sparse_24_unit.rudy with 100 iterations (F07-C2, section 5.7).
SPARSE_24 = {"lambda_min": -4.359170003893492, "lambda_max": 3.121268901575628,
             "dt_auto": 1.1418973, "c0_lambda": 0.32038254}


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"CAMPAIGN {name}: {'PASS' if ok else 'FAIL'}" + (f" ({detail})" if detail else ""))
        if not ok:
            self.failures += 1


def close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(abs(a), abs(b))


def campaign_text(name: str, output: Path, tools: dict[str, str], instances: str, methods: str,
                  replicas: int = 8, extra: str = "") -> str:
    return (f'[campaign]\nname = "{name}"\noutput = "{output}"\nsbmc_run = "{tools["run"]}"\n'
            f'sbmc_spectrum = "{tools["spectrum"]}"\nreplicas = {replicas}\nthreads = 2\n'
            f"reps = 2\nwarmup = 0\nseed = 1\n{extra}\n{instances}\n{methods}\n")


def instance_text(instance_id: str, path: Path, target: str = "40", family: str = "") -> str:
    text = f'[[instance]]\nid = "{instance_id}"\nargs = ["--instance", "{path}"]\n'
    text += f"target = {target}\n"
    if family:
        text += f'family = "{family}"\n'
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Checks of python/sbmc_campaign.py.")
    parser.add_argument("--pyroot", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--sbmc-run", required=True)
    parser.add_argument("--sbmc-spectrum", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args()
    print(f"python: {platform.python_version()} ({sys.executable})")
    root = Path(args.root).resolve()
    tools = {"run": str(Path(args.sbmc_run).resolve()),
             "spectrum": str(Path(args.sbmc_spectrum).resolve())}
    tool = Path(args.pyroot).resolve() / "sbmc_campaign.py"
    missing = [p for p in (tools["run"], tools["spectrum"], str(tool)) if not Path(p).exists()]
    if missing:
        print(f"missing: {', '.join(missing)}")
        print("CAMPAIGN_VERDICT: SKIP")
        return EXIT_SKIP
    sys.path.insert(0, str(Path(args.pyroot).resolve()))
    import sbmc_campaign as sc

    work = Path(args.workdir).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    report = Report()

    def run(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-I", "-B", str(tool), "--root", str(root),
                               *arguments], capture_output=True, text=True, check=False,
                              timeout=600)

    # 1. Scales, then beta0 in the commands of --dry-run.
    for name, (h, w) in SCALES.items():
        _, _, got_h, got_w = sc.rudy_scales(root / name)
        report.check(f"scales of {name}", close(got_h, h) and got_w == w,
                     f"h {got_h!r} (expected {h!r}), w {got_w} (expected {w})")
    dense = sc.Instance("dense_n200_s1", ("--dense", "200", "--instance-seed", "1"), "none", None,
                        None)
    got = sc.instance_scales(dense, root)
    report.check("scales of --dense 200", close(got[0], DENSE_200[0]) and got[1] == DENSE_200[1],
                 f"h {got[0]!r}, w {got[1]}")
    toml = work / "beta.toml"
    toml.write_text(campaign_text(
        "beta", work / "beta_out", tools,
        instance_text("G12", root / "data/gset/G12", '"best-known"')
        + instance_text("G2", root / "data/gset/G2", '"best-known"'),
        '[[method]]\nid = "sa"\nvariant = "c2"\na = [0.5, 1.0]\nb = [2.0]\n'
        "steps = { G12 = [10], G2 = [10] }\n"), encoding="utf-8")
    result = run("run", str(toml), "--notes", "beta", "--dry-run")
    lines = [shlex.split(line[len("run: "):]) for line in result.stdout.splitlines()]
    beta0 = {(words[words.index("--name") + 1], words[words.index("--tag") + 1]):
             float(words[words.index("--beta0") + 1]) for words in lines}
    ok = (result.returncode == 0 and len(lines) == 4
          and close(beta0.get(("G12", "sa_a0.5_b2.0_S10"), 0), 0.25)
          and close(beta0.get(("G2", "sa_a1.0_b2.0_S10"), 0), 0.1444278629379538))
    report.check("beta0 of G12 with a 0.5 and of G2 with a 1 in --dry-run", ok,
                 f"exit {result.returncode}, {beta0}")

    # 2. Validation: exit status 2, no file written.
    base_instance = instance_text("fx", root / "tests/data/torus_4x6_pm1.rudy")
    sa = '[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\nb = [2.0]\nsteps = { fx = [10] }\n'
    cases = {
        "unknown key": (base_instance, sa, "", "x = 1"),
        "missing key": (base_instance, sa, "", "DROP seed"),
        "wrong type": (base_instance, sa.replace("a = [1.0]", 'a = ["1"]'), "", ""),
        "a <= 0": (base_instance, sa.replace("a = [1.0]", "a = [0.0]"), "", ""),
        "empty steps": (base_instance, sa.replace("fx = [10]", "fx = []"), "", ""),
        "repeated configuration id": (base_instance, sa.replace("a = [1.0]", "a = [1.0, 1.0]"),
                                      "", ""),
        "unknown method": (base_instance, sa.replace('id = "sa"', 'id = "bsb"'), "", ""),
        "variant of another method": (base_instance, sa.replace('"c2"', '"c1"'), "", ""),
        "unknown target": (base_instance.replace("target = 40", 'target = "best"'), sa, "", ""),
        "stage called grid": (base_instance, sa, '[[stage]]\nname = "grid"\ninstance = "fx"\n'
                              'runs = [ { method = "sa", a = 1.0, b = 2.0, steps = 5 } ]\n', ""),
        "best-found without a stage": (base_instance.replace("target = 40",
                                                             'target = "best-found"'), sa, "", ""),
        "repeated instance id": (base_instance + base_instance, sa, "", ""),
    }
    for k, (name, (instances, methods, stages, campaign_change)) in enumerate(cases.items()):
        output = work / f"bad_{k}"
        text = campaign_text("bad", output, tools, instances, methods + stages)
        if campaign_change == "x = 1":
            text = text.replace("seed = 1\n", "seed = 1\nx = 1\n", 1)
        elif campaign_change == "DROP seed":
            text = text.replace("seed = 1\n", "", 1)
        path = work / f"bad_{k}.toml"
        path.write_text(text, encoding="utf-8")
        result = run("run", str(path), "--notes", "bad")
        first = result.stderr.strip().splitlines()[0:1]
        ok = (result.returncode == 2 and bool(first) and first[0].startswith("sbmc_campaign: ")
              and str(path) in first[0] and not output.exists() and result.stdout == "")
        report.check(f"validation, {name}: exit 2, no file", ok,
                     f"exit {result.returncode}: {first}")

    # 3. --dry-run: 16 lines in the order of section 5.2.
    output = work / "dry_out"
    toml = work / "dry.toml"
    toml.write_text(campaign_text(
        "dry", output, tools, base_instance,
        '[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0, 2.0]\nb = [2.0, 3.0]\n'
        "steps = { fx = [10, 20] }\n"
        '[[method]]\nid = "dsb"\nvariant = "c1"\nf = [0.5, 1.0]\nc0 = ["sigma", "lambda"]\n'
        "steps = { fx = [10, 20] }\n"), encoding="utf-8")
    result = run("run", str(toml), "--notes", "dry run", "--dry-run")
    lines = result.stdout.splitlines()
    expected = []
    for a in (1.0, 2.0):
        for b in (2.0, 3.0):
            for s in (10, 20):
                expected.append((f"sa_a{a!r}_b{b!r}_S{s}",
                                 ["--algo", "sa", "--variant", "c2", "--sweeps", str(s),
                                  "--beta0", repr(a / 2.0), "--beta1", repr(b)]))
    for f in (0.5, 1.0):
        for c0, value in (("sigma", "auto"), ("lambda", "lambda")):
            for m in (10, 20):
                expected.append((f"dsb_f{f!r}_c{c0}_M{m}",
                                 ["--algo", "dsb", "--variant", "c1", "--steps", str(m), "--dt",
                                  "auto", "--dt-factor", repr(f), "--c0", value]))
    wrong = []
    for k, (config, parameters) in enumerate(expected):
        if k >= len(lines) or not lines[k].startswith("run: "):
            wrong.append(f"line {k + 1} missing")
            continue
        words = shlex.split(lines[k][len("run: "):])
        tail = ["--replicas", "8", "--threads", "2", "--seed", "1", "--reps", "2", "--warmup",
                "0", "--target", "40", "--notes", "dry run", "--tag", config]
        method = config.split("_")[0]
        want = ([tools["run"], "--instance", str(root / "tests/data/torus_4x6_pm1.rudy"), "--name",
                 "fx"] + parameters + tail
                + ["--out", f"{output}/fx/{method}/{config}.csv", "--replica-out",
                   f"{output}/raw/fx/{method}/{config}_replicas.csv"])
        if words != want:
            wrong.append(f"line {k + 1}: {lines[k]}")
    report.check("--dry-run: 16 lines in the order of section 5.2, with --beta0 1.0 for a 2",
                 result.returncode == 0 and len(lines) == 16 and not wrong
                 and "--beta0 1.0 " in lines[4] and not output.exists(),
                 f"exit {result.returncode}, {len(lines)} lines; " + "; ".join(wrong[:3]))

    # 4. Real runs on sparse_24_unit.rudy.
    output = work / "real_out"
    toml = work / "real.toml"
    sparse = root / "tests/data/sparse_24_unit.rudy"
    toml.write_text(campaign_text(
        "real", output, tools, instance_text("sparse_24_unit", sparse, "40", "fam"),
        '[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\nb = [2.0]\n'
        "steps = { sparse_24_unit = [10, 20] }\n"
        '[[method]]\nid = "dsb"\nvariant = "c1"\nf = [1.0]\nc0 = ["sigma"]\n'
        "steps = { sparse_24_unit = [10, 20] }\n"), encoding="utf-8")
    log = output / "campaign.log"

    def events(text: str, event: str) -> list[str]:
        return [line for line in text.splitlines() if line.split(" ")[1:2] == [event]]

    result = run("run", str(toml), "--notes", "real runs")
    done = events(result.stdout, "done")
    report.check("first run: exit 0, 4 configurations done", result.returncode == 0
                 and len(done) == 4 and not events(result.stdout, "fail"),
                 f"exit {result.returncode}, {len(done)} done; {result.stderr.strip()[:200]}")
    spectrum = subprocess.run([tools["spectrum"], "--instance", str(sparse)], capture_output=True,
                              text=True, check=False).stdout.split()
    fields = dict(w.split("=", 1) for w in spectrum[1:])
    try:
        with open(output / "instances.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        rows = []
    row = rows[0] if len(rows) == 1 else {}
    ok = (bool(row) and row["instance"] == "sparse_24_unit" and row["n"] == "24"
          and row["m"] == "48" and row["h"] == "2.0" and row["w"] == "1"
          and all(abs(float(row[k]) - v) <= 1e-9 * abs(v) for k, v in SPARSE_24.items())
          and row["c0_sigma"] == fields.get("c0_sigma") and row["target"] == "40"
          and row["target_source"] == "given" and row["family"] == "fam")
    report.check("instances.csv: the row of sparse_24_unit", ok, f"{row}")
    result = run("run", str(toml), "--notes", "real runs")
    report.check("second run: every configuration skip", result.returncode == 0
                 and len(events(result.stdout, "skip")) == 4
                 and not events(result.stdout, "run"), f"exit {result.returncode}")
    results_file = output / "sparse_24_unit/sa/sa_a1.0_b2.0_S20.csv"
    replica_file = output / "raw/sparse_24_unit/dsb/dsb_f1.0_csigma_M10_replicas.csv"
    lines = results_file.read_text(encoding="utf-8").splitlines(keepends=True)
    results_file.write_text("".join(lines[:-1]), encoding="utf-8")
    replica_file.unlink()
    result = run("run", str(toml), "--notes", "real runs")
    repeated = sorted(line.split(" ")[3] for line in events(result.stdout, "run"))
    report.check("third run: only the two incomplete configurations repeated",
                 result.returncode == 0 and repeated == ["config=dsb_f1.0_csigma_M10",
                                                         "config=sa_a1.0_b2.0_S20"]
                 and len(events(result.stdout, "skip")) == 2
                 and all("incomplete files removed" in line
                         for line in events(result.stdout, "run")), f"{repeated}")
    raw = output / "raw"
    expected_manifest = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(raw).as_posix()}\n"
        for p in sorted(raw.rglob("*_replicas.csv"), key=lambda p: p.relative_to(raw).as_posix()))
    manifest = (raw / "MANIFEST.sha256").read_text(encoding="utf-8")
    copy = (output / "raw_manifest.sha256").read_text(encoding="utf-8")
    report.check("MANIFEST.sha256: the SHA-256 of the 4 replica files, and its copy",
                 manifest == expected_manifest and copy == manifest
                 and manifest.count("\n") == 4, manifest.replace("\n", " | "))
    result = run("status", str(toml))
    report.check("status: 4 done", result.returncode == 0 and
                 "STATUS_TOTAL configurations=4 done=4 partial=0 todo=0" in result.stdout,
                 result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "")
    log_text = log.read_text(encoding="utf-8") if log.exists() else ""
    report.check("campaign.log: the events of the three runs",
                 len(events(log_text, "done")) == 6 and len(events(log_text, "start")) == 3
                 and len(events(log_text, "stop")) == 3, f"{len(log_text.splitlines())} lines")

    # 5. Failure: sbmc_run exits with 1 on the first configuration.
    output = work / "fail_out"
    toml = work / "fail.toml"
    toml.write_text(campaign_text(
        "fail", output, tools, instance_text("not_in_table", sparse, '"best-known"'),
        '[[method]]\nid = "sa"\nvariant = "c2"\na = [1.0]\nb = [2.0]\n'
        "steps = { not_in_table = [10, 20] }\n"), encoding="utf-8")
    result = run("run", str(toml), "--notes", "failure")
    log_text = (output / "campaign.log").read_text(encoding="utf-8") if output.exists() else ""
    ok = (result.returncode == 1 and len(events(log_text, "fail")) == 1
          and len(events(log_text, "run")) == 1
          and "config=sa_a1.0_b2.0_S10" in events(log_text, "fail")[0]
          and not (output / "not_in_table/sa/sa_a1.0_b2.0_S20.csv").exists()
          and "best_known" in result.stderr)
    report.check("failure: exit 1, the next configuration not started, one fail line", ok,
                 f"exit {result.returncode}, {result.stderr.strip()[:200]}")

    verdict = "PASS" if report.failures == 0 else "FAIL"
    print(f"CAMPAIGN_VERDICT: {verdict}")
    return EXIT_PASS if report.failures == 0 else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
