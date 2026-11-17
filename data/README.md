# Data

Instances and target values of the experiments. Everything here is read-only: no script or test
of the project rewrites these files.

## Layout

| Path             | Content | In git |
|------------------|---------|--------|
| `gset/`          | 13 G-set instances used by the experiments, byte for byte as published at Stanford, and `SHA256SUMS` with the SHA-256 of all 71 G-set files | yes |
| `best_known.csv` | Best-known cut values, with sources and the date of the check | yes |
| `raw/`           | The other 58 G-set files (`raw/gset/`) and K2000 (`raw/k2000/WK2000_1.rud`), downloaded by `scripts/fetch_data.sh` | no (ignored by git, excluded by `scripts/sync_from_zip.sh`) |

`scripts/fetch_data.sh` checks the committed files against `gset/SHA256SUMS`, downloads the
missing ones into `raw/` and checks every download against its SHA-256 before keeping it:

```bash
scripts/fetch_data.sh            # both parts
scripts/fetch_data.sh --gset     # the 13 committed files, then the other 58 into raw/gset/
scripts/fetch_data.sh --k2000    # raw/k2000/WK2000_1.rud
```

## G-set

Origin: the instances of C. Helmberg and F. Rendl, "A spectral bundle method for semidefinite
programming", SIAM Journal on Optimization 10 (2000), generated with the graph generator rudy by
G. Rinaldi. They became the standard benchmark of Max-Cut.

Download: the 71 files G1 ... G81 and `rudy.c` from <https://web.stanford.edu/~yyye/yyye/Gset/>,
downloaded on 2026-11-15T21:19:01Z. Their SHA-256 are in `gset/SHA256SUMS` (the 71 G files,
sorted by instance number, in the format of `sha256sum`).

Cross-checks:
- An independent copy, <https://github.com/heri-espino/maxkcut> at commit
  `c7c6c9d6421b13ed38fec450d026c41fadb34a9a` (`data/G*.txt`): 68 files are identical byte for
  byte; G56, G60 and G61 differ only by line endings (CRLF in the Stanford originals); the content
  is identical for all 71.
- The structural relations listed in the SuiteSparse notes of the G-set (G1 = spones(G6) and
  similar) hold on 22 pairs.
- The C++ reader (`sbmc::read_rudy`) and the reader of the Python reference P0
  (`sbmc_ref.read_rudy`) read all 71 files with the same n, m and W.

Real format of the files (it differs from the minimal description "N M, then M lines i j w"):
- the header is "N M " with a trailing space, in every file;
- every edge line has i < j, but the edges are not sorted;
- no repeated edges, no self-loops, indices in [1, N];
- weights are all 1, or all +1 and -1;
- G56, G60 and G61 have CRLF line endings, the other files LF;
- every file ends with '\n'.

The 13 instances in `gset/` (roles from section 14.1 of the reference document: evaluation, and
tuning on different instances of the same families):

| Instance | n      | m      | Weights | W      | Role       |
|----------|--------|--------|---------|--------|------------|
| G1       | 800    | 19176  | 1       | 19176  | evaluation |
| G2       | 800    | 19176  | 1       | 19176  | tuning     |
| G11      | 800    | 1600   | ±1      | 34     | evaluation |
| G12      | 800    | 1600   | ±1      | -4     | tuning     |
| G14      | 800    | 4694   | 1       | 4694   | evaluation |
| G15      | 800    | 4661   | 1       | 4661   | tuning     |
| G22      | 2000   | 19990  | 1       | 19990  | evaluation |
| G43      | 1000   | 9990   | 1       | 9990   | evaluation |
| G48      | 3000   | 6000   | 1       | 6000   | evaluation |
| G55      | 5000   | 12498  | 1       | 12498  | evaluation |
| G63      | 7000   | 41459  | 1       | 41459  | evaluation |
| G70      | 10000  | 9999   | 1       | 9999   | evaluation |
| G81      | 20000  | 40000  | ±1      | 34     | evaluation |

Together they take 2,126,360 bytes; every file is far below the 5 MB limit of the repository.

Line endings: `.gitattributes` marks `data/gset/G*` as `-text`, so git never converts the line
endings of these files, whatever the global rule (`* text=auto eol=lf`) says. The 13 files have
LF only and would survive the global rule too; the explicit rule documents that the files are
kept exactly as published and also covers G56, G60 and G61 (CRLF) should they enter `gset/`.

License: the G-set files are third-party data, distributed by their authors. They are not covered
by the MIT license of this project.

## K2000

Complete graph with 2000 nodes and weights +1 and -1, the classic dense instance of Max-Cut. Origin:
T. Inagaki et al., "A coherent Ising machine for 2000-node optimization problems", Science 354
(2016), as stated by the README of the repository below.

| Field      | Value |
|------------|-------|
| Repository | <https://github.com/haribara/SA-complete-graph> |
| Tag        | `WK2000`, commit `785d664eddb65a5c90a03f5c2253a7cb2cf6ef8b` (2016-10-06) |
| License    | MIT (stated in the README of the repository) |
| File       | `WK2000_1.rud`, 22,775,141 bytes |
| SHA-256    | `9ed615e5e18726914f12740b7f9bedb6b69676477ed5958fac42c8ab0c252ba7` |
| URL        | <https://raw.githubusercontent.com/haribara/SA-complete-graph/WK2000/WK2000_1.rud> |
| Content    | n = 2000, m = 1,999,000 edges: 998,980 of weight +1 and 1,000,020 of weight -1; W = -1040 |

The file is above the 5 MB limit, so it is not in git: `scripts/fetch_data.sh --k2000` puts it in
`raw/k2000/WK2000_1.rud`. The test `test_k2000` exits with 77 (skipped) until the file is there.

## best_known.csv

CSV following RFC 4180: comma separator, '\n' line endings, fields with commas or quotes enclosed
in double quotes. Columns:

| Column       | Meaning |
|--------------|---------|
| `instance`   | Name of the instance, as written in the `instance` column of the results (`sbmc_run --name`) |
| `n`, `m`     | Number of nodes and of edges |
| `best_known` | Best-known cut value |
| `status`     | `best_known`: best published value, not proven optimal; `optimal`: optimal by an argument that can be checked here; `reported_optimal`: declared optimal by a source |
| `source_1`, `source_2` | Sources of the value (`source_2` empty when only one source was found) |
| `checked`    | Date on which the sources were checked |
| `notes`      | Remarks on the value or on the sources |

Policy:
- two independent sources where possible, and the date of the check for every row;
- best-known values change: the values of G63 and G81 moved in 2025-2026 (G63: 27047 in 2025,
  then 27052 in 2026; G81: 14060, reported optimal in 2026), so a value holds at the date in
  `checked`, and an update brings new sources and a new date;
- `optimal` is used only when the optimality can be checked here: G48 has 6000 edges, all of
  weight +1, and its best-known cut is 6000, so every edge is cut.

Rows with a single source: G55 (one table found; an earlier value, 10294, was reported by Benlic
and Hao in 2013), G63 (the new value comes with a public certificate not yet checked here) and G70
(value attributed by Zick to Chen et al. 2023).

The K2000 row is named `K2000`, while the file is `WK2000_1.rud`: `sbmc_run` takes the instance
name from the file name (`WK2000_1`), so a run on K2000 needs `--name K2000` to find its target
with `--target best-known`.
