# Third-party code

Dependencies copied into the repository without modifications.

## doctest

| Field    | Value |
|----------|-------|
| Name     | doctest, single-header C++ testing framework |
| Version  | 2.5.3 (latest stable release, 2026-07-06) |
| Release  | https://github.com/doctest/doctest/releases/tag/v2.5.3 |
| Commit   | 2d0a9359a60c51affe2a9bebb1be1dca47868151 (tag `v2.5.3`) |
| License  | MIT (`doctest/LICENSE.txt`) |

| File                  | Upstream path        | SHA-256 |
|-----------------------|----------------------|---------|
| `doctest/doctest.h`   | `doctest/doctest.h`  | `cfd518a3ef90f67e1f3ba514df23fb3627437de1a2feeba78cf5062a40021421` |
| `doctest/LICENSE.txt` | `LICENSE.txt`        | `0fe0b331fa1513dcce8604ff1fa925f32d1cea17d8aeb1c2471fad40d291adc5` |

Usage: the CMake interface target `doctest` (in `tests/CMakeLists.txt`) adds `third_party/` as
a system include directory; tests include `<doctest/doctest.h>`.

## CLI11

| Field    | Value |
|----------|-------|
| Name     | CLI11, command line parser for C++11 and later (single-header edition) |
| Version  | 2.7.2 (latest release, 2026-08-02: date of the annotated tag) |
| Release  | https://github.com/CLIUtils/CLI11/releases/tag/v2.7.2 |
| Commit   | cbd58a3696887b34c70949aef21a71735a0c2ad5 (tag `v2.7.2`) |
| License  | BSD-3-Clause (`CLI11/LICENSE`) |

| File              | Upstream path                                                   | SHA-256 |
|-------------------|-----------------------------------------------------------------|---------|
| `CLI11/CLI11.hpp` | release asset `CLI11.hpp` of v2.7.2 (single header, 493,427 bytes) | `ffa9a30da295c5858fb5f91f9f45771bab09471d7010a34c7c68c857a330dd76` |
| `CLI11/LICENSE`   | `LICENSE` at tag `v2.7.2`                                       | `cfc76368aef8f51868fa1cda6bcfa26e583406a4cf8e48e725e7624df24d8855` |

Usage: the CMake interface target `sbmc_cli11` (alias `sbmc::cli11`, in `apps/CMakeLists.txt`)
adds `third_party/` as a system include directory; the applications in `apps/` include
`<CLI11/CLI11.hpp>`. No public header of the project includes it.
