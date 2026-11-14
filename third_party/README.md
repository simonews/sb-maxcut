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
