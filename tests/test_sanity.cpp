// Sanity checks of the CPU toolchain: OpenMP and the build configuration. No CUDA here.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <array>
#include <cstddef>
#include <string>

#ifndef SBMC_STRICT_FP_BUILD
#error "SBMC_STRICT_FP_BUILD must be defined by CMake (1 when SBMC_STRICT_FP is ON, 0 otherwise)"
#endif

namespace {

constexpr bool kStrictFpBuild = (SBMC_STRICT_FP_BUILD != 0);

}  // namespace

TEST_CASE("OpenMP: at least one thread is available") {
  CHECK(omp_get_max_threads() >= 1);
}

TEST_CASE("OpenMP: a parallel region with 2 requested threads runs on 2 threads") {
  omp_set_dynamic(0);  // the runtime must not shrink the team below the requested size
  int team_size = 0;
  std::array<int, 2> ran{0, 0};
#pragma omp parallel num_threads(2)
  {
#pragma omp single
    team_size = omp_get_num_threads();

    const int id = omp_get_thread_num();
    if (id >= 0 && id < static_cast<int>(ran.size())) {
      ran[static_cast<std::size_t>(id)] = 1;  // each thread writes only its own slot
    }
  }
  CHECK(team_size == 2);
  CHECK(ran[0] == 1);
  CHECK(ran[1] == 1);
}

TEST_CASE("Build: floating-point contraction mode") {
  // std::string, not const char*: doctest 2.5.3 prints a char pointer in MESSAGE as "1".
  const std::string state = kStrictFpBuild ? "ON" : "OFF";
  const std::string flag = kStrictFpBuild ? "-ffp-contract=off" : "-ffp-contract=fast";
  MESSAGE("SBMC_STRICT_FP is " << state << " in this build (" << flag << ")");
}
