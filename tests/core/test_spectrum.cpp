// The estimate of the spectrum of J and the automatic dt and c0 (sbmc/spectrum.hpp, phase 07): the
// extremes with the 100 default iterations and with a few iterations, against the reference
// values of F07-C2, section 5.7 (numpy.linalg.eigvalsh and an independent implementation of the
// algorithm, by cldArchitect); monotone extremes; the extremes of G14 with iterations = n;
// independence from the number of threads, bit for bit; the edge cases and the preconditions.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/philox.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/spectrum.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;
using sbmc::SpectrumEstimate;

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Instance test_rudy(const std::string& name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

Instance gset(const std::string& name) {
  return sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/" + name);
}

// |estimate - reference| <= 1e-9 max(|lambda_min|, |lambda_max|) of the reference.
void check_lambdas(const SpectrumEstimate& s, double lambda_min, double lambda_max) {
  const double tolerance = 1e-9 * std::max(std::abs(lambda_min), std::abs(lambda_max));
  INFO("lambda_min " << s.lambda_min << " (reference " << lambda_min << "), lambda_max "
                     << s.lambda_max << " (reference " << lambda_max << ")");
  CHECK(std::abs(s.lambda_min - lambda_min) <= tolerance);
  CHECK(std::abs(s.lambda_max - lambda_max) <= tolerance);
}

// Relative 1e-6.
void check_relative(float value, double reference) {
  INFO("value " << value << ", reference " << reference);
  CHECK(std::abs(static_cast<double>(value) - reference) <= 1e-6 * std::abs(reference));
}

struct Row {
  const char* name;
  Instance instance;
  double lambda_min;
  double lambda_max;
  int iterations;
  int restarts;
  double dt;
  double c0;
};

bool same_bits(const SpectrumEstimate& a, const SpectrumEstimate& b) {
  return std::bit_cast<std::uint64_t>(a.lambda_min) == std::bit_cast<std::uint64_t>(b.lambda_min) &&
         std::bit_cast<std::uint64_t>(a.lambda_max) == std::bit_cast<std::uint64_t>(b.lambda_max) &&
         a.iterations == b.iterations && a.restarts == b.restarts;
}

}  // namespace

TEST_CASE("SPECTRUM: extremes with the default 100 iterations") {
  const std::vector<Row> rows = {
      {"four_nodes.rudy", test_rudy("four_nodes.rudy"), -2.56155281280883, 1.5615528128088298, 4, 0,
       1.0879054, 0.6403882},
      {"torus_4x6_pm1.rudy", test_rudy("torus_4x6_pm1.rudy"), -3.3364851961066266,
       3.336485196106626, 24, 0, 1.25, 0.2997166},
      {"sparse_24_unit.rudy", test_rudy("sparse_24_unit.rudy"), -4.359170003893492,
       3.121268901575628, 24, 0, 1.1418973, 0.32038254},
      {"weighted_isolated_13.rudy", test_rudy("weighted_isolated_13.rudy"), -3474701083.9172173,
       3474701978.344408, 13, 2, 1.2500001, 2.8779445e-10},
      {"G1", gset("G1"), -48.78749417418736, 13.27415171569161, 100, 0, 0.8175546, 0.07533438},
      {"G11", gset("G11"), -3.4464609249401, 3.446460924940104, 100, 0, 1.25, 0.29015273},
      {"G14", gset("G14"), -22.427689180607906, 10.202365310683353, 100, 0, 0.98847693,
       0.098016486},
      {"DenseSpinGlass{200, 7}", DenseSpinGlass{200, 7}, -26.77769686885779, 27.717738844425185,
       100, 0, 1.2607352, 0.03607798},
  };
  for (const Row& row : rows) {
    INFO(row.name);
    const SpectrumEstimate s = sbmc::estimate_spectrum(row.instance);
    check_lambdas(s, row.lambda_min, row.lambda_max);
    CHECK(s.iterations == row.iterations);
    CHECK(s.restarts == row.restarts);
    check_relative(sbmc::spectral_dt(s), row.dt);
    check_relative(sbmc::spectral_c0(s), row.c0);
    CHECK(sbmc::kSpectrumIterations == 100);
  }
}

TEST_CASE("SPECTRUM: a few iterations (start vector and algorithm before convergence)") {
  struct Few {
    const char* name;
    Instance instance;
    int iterations;
    double lambda_min;
    double lambda_max;
  };
  const std::vector<Few> rows = {
      {"G1", gset("G1"), 1, 0.0111263890796685, 0.0111263890796685},
      {"G1", gset("G1"), 10, -48.7874941741805, 12.7715488120668},
      {"G1", gset("G1"), 20, -48.7874941741874, 13.2405492870742},
      {"sparse_24_unit.rudy", test_rudy("sparse_24_unit.rudy"), 5, -4.24351492088421,
       2.65704104661084},
      {"DenseSpinGlass{200, 7}", DenseSpinGlass{200, 7}, 10, -26.5564011409704, 27.4463192377289},
  };
  for (const Few& row : rows) {
    INFO(row.name << " with " << row.iterations << " iterations");
    const SpectrumEstimate s = sbmc::estimate_spectrum(row.instance, row.iterations);
    check_lambdas(s, row.lambda_min, row.lambda_max);
    CHECK(s.iterations == row.iterations);
    CHECK(s.restarts == 0);
  }
}

TEST_CASE("SPECTRUM: properties") {
  const Instance g1 = gset("G1");
  SUBCASE("G1: the extremes widen with 32, 64 and 100 iterations") {
    const SpectrumEstimate s32 = sbmc::estimate_spectrum(g1, 32);
    const SpectrumEstimate s64 = sbmc::estimate_spectrum(g1, 64);
    const SpectrumEstimate s100 = sbmc::estimate_spectrum(g1, 100);
    const double tolerance = 1e-12 * std::max(std::abs(s100.lambda_min), s100.lambda_max);
    CHECK(s64.lambda_max >= s32.lambda_max - tolerance);
    CHECK(s100.lambda_max >= s64.lambda_max - tolerance);
    CHECK(s64.lambda_min <= s32.lambda_min + tolerance);
    CHECK(s100.lambda_min <= s64.lambda_min + tolerance);
  }
  SUBCASE("the same bits with 1 and 3 threads") {
    const int before = omp_get_max_threads();
    for (const Instance& inst : {g1, Instance{DenseSpinGlass{200, 7}}}) {
      omp_set_num_threads(1);
      const SpectrumEstimate one = sbmc::estimate_spectrum(inst);
      omp_set_num_threads(3);
      const SpectrumEstimate three = sbmc::estimate_spectrum(inst);
      CHECK(same_bits(one, three));
    }
    omp_set_num_threads(before);
    CHECK(omp_get_max_threads() == before);
  }
  // Added in F07-R2 (mutation test): with iterations = n the Krylov space is the whole space and
  // the Ritz values are the eigenvalues; without a full reorthogonalization (one pass of
  // Gram-Schmidt instead of two) the basis loses its orthogonality and on G14 the extremes fall
  // outside the spectrum (-31.1 and 33.4). The references are those of G14 with 100 iterations
  // (the eigenvalues, within 5e-14).
  SUBCASE("G14 with iterations = n: the extremes of the spectrum") {
    const SpectrumEstimate s = sbmc::estimate_spectrum(gset("G14"), 1000);
    CHECK(s.iterations == 800);
    CHECK(s.restarts == 0);
    check_lambdas(s, -22.427689180607906, 10.202365310683353);
  }
  SUBCASE("spectral_dt of G1 with the factor 0.75") {
    check_relative(sbmc::spectral_dt(sbmc::estimate_spectrum(g1), 0.75), 0.6131659);
  }
  SUBCASE("the domain of the start vectors") {
    CHECK(static_cast<std::uint32_t>(sbmc::RngDomain::spectrum_start) == 5u);
  }
}

TEST_CASE("SPECTRUM: edge cases") {
  SUBCASE("a graph without edges: J = 0") {
    const SpectrumEstimate s = sbmc::estimate_spectrum(Graph{5, {}});
    CHECK(s.lambda_min == 0.0);
    CHECK(s.lambda_max == 0.0);
    CHECK(s.iterations == 5);
    CHECK_THROWS_AS(sbmc::spectral_dt(s), InvalidArgument);
    CHECK_THROWS_AS(sbmc::spectral_c0(s), InvalidArgument);
  }
  SUBCASE("one spin") {
    for (const Instance& inst : {Instance{Graph{1, {}}}, Instance{DenseSpinGlass{1, 3}}}) {
      const SpectrumEstimate s = sbmc::estimate_spectrum(inst);
      CHECK(s.iterations == 1);
      CHECK(s.restarts == 0);
      CHECK(s.lambda_min == 0.0);
      CHECK(s.lambda_max == 0.0);
    }
  }
}

TEST_CASE("SPECTRUM: preconditions") {
  const Instance four = test_rudy("four_nodes.rudy");
  CHECK_THROWS_AS(sbmc::estimate_spectrum(four, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::estimate_spectrum(four, -1), InvalidArgument);
  CHECK_THROWS_WITH_AS(sbmc::estimate_spectrum(Graph{3, {{0, 3, 1}}}),
                       doctest::Contains("estimate_spectrum"), InvalidArgument);
  // n = 32769 is refused before the dense copy of J (about 1 GiB) is allocated.
  CHECK_THROWS_WITH_AS(sbmc::estimate_spectrum(DenseSpinGlass{32769, 1}),
                       doctest::Contains("32768"), InvalidArgument);

  const SpectrumEstimate s = sbmc::estimate_spectrum(four);
  for (const double factor : {0.0, -1.0, kInf, kNaN}) {
    INFO("factor " << factor);
    CHECK_THROWS_AS(sbmc::spectral_dt(s, factor), InvalidArgument);
  }
  for (const double lambda_max : {0.0, -1.0, kInf, kNaN}) {
    INFO("lambda_max " << lambda_max);
    SpectrumEstimate bad = s;
    bad.lambda_max = lambda_max;
    CHECK_THROWS_AS(sbmc::spectral_dt(bad), InvalidArgument);
    CHECK_THROWS_AS(sbmc::spectral_c0(bad), InvalidArgument);
  }
  for (const double lambda_min : {0.5, kInf, -kInf, kNaN}) {
    INFO("lambda_min " << lambda_min);
    SpectrumEstimate bad = s;
    bad.lambda_min = lambda_min;
    CHECK_THROWS_AS(sbmc::spectral_dt(bad), InvalidArgument);
  }
}
