// Parts of simulated bifurcation common to every variant (sbmc/sb.hpp): pressure schedule,
// automatic c0, sign and update of one spin.
//
// The reference values were computed by cldArchitect with two independent implementations
// (NumPy in float32 and scalar Python rounding to float32 after every operation). "Strict" means
// SBMC_STRICT_FP_BUILD = 1 (no FMA contraction): there the floating-point values are compared bit
// for bit; in the production presets FMA contraction may change the last bits, and those cases
// check a tolerance instead.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sb.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_STRICT_FP_BUILD
#error "SBMC_STRICT_FP_BUILD must be defined by CMake (1 in the strict presets, 0 otherwise)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;

constexpr bool kStrict = SBMC_STRICT_FP_BUILD != 0;

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

float from_bits(std::uint32_t value) {
  return std::bit_cast<float>(value);
}

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

Graph rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

// auto_c0 against its reference bits: exact in the strict presets, relative error <= 1e-6 in the
// others.
void check_c0(float c0, std::uint32_t expected) {
  INFO("expected " << hex32(expected) << ", got " << hex32(bits(c0)) << " (strict = " << kStrict
                   << ")");
  if (kStrict) {
    CHECK(bits(c0) == expected);
  } else {
    const double reference = from_bits(expected);
    CHECK(std::abs(static_cast<double>(c0) - reference) <= 1e-6 * reference);
  }
}

}  // namespace

static_assert(sbmc::sb_sign(-0.0f) == 1, "sgn(-0) = +1");

TEST_CASE("SB: pressure_schedule gives the reference bits in every preset") {
  struct Probe {
    std::size_t m;
    std::uint32_t bits;
  };
  struct Case {
    float a0;
    int steps;
    std::vector<Probe> probes;
  };
  const std::vector<Case> cases = {
      {1.0f, 1, {{0, 0x00000000u}}},
      {1.0f, 3, {{0, 0x3F2AAAABu}, {1, 0x3EAAAAABu}, {2, 0x00000000u}}},
      {1.0f, 4, {{0, bits(0.75f)}, {1, bits(0.5f)}, {2, bits(0.25f)}, {3, bits(0.0f)}}},
      {0.7f,
       10,
       {{0, 0x3F2147AEu}, {1, 0x3F0F5C29u}, {4, 0x3EB33333u}, {8, 0x3D8F5C29u}, {9, 0x00000000u}}},
      {1.0f,
       1000,
       {{0, 0x3F7FBE77u},
        {1, 0x3F7F7CEEu},
        {499, 0x3F000000u},
        {998, 0x3A83126Fu},
        {999, 0x00000000u}}},
  };
  for (const Case& c : cases) {
    const std::vector<float> p = sbmc::pressure_schedule(c.a0, c.steps);
    INFO("a0 = " << c.a0 << ", steps = " << c.steps);
    REQUIRE(p.size() == static_cast<std::size_t>(c.steps));
    for (const Probe& probe : c.probes) {
      INFO("p[" << probe.m << "]: expected " << hex32(probe.bits) << ", got "
                << hex32(bits(p[probe.m])));
      CHECK(bits(p[probe.m]) == probe.bits);
    }
  }
}

TEST_CASE("SB: pressure_schedule rejects invalid arguments") {
  CHECK_THROWS_AS(sbmc::pressure_schedule(1.0f, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::pressure_schedule(1.0f, -1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::pressure_schedule(0.0f, 10), InvalidArgument);
  CHECK_THROWS_AS(sbmc::pressure_schedule(-1.0f, 10), InvalidArgument);
  CHECK_THROWS_AS(sbmc::pressure_schedule(std::numeric_limits<float>::quiet_NaN(), 10),
                  InvalidArgument);
  CHECK_THROWS_AS(sbmc::pressure_schedule(std::numeric_limits<float>::infinity(), 10),
                  InvalidArgument);
}

TEST_CASE("SB: auto_c0 reproduces the reference values") {
  SUBCASE("rudy files") {
    check_c0(sbmc::auto_c0(rudy("four_nodes.rudy")), 0x3F2BBAE3u);
    check_c0(sbmc::auto_c0(rudy("torus_4x6_pm1.rudy")), 0x3E7D1123u);
    check_c0(sbmc::auto_c0(rudy("torus_4x6_unit.rudy")), 0x3E89DD8Du);
  }
  SUBCASE("dense spin glasses") {
    check_c0(sbmc::auto_c0(DenseSpinGlass{20, 20}), 0x3DE54A89u);
    check_c0(sbmc::auto_c0(DenseSpinGlass{24, 24}), 0x3DD12914u);
    check_c0(sbmc::auto_c0(DenseSpinGlass{64, 1}), 0x3D800108u);
  }
  SUBCASE("graphs with one edge, also of weight 2^31 - 1") {
    check_c0(sbmc::auto_c0(Graph{3, {{0, 1, 1}}}), 0x3F1CC471u);
    check_c0(sbmc::auto_c0(Graph{3, {{0, 1, 2147483647}}}), 0x2F9CC471u);
  }
  SUBCASE("the Instance versions give the same bits") {
    const std::vector<Instance> instances = {
        rudy("four_nodes.rudy"), rudy("torus_4x6_pm1.rudy"),     rudy("torus_4x6_unit.rudy"),
        DenseSpinGlass{20, 20},  DenseSpinGlass{24, 24},         DenseSpinGlass{64, 1},
        Graph{3, {{0, 1, 1}}},   Graph{3, {{0, 1, 2147483647}}},
    };
    for (const Instance& inst : instances) {
      const float direct = std::visit([](const auto& g) { return sbmc::auto_c0(g); }, inst);
      CHECK(bits(sbmc::auto_c0(inst)) == bits(direct));
    }
  }
}

TEST_CASE("SB: auto_c0 rejects instances where c0 is not defined") {
  constexpr std::int32_t kW = 2147483647;
  CHECK_THROWS_AS(sbmc::auto_c0(Graph{1, {}}), InvalidArgument);
  CHECK_THROWS_AS(sbmc::auto_c0(Graph{2, {{0, 1, 1}}}), InvalidArgument);  // zero variance
  CHECK_THROWS_AS(
      sbmc::auto_c0(Graph{4, {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1}, {1, 3, 1}, {2, 3, 1}}}),
      InvalidArgument);  // K4 with equal weights: zero variance
  CHECK_THROWS_AS(sbmc::auto_c0(DenseSpinGlass{1, 7}), InvalidArgument);
  CHECK_THROWS_AS(sbmc::auto_c0(DenseSpinGlass{2, 0}), InvalidArgument);                // J_01 = 1
  CHECK_THROWS_AS(sbmc::auto_c0(Graph{3, {{0, 1, kW}, {0, 2, kW}}}), InvalidArgument);  // 2Q
  CHECK_THROWS_AS(sbmc::auto_c0(Graph{3, {{0, 1, kW}, {0, 2, kW}, {1, 2, kW}}}),
                  InvalidArgument);  // Q itself overflows int64
  CHECK_THROWS_AS(sbmc::auto_c0(Graph{3, {{0, 2, 1}, {0, 1, 1}}}), InvalidArgument);  // order
  CHECK_THROWS_AS(sbmc::auto_c0(Instance{Graph{2, {{0, 1, 1}}}}), InvalidArgument);
  CHECK(static_cast<int>(sbmc::coupling(DenseSpinGlass{2, 0}, 0, 1)) == 1);
}

TEST_CASE("SB: sb_sign is +1 for +0, -0 and positive values, -1 for negative values") {
  constexpr float kTiny = std::numeric_limits<float>::denorm_min();
  CHECK(static_cast<int>(sbmc::sb_sign(0.0f)) == 1);
  CHECK(static_cast<int>(sbmc::sb_sign(-0.0f)) == 1);
  CHECK(static_cast<int>(sbmc::sb_sign(kTiny)) == 1);
  CHECK(static_cast<int>(sbmc::sb_sign(-kTiny)) == -1);
  CHECK(static_cast<int>(sbmc::sb_sign(1.0f)) == 1);
  CHECK(static_cast<int>(sbmc::sb_sign(-1.0f)) == -1);
}

TEST_CASE("SB: sb_update follows the specified operations and walls") {
  struct Case {
    float x, y, field, p, dt, c0, dt_a0;
    std::uint32_t x_bits, y_bits;
    bool every_preset;
  };
  const std::vector<Case> cases = {
      // x reaches exactly 1: the wall does not trigger.
      {0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0x3F800000u, 0x3F800000u, true},
      // x = 1 + 2^-23: the wall triggers.
      {0.5f, from_bits(0x3F800002u), 0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0x3F800000u, 0x00000000u, true},
      {-0.5f, from_bits(0xBF800002u), 0.0f, 0.0f, 1.0f, 1.0f, 0.5f, 0xBF800000u, 0x00000000u, true},
      {0.9f, 0.0f, 10.0f, 0.5f, 1.25f, 0.1f, 1.25f, 0x3F800000u, 0x00000000u, true},
      // Strict only: with FMA contraction y is 0x3E200000.
      {0.5f, 0.25f, 3.0f, 0.75f, 1.25f, 0.1f, 1.25f, 0x3F320000u, 0x3E200001u, false},
  };
  for (const Case& c : cases) {
    float x = c.x;
    float y = c.y;
    sbmc::sb_update(x, y, c.field, c.p, c.dt, c.c0, c.dt_a0);
    INFO("input x " << hex32(bits(c.x)) << ", y " << hex32(bits(c.y)) << ": expected ("
                    << hex32(c.x_bits) << ", " << hex32(c.y_bits) << "), got (" << hex32(bits(x))
                    << ", " << hex32(bits(y)) << ")");
    if (c.every_preset || kStrict) {
      CHECK(bits(x) == c.x_bits);
      CHECK(bits(y) == c.y_bits);
    } else {
      CHECK(std::abs(static_cast<double>(x) - 0.6953125) <= 1e-6);
      CHECK(std::abs(static_cast<double>(y) - 0.15625) <= 1e-6);
    }
  }
}
