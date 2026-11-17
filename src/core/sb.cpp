// Parts of simulated bifurcation common to every variant (sbmc/sb.hpp): the pressure schedule
// and the automatic c0.
#include <sbmc/sb.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>

namespace sbmc {
namespace {

constexpr std::int64_t kInt64Max = std::numeric_limits<std::int64_t>::max();

[[noreturn]] void invalid(const char* function, const std::string& reason) {
  throw InvalidArgument(std::string(function) + ": " + reason);
}

// c0 from n, W and Q = sum_{i<j} w_ij^2 of a valid instance, in the order of sbmc/sb.hpp. The
// int64 operations are checked for overflow before they are done.
float c0_from_sums(std::int64_t n, std::int64_t w, std::int64_t q) {
  constexpr const char* kFunction = "auto_c0";
  if (n < 2) {
    invalid(kFunction, "n = " + std::to_string(n) + ": at least 2 spins are needed");
  }
  // Defensive check without a test (answer 1 A to F04-R1): with a valid instance |W| < 2^62
  // (a Graph would need more than 2^31 edges, a DenseSpinGlass has |W| < 2^61), so it cannot fire.
  if (w > kInt64Max / 2 || w < -(kInt64Max / 2)) {
    invalid(kFunction, "S1 = -2 W overflows int64 (W = " + std::to_string(w) + ")");
  }
  const std::int64_t s1 = -2 * w;
  if (q > kInt64Max / 2) {
    invalid(kFunction, "S2 = 2 Q overflows int64 (Q = " + std::to_string(q) + ")");
  }
  const std::int64_t s2 = 2 * q;
  const std::int64_t pairs = n * (n - 1);  // n <= 2^31 - 1: below 2^62

  const double mean = static_cast<double>(s1) / static_cast<double>(pairs);
  const double ms = static_cast<double>(s2) / static_cast<double>(pairs);
  const double var = ms - mean * mean;
  // var not > 0 is tested (zero variance). var not finite is a defensive check without a test
  // (answer 1 A to F04-R1): the operands are finite, so it cannot fire with a valid instance.
  if (!std::isfinite(var) || !(var > 0.0)) {
    invalid(kFunction, "the variance of the off-diagonal couplings is not positive (var = " +
                           std::to_string(var) + ")");
  }
  const double sigma = std::sqrt(var);
  const double c0 = 1.0 / (2.0 * sigma * std::sqrt(static_cast<double>(n)));
  const auto result = static_cast<float>(c0);
  // Defensive check without a test (answer 1 A to F04-R1): with var > 0 computed from int64
  // sums, 2 sigma sqrt(n) stays above about 2^-56, so c0 < 2^56 is finite.
  if (!std::isfinite(result)) {
    invalid(kFunction, "c0 is not finite");
  }
  return result;
}

}  // namespace

std::vector<float> pressure_schedule(float a0, int steps) {
  constexpr const char* kFunction = "pressure_schedule";
  if (steps < 1) {
    invalid(kFunction, "steps = " + std::to_string(steps) + " must be at least 1");
  }
  if (!std::isfinite(a0) || !(a0 > 0.0f)) {
    invalid(kFunction, "a0 = " + std::to_string(a0) + " must be finite and > 0");
  }
  std::vector<float> p(static_cast<std::size_t>(steps));
  const auto a = static_cast<double>(a0);
  const auto total = static_cast<double>(steps);
  for (int m = 0; m < steps; ++m) {
    p[static_cast<std::size_t>(m)] =
        static_cast<float>(a * (static_cast<double>(steps - 1 - m) / total));
  }
  return p;
}

float auto_c0(const Graph& g) {
  const std::int64_t w = total_weight(g);  // full check of the graph
  std::int64_t q = 0;
  for (const Edge& e : g.edges) {
    const std::int64_t square = std::int64_t{e.w} * e.w;  // below 2^62
    if (q > kInt64Max - square) {
      invalid("auto_c0", "Q = sum of the squared weights overflows int64");
    }
    q += square;
  }
  return c0_from_sums(g.n, w, q);
}

float auto_c0(const DenseSpinGlass& g) {
  const std::int64_t w = total_weight(g);  // checks n
  const std::int64_t q = g.n * (g.n - 1) / 2;
  return c0_from_sums(g.n, w, q);
}

float auto_c0(const Instance& inst) {
  return std::visit([](const auto& g) { return auto_c0(g); }, inst);
}

}  // namespace sbmc
