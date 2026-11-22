// Estimate of the extremes of the spectrum of J with the Lanczos method, and the automatic dt and
// c0 derived from them (sbmc/spectrum.hpp).
//
// Products y = J x in double: a Graph on compressed rows built once (both directions of every
// edge, increasing columns, values -w as double); a DenseSpinGlass on its dense INT8 rows from
// dense_rows_int8 (ld = n). y_i = sum_j J_ij x_j with j increasing, the rows in parallel with
// OpenMP (default number of threads): every row is summed by one thread in the same order, so the
// products do not depend on the number of threads. Dot products, norms and the sums of
// Gram-Schmidt are sequential loops with increasing index. The extremes of the tridiagonal matrix
// come from bisection with Sturm counts, to an absolute error <= 1e-14 * max_j (|alpha_j| +
// |beta_j| + |beta_{j-1}|).
#include <sbmc/spectrum.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/philox.hpp>

namespace sbmc {
namespace {

constexpr std::int64_t kMaxDenseSpins = 32768;  // dense INT8 copy of J: at most 1 GiB
constexpr double kBreakdown = 1e-12;            // beta_j <= kBreakdown * scale: invariant subspace
constexpr double kMinStartNorm = 1e-6;          // a new start vector shorter than this is skipped
constexpr int kStartAttempts = 9;               // the first new start vector and 8 more
constexpr double kBisectionTolerance = 1e-14;   // relative to max_j (|alpha_j| + |beta_j| + ...)
constexpr double kTwoTo32 = 4294967296.0;

[[noreturn]] void invalid(const char* function, const std::string& reason) {
  throw InvalidArgument(std::string(function) + ": " + reason);
}

// Shortest form that reads back as the same double ("inf", "nan" included).
std::string number(double v) {
  std::array<char, 32> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), v);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("?");
}

void check_iterations(int iterations) {
  if (iterations < 1) {
    invalid("estimate_spectrum",
            "iterations = " + std::to_string(iterations) + " must be at least 1");
  }
}

// J of a Graph as compressed rows: row i has the entries start[i] .. start[i + 1] - 1, with
// increasing columns.
struct SparseRows {
  std::int64_t n = 0;
  std::vector<std::int64_t> start;
  std::vector<std::int32_t> column;
  std::vector<double> value;

  explicit SparseRows(const Graph& g) : n(g.n), start(static_cast<std::size_t>(g.n) + 1, 0) {
    for (const Edge& e : g.edges) {
      ++start[static_cast<std::size_t>(e.u) + 1];
      ++start[static_cast<std::size_t>(e.v) + 1];
    }
    for (std::size_t i = 1; i < start.size(); ++i) {
      start[i] += start[i - 1];
    }
    column.resize(static_cast<std::size_t>(start.back()));
    value.resize(static_cast<std::size_t>(start.back()));
    // The edges are in increasing (u, v) order (checked by total_weight): row i receives first the
    // columns u < i of the edges (u, i), then the columns v > i of the edges (i, v), both in
    // increasing order.
    std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
    for (const Edge& e : g.edges) {
      const double j = -static_cast<double>(e.w);
      const auto u = static_cast<std::size_t>(e.u);
      const auto v = static_cast<std::size_t>(e.v);
      column[static_cast<std::size_t>(fill[u])] = e.v;
      value[static_cast<std::size_t>(fill[u]++)] = j;
      column[static_cast<std::size_t>(fill[v])] = e.u;
      value[static_cast<std::size_t>(fill[v]++)] = j;
    }
  }

  void apply(const double* x, double* y) const {
    const std::int64_t rows = n;
    const std::int64_t* const first = start.data();
    const std::int32_t* const col = column.data();
    const double* const val = value.data();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (std::int64_t p = first[i]; p < first[i + 1]; ++p) {
        sum += val[p] * x[col[p]];
      }
      y[i] = sum;
    }
  }
};

// J of a DenseSpinGlass as dense INT8 rows, ld = n.
struct DenseRows {
  std::int64_t n = 0;
  std::vector<std::int8_t> rows;

  explicit DenseRows(const DenseSpinGlass& g)
      : n(g.n), rows(static_cast<std::size_t>(g.n) * static_cast<std::size_t>(g.n)) {
    dense_rows_int8(g, 0, g.n, g.n, rows);
  }

  void apply(const double* x, double* y) const {
    const std::int64_t size = n;
    const std::int8_t* const j = rows.data();
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < size; ++i) {
      const std::int8_t* const row = j + i * size;
      double sum = 0.0;
      for (std::int64_t c = 0; c < size; ++c) {
        sum += static_cast<double>(row[c]) * x[c];
      }
      y[i] = sum;
    }
  }
};

double dot(const double* a, const double* b, std::size_t n) {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

// w <- w - sum_{i < count} (q_i . w) q_i, classical Gram-Schmidt (all the coefficients from the
// same w), twice. q holds the vectors q_0, q_1, ... one after the other.
void orthogonalize(double* w, const std::vector<double>& q, int count, std::size_t n,
                   std::vector<double>& coefficient) {
  for (int pass = 0; pass < 2; ++pass) {
    for (int i = 0; i < count; ++i) {
      coefficient[static_cast<std::size_t>(i)] =
          dot(q.data() + static_cast<std::size_t>(i) * n, w, n);
    }
    for (int i = 0; i < count; ++i) {
      const double c = coefficient[static_cast<std::size_t>(i)];
      const double* const qi = q.data() + static_cast<std::size_t>(i) * n;
      for (std::size_t t = 0; t < n; ++t) {
        w[t] -= c * qi[t];
      }
    }
  }
}

// Start vector number s: v_i = (double(w_i) + 0.5) / 2^32 - 0.5, w_i = stream_word(0,
// spectrum_start, s, i); then normalized. Its norm before the normalization is about sqrt(n / 12),
// never 0.
void start_vector(std::uint32_t s, double* v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t w = stream_word(0, RngDomain::spectrum_start, s, i);
    v[i] = (static_cast<double>(w) + 0.5) / kTwoTo32 - 0.5;
  }
  const double norm = std::sqrt(dot(v, v, n));
  for (std::size_t i = 0; i < n; ++i) {
    v[i] /= norm;
  }
}

// Number of eigenvalues below x of the symmetric tridiagonal matrix with diagonal a[0 .. k-1] and
// off-diagonal b[0 .. k-2] (Sturm count of the LDL^T factorization of T - x I).
int count_below(const std::vector<double>& a, const std::vector<double>& b, int k, double x,
                double pivot_min) {
  int count = 0;
  double d = 1.0;
  for (int i = 0; i < k; ++i) {
    const double off = i > 0 ? b[static_cast<std::size_t>(i - 1)] : 0.0;
    d = (a[static_cast<std::size_t>(i)] - x) - (i > 0 ? off * off / d : 0.0);
    if (std::abs(d) < pivot_min) {
      d = -pivot_min;
    }
    if (d < 0.0) {
      ++count;
    }
  }
  return count;
}

// The smallest (rank 1) or the largest (rank k) eigenvalue of the tridiagonal matrix, by bisection
// on [lo, hi] with count_below(lo) < rank <= count_below(hi).
double bisect(const std::vector<double>& a, const std::vector<double>& b, int k, int rank,
              double lo, double hi, double tolerance, double pivot_min) {
  for (int step = 0; step < 4096 && hi - lo > tolerance; ++step) {
    const double mid = lo + (hi - lo) / 2.0;
    if (mid <= lo || mid >= hi) {
      break;
    }
    if (count_below(a, b, k, mid, pivot_min) >= rank) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return lo + (hi - lo) / 2.0;
}

template <typename Rows>
SpectrumEstimate lanczos(const Rows& rows, int iterations) {
  const auto n = static_cast<std::size_t>(rows.n);
  const int k = static_cast<int>(std::min<std::int64_t>(iterations, rows.n));
  std::vector<double> q(static_cast<std::size_t>(k) * n);  // q_0, q_1, ..., q_{k-1}
  std::vector<double> w(n);
  std::vector<double> alpha(static_cast<std::size_t>(k));
  std::vector<double> beta(static_cast<std::size_t>(k));
  std::vector<double> coefficient(static_cast<std::size_t>(k));

  SpectrumEstimate result;
  std::uint32_t stream = 0;
  start_vector(stream, q.data(), n);
  double scale = 0.0;
  for (int j = 0; j < k; ++j) {
    const double* const qj = q.data() + static_cast<std::size_t>(j) * n;
    rows.apply(qj, w.data());
    alpha[static_cast<std::size_t>(j)] = dot(qj, w.data(), n);
    orthogonalize(w.data(), q, j + 1, n, coefficient);
    const double b = std::sqrt(dot(w.data(), w.data(), n));
    beta[static_cast<std::size_t>(j)] = b;
    scale = std::max(scale, std::abs(alpha[static_cast<std::size_t>(j)]) + b);
    if (j == k - 1) {
      break;
    }
    double* const next = q.data() + static_cast<std::size_t>(j + 1) * n;
    if (b <= kBreakdown * scale) {
      // Breakdown: an invariant subspace. Restart with the next start vector, orthogonal to the
      // vectors found so far.
      beta[static_cast<std::size_t>(j)] = 0.0;
      ++result.restarts;
      bool found = false;
      for (int attempt = 0; attempt < kStartAttempts && !found; ++attempt) {
        ++stream;
        start_vector(stream, next, n);
        orthogonalize(next, q, j + 1, n, coefficient);
        const double norm = std::sqrt(dot(next, next, n));
        if (norm >= kMinStartNorm) {
          for (std::size_t t = 0; t < n; ++t) {
            next[t] /= norm;
          }
          found = true;
        }
      }
      if (!found) {
        throw Error(
            "estimate_spectrum: internal error: no new start vector after a breakdown at "
            "step " +
            std::to_string(j) + " (" + std::to_string(kStartAttempts) + " attempts)");
      }
    } else {
      for (std::size_t t = 0; t < n; ++t) {
        next[t] = w[t] / b;
      }
    }
  }

  // Extremes of the tridiagonal matrix: Gershgorin bounds, then bisection.
  double norm = 0.0;
  double lo = 0.0;
  double hi = 0.0;
  for (int j = 0; j < k; ++j) {
    const double a = alpha[static_cast<std::size_t>(j)];
    const double radius = (j + 1 < k ? std::abs(beta[static_cast<std::size_t>(j)]) : 0.0) +
                          (j > 0 ? std::abs(beta[static_cast<std::size_t>(j - 1)]) : 0.0);
    norm = std::max(norm, std::abs(a) + radius);
    lo = j == 0 ? a - radius : std::min(lo, a - radius);
    hi = j == 0 ? a + radius : std::max(hi, a + radius);
  }
  result.iterations = k;
  if (norm == 0.0) {  // J q = 0 in every step: every Ritz value is 0
    return result;
  }
  const double tolerance = kBisectionTolerance * norm;
  const double pivot_min = std::numeric_limits<double>::min() * std::max(1.0, norm * norm);
  lo -= tolerance;
  hi += tolerance;
  result.lambda_min = bisect(alpha, beta, k, 1, lo, hi, tolerance, pivot_min);
  result.lambda_max = bisect(alpha, beta, k, k, lo, hi, tolerance, pivot_min);
  return result;
}

}  // namespace

SpectrumEstimate estimate_spectrum(const Graph& g, int iterations) {
  check_iterations(iterations);
  try {
    total_weight(g);  // full check of the graph
  } catch (const InvalidArgument& error) {
    throw InvalidArgument(std::string("estimate_spectrum: ") + error.what());
  }
  return lanczos(SparseRows(g), iterations);
}

SpectrumEstimate estimate_spectrum(const DenseSpinGlass& g, int iterations) {
  check_iterations(iterations);
  try {
    num_spins(g);  // checks n (the full check of a DenseSpinGlass)
  } catch (const InvalidArgument& error) {
    throw InvalidArgument(std::string("estimate_spectrum: ") + error.what());
  }
  if (g.n > kMaxDenseSpins) {
    invalid("estimate_spectrum", "DenseSpinGlass with n = " + std::to_string(g.n) +
                                     " > 32768: its dense INT8 copy of J would exceed 1 GiB");
  }
  return lanczos(DenseRows(g), iterations);
}

SpectrumEstimate estimate_spectrum(const Instance& inst, int iterations) {
  return std::visit([&](const auto& g) { return estimate_spectrum(g, iterations); }, inst);
}

float spectral_dt(const SpectrumEstimate& s, double factor) {
  constexpr const char* kFunction = "spectral_dt";
  if (!std::isfinite(s.lambda_max) || !(s.lambda_max > 0.0)) {
    invalid(kFunction, "lambda_max = " + number(s.lambda_max) + " must be finite and > 0");
  }
  if (!std::isfinite(s.lambda_min) || !(s.lambda_min <= 0.0)) {
    invalid(kFunction, "lambda_min = " + number(s.lambda_min) + " must be finite and <= 0");
  }
  if (!std::isfinite(factor) || !(factor > 0.0)) {
    invalid(kFunction, "factor = " + number(factor) + " must be finite and > 0");
  }
  return static_cast<float>(factor * 1.25 * std::sqrt(2.0 / (1.0 - s.lambda_min / s.lambda_max)));
}

float spectral_c0(const SpectrumEstimate& s) {
  if (!std::isfinite(s.lambda_max) || !(s.lambda_max > 0.0)) {
    invalid("spectral_c0", "lambda_max = " + number(s.lambda_max) + " must be finite and > 0");
  }
  return static_cast<float>(1.0 / s.lambda_max);
}

}  // namespace sbmc
