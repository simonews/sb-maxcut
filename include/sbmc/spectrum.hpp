// Estimate of the extremes of the spectrum of J, and the automatic dt and c0 of simulated
// bifurcation derived from them (phase 07, decision L4; section 6 of the reference document).
//
// estimate_spectrum runs k = min(iterations, n) steps of the Lanczos method in double on J
// (J_ij = -w_ij, zero diagonal), with full reorthogonalization: two passes of classical
// Gram-Schmidt against every previous Lanczos vector at every step. The start vector s = 0 has the
// components v_i = (double(w_i) + 0.5) / 2^32 - 0.5, w_i = stream_word(0,
// RngDomain::spectrum_start, s, i), normalized: continuous components, because a start vector of
// +-1 can stay in an invariant subspace (on tests/data/four_nodes.rudy it gives 0 and 1 instead of
// -2.5616 and 1.5616). A breakdown (beta_j <= 1e-12 * scale, an invariant subspace, for example
// with isolated nodes) restarts with the next start vector, s + 1, orthogonalized against the
// Lanczos vectors found so far. The extremes of the tridiagonal matrix of the k steps (its Ritz
// values) are found by bisection with Sturm counts. The Ritz values lie inside the spectrum:
// lambda_min >= the smallest eigenvalue of J and lambda_max <= the largest. The products J v may
// run with OpenMP over the rows (default number of threads), each row summed in increasing column
// order; every other sum is sequential, so the result does not depend on the number of threads,
// bit for bit. The Lanczos method estimates both extremes with the same products J v: the power
// method suggested by section 6 of the reference document converges badly on the second extreme
// when the spectrum is crowded at that end.
//
// Standard C++ only: no CUDA type or header.
#pragma once

#include <sbmc/instance.hpp>

namespace sbmc {

// Estimate of the extremes of the spectrum of J (phase 07, decision L4).
struct SpectrumEstimate {
  double lambda_min = 0.0;  // smallest Ritz value, >= the smallest eigenvalue of J
  double lambda_max = 0.0;  // largest Ritz value, <= the largest eigenvalue of J
  int iterations = 0;       // Lanczos steps done: min(requested, n)
  int restarts = 0;         // new start vectors after a breakdown
};

inline constexpr int kSpectrumIterations = 100;

// InvalidArgument for iterations < 1, for an invalid instance (full check) and for a
// DenseSpinGlass with n > 32768 (its dense INT8 copy would exceed 1 GiB; checked before any
// allocation). A graph without edges (J = 0) gives lambda_min = lambda_max = 0, without
// exceptions. sbmc::Error with "internal" if no new start vector survives the orthogonalization
// after a breakdown (9 attempts).
SpectrumEstimate estimate_spectrum(const Graph& g, int iterations = kSpectrumIterations);
SpectrumEstimate estimate_spectrum(const DenseSpinGlass& g, int iterations = kSpectrumIterations);
SpectrumEstimate estimate_spectrum(const Instance& inst, int iterations = kSpectrumIterations);

// dt = factor * 1.25 * sqrt(2 / (1 - lambda_min / lambda_max)), computed in double and rounded
// once to float (section 6 of the reference document). InvalidArgument unless lambda_max is
// finite and > 0, lambda_min finite and <= 0, factor finite and > 0.
float spectral_dt(const SpectrumEstimate& s, double factor = 1.0);

// c0 = 1 / lambda_max, computed in double and rounded once to float. InvalidArgument unless
// lambda_max is finite and > 0.
float spectral_c0(const SpectrumEstimate& s);

}  // namespace sbmc
