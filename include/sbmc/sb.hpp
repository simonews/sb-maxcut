// Simulated bifurcation: the parts common to every variant (reference document, section 6).
//
// Equations of motion, discretised with the symplectic Euler scheme of Goto et al.: at step m,
// with the field F computed from the positions before the step,
//   y_i <- y_i + dt * (c0 * F_i - p[m] * x_i),   x_i <- x_i + dt * a0 * y_i,
// then the inelastic walls: if |x_i| > 1, x_i <- sgn(x_i) and y_i <- 0. F = J * sgn(x) in dSB
// (discrete) and F = J * x in bSB (ballistic), with sgn(0) = +1. Positions, momenta, fields,
// p[m], dt, a0 and c0 are float (FP32), as on the GPU. Every variant uses sb_sign and sb_update,
// so that the floating-point operations are the same and in the same order: in the strict
// builds (no FMA contraction) dSB gives bit-for-bit identical trajectories in every variant
// (test T7), with the sequential reference C0 (sbmc/sb_sequential.hpp) as the reference.
#pragma once

#include <cstdint>
#include <vector>

#include <sbmc/instance.hpp>
#include <sbmc/philox.hpp>

namespace sbmc {

enum class SbAlgorithm : std::uint8_t { ballistic = 0, discrete = 1 };  // bSB, dSB

struct SbParams {
  SbAlgorithm algorithm = SbAlgorithm::discrete;
  int replicas = 1;                // R >= 1
  int steps = 1;                   // M >= 1
  float dt = 1.25f;                // finite, > 0
  float a0 = 1.0f;                 // finite, > 0
  float c0 = 0.0f;                 // 0: automatic (auto_c0); otherwise finite, > 0
  int eval_every = 0;              // bSB only: K >= 0; 0 = only the final configuration
  std::uint64_t seed = 0;          // run seed of x0 (RngDomain::sb_initial_positions)
  std::int64_t first_replica = 0;  // global index of local replica 0
};

// p[m] = a0 * (1 - (m + 1) / M), m = 0 .. M - 1, computed as
// (float)((double)a0 * ((double)(M - 1 - m) / (double)M)): one division and one multiplication
// in double, one rounding to float, no addition, so the same bits in every preset; p[M - 1] = 0.
// InvalidArgument if steps < 1 or if a0 is not finite or not > 0.
std::vector<float> pressure_schedule(float a0, int steps);

// c0 = 1 / (2 sigma sqrt(n)), sigma = standard deviation of the n(n - 1) off-diagonal J_ij.
// From W = total_weight and Q = sum_{i<j} w_ij^2 (n(n - 1)/2 for a DenseSpinGlass), in int64:
// S1 = -2 W, S2 = 2 Q, P = n (n - 1); then in double, in this order: mean = S1 / P, ms = S2 / P,
// var = ms - mean * mean, sigma = sqrt(var), c0 = (float)(1 / (2 * sigma * sqrt(n))).
// InvalidArgument for an invalid instance (full check), n < 2, an int64 overflow of Q, S1 or S2,
// a variance that is not finite or not > 0, or a result that is not finite. In the production
// builds FMA contraction may change the last bit of var and of c0.
float auto_c0(const Graph& g);
float auto_c0(const DenseSpinGlass& g);
float auto_c0(const Instance& inst);

// sgn with sgn(0) = sgn(-0) = +1.
SBMC_HD constexpr Spin sb_sign(float x) noexcept {
  return x < 0.0f ? Spin{-1} : Spin{1};
}

// Symplectic Euler update of one spin of one replica, then the inelastic walls; dt_a0 = dt * a0
// is computed once in float by the caller. Exactly:
//   1. y = y + dt * (c0 * field - p * x)
//   2. x = x + dt_a0 * y
//   3. if x > 1 or x < -1: x = (x < 0) ? -1 : 1 and y = 0 (x exactly +-1 does not trigger it).
// c0 * field - p * x has the value of -p * x + c0 * F of the reference document in IEEE
// arithmetic.
SBMC_HD constexpr void sb_update(float& x, float& y, float field, float p, float dt, float c0,
                                 float dt_a0) noexcept {
  y = y + dt * (c0 * field - p * x);
  x = x + dt_a0 * y;
  if (x > 1.0f || x < -1.0f) {
    x = x < 0.0f ? -1.0f : 1.0f;
    y = 0.0f;
  }
}

struct SbReplicaResult {
  std::int64_t best_energy = 0;  // minimum E over the evaluated configurations
  std::int64_t best_cut = 0;     // (W - best_energy) / 2, checked with cut_direct
  int best_step = 0;             // m of the configuration s(m) with best_energy; ties: smallest m
  std::int64_t final_cut = 0;    // cut of s(M), the signs after the last step
};

struct SbResult {
  std::int64_t n = 0;
  int replicas = 0;
  int steps = 0;
  std::int64_t first_replica = 0;
  float c0 = 0.0f;                           // c0 actually used
  std::int64_t total_weight = 0;             // W
  std::vector<SbReplicaResult> per_replica;  // R elements
  std::vector<Spin> best_spins;              // replica-major, R * n: s(best_step)
  std::vector<float> final_x, final_y;       // replica-major, R * n
  int trace_steps = 0;                       // K
  std::vector<float> trace_x, trace_y;       // [k][r][i], K * R * n: state after step k + 1
};

}  // namespace sbmc
