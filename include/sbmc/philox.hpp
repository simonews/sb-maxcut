// Philox4x32-10, the random streams of the project and the conversions of their words.
//
// Philox4x32-10 is the counter-based generator of Salmon, Moraes, Dror and Shaw ("Parallel random
// numbers: as easy as 1, 2, 3", SC 2011), identical to Random123 and to curand_Philox4x32_10 of
// cuRAND. This header depends only on <cstdint>. Every function is constexpr, noexcept and, under
// nvcc, __host__ __device__ (macro SBMC_HD); no intrinsic is used, so the same code compiles for
// the host and for the device and gives the same bits on both.
//
// Streams. A stream is identified by (seed, domain d, stream index). Its block b (four words) and
// its 32-bit word k are
//   stream_block(seed, d, stream, b) = philox4x32_10({lo32(b), hi32(b), stream, d},
//                                                    {lo32(seed), hi32(seed)}),
//   stream_word(seed, d, stream, k)  = component k mod 4 (x, y, z, w) of block k / 4.
// Equivalence with cuRAND (checked on the GPU by tests/gpu/test_curand_philox.cu): after
// curand_init(seed, ((uint64)d << 32) | stream, k, &state), the t-th call to curand(&state),
// t = 0, 1, ..., returns stream_word(seed, d, stream, k + t); with k a multiple of 4,
// curand4(&state) returns stream_block(seed, d, stream, k / 4).
#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define SBMC_HD __host__ __device__
#else
#define SBMC_HD
#endif

namespace sbmc {

// Four 32-bit words: counter and output block of Philox4x32.
struct U32x4 {
  std::uint32_t x, y, z, w;

  constexpr bool operator==(const U32x4&) const noexcept = default;
};

// Two 32-bit words: key of Philox4x32.
struct U32x2 {
  std::uint32_t x, y;

  constexpr bool operator==(const U32x2&) const noexcept = default;
};

namespace detail {

// One round R(c, k): p0 = M0 * c0 and p1 = M1 * c2 as 64-bit products, then
// R(c, k) = (hi(p1) ^ c1 ^ k0, lo(p1), hi(p0) ^ c3 ^ k1, lo(p0)).
SBMC_HD constexpr U32x4 philox4x32_round(U32x4 c, U32x2 k) noexcept {
  constexpr std::uint64_t kM0 = 0xD2511F53u;
  constexpr std::uint64_t kM1 = 0xCD9E8D57u;
  const std::uint64_t p0 = kM0 * c.x;
  const std::uint64_t p1 = kM1 * c.z;
  return U32x4{static_cast<std::uint32_t>(p1 >> 32) ^ c.y ^ k.x, static_cast<std::uint32_t>(p1),
               static_cast<std::uint32_t>(p0 >> 32) ^ c.w ^ k.y, static_cast<std::uint32_t>(p0)};
}

}  // namespace detail

// Philox4x32-10: a first round with the key, then 9 times key bump B(k) = (k0 + W0, k1 + W1),
// modulo 2^32, followed by a round.
SBMC_HD constexpr U32x4 philox4x32_10(U32x4 ctr, U32x2 key) noexcept {
  constexpr std::uint32_t kW0 = 0x9E3779B9u;
  constexpr std::uint32_t kW1 = 0xBB67AE85u;
  ctr = detail::philox4x32_round(ctr, key);
  for (int round = 1; round < 10; ++round) {
    key.x = static_cast<std::uint32_t>(key.x + kW0);
    key.y = static_cast<std::uint32_t>(key.y + kW1);
    ctr = detail::philox4x32_round(ctr, key);
  }
  return ctr;
}

// Domains of the random streams. The values are fixed: 0 is never used, a new domain takes the
// next number, and numbers are never reassigned.
//
// | Domain               | Value | Seed (key)        | Stream index        | Word k            |
// |----------------------|-------|-------------------|---------------------|-------------------|
// | couplings            | 1     | instance seed     | min(i, j)           | max(i, j)         |
// | sb_initial_positions | 2     | run seed          | global replica r    | spin index i      |
// | sa_initial_spins     | 3     | run seed          | r                   | i                 |
// | sa_acceptance        | 4     | run seed          | r                   | sweep * n + spin  |
//
// Uses: couplings, J_ij = coupling_from_word(word); sb_initial_positions,
// x0 = position_from_word(word); sa_initial_spins, initial spin s_i = coupling_from_word(word)
// (sa_initial_spin of sbmc/sa.hpp); sa_acceptance, word of the proposal of spin i in sweep k,
// consumed by every proposal, accepted or not: an uphill move is accepted when the word is below
// sa_threshold(beta_k, delta_e) (sa_acceptance_word and sa_threshold of sbmc/sa.hpp).
enum class RngDomain : std::uint32_t {
  couplings = 1,
  sb_initial_positions = 2,
  sa_initial_spins = 3,
  sa_acceptance = 4,
};

// Block b of the stream (seed, d, stream).
SBMC_HD constexpr U32x4 stream_block(std::uint64_t seed, RngDomain d, std::uint32_t stream,
                                     std::uint64_t block) noexcept {
  return philox4x32_10(
      U32x4{static_cast<std::uint32_t>(block), static_cast<std::uint32_t>(block >> 32), stream,
            static_cast<std::uint32_t>(d)},
      U32x2{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)});
}

// Word k of the stream (seed, d, stream): component k mod 4 of block k / 4. The component is
// chosen with selects, not with an indexed array, which would go to local memory on the device.
SBMC_HD constexpr std::uint32_t stream_word(std::uint64_t seed, RngDomain d, std::uint32_t stream,
                                            std::uint64_t k) noexcept {
  const U32x4 b = stream_block(seed, d, stream, k >> 2);
  const bool odd = (k & 1u) != 0;
  return (k & 2u) != 0 ? (odd ? b.w : b.z) : (odd ? b.y : b.x);
}

// J_ij from a word of the couplings domain: +1 when the top bit is 0, -1 otherwise.
SBMC_HD constexpr std::int8_t coupling_from_word(std::uint32_t w) noexcept {
  return (w >> 31) == 0 ? std::int8_t{1} : std::int8_t{-1};
}

// Initial position from a word of the sb_initial_positions domain, uniform in [-0.1f, 0.1f):
// t = (int32)(w >> 8) - 2^23, an exact integer in [-2^23, 2^23 - 1], converted exactly to float
// and multiplied once by S = 0.1f / 2^23 = 0x1.99999ap-27f (bits 0x324CCCCD). One rounded
// multiplication and no other floating-point operation: no FMA contraction is possible, so the
// bits are the same in every preset, strict or not, on the host and on the device. w = 0 gives
// -0.1f (bits 0xBDCCCCCD); w = 0xFFFFFFFF gives the float with bits 0x3DCCCCCB.
SBMC_HD constexpr float position_from_word(std::uint32_t w) noexcept {
  constexpr float kScale = 0x1.99999ap-27f;
  const std::int32_t t = static_cast<std::int32_t>(w >> 8) - (std::int32_t{1} << 23);
  return static_cast<float>(t) * kScale;
}

}  // namespace sbmc
