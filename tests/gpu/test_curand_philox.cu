// T11, GPU part: cuRAND's Philox4x32-10 against the stream convention of sbmc/philox.hpp, the
// project's Philox compiled for the device, and x0 computed on the device.
//
// Every case is prepared on the host, and its expected value is computed on the host with
// sbmc/philox.hpp and sbmc/initial_state.hpp; on device 0, one thread per case computes the
// value to compare. Checks:
// - T11_CURAND_WORD: after curand_init(seed, ((uint64)d << 32) | stream, k, &state), curand()
//   equals stream_word(seed, d, stream, k); the 17 reference cases of test_philox.cpp and 200
//   pseudo-random cases (k with 2, 34, 40 or 64 random bits);
// - T11_CURAND_SEQUENCE: 16 consecutive curand() after one curand_init at k0 equal
//   stream_word(..., k0 + t), for k0 = 0, 1, 2, 3, 2^34 - 6, 2^64 - 20 (one case per word);
// - T11_CURAND4: curand4() right after curand_init at k equals stream_block(..., k / 4), for
//   k = 0, 4, 2^34, 2^34 + 8;
// - T11_OWN_PHILOX_DEVICE: stream_word on the device for the word cases and philox4x32_10 on the
//   device for the three Random123 vectors, equal to the host;
// - T11_X0_DEVICE: initial_position on the device, bit for bit equal to the host (the 9 reference
//   cases of test_initial_state.cpp and 1000 pseudo-random cases).
//
// Output: "device: <name> (sm_XY)", then one line per check, "<KEY>: PASS|FAIL (<different>/
// <cases>)", followed for a FAIL by the first 5 different cases with their arguments and both
// values in hexadecimal; the last line is "T11_GPU_VERDICT: PASS" only when all five checks
// pass, "T11_GPU_VERDICT: FAIL" otherwise. Exit codes: 0 passed, 1 failed or CUDA error,
// 77 skipped (no usable device).
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <sbmc/initial_state.hpp>
#include <sbmc/philox.hpp>

#include "../smoke/smoke_common.cuh"

namespace {

using sbmc::RngDomain;
using sbmc::U32x2;
using sbmc::U32x4;
using sbmc::smoke::DeviceBuffer;

constexpr unsigned kThreadsPerBlock = 128;
constexpr int kSequenceLength = 16;
constexpr std::size_t kReportedCases = 5;
constexpr std::uint64_t kCaseSeed = 20261115u;  // std::mt19937_64 seed of the pseudo-random cases

// Word k of the stream (seed, domain, stream); also the start of a sequence or of a block.
struct WordCase {
  std::uint64_t seed;
  std::uint32_t domain;
  std::uint32_t stream;
  std::uint64_t k;
};

struct KatCase {
  U32x4 ctr;
  U32x2 key;
};

struct PositionCase {
  std::uint64_t seed;
  std::uint32_t replica;
  std::uint32_t spin;
};

// --- Kernels: one thread per case ----------------------------------------------------------

__device__ void init_state(const WordCase& c, curandStatePhilox4_32_10_t* state) {
  const unsigned long long subsequence =
      (static_cast<unsigned long long>(c.domain) << 32) | static_cast<unsigned long long>(c.stream);
  curand_init(c.seed, subsequence, c.k, state);
}

__global__ void curand_word_kernel(const WordCase* cases, std::uint32_t* out, std::size_t count) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    curandStatePhilox4_32_10_t state;
    init_state(cases[t], &state);
    out[t] = curand(&state);
  }
}

__global__ void curand_sequence_kernel(const WordCase* cases, std::uint32_t* out,
                                       std::size_t count) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    curandStatePhilox4_32_10_t state;
    init_state(cases[t], &state);
    for (int q = 0; q < kSequenceLength; ++q) {
      out[t * kSequenceLength + static_cast<std::size_t>(q)] = curand(&state);
    }
  }
}

__global__ void curand4_kernel(const WordCase* cases, U32x4* out, std::size_t count) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    curandStatePhilox4_32_10_t state;
    init_state(cases[t], &state);
    const uint4 block = curand4(&state);
    out[t] = U32x4{block.x, block.y, block.z, block.w};
  }
}

__global__ void own_word_kernel(const WordCase* cases, std::uint32_t* out, std::size_t count) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    const WordCase c = cases[t];
    out[t] = sbmc::stream_word(c.seed, static_cast<RngDomain>(c.domain), c.stream, c.k);
  }
}

__global__ void own_kat_kernel(const KatCase* cases, U32x4* out, std::size_t count) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    out[t] = sbmc::philox4x32_10(cases[t].ctr, cases[t].key);
  }
}

__global__ void x0_kernel(const PositionCase* cases, float* out, std::size_t count) {
  const std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t < count) {
    const PositionCase c = cases[t];
    out[t] = sbmc::initial_position(c.seed, c.replica, c.spin);
  }
}

unsigned grid_for(std::size_t count) {
  return static_cast<unsigned>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

// --- Cases ---------------------------------------------------------------------------------

std::vector<WordCase> make_word_cases(std::mt19937_64& rng) {
  // The 17 reference cases of tests/core/test_philox.cpp.
  std::vector<WordCase> cases = {
      {0x0u, 1, 0, 0},
      {0x0u, 1, 0, 1},
      {0x0u, 1, 0, 2},
      {0x0u, 1, 0, 3},
      {0x0u, 1, 0, 4},
      {0x1u, 2, 0, 0},
      {0x1u, 2, 7, 5},
      {0x2Au, 1, 3, 17179869180u},
      {0x2Au, 1, 3, 17179869183u},
      {0x2Au, 1, 3, 17179869184u},
      {0xFFFFFFFFFFFFFFFFu, 4, 4294967295u, 18446744073709551615u},
      {0x243F6A8885A308D3u, 3, 320440878u, 247824715705767544u},
      {0x5u, 1, 9, 11},
      {0x5u, 2, 9, 11},
      {0x5u, 3, 9, 11},
      {0x5u, 4, 9, 11},
      {0x100000000u, 1, 0, 0},
  };
  constexpr int kBits[4] = {2, 34, 40, 64};
  for (int k = 0; k < 200; ++k) {
    WordCase c{};
    c.seed = rng();
    c.domain = 1 + static_cast<std::uint32_t>(rng() % 4);
    c.stream = static_cast<std::uint32_t>(rng());
    const int bits = kBits[rng() % 4];
    c.k = rng() >> (64 - bits);
    cases.push_back(c);
  }
  return cases;
}

// Starts k0 = 0, 1, 2, 3, 2^34 - 6 and 2^64 - 20.
std::vector<WordCase> make_sequence_cases() {
  return {
      {0x0123456789ABCDEFu, 1, 0, 0},
      {0x2Au, 2, 7, 1},
      {0xFFFFFFFFFFFFFFFFu, 3, 0xFFFFFFFFu, 2},
      {0x9E3779B97F4A7C15u, 4, 123456789u, 3},
      {0x243F6A8885A308D3u, 1, 3, (std::uint64_t{1} << 34) - 6},
      {0x100000000u, 2, 0x80000000u, 0xFFFFFFFFFFFFFFECu},
  };
}

// k = 0, 4, 2^34 and 2^34 + 8.
std::vector<WordCase> make_block_cases() {
  return {
      {0x5u, 1, 9, 0},
      {0x2Au, 2, 1, 4},
      {0x243F6A8885A308D3u, 3, 0xFFFFFFFFu, std::uint64_t{1} << 34},
      {0xFFFFFFFFFFFFFFFFu, 4, 77, (std::uint64_t{1} << 34) + 8},
  };
}

// The three Philox4x32-10 vectors of Random123 (tests/kat_vectors).
std::vector<KatCase> make_kat_cases() {
  return {
      {{0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u}},
      {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu}},
      {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u}},
  };
}

std::vector<PositionCase> make_position_cases(std::mt19937_64& rng) {
  // The 9 reference cases of tests/core/test_initial_state.cpp.
  std::vector<PositionCase> cases = {
      {0x2Au, 0, 0},     {0x2Au, 0, 1},
      {0x2Au, 0, 2},     {0x2Au, 0, 3},
      {0x2Au, 1, 0},     {0x2Au, 1023, 149999},
      {0x0u, 0, 0},      {0xFFFFFFFFFFFFFFFFu, 4294967295u, 2147483646u},
      {0x7u, 5, 123456},
  };
  for (int k = 0; k < 1000; ++k) {
    PositionCase c{};
    c.seed = rng();
    c.replica = static_cast<std::uint32_t>(rng());
    c.spin = static_cast<std::uint32_t>(rng() >> 33);  // i < 2^31
    cases.push_back(c);
  }
  return cases;
}

// --- Reports -------------------------------------------------------------------------------

std::string word_case_text(const char* label, std::size_t index, const WordCase& c) {
  char text[160];
  std::snprintf(text, sizeof(text), "%s[%zu] seed=0x%016llx domain=%u stream=0x%08x k=0x%016llx",
                label, index, static_cast<unsigned long long>(c.seed), c.domain, c.stream,
                static_cast<unsigned long long>(c.k));
  return text;
}

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08x", value);
  return text;
}

std::string hex_block(const U32x4& v) {
  return "{" + hex32(v.x) + "," + hex32(v.y) + "," + hex32(v.z) + "," + hex32(v.w) + "}";
}

class Check {
 public:
  explicit Check(const char* key) : key_(key) {}

  void add(bool equal, const std::string& arguments, const std::string& expected,
           const std::string& got) {
    ++cases_;
    if (!equal) {
      ++different_;
      if (examples_.size() < kReportedCases) {
        examples_.push_back(arguments + ": expected " + expected + ", got " + got);
      }
    }
  }

  [[nodiscard]] bool passed() const noexcept { return cases_ > 0 && different_ == 0; }

  void print() const {
    std::printf("%s: %s (%zu/%zu)\n", key_, passed() ? "PASS" : "FAIL", different_, cases_);
    for (const std::string& example : examples_) {
      std::printf("  %s\n", example.c_str());
    }
  }

 private:
  const char* key_;
  std::size_t cases_ = 0;
  std::size_t different_ = 0;
  std::vector<std::string> examples_;
};

std::uint32_t host_word(const WordCase& c, std::uint64_t offset = 0) {
  return sbmc::stream_word(c.seed, static_cast<RngDomain>(c.domain), c.stream, c.k + offset);
}

}  // namespace

int main() {
  sbmc::smoke::set_verdict_key("T11_GPU_VERDICT");
  (void)sbmc::smoke::require_cuda_devices();
  SMOKE_CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  SMOKE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  std::mt19937_64 rng(kCaseSeed);
  const std::vector<WordCase> words = make_word_cases(rng);
  const std::vector<WordCase> sequences = make_sequence_cases();
  const std::vector<WordCase> blocks = make_block_cases();
  const std::vector<KatCase> kats = make_kat_cases();
  const std::vector<PositionCase> positions = make_position_cases(rng);

  std::vector<std::uint32_t> curand_words(words.size());
  std::vector<std::uint32_t> curand_sequences(sequences.size() * kSequenceLength);
  std::vector<U32x4> curand_blocks(blocks.size());
  std::vector<std::uint32_t> own_words(words.size());
  std::vector<U32x4> own_kats(kats.size());
  std::vector<float> device_positions(positions.size());
  {
    DeviceBuffer<WordCase> d_words(words.size());
    DeviceBuffer<WordCase> d_sequences(sequences.size());
    DeviceBuffer<WordCase> d_blocks(blocks.size());
    DeviceBuffer<KatCase> d_kats(kats.size());
    DeviceBuffer<PositionCase> d_positions(positions.size());
    d_words.upload(words);
    d_sequences.upload(sequences);
    d_blocks.upload(blocks);
    d_kats.upload(kats);
    d_positions.upload(positions);

    DeviceBuffer<std::uint32_t> d_curand_words(curand_words.size());
    DeviceBuffer<std::uint32_t> d_curand_sequences(curand_sequences.size());
    DeviceBuffer<U32x4> d_curand_blocks(curand_blocks.size());
    DeviceBuffer<std::uint32_t> d_own_words(own_words.size());
    DeviceBuffer<U32x4> d_own_kats(own_kats.size());
    DeviceBuffer<float> d_x0(device_positions.size());

    curand_word_kernel<<<grid_for(words.size()), kThreadsPerBlock>>>(
        d_words.data(), d_curand_words.data(), words.size());
    SMOKE_CUDA_CHECK_LAUNCH();
    curand_sequence_kernel<<<grid_for(sequences.size()), kThreadsPerBlock>>>(
        d_sequences.data(), d_curand_sequences.data(), sequences.size());
    SMOKE_CUDA_CHECK_LAUNCH();
    curand4_kernel<<<grid_for(blocks.size()), kThreadsPerBlock>>>(
        d_blocks.data(), d_curand_blocks.data(), blocks.size());
    SMOKE_CUDA_CHECK_LAUNCH();
    own_word_kernel<<<grid_for(words.size()), kThreadsPerBlock>>>(d_words.data(),
                                                                  d_own_words.data(), words.size());
    SMOKE_CUDA_CHECK_LAUNCH();
    own_kat_kernel<<<grid_for(kats.size()), kThreadsPerBlock>>>(d_kats.data(), d_own_kats.data(),
                                                                kats.size());
    SMOKE_CUDA_CHECK_LAUNCH();
    x0_kernel<<<grid_for(positions.size()), kThreadsPerBlock>>>(d_positions.data(), d_x0.data(),
                                                                positions.size());
    SMOKE_CUDA_CHECK_LAUNCH();

    d_curand_words.download(curand_words);
    d_curand_sequences.download(curand_sequences);
    d_curand_blocks.download(curand_blocks);
    d_own_words.download(own_words);
    d_own_kats.download(own_kats);
    d_x0.download(device_positions);
  }

  Check curand_word("T11_CURAND_WORD");
  for (std::size_t c = 0; c < words.size(); ++c) {
    const std::uint32_t expected = host_word(words[c]);
    curand_word.add(curand_words[c] == expected, word_case_text("W", c, words[c]), hex32(expected),
                    hex32(curand_words[c]));
  }

  Check curand_sequence("T11_CURAND_SEQUENCE");
  for (std::size_t c = 0; c < sequences.size(); ++c) {
    for (int q = 0; q < kSequenceLength; ++q) {
      const std::uint32_t expected = host_word(sequences[c], static_cast<std::uint64_t>(q));
      const std::uint32_t got = curand_sequences[c * kSequenceLength + static_cast<std::size_t>(q)];
      curand_sequence.add(got == expected,
                          word_case_text("S", c, sequences[c]) + " t=" + std::to_string(q),
                          hex32(expected), hex32(got));
    }
  }

  Check curand4_check("T11_CURAND4");
  for (std::size_t c = 0; c < blocks.size(); ++c) {
    const WordCase& b = blocks[c];
    const U32x4 expected =
        sbmc::stream_block(b.seed, static_cast<RngDomain>(b.domain), b.stream, b.k / 4);
    curand4_check.add(curand_blocks[c] == expected, word_case_text("B", c, b), hex_block(expected),
                      hex_block(curand_blocks[c]));
  }

  Check own_philox("T11_OWN_PHILOX_DEVICE");
  for (std::size_t c = 0; c < words.size(); ++c) {
    const std::uint32_t expected = host_word(words[c]);
    own_philox.add(own_words[c] == expected, word_case_text("W", c, words[c]), hex32(expected),
                   hex32(own_words[c]));
  }
  for (std::size_t c = 0; c < kats.size(); ++c) {
    const U32x4 expected = sbmc::philox4x32_10(kats[c].ctr, kats[c].key);
    own_philox.add(own_kats[c] == expected,
                   "KAT[" + std::to_string(c) + "] ctr=" + hex_block(kats[c].ctr) + " key={" +
                       hex32(kats[c].key.x) + "," + hex32(kats[c].key.y) + "}",
                   hex_block(expected), hex_block(own_kats[c]));
  }

  Check x0_device("T11_X0_DEVICE");
  for (std::size_t c = 0; c < positions.size(); ++c) {
    const PositionCase& p = positions[c];
    const auto expected =
        std::bit_cast<std::uint32_t>(sbmc::initial_position(p.seed, p.replica, p.spin));
    const auto got = std::bit_cast<std::uint32_t>(device_positions[c]);
    char arguments[128];
    std::snprintf(arguments, sizeof(arguments), "X[%zu] seed=0x%016llx r=%u i=%u", c,
                  static_cast<unsigned long long>(p.seed), p.replica, p.spin);
    x0_device.add(got == expected, arguments, hex32(expected), hex32(got));
  }

  const Check* const checks[] = {&curand_word, &curand_sequence, &curand4_check, &own_philox,
                                 &x0_device};
  bool all_passed = true;
  for (const Check* check : checks) {
    check->print();
    all_passed = all_passed && check->passed();
  }
  std::printf("T11_GPU_VERDICT: %s\n", all_passed ? "PASS" : "FAIL");
  return all_passed ? sbmc::smoke::kExitPass : sbmc::smoke::kExitFail;
}
