// CUDA Tile C++ smoke test (sm_89 only): vector addition and an INT8 product with ct::mma.
//
// Answers the question "does CUDA Tile C++ compile and run correctly on this machine, INT8
// ct::mma included?". Output: "key: value" lines, TILE_VECADD, TILE_MMA_INT8_FULLRANGE and
// TILE_MMA_INT8_TERNARY (PASS|FAIL), then "TILE_VERDICT: OK|FAIL" (SKIP when no suitable
// device). Exit codes: 0 OK, 1 FAIL, 77 skipped.
//
// Tile kernels are launched as kernel<<<blocks, 1>>>: the compiler chooses the threads of a
// block, and the block size argument must be 1 (CUDA Tile C++ samples and blog).
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "smoke_common.cuh"

namespace {

constexpr std::uint32_t kSeed = 20261113u;

// Vector addition: kVectorSize is a multiple of the tile size.
constexpr std::size_t kVectorSize = std::size_t{1} << 20;
constexpr int kVectorTile = 1024;
static_assert(kVectorSize % kVectorTile == 0);

// INT8 product C = A * B, row-major operands, tiles of kTileM x kTileK and kTileK x kTileN.
constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kK = 128;
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 32;
static_assert(kM % kTileM == 0 && kN % kTileN == 0 && kK % kTileK == 0);
static_assert(kK / kTileK >= 2, "the K loop must run at least two iterations");

}  // namespace

// Follows the official vector addition sample: tensor_span, partition_view, masked load/store.
__tile_global__ void tile_vector_add(const float* __restrict__ a_in, const float* __restrict__ b_in,
                                     float* __restrict__ c_out, std::size_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const float* a = ct::assume_aligned(a_in, 16_ic);
  const float* b = ct::assume_aligned(b_in, 16_ic);
  float* c = ct::assume_aligned(c_out, 16_ic);

  const auto block = ct::bid().x;
  const ct::tensor_span a_span{a, ct::extents{n}};
  const ct::tensor_span b_span{b, ct::extents{n}};
  const ct::tensor_span c_span{c, ct::extents{n}};
  const auto a_view = ct::partition_view{a_span, ct::shape<kVectorTile>{}};
  const auto b_view = ct::partition_view{b_span, ct::shape<kVectorTile>{}};
  const auto c_view = ct::partition_view{c_span, ct::shape<kVectorTile>{}};

  const auto a_tile = a_view.load_masked(block);
  const auto b_tile = b_view.load_masked(block);
  c_view.store_masked(a_tile + b_tile, block);
}

// One block per kTileM x kTileN output tile; block (x, y) computes rows x * kTileM and columns
// y * kTileN. The K loop carries only the accumulator produced by ct::mma: the A and B tiles are
// loaded and consumed in the same iteration (no manual prefetch), as required by the known
// issue on loop-carried loaded tiles in Tile IR 13.4.
__tile_global__ void tile_mma_int8(const std::int8_t* __restrict__ a_in,
                                   const std::int8_t* __restrict__ b_in,
                                   std::int32_t* __restrict__ c_out, int m, int n, int k) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const std::int8_t* a = ct::assume_aligned(a_in, 16_ic);
  const std::int8_t* b = ct::assume_aligned(b_in, 16_ic);
  std::int32_t* c = ct::assume_aligned(c_out, 16_ic);

  // Default layout (layout_right): row-major, A[i][p] at i * k + p, B[p][j] at p * n + j.
  const auto a_span = ct::tensor_span{a, ct::extents{m, k}};
  const auto b_span = ct::tensor_span{b, ct::extents{k, n}};
  const auto c_span = ct::tensor_span{c, ct::extents{m, n}};
  const auto a_view = ct::partition_view{a_span, ct::shape<kTileM, kTileK>{}};
  const auto b_view = ct::partition_view{b_span, ct::shape<kTileK, kTileN>{}};
  const auto c_view = ct::partition_view{c_span, ct::shape<kTileM, kTileN>{}};

  const auto block = ct::bid();
  auto acc = ct::zeros<ct::tile<std::int32_t, ct::shape<kTileM, kTileN>>>();
  for (auto kb : ct::irange(0, k / kTileK)) {
    const auto a_tile = a_view.load(block.x, kb);
    const auto b_tile = b_view.load(kb, block.y);
    acc = ct::mma(a_tile, b_tile, acc);
  }
  c_view.store(acc, block.x, block.y);
}

namespace {

using sbmc::smoke::DeviceBuffer;

bool run_vector_add(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a(kVectorSize);
  std::vector<float> b(kVectorSize);
  std::vector<float> expected(kVectorSize);
  for (std::size_t i = 0; i < kVectorSize; ++i) {
    a[i] = dist(rng);
    b[i] = dist(rng);
    expected[i] = a[i] + b[i];
  }

  DeviceBuffer<float> d_a(kVectorSize);
  DeviceBuffer<float> d_b(kVectorSize);
  DeviceBuffer<float> d_c(kVectorSize);
  d_a.upload(a);
  d_b.upload(b);
  SMOKE_CUDA_CHECK(cudaMemset(d_c.data(), 0xFF, d_c.bytes()));  // NaN pattern: catches no-ops

  const auto blocks = static_cast<unsigned>(kVectorSize / kVectorTile);
  tile_vector_add<<<blocks, 1>>>(d_a.data(), d_b.data(), d_c.data(), kVectorSize);
  SMOKE_CUDA_CHECK_LAUNCH();

  std::vector<float> c(kVectorSize);
  d_c.download(c);
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < kVectorSize; ++i) {
    if (std::bit_cast<std::uint32_t>(c[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
      ++mismatches;
    }
  }
  std::printf("tile_vector_add_n: %zu (tile %d, %u blocks)\n", kVectorSize, kVectorTile, blocks);
  std::printf("tile_vector_add_mismatches: %zu\n", mismatches);
  std::printf("TILE_VECADD: %s\n", mismatches == 0 ? "PASS" : "FAIL");
  return mismatches == 0;
}

// Runs C = A * B on the device and compares it exactly with an int64 host reference.
bool run_mma_int8(const char* label, std::uniform_int_distribution<int> dist, std::mt19937& rng) {
  std::vector<std::int8_t> a(static_cast<std::size_t>(kM) * kK);  // row-major
  std::vector<std::int8_t> b(static_cast<std::size_t>(kK) * kN);  // row-major
  for (auto& value : a) {
    value = static_cast<std::int8_t>(dist(rng));
  }
  for (auto& value : b) {
    value = static_cast<std::int8_t>(dist(rng));
  }

  std::vector<std::int64_t> expected(static_cast<std::size_t>(kM) * kN, 0);
  for (int i = 0; i < kM; ++i) {
    for (int p = 0; p < kK; ++p) {
      const auto a_ip = static_cast<std::int64_t>(a[static_cast<std::size_t>(i) * kK + p]);
      for (int j = 0; j < kN; ++j) {
        expected[static_cast<std::size_t>(i) * kN + j] +=
            a_ip * static_cast<std::int64_t>(b[static_cast<std::size_t>(p) * kN + j]);
      }
    }
  }

  DeviceBuffer<std::int8_t> d_a(a.size());
  DeviceBuffer<std::int8_t> d_b(b.size());
  DeviceBuffer<std::int32_t> d_c(expected.size());
  d_a.upload(a);
  d_b.upload(b);
  SMOKE_CUDA_CHECK(cudaMemset(d_c.data(), 0x7F, d_c.bytes()));  // garbage: catches no-ops

  const dim3 grid(kM / kTileM, kN / kTileN);
  tile_mma_int8<<<grid, 1>>>(d_a.data(), d_b.data(), d_c.data(), kM, kN, kK);
  SMOKE_CUDA_CHECK_LAUNCH();

  std::vector<std::int32_t> c(expected.size());
  d_c.download(c);
  std::size_t mismatches = 0;
  std::size_t first_bad = c.size();
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (static_cast<std::int64_t>(c[i]) != expected[i]) {
      if (mismatches == 0) {
        first_bad = i;
      }
      ++mismatches;
    }
  }
  std::printf("tile_mma_int8_%s_mismatches: %zu of %zu\n", label, mismatches, c.size());
  if (mismatches != 0) {
    std::printf("tile_mma_int8_%s_first_mismatch: C[%zu][%zu] = %d, expected %lld\n", label,
                first_bad / kN, first_bad % kN, c[first_bad],
                static_cast<long long>(expected[first_bad]));
  }
  return mismatches == 0;
}

}  // namespace

int main() {
  sbmc::smoke::set_verdict_key("TILE_VERDICT");
  (void)sbmc::smoke::require_cuda_devices();

  cudaDeviceProp prop{};
  SMOKE_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device0_name: %s\n", prop.name);
  std::printf("device0_compute_capability: %d.%d\n", prop.major, prop.minor);
  if (prop.major < 8) {
    sbmc::smoke::skip_and_exit("CUDA Tile needs compute capability 8.0 or newer on device 0");
  }
  SMOKE_CUDA_CHECK(cudaSetDevice(0));

  std::mt19937 rng(kSeed);
  const bool vector_add_ok = run_vector_add(rng);

  std::printf("tile_mma_problem: C(int32) = A(int8) * B(int8), M=%d N=%d K=%d, row-major\n", kM, kN,
              kK);
  std::printf("tile_mma_tiles: %dx%d (A) * %dx%d (B), %d K iterations, grid %dx%d\n", kTileM,
              kTileK, kTileK, kTileN, kK / kTileK, kM / kTileM, kN / kTileN);
  const bool full_ok =
      run_mma_int8("fullrange", std::uniform_int_distribution<int>(-128, 127), rng);
  std::printf("TILE_MMA_INT8_FULLRANGE: %s\n", full_ok ? "PASS" : "FAIL");
  const bool ternary_ok = run_mma_int8("ternary", std::uniform_int_distribution<int>(-1, 1), rng);
  std::printf("TILE_MMA_INT8_TERNARY: %s\n", ternary_ok ? "PASS" : "FAIL");

  const bool ok = vector_add_ok && full_ok && ternary_ok;
  std::printf("TILE_VERDICT: %s\n", ok ? "OK" : "FAIL");
  return ok ? sbmc::smoke::kExitPass : sbmc::smoke::kExitFail;
}
