// FMA-contraction probe for CUDA Tile (sm_89 only).
//
// Compiled twice from this source: fma_probe_tile_fmad_on (--fmad=true) and
// fma_probe_tile_fmad_off (--fmad=false); the host code is always built with
// -ffp-contract=off, so the host reference is never contracted. The probe evaluates the SB
// momentum update, written everywhere with the same order and parentheses,
//     y_new = y + dt * ((-p) * x + c0 * F)
// with a Tile kernel, a SIMT kernel and the host, and counts the results identical bit for bit.
//
// Method check (makes the test fail, exit code 1):
//   fmad_off: the SIMT kernel must match the host on n/n elements, else PROBE_METHOD: BROKEN;
//   fmad_on:  the SIMT kernel must differ on at least one element, else NOT_DISCRIMINATING.
// The Tile outcome (TILE_CONTRACTION) is informative only and never fails the test.
#include <cuda_runtime.h>
#include <cuda_tile.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "smoke_common.cuh"

#ifndef SBMC_PROBE_FMAD_ON
#error "SBMC_PROBE_FMAD_ON must be defined: 1 for the --fmad=true build, 0 for --fmad=false"
#endif

namespace {

constexpr std::uint32_t kSeed = 20261113u;
constexpr std::size_t kSize = std::size_t{1} << 20;
constexpr int kTile = 1024;
static_assert(kSize % kTile == 0);

constexpr float kP = 0.73f;
constexpr float kC0 = 0.0123f;
constexpr float kDt = 1.25f;

}  // namespace

__tile_global__ void probe_update_tile(const float* __restrict__ x_in,
                                       const float* __restrict__ y_in,
                                       const float* __restrict__ f_in, float* __restrict__ out,
                                       float p, float c0, float dt, std::size_t n) {
  namespace ct = cuda::tiles;
  using namespace ct::literals;

  const float* x = ct::assume_aligned(x_in, 16_ic);
  const float* y = ct::assume_aligned(y_in, 16_ic);
  const float* f = ct::assume_aligned(f_in, 16_ic);
  float* y_new = ct::assume_aligned(out, 16_ic);

  const auto block = ct::bid().x;
  const auto x_view = ct::partition_view{ct::tensor_span{x, ct::extents{n}}, ct::shape<kTile>{}};
  const auto y_view = ct::partition_view{ct::tensor_span{y, ct::extents{n}}, ct::shape<kTile>{}};
  const auto f_view = ct::partition_view{ct::tensor_span{f, ct::extents{n}}, ct::shape<kTile>{}};
  const auto out_view =
      ct::partition_view{ct::tensor_span{y_new, ct::extents{n}}, ct::shape<kTile>{}};

  const auto x_tile = x_view.load(block);
  const auto y_tile = y_view.load(block);
  const auto f_tile = f_view.load(block);
  out_view.store(y_tile + dt * ((-p) * x_tile + c0 * f_tile), block);
}

__global__ void probe_update_simt(const float* __restrict__ x, const float* __restrict__ y,
                                  const float* __restrict__ f, float* __restrict__ y_new, float p,
                                  float c0, float dt, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    y_new[i] = y[i] + dt * ((-p) * x[i] + c0 * f[i]);
  }
}

namespace {

std::size_t count_identical(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  std::size_t identical = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::bit_cast<std::uint32_t>(lhs[i]) == std::bit_cast<std::uint32_t>(rhs[i])) {
      ++identical;
    }
  }
  return identical;
}

}  // namespace

int main() {
  constexpr bool kFmadOn = (SBMC_PROBE_FMAD_ON != 0);
  std::printf("PROBE_VARIANT: %s\n", kFmadOn ? "fmad_on" : "fmad_off");
  std::fflush(stdout);
  sbmc::smoke::set_verdict_key("PROBE_VERDICT");
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
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::uniform_int_distribution<int> field(-1000, 1000);
  std::vector<float> x(kSize);
  std::vector<float> y(kSize);
  std::vector<float> f(kSize);
  for (std::size_t i = 0; i < kSize; ++i) {
    x[i] = unit(rng);
    y[i] = unit(rng);
    f[i] = static_cast<float>(field(rng));
  }

  // Host reference: this translation unit is compiled with -ffp-contract=off.
  const float p = kP;
  const float c0 = kC0;
  const float dt = kDt;
  std::vector<float> host(kSize);
  for (std::size_t i = 0; i < kSize; ++i) {
    host[i] = y[i] + dt * ((-p) * x[i] + c0 * f[i]);
  }

  sbmc::smoke::DeviceBuffer<float> d_x(kSize);
  sbmc::smoke::DeviceBuffer<float> d_y(kSize);
  sbmc::smoke::DeviceBuffer<float> d_f(kSize);
  sbmc::smoke::DeviceBuffer<float> d_out(kSize);
  d_x.upload(x);
  d_y.upload(y);
  d_f.upload(f);

  std::vector<float> tile_result(kSize);
  SMOKE_CUDA_CHECK(cudaMemset(d_out.data(), 0xFF, d_out.bytes()));
  probe_update_tile<<<static_cast<unsigned>(kSize / kTile), 1>>>(d_x.data(), d_y.data(), d_f.data(),
                                                                 d_out.data(), p, c0, dt, kSize);
  SMOKE_CUDA_CHECK_LAUNCH();
  d_out.download(tile_result);

  std::vector<float> simt_result(kSize);
  SMOKE_CUDA_CHECK(cudaMemset(d_out.data(), 0xFF, d_out.bytes()));
  constexpr unsigned kThreads = 256;
  probe_update_simt<<<static_cast<unsigned>((kSize + kThreads - 1) / kThreads), kThreads>>>(
      d_x.data(), d_y.data(), d_f.data(), d_out.data(), p, c0, dt, kSize);
  SMOKE_CUDA_CHECK_LAUNCH();
  d_out.download(simt_result);

  const std::size_t tile_identical = count_identical(tile_result, host);
  const std::size_t simt_identical = count_identical(simt_result, host);
  const std::size_t tile_simt_identical = count_identical(tile_result, simt_result);

  std::printf("probe_expression: y_new = y + dt * ((-p) * x + c0 * F)\n");
  std::printf("probe_parameters: n=%zu p=%.9g c0=%.9g dt=%.9g seed=%u\n", kSize,
              static_cast<double>(p), static_cast<double>(c0), static_cast<double>(dt), kSeed);
  std::printf("tile_identical_to_host: %zu/%zu\n", tile_identical, kSize);
  std::printf("simt_identical_to_host: %zu/%zu\n", simt_identical, kSize);
  std::printf("tile_identical_to_simt: %zu/%zu\n", tile_simt_identical, kSize);

  bool method_ok = true;
  if (kFmadOn) {
    method_ok = simt_identical < kSize;
    std::printf("PROBE_METHOD: %s\n", method_ok ? "OK" : "NOT_DISCRIMINATING");
  } else {
    method_ok = simt_identical == kSize;
    std::printf("PROBE_METHOD: %s\n", method_ok ? "OK" : "BROKEN");
  }

  if (tile_identical == kSize) {
    std::printf("TILE_CONTRACTION: none\n");
  } else {
    std::printf("TILE_CONTRACTION: present (%zu elements differ)\n", kSize - tile_identical);
  }

  std::printf("PROBE_VERDICT: %s\n", method_ok ? "PASS" : "FAIL");
  return method_ok ? sbmc::smoke::kExitPass : sbmc::smoke::kExitFail;
}
