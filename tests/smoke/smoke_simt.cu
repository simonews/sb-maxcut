// SIMT smoke test: device report, vector addition, cuBLAS SGEMM and a cuBLASLt handle.
//
// Answers the question "do SIMT code and cuBLAS run on this machine with the PyPI toolkit?".
// Built for every architecture of the preset. Output: "key: value" lines, then
// "SIMT_VERDICT: PASS|FAIL" (SKIP when no device is usable). Exit codes: 0, 1, 77.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include "smoke_common.cuh"

namespace {

using sbmc::smoke::DeviceBuffer;

constexpr std::uint32_t kSeed = 20261113u;
constexpr std::size_t kVectorSize = std::size_t{1} << 20;
constexpr int kGemmSize = 256;  // M = N = K
constexpr int kGemmMaxAbsValue = 8;

struct CublasDeleter {
  void operator()(cublasHandle_t handle) const noexcept { cublasDestroy(handle); }
};
using CublasHandle = std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, CublasDeleter>;

struct CublasLtDeleter {
  void operator()(cublasLtHandle_t handle) const noexcept { cublasLtDestroy(handle); }
};
using CublasLtHandle = std::unique_ptr<std::remove_pointer_t<cublasLtHandle_t>, CublasLtDeleter>;

__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// CUDA encodes versions as 1000 * major + 10 * minor. A failed query prints "n/a" instead of
// failing: without a driver the test must still reach the device check and exit with 77.
void print_version(const char* key, cudaError_t status, int encoded) {
  if (status != cudaSuccess) {
    (void)cudaGetLastError();
    std::printf("%s: n/a (%s)\n", key, cudaGetErrorName(status));
    return;
  }
  std::printf("%s: %d (%d.%d)\n", key, encoded, encoded / 1000, (encoded % 1000) / 10);
}

// Prints an integer device attribute, or "n/a" when the runtime cannot report it.
void print_attribute(int device, const char* key, cudaDeviceAttr attribute) {
  int value = 0;
  if (cudaDeviceGetAttribute(&value, attribute, device) != cudaSuccess) {
    (void)cudaGetLastError();  // clear it, so that later launch checks do not report it
    std::printf("device%d_%s: n/a\n", device, key);
    return;
  }
  std::printf("device%d_%s: %d\n", device, key, value);
}

void print_device(int device) {
  cudaDeviceProp prop{};
  SMOKE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  std::printf("device%d_name: %s\n", device, prop.name);
  std::printf("device%d_compute_capability: %d.%d\n", device, prop.major, prop.minor);
  print_attribute(device, "sm_count", cudaDevAttrMultiProcessorCount);
  print_attribute(device, "l2_cache_bytes", cudaDevAttrL2CacheSize);
  std::printf("device%d_global_memory_bytes: %zu\n", device, prop.totalGlobalMem);
  print_attribute(device, "memory_bus_width_bits", cudaDevAttrGlobalMemoryBusWidth);
  print_attribute(device, "sm_clock_khz", cudaDevAttrClockRate);
  print_attribute(device, "memory_clock_khz", cudaDevAttrMemoryClockRate);
  char bus_id[64] = {};
  if (cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), device) == cudaSuccess) {
    std::printf("device%d_pci_bus_id: %s\n", device, bus_id);
  } else {
    (void)cudaGetLastError();
    std::printf("device%d_pci_bus_id: n/a\n", device);
  }
}

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

  constexpr unsigned kThreads = 256;
  const auto blocks = static_cast<unsigned>((kVectorSize + kThreads - 1) / kThreads);
  vector_add_kernel<<<blocks, kThreads>>>(d_a.data(), d_b.data(), d_c.data(), kVectorSize);
  SMOKE_CUDA_CHECK_LAUNCH();

  std::vector<float> c(kVectorSize);
  d_c.download(c);
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < kVectorSize; ++i) {
    if (std::bit_cast<std::uint32_t>(c[i]) != std::bit_cast<std::uint32_t>(expected[i])) {
      ++mismatches;
    }
  }
  std::printf("simt_vector_add_n: %zu\n", kVectorSize);
  std::printf("simt_vector_add_mismatches: %zu\n", mismatches);
  std::printf("SIMT_VECTOR_ADD: %s\n", mismatches == 0 ? "PASS" : "FAIL");
  return mismatches == 0;
}

// C = A * B with small integers: every partial sum is an integer below 2^24 in magnitude, hence
// exact in FP32 whatever the summation order, and the result must match an int64 reference.
bool run_sgemm(cublasHandle_t handle, std::mt19937& rng) {
  constexpr int m = kGemmSize;
  constexpr int n = kGemmSize;
  constexpr int k = kGemmSize;
  std::uniform_int_distribution<int> dist(-kGemmMaxAbsValue, kGemmMaxAbsValue);
  std::vector<float> a(static_cast<std::size_t>(m) * k);  // column-major, lda = m
  std::vector<float> b(static_cast<std::size_t>(k) * n);  // column-major, ldb = k
  for (auto& value : a) {
    value = static_cast<float>(dist(rng));
  }
  for (auto& value : b) {
    value = static_cast<float>(dist(rng));
  }

  std::vector<std::int64_t> expected(static_cast<std::size_t>(m) * n, 0);
  for (int j = 0; j < n; ++j) {
    for (int p = 0; p < k; ++p) {
      const auto b_pj = static_cast<std::int64_t>(b[p + static_cast<std::size_t>(j) * k]);
      for (int i = 0; i < m; ++i) {
        expected[i + static_cast<std::size_t>(j) * m] +=
            static_cast<std::int64_t>(a[i + static_cast<std::size_t>(p) * m]) * b_pj;
      }
    }
  }

  DeviceBuffer<float> d_a(a.size());
  DeviceBuffer<float> d_b(b.size());
  DeviceBuffer<float> d_c(expected.size());
  d_a.upload(a);
  d_b.upload(b);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  SMOKE_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_a.data(), m,
                                 d_b.data(), k, &beta, d_c.data(), m));
  SMOKE_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> c(expected.size());
  d_c.download(c);
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i] != static_cast<float>(expected[i]) || static_cast<std::int64_t>(c[i]) != expected[i]) {
      ++mismatches;
    }
  }
  std::printf("cublas_sgemm_shape: %dx%dx%d, values in [-%d, %d]\n", m, n, k, kGemmMaxAbsValue,
              kGemmMaxAbsValue);
  std::printf("cublas_sgemm_mismatches: %zu\n", mismatches);
  std::printf("CUBLAS_SGEMM: %s\n", mismatches == 0 ? "PASS" : "FAIL");
  return mismatches == 0;
}

}  // namespace

int main() {
  sbmc::smoke::set_verdict_key("SIMT_VERDICT");

  int runtime_version = 0;
  int driver_version = 0;
  const cudaError_t runtime_status = cudaRuntimeGetVersion(&runtime_version);
  print_version("cuda_runtime_version", runtime_status, runtime_version);
  const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);
  print_version("cuda_driver_version", driver_status, driver_version);

  int cublas_major = 0;
  int cublas_minor = 0;
  int cublas_patch = 0;
  SMOKE_CUBLAS_CHECK(cublasGetProperty(MAJOR_VERSION, &cublas_major));
  SMOKE_CUBLAS_CHECK(cublasGetProperty(MINOR_VERSION, &cublas_minor));
  SMOKE_CUBLAS_CHECK(cublasGetProperty(PATCH_LEVEL, &cublas_patch));
  std::printf("cublas_version: %d.%d.%d\n", cublas_major, cublas_minor, cublas_patch);
  std::printf("cublaslt_version: %zu\n", cublasLtGetVersion());

  const int device_count = sbmc::smoke::require_cuda_devices();
  std::printf("device_count: %d\n", device_count);
  for (int device = 0; device < device_count; ++device) {
    print_device(device);
  }

  if (device_count >= 2) {
    for (int from = 0; from < device_count; ++from) {
      for (int to = 0; to < device_count; ++to) {
        if (from == to) {
          continue;
        }
        int can_access = 0;
        SMOKE_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, from, to));
        std::printf("p2p_access_%d_to_%d: %s\n", from, to, can_access != 0 ? "yes" : "no");
      }
    }
  }

  SMOKE_CUDA_CHECK(cudaSetDevice(0));
  std::mt19937 rng(kSeed);
  const bool vector_add_ok = run_vector_add(rng);

  cublasHandle_t raw_cublas = nullptr;
  SMOKE_CUBLAS_CHECK(cublasCreate(&raw_cublas));
  const CublasHandle cublas(raw_cublas);
  const bool sgemm_ok = run_sgemm(cublas.get(), rng);

  cublasLtHandle_t raw_lt = nullptr;
  SMOKE_CUBLAS_CHECK(cublasLtCreate(&raw_lt));
  CublasLtHandle lt(raw_lt);
  SMOKE_CUBLAS_CHECK(cublasLtDestroy(lt.release()));
  std::printf("CUBLASLT_HANDLE: PASS\n");

  const bool ok = vector_add_ok && sgemm_ok;
  std::printf("SIMT_VERDICT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? sbmc::smoke::kExitPass : sbmc::smoke::kExitFail;
}
