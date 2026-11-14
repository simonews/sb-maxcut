// Shared helpers for the CUDA smoke tests (included by .cu files only).
//
// Output convention: "key: value" lines followed by a final verdict line. Exit codes: 0 passed,
// 1 failed, 77 skipped (no usable device), registered in CTest as SKIP_RETURN_CODE.
//
// The project-wide error-checking macros (CUDA_CHECK and friends) arrive in phase 08; these
// local versions only serve the smoke tests.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>

namespace sbmc::smoke {

inline constexpr int kExitPass = 0;
inline constexpr int kExitFail = 1;
inline constexpr int kExitSkip = 77;

// Key of the verdict line printed by fail_and_exit() and skip_and_exit(), e.g. "SIMT_VERDICT".
inline const char* g_verdict_key = "VERDICT";

inline void set_verdict_key(const char* key) noexcept {
  g_verdict_key = key;
}

[[noreturn]] inline void fail_and_exit(const char* library, int code, const char* name,
                                       const char* message, const char* expr, const char* file,
                                       int line) {
  std::fflush(stdout);
  std::fprintf(stderr, "%s:%d: %s error %d (%s: %s) in '%s'\n", file, line, library, code, name,
               message, expr);
  std::printf("%s: FAIL\n", g_verdict_key);
  std::fflush(stdout);
  std::exit(kExitFail);
}

[[noreturn]] inline void skip_and_exit(const char* reason) {
  std::printf("skip_reason: %s\n", reason);
  std::printf("%s: SKIP\n", g_verdict_key);
  std::fflush(stdout);
  std::exit(kExitSkip);
}

// Exits with 77 when no CUDA device is usable (no driver, no device, driver too old, ...).
// Returns the number of devices otherwise.
inline int require_cuda_devices() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    char reason[256];
    std::snprintf(reason, sizeof(reason), "cudaGetDeviceCount failed with %s (%s)",
                  cudaGetErrorName(err), cudaGetErrorString(err));
    skip_and_exit(reason);
  }
  if (count == 0) {
    skip_and_exit("cudaGetDeviceCount returned 0 devices");
  }
  return count;
}

}  // namespace sbmc::smoke

#define SMOKE_CUDA_CHECK(expr)                                                                   \
  do {                                                                                           \
    const cudaError_t smoke_err_ = (expr);                                                       \
    if (smoke_err_ != cudaSuccess) {                                                             \
      ::sbmc::smoke::fail_and_exit("CUDA", static_cast<int>(smoke_err_),                         \
                                   cudaGetErrorName(smoke_err_), cudaGetErrorString(smoke_err_), \
                                   #expr, __FILE__, __LINE__);                                   \
    }                                                                                            \
  } while (0)

// Checks a launch: configuration errors right away, execution errors after synchronization.
#define SMOKE_CUDA_CHECK_LAUNCH()              \
  do {                                         \
    SMOKE_CUDA_CHECK(cudaGetLastError());      \
    SMOKE_CUDA_CHECK(cudaDeviceSynchronize()); \
  } while (0)

// For cuBLAS and cuBLASLt (both return cublasStatus_t); usable where cuBLAS is included.
#define SMOKE_CUBLAS_CHECK(expr)                                                         \
  do {                                                                                   \
    const cublasStatus_t smoke_status_ = (expr);                                         \
    if (smoke_status_ != CUBLAS_STATUS_SUCCESS) {                                        \
      ::sbmc::smoke::fail_and_exit(                                                      \
          "cuBLAS", static_cast<int>(smoke_status_), cublasGetStatusName(smoke_status_), \
          cublasGetStatusString(smoke_status_), #expr, __FILE__, __LINE__);              \
    }                                                                                    \
  } while (0)

namespace sbmc::smoke {

// Owning device buffer (RAII): allocated with cudaMalloc, released with cudaFree.
template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    SMOKE_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        cudaFree(ptr_);
      }
      ptr_ = std::exchange(other.ptr_, nullptr);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  [[nodiscard]] T* data() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

  void upload(std::span<const T> host) {
    if (host.size() != count_) {
      std::fprintf(stderr, "DeviceBuffer::upload: size mismatch (%zu vs %zu)\n", host.size(),
                   count_);
      std::printf("%s: FAIL\n", g_verdict_key);
      std::exit(kExitFail);
    }
    SMOKE_CUDA_CHECK(cudaMemcpy(ptr_, host.data(), bytes(), cudaMemcpyHostToDevice));
  }

  void download(std::span<T> host) const {
    if (host.size() != count_) {
      std::fprintf(stderr, "DeviceBuffer::download: size mismatch (%zu vs %zu)\n", host.size(),
                   count_);
      std::printf("%s: FAIL\n", g_verdict_key);
      std::exit(kExitFail);
    }
    SMOKE_CUDA_CHECK(cudaMemcpy(host.data(), ptr_, bytes(), cudaMemcpyDeviceToHost));
  }

 private:
  T* ptr_ = nullptr;
  std::size_t count_ = 0;
};

}  // namespace sbmc::smoke
