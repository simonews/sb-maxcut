cmake_minimum_required(VERSION 3.28)

project(sbmc VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SBMC_ENABLE_CUDA "Build the CUDA components (requires a CUDA toolkit)" OFF)
option(SBMC_ENABLE_TILE
       "Build the CUDA Tile C++ components (CUDA >= 13.3, sm_89 only; needs SBMC_ENABLE_CUDA)" ON)
option(SBMC_ENABLE_MPI "MPI multi-GPU driver (not available yet)" OFF)
option(SBMC_ENABLE_NCCL "NCCL all-gather of the signs (not available yet)" OFF)
option(SBMC_STRICT_FP "No FMA contraction on host and device (bit-for-bit equality builds)" OFF)

if(SBMC_ENABLE_MPI)
  message(FATAL_ERROR "SBMC_ENABLE_MPI=ON: not available yet: planned for phase 16/17")
endif()
if(SBMC_ENABLE_NCCL)
  message(FATAL_ERROR "SBMC_ENABLE_NCCL=ON: not available yet: planned for phase 16/17")
endif()

if(SBMC_ENABLE_CUDA)
  if(DEFINED CMAKE_CUDA_COMPILER AND IS_ABSOLUTE "${CMAKE_CUDA_COMPILER}"
     AND NOT EXISTS "${CMAKE_CUDA_COMPILER}")
    message(FATAL_ERROR
      "SBMC_ENABLE_CUDA=ON but CMAKE_CUDA_COMPILER does not exist: ${CMAKE_CUDA_COMPILER}\n"
      "Run scripts/setup_cuda_pypi.sh to install the pinned toolkit and regenerate "
      "CMakeUserPresets.json, then configure with the 'cuda' or 'cuda-strict' preset.")
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "SBMC_ENABLE_CUDA=ON but no CUDA compiler was found.\n"
      "Run scripts/setup_cuda_pypi.sh to install the pinned toolkit from PyPI, then configure "
      "with the generated 'cuda' or 'cuda-strict' preset (or pass -DCMAKE_CUDA_COMPILER=...).")
  endif()

  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # Every CUDA target links the shared runtime explicitly (CUDA::cudart): no implicit runtime,
  # so the static and the shared cudart are never mixed in one executable.
  set(CMAKE_CUDA_RUNTIME_LIBRARY None)

  if(SBMC_ENABLE_TILE AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.3)
    message(FATAL_ERROR
      "SBMC_ENABLE_TILE=ON requires CUDA 13.3 or newer, found ${CMAKE_CUDA_COMPILER_VERSION}.\n"
      "Use the pinned toolkit (scripts/setup_cuda_pypi.sh) or configure with "
      "-DSBMC_ENABLE_TILE=OFF.")
  endif()
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

include(cmake/SbmcFlags.cmake)

add_subdirectory(src/core)

enable_testing()
add_subdirectory(tests)

# Configuration summary.
if(SBMC_ENABLE_CUDA)
  set(_sbmc_cuda_compiler
      "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER})")
  set(_sbmc_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
  if(SBMC_ENABLE_TILE)
    string(APPEND _sbmc_cuda_archs " (Tile targets: 89)")
  endif()
  set(_sbmc_tile_state "${SBMC_ENABLE_TILE}")
else()
  set(_sbmc_cuda_compiler "disabled")
  set(_sbmc_cuda_archs "n/a")
  set(_sbmc_tile_state "${SBMC_ENABLE_TILE} (no effect without SBMC_ENABLE_CUDA)")
endif()
message(STATUS "")
message(STATUS "sbmc configuration summary")
message(STATUS "  project version    : ${PROJECT_VERSION}")
message(STATUS "  C++ compiler       : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
message(STATUS "  CUDA compiler      : ${_sbmc_cuda_compiler}")
message(STATUS "  CUDA architectures : ${_sbmc_cuda_archs}")
message(STATUS "  SBMC_ENABLE_CUDA   : ${SBMC_ENABLE_CUDA}")
message(STATUS "  SBMC_ENABLE_TILE   : ${_sbmc_tile_state}")
message(STATUS "  SBMC_ENABLE_MPI    : ${SBMC_ENABLE_MPI}")
message(STATUS "  SBMC_ENABLE_NCCL   : ${SBMC_ENABLE_NCCL}")
message(STATUS "  SBMC_STRICT_FP     : ${SBMC_STRICT_FP}")
message(STATUS "")
