# Project-wide compile options, exposed as the INTERFACE target sbmc_flags (alias sbmc::flags).
#
# Every target of the project links sbmc::flags, except the FMA probes, which set --fmad on
# their own. No optimisation or floating-point flag goes into CMAKE_CXX_FLAGS or
# CMAKE_CUDA_FLAGS. The generator expressions select the options per language, so this module
# also works when CUDA is not enabled.
include_guard(GLOBAL)

add_library(sbmc_flags INTERFACE)
add_library(sbmc::flags ALIAS sbmc_flags)

# Warnings: the same set for C++ and for the host compiler of CUDA files.
target_compile_options(sbmc_flags INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall;-Xcompiler=-Wextra;-Xcompiler=-Wpedantic>"
)

# Release optimisation.
target_compile_options(sbmc_flags INTERFACE
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3;-march=native>"
  "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3;-Xcompiler=-march=native>"
)

# FMA contraction, always explicit: off in the strict builds (bit-for-bit equality tests), on in
# the production builds.
if(SBMC_STRICT_FP)
  target_compile_options(sbmc_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off;--fmad=false>"
  )
else()
  target_compile_options(sbmc_flags INTERFACE
    "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=fast>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=fast;--fmad=true>"
  )
endif()
