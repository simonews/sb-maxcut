# CPU solvers: sbmc_cpu (alias sbmc::cpu). Phase 04: the sequential reference C0 of simulated
# bifurcation (sb_sequential.cpp), without OpenMP directives of its own.

add_library(sbmc_cpu STATIC sb_sequential.cpp)
add_library(sbmc::cpu ALIAS sbmc_cpu)
target_link_libraries(sbmc_cpu PUBLIC sbmc::core PRIVATE sbmc::flags)
