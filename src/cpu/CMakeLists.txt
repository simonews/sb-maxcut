# CPU solvers: sbmc_cpu (alias sbmc::cpu). Phase 04: the sequential reference C0 of simulated
# bifurcation (sb_sequential.cpp). Phase 05: C1, simulated bifurcation with OpenMP over blocks of
# replicas and SIMD over the replicas of a block (sb_openmp.cpp), and the exact solution by
# Gray-code enumeration C3 (bruteforce.cpp), parallel with OpenMP. Phase 06: the sequential
# reference of simulated annealing (sa_sequential.cpp). OpenMP directives only in sb_openmp.cpp
# and bruteforce.cpp: sb_sequential.cpp and sa_sequential.cpp stay without directives of their
# own.

add_library(sbmc_cpu STATIC sb_sequential.cpp sb_openmp.cpp sa_sequential.cpp bruteforce.cpp)
add_library(sbmc::cpu ALIAS sbmc_cpu)
target_link_libraries(sbmc_cpu PUBLIC sbmc::core PRIVATE sbmc::flags OpenMP::OpenMP_CXX)
