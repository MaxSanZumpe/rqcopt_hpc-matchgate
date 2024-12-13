cmake_minimum_required(VERSION 3.8)

project(rqcopt_hpc_matchgate)


# Find necessary packages
find_package(MPI REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)

# Print the HDF5 include directories
message(STATUS "HDF5 include directories: ${HDF5_INCLUDE_DIRS}")

# Print the HDF5 libraries
message(STATUS "HDF5 libraries: ${HDF5_C_LIBRARIES}")
 

# Compiler flags
set(CMAKE_C_FLAGS "-O3 -fopenmp")

# Definitions
add_compile_definitions(COMPLEX_CIRCUIT)

# Define source files
set(SOURCES 
    "src/matchgate.c" 
    "src/util.c" 
    "src/gate.c" 
    "src/statevector.c" 
    "src/quantum_circuit.c" 
    "src/matchgate_brickwall.c" 
    "src/matchgate_target.c" 
    "src/mg_brickwall_opt.c" 
    "src/trust_region.c"
)

# Define test sources
set(TEST_SOURCES 
    "test/test_matchgate.c" 
    "test/test_gate.c" 
    "test/numerical_gradient.c" 
    "test/test_matchgate_brickwall.c" 
    "test/test_matchgate_target.c" 
    "test/run_custom_test.c"
)

# add_executable(            spl_hubbard1d ${SOURCES} "examples/spl_hubbard1d/spl_hubbard1d_opt.c" "examples/timing/timing.c")
# target_include_directories(spl_hubbard1d PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(     spl_hubbard1d PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
# target_compile_definitions(spl_hubbard1d PRIVATE TRANSLATIONAL_INVARIANCE STATEVECTOR_PARALLELIZATION)


# add_executable(spinhubbard.exe ${SOURCES} "examples/spinhubbard/spinhubbard_opt.c" "examples/timing/timing.c")
# target_include_directories(spinhubbard.exe PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(spinhubbard.exe PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})


add_executable(            benchmark_threads ${SOURCES} "benchmark_hpc/benchmark_threads.c" "examples/timing/timing.c")
target_include_directories(benchmark_threads PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
target_link_libraries(     benchmark_threads ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
target_compile_definitions(benchmark_threads PRIVATE STATEVECTOR_PARALLELIZATION TRANSLATIONAL_INVARIANCE)


add_executable(		       benchmark_mpi ${SOURCES} "benchmark_hpc/benchmark_mpi.c" "examples/timing/timing.c")
target_include_directories(benchmark_mpi PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} "src" "examples/timing")
target_link_libraries(     benchmark_mpi ${BLAS_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES})
target_compile_definitions(benchmark_mpi PRIVATE MPI STATEVECTOR_PARALLELIZATION TRANSLATIONAL_INVARIANCE)


# add_executable(benchmark_serial.exe ${SOURCES} "benchmark/benchmark_serial.c" "examples/timing/timing.c")
# target_include_directories(benchmark_serial.exe PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(benchmark_serial.exe ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
# # target_compile_definitions(benchmark_serial.exe PRIVATE TRANSLATIONAL_INVARIANCE)

add_executable(            spl_hubbard2d ${SOURCES} "examples/spl_hubbard2d/spl_hubbard2d.c" "examples/timing/timing.c")
target_include_directories(spl_hubbard2d PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${MPI_INCLUDE_PATH} "src" "examples/timing")
target_link_libraries(     spl_hubbard2d ${BLAS_LIBRARIES} ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES})
target_compile_definitions(spl_hubbard2d PRIVATE MPI TRANSLATIONAL_INVARIANCE STATEVECTOR_PARALLELIZATION)


# add_executable(test_2d_invar.exe ${SOURCES} "examples/hubbard2d/test_invariance.c" "examples/timing/timing.c")
# target_include_directories(test_2d_invar.exe PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(test_2d_invar.exe ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
# target_compile_definitions(test_2d_invar.exe PRIVATE TRANSLATIONAL_INVARIANCE)

 
# add_executable(            corr_splh1d ${SOURCES} "examples/spl_hubbard1d/correlations.c" "examples/timing/timing.c")
# target_include_directories(corr_splh1d PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(     corr_splh1d PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
 
# add_executable(            fid_splh1d ${SOURCES} "examples/spl_hubbard1d/fidelity.c" "examples/timing/timing.c")
# target_include_directories(fid_splh1d PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(     fid_splh1d PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})

# add_executable(            bench_grad ${SOURCES} "examples/hubbard1d/benchmark_gradient.c" "examples/timing/timing.c")
# target_include_directories(bench_grad PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(     bench_grad PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})
# 
# add_executable(            bench_hess_partial ${SOURCES} "examples/hubbard1d/benchmark_hess.c" "examples/timing/timing.c")
# target_include_directories(bench_hess_partial PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
# target_link_libraries(     bench_hess_partial PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})


#add_executable(            bench_hess_partial ${SOURCES} "examples/hubbard1d/benchmark_hess.c" "examples/timing/timing.c")
#target_include_directories(bench_hess_partial PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
#target_link_libraries(     bench_hess_partial PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})


add_executable(            splh1d_errors ${SOURCES} "examples/spl_hubbard1d/test_splitting/splitting_error.c" "examples/timing/timing.c")
target_include_directories(splh1d_errors PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
target_link_libraries(     splh1d_errors PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})


add_executable(            splh2d_errors ${SOURCES} "examples/spl_hubbard2d/test_splitting/splitting_error.c" "examples/timing/timing.c")
target_include_directories(splh2d_errors PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
target_link_libraries(     splh2d_errors PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})


add_executable(            h1d_errors ${SOURCES} "examples/hubbard1d/test_splitting/splitting_error.c" "examples/timing/timing.c")
target_include_directories(h1d_errors PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src" "examples/timing")
target_link_libraries(     h1d_errors PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})

add_executable(run_tests ${SOURCES} ${TEST_SOURCES})
target_include_directories(run_tests PRIVATE ${BLAS_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} "src")
target_link_libraries(run_tests PRIVATE ${BLAS_LIBRARIES} ${HDF5_LIBRARIES})

