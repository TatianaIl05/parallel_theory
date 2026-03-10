cmake_minimum_required(VERSION 3.10)
project(MatrixVectorProduct)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(matrix_vector task1.c)

target_link_libraries(matrix_vector OpenMP::OpenMP_C)
target_compile_options(matrix_vector PRIVATE -O2)

message(STATUS "Project configured for matrix-vector multiplication")
