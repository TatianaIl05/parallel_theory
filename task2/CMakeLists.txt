cmake_minimum_required(VERSION 3.10)
project(NumericalIntegration)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(integral task2.c)

target_link_libraries(integral OpenMP::OpenMP_C m)
target_compile_options(integral PRIVATE -O2)

message(STATUS "Project configured for integration")
