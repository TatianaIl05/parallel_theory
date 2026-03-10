cmake_minimum_required(VERSION 3.10)
project(ParallelIteration)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(scalability task3.cpp)
target_link_libraries(scalability OpenMP::OpenMP_CXX)
target_compile_options(scalability PRIVATE -O2)

add_executable(scheduling scheduling.cpp)
target_link_libraries(scheduling OpenMP::OpenMP_CXX)
target_compile_options(scheduling PRIVATE -O2)

message(STATUS "Project configured")
