cmake_minimum_required(VERSION 3.10)
project(SimpleIteration) 

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_executable(simple_iteration task3.cpp)  

target_link_libraries(simple_iteration OpenMP::OpenMP_CXX)
target_compile_options(simple_iteration PRIVATE -O2)

message(STATUS "Project configured")
