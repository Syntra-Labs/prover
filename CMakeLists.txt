include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.5)
project(prover)

check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)

if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} does not support C++11")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/my_bin)

add_executable(prover
    src/process.cpp
    src/parseJsonl.cpp
)
