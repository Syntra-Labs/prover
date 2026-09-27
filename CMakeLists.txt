cmake_minimum_required(VERSION 3.5)

project(lawful LANGUAGES CXX)

# Creates the library where python module will be placed.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Set python paths.
# Cmake's find() was not able to find them.
set(Python3_EXECUTABLE "/usr/bin/python3")
set(Python3_INCLUDE_DIR "/usr/include/python3.12")
set(Python3_LIBRARY "/lib/python3.12")

set(SOURCE
    tests.cpp
    symbolic.cpp
    symbolic.hpp
)

# Automatically locate pybind11
find_package( pybind11 CONFIG REQUIRED )

target_link_libraries(lawful PUBLIC pybind::embed)
