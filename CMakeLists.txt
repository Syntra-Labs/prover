cmake_minimum_required(VERSION 3.12)
project(r4_coding_engine CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/my_bin)

# Include C++ categorized headers and project root
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/manifold
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/oracle
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/geometry
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/ir
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/search
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/io
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/cli
)

# 1. Manifold Substrate & Lifecycle Router
add_executable(manifold
    src/cpp/manifold/manifold.cpp
)

# 2. Geometry & Rotor Engine
add_library(tree_rotor
    src/cpp/geometry/tree_rotor.cpp
)

add_library(tree_rotor_shared SHARED
    src/cpp/geometry/tree_rotor.cpp
)
set_target_properties(tree_rotor_shared PROPERTIES OUTPUT_NAME "tree_rotor")

add_library(r4_runner
    src/cpp/geometry/r4_runner.cpp
)
target_link_libraries(r4_runner tree_rotor)
set_target_properties(r4_runner PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 3. Deterministic Code Oracle & Test-Driven Verification
add_library(code_oracle
    src/cpp/oracle/code_oracle.cpp
)

# 4. Search & Macro Routing
add_library(macro_decider
    src/cpp/search/macro_decider.cpp
)
target_link_libraries(macro_decider tree_rotor)

add_library(geodesic_search
    src/cpp/search/geodesic_search.cpp
)

# 5. CLI & Engine Executables
add_executable(prover
    src/cpp/cli/process.cpp
    src/cpp/io/parseJsonl.cpp
)
target_link_libraries(prover tree_rotor macro_decider geodesic_search)

# 6. Test Executables
add_executable(test_tree_rotor
    tests/test_tree_rotor.cpp
)
target_link_libraries(test_tree_rotor tree_rotor)

add_executable(test_r4_runner_cpp
    tests/test_r4_runner_cpp.cpp
)
target_link_libraries(test_r4_runner_cpp r4_runner tree_rotor)

add_executable(test_speculative_geodesic_search
    tests/test_speculative_geodesic_search.cpp
)
target_link_libraries(test_speculative_geodesic_search geodesic_search)

add_executable(test_macro_decider
    tests/test_macro_decider.cpp
)
target_link_libraries(test_macro_decider macro_decider tree_rotor)

add_executable(test_code_oracle
    tests/test_code_oracle.cpp
)
target_link_libraries(test_code_oracle code_oracle)
