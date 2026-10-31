cmake_minimum_required(VERSION 3.10)
project(prover VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/my_bin)

# --- Core Libraries ---
add_library(tree_rotor
    src/tree_rotor.cpp
)

add_library(tree_rotor_shared SHARED
    src/tree_rotor.cpp
)
set_target_properties(tree_rotor_shared PROPERTIES OUTPUT_NAME "tree_rotor")

add_library(macro_decider
    src/search/macro_decider.cpp
)
target_link_libraries(macro_decider tree_rotor)

add_library(geodesic_search
    src/search/geodesic_search.cpp
)

# --- Executables ---
add_executable(prover
    src/process.cpp
    src/parseJsonl.cpp
)
target_link_libraries(prover tree_rotor macro_decider geodesic_search)

add_executable(eval_least_action_prover
    src/eval_least_action_prover.cpp
)
target_link_libraries(eval_least_action_prover tree_rotor geodesic_search)

add_executable(eval_live_lean_prover
    src/eval_live_lean_prover.cpp
)
target_link_libraries(eval_live_lean_prover tree_rotor macro_decider geodesic_search)

# --- Test Suite ---
enable_testing()

add_executable(test_tree_rotor
    tests/test_tree_rotor.cpp
)
target_link_libraries(test_tree_rotor tree_rotor)
add_test(NAME test_tree_rotor COMMAND test_tree_rotor)

add_executable(test_speculative_geodesic_search
    tests/test_speculative_geodesic_search.cpp
)
target_link_libraries(test_speculative_geodesic_search geodesic_search)
add_test(NAME test_speculative_geodesic_search COMMAND test_speculative_geodesic_search)

add_executable(test_macro_decider
    tests/test_macro_decider.cpp
)
target_link_libraries(test_macro_decider macro_decider tree_rotor)
add_test(NAME test_macro_decider COMMAND test_macro_decider)

add_test(NAME eval_live_lean_prover COMMAND eval_live_lean_prover)
