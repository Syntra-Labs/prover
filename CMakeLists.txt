include(CheckCXXCompilerFlag)

cmake_minimum_required(VERSION 3.5)
project(prover)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

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
target_link_libraries(prover tree_rotor macro_decider geodesic_search)

add_library(tree_rotor
    src/tree_rotor.cpp
)

add_library(tree_rotor_shared SHARED
    src/tree_rotor.cpp
)
set_target_properties(tree_rotor_shared PROPERTIES OUTPUT_NAME "tree_rotor")

add_executable(test_tree_rotor
    tests/test_tree_rotor.cpp
)
target_link_libraries(test_tree_rotor tree_rotor)

add_library(geodesic_search
    src/search/geodesic_search.cpp
)

add_executable(test_speculative_geodesic_search
    tests/test_speculative_geodesic_search.cpp
)
target_link_libraries(test_speculative_geodesic_search geodesic_search)

add_executable(eval_least_action_prover
    src/eval_least_action_prover.cpp
)
target_link_libraries(eval_least_action_prover tree_rotor geodesic_search)

add_library(macro_decider
    src/search/macro_decider.cpp
)
target_link_libraries(macro_decider tree_rotor)

add_executable(test_macro_decider
    tests/test_macro_decider.cpp
)
target_link_libraries(test_macro_decider macro_decider tree_rotor)

add_executable(eval_live_lean_prover
    src/eval_live_lean_prover.cpp
)
target_link_libraries(eval_live_lean_prover tree_rotor macro_decider geodesic_search)

