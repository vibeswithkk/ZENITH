# Tests CMakeLists.txt
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

# ============================================================================
# Fetch GoogleTest
# ============================================================================

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ============================================================================
# Test Executables
# ============================================================================

# Core library tests
add_executable(test_core
    test_types.cpp
    test_tensor.cpp
    test_node.cpp
    test_graph_ir.cpp
)

target_link_libraries(test_core PRIVATE
    zenith::core
    GTest::gtest_main
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(test_core)
