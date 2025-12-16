cmake_minimum_required(VERSION 3.20)
project(zenith
    VERSION 0.1.0
    DESCRIPTION "Cross-Platform ML Optimization Framework"
    LANGUAGES CXX
)

# ============================================================================
# Project Configuration
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# Options
# ============================================================================

option(ZENITH_BUILD_TESTS "Build test suite" ON)
option(ZENITH_BUILD_PYTHON "Build Python bindings" ON)
option(ZENITH_ENABLE_CUDA "Enable CUDA backend" OFF)
option(ZENITH_ENABLE_SIMD "Enable SIMD optimizations" ON)

# ============================================================================
# Dependencies
# ============================================================================

# Find pybind11 for Python bindings
if(ZENITH_BUILD_PYTHON)
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, will use FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ============================================================================
# Core Library
# ============================================================================

add_subdirectory(core)

# ============================================================================
# Python Bindings
# ============================================================================

if(ZENITH_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ============================================================================
# Tests
# ============================================================================

if(ZENITH_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS zenith_core
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY core/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zenith
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Zenith Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python:         ${ZENITH_BUILD_PYTHON}")
message(STATUS "  CUDA:           ${ZENITH_ENABLE_CUDA}")
message(STATUS "  SIMD:           ${ZENITH_ENABLE_SIMD}")
message(STATUS "  Tests:          ${ZENITH_BUILD_TESTS}")
message(STATUS "")
