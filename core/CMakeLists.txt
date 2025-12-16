# Core Library CMakeLists.txt
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

# ============================================================================
# Zenith Core Library
# ============================================================================

set(ZENITH_CORE_HEADERS
    include/zenith/zenith.hpp
    include/zenith/types.hpp
    include/zenith/tensor.hpp
    include/zenith/node.hpp
    include/zenith/graph_ir.hpp
    include/zenith/backend.hpp
    include/zenith/kernels.hpp
    include/zenith/cpu_backend.hpp
)

set(ZENITH_CORE_SOURCES
    src/graph_ir.cpp
    src/cpu_kernels.cpp
)

# Create the library
add_library(zenith_core STATIC
    ${ZENITH_CORE_HEADERS}
    ${ZENITH_CORE_SOURCES}
)

# Set include directories
target_include_directories(zenith_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Set compile features
target_compile_features(zenith_core PUBLIC cxx_std_20)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(zenith_core PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )
elseif(MSVC)
    target_compile_options(zenith_core PRIVATE /W4)
endif()

# SIMD support - Enable by default on x86_64
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        include(CheckCXXCompilerFlag)
        
        # Check for AVX2 support
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(zenith_core PRIVATE -mavx2)
            target_compile_definitions(zenith_core PRIVATE ZENITH_HAS_AVX2)
        endif()
        
        # Check for FMA support
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(zenith_core PRIVATE -mfma)
            target_compile_definitions(zenith_core PRIVATE ZENITH_HAS_FMA)
        endif()
    endif()
endif()

# Set properties
set_target_properties(zenith_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)

# Create alias for use in subdirectories
add_library(zenith::core ALIAS zenith_core)
