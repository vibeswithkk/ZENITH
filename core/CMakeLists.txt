# Core Library CMakeLists.txt
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

# ============================================================================
# Zenith Core Library
# ============================================================================

set(ZENITH_CORE_HEADERS
    include/zenith/zenith.hpp
    include/zenith/types.hpp
    include/zenith/rocm_backend.hpp
    include/zenith/rocblas_ops.hpp
    include/zenith/miopen_ops.hpp
    include/zenith/tensor.hpp
    include/zenith/node.hpp
    include/zenith/graph_ir.hpp
    include/zenith/backend.hpp
    include/zenith/kernels.hpp
    include/zenith/cpu_backend.hpp
    include/zenith/cuda_backend.hpp
    include/zenith/op_signature.hpp
    include/zenith/target_descriptor.hpp
    include/zenith/kernel.hpp
    include/zenith/kernel_registry.hpp
    include/zenith/compiled_artifact.hpp
    include/zenith/compilation_session.hpp
    include/zenith/cublas_ops.hpp
    include/zenith/cudnn_ops.hpp
    include/zenith/dispatcher.hpp
)

set(ZENITH_CORE_SOURCES
    src/graph_ir.cpp
    src/cpu_kernels.cpp
    src/compilation_session.cpp
)

# ============================================================================
# CUDA Support
# ============================================================================

if(ZENITH_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    list(APPEND ZENITH_CORE_SOURCES 
        src/cuda_kernels.cu
        src/transformer_kernels.cu
        src/flash_attention.cu
        src/cublas_attention.cu
        src/fp16_ops.cu
        src/fp16_kernels.cu
    )
    
    # Export CUDA variables to parent scope for python bindings
    set(CUDA_cudart_LIBRARY ${CUDAToolkit_LIBRARY_DIR}/libcudart.so CACHE FILEPATH "CUDA runtime lib")
    set(CUDA_cublas_LIBRARY ${CUDAToolkit_LIBRARY_DIR}/libcublas.so CACHE FILEPATH "cuBLAS lib")
    
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include: ${CUDAToolkit_INCLUDE_DIRS}")
endif()

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

# ============================================================================
# CUDA/cuDNN/cuBLAS Linking
# ============================================================================

if(ZENITH_ENABLE_CUDA)
    target_compile_definitions(zenith_core PUBLIC ZENITH_HAS_CUDA)
    target_link_libraries(zenith_core PUBLIC CUDA::cudart)
    
    # cuBLAS
    if(TARGET CUDA::cublas)
        target_link_libraries(zenith_core PUBLIC CUDA::cublas)
        message(STATUS "cuBLAS found and enabled")
    endif()
    
    # cuDNN (optional, may need manual path)
    find_library(CUDNN_LIBRARY cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    )
    if(CUDNN_LIBRARY)
        target_compile_definitions(zenith_core PUBLIC ZENITH_HAS_CUDNN)
        target_link_libraries(zenith_core PUBLIC ${CUDNN_LIBRARY})
        message(STATUS "cuDNN found: ${CUDNN_LIBRARY}")
    else()
        message(STATUS "cuDNN not found, cuDNN ops will be disabled")
    endif()
endif()

# ============================================================================
# ROCm/HIP Linking
# ============================================================================

if(ZENITH_ENABLE_ROCM)
    # Find HIP package
    find_package(hip QUIET)
    if(hip_FOUND)
        target_compile_definitions(zenith_core PUBLIC ZENITH_HAS_ROCM)
        target_link_libraries(zenith_core PUBLIC hip::device)
        message(STATUS "HIP found and enabled")
        
        # Find rocBLAS
        find_package(rocblas QUIET)
        if(rocblas_FOUND)
            target_compile_definitions(zenith_core PUBLIC ZENITH_HAS_ROCBLAS)
            target_link_libraries(zenith_core PUBLIC roc::rocblas)
            message(STATUS "rocBLAS found and enabled")
        else()
            message(STATUS "rocBLAS not found, rocBLAS ops will be disabled")
        endif()
        
        # Find MIOpen
        find_package(miopen QUIET)
        if(miopen_FOUND)
            target_compile_definitions(zenith_core PUBLIC ZENITH_HAS_MIOPEN)
            target_link_libraries(zenith_core PUBLIC MIOpen)
            message(STATUS "MIOpen found and enabled")
        else()
            message(STATUS "MIOpen not found, MIOpen ops will be disabled")
        endif()
    else()
        message(WARNING "ROCm enabled but HIP not found. Install ROCm toolkit.")
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

