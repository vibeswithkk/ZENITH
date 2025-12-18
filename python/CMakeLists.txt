# Python Bindings CMakeLists.txt
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

# ============================================================================
# Python Bindings using pybind11
# ============================================================================

pybind11_add_module(_zenith_core
    src/bindings.cpp
)

target_link_libraries(_zenith_core PRIVATE zenith::core)

# Link CUDA libraries if enabled
if(ZENITH_ENABLE_CUDA)
    target_compile_definitions(_zenith_core PRIVATE ZENITH_HAS_CUDA)
    target_include_directories(_zenith_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(_zenith_core PRIVATE ${CUDA_cudart_LIBRARY} ${CUDA_cublas_LIBRARY})
    if(CUDNN_LIBRARY)
        target_compile_definitions(_zenith_core PRIVATE ZENITH_HAS_CUDNN)
        target_link_libraries(_zenith_core PRIVATE ${CUDNN_LIBRARY})
    endif()
endif()

# Link ROCm libraries if enabled
if(ZENITH_ENABLE_ROCM)
    find_package(hip QUIET)
    if(hip_FOUND)
        target_compile_definitions(_zenith_core PRIVATE ZENITH_HAS_ROCM)
        target_link_libraries(_zenith_core PRIVATE hip::device)
        
        find_package(rocblas QUIET)
        if(rocblas_FOUND)
            target_compile_definitions(_zenith_core PRIVATE ZENITH_HAS_ROCBLAS)
            target_link_libraries(_zenith_core PRIVATE roc::rocblas)
        endif()
        
        find_package(miopen QUIET)
        if(miopen_FOUND)
            target_compile_definitions(_zenith_core PRIVATE ZENITH_HAS_MIOPEN)
            target_link_libraries(_zenith_core PRIVATE MIOpen)
        endif()
    endif()
endif()

# Set output directory
set_target_properties(_zenith_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/zenith"
)

# Install Python module
install(TARGETS _zenith_core
    LIBRARY DESTINATION zenith
)
