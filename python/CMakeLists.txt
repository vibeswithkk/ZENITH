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

# Set output directory
set_target_properties(_zenith_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/zenith"
)

# Install Python module
install(TARGETS _zenith_core
    LIBRARY DESTINATION lib/python/zenith
)
