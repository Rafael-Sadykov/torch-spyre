# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ============================================================================
# Spyre Profiler Feature Flag
# ============================================================================
# This option gates all profiling infrastructure. When OFF, no profiling
# code is compiled or linked. When ON, the build system detects and links
# against libAIUpti and libKineto.
#
# Usage:
#   cmake -DUSE_SPYRE_PROFILER=ON -DSPYRE_KINETO_MODE=WHEEL .
#   cmake -DUSE_SPYRE_PROFILER=ON -DSPYRE_KINETO_MODE=UPSTREAM .
#   cmake -DUSE_SPYRE_PROFILER=OFF .  # default
option(USE_SPYRE_PROFILER "Enable Spyre profiler support via libAIUpti" OFF)

# ============================================================================
# Kineto Dependency Mode Selection
# ============================================================================
# Determines which Kineto library to link against:
#
# AUTO (default): Try WHEEL first, fall back to UPSTREAM
#   - Searches for kineto-spyre wheel installation first
#   - Falls back to upstream PyTorch kineto if wheel not found
#
# WHEEL: Use IBM/kineto-spyre pre-built wheel (torch-2.9.1+aiu.kineto.1.1)
#   - Links against wheel's libkineto.so with AiuptiActivityApi support
#   - Install: pip install torch-2.9.1+aiu.kineto.1.1-*.whl
#   - Will be deprecated after pytorch/kineto#928 merges
#
# UPSTREAM: Use upstream PyTorch kineto (PyTorch 2.10+)
#   - Links against PyTorch's libkineto.so built with USE_AIUPTI=ON
#   - Requires: Build PyTorch with -DUSE_KINETO=ON -DUSE_AIUPTI=ON
#   - Future-proof path for PyTorch 2.10+
set(SPYRE_KINETO_MODE "AUTO" CACHE STRING "Kineto dependency mode for Spyre profiler")
set_property(CACHE SPYRE_KINETO_MODE PROPERTY STRINGS AUTO WHEEL UPSTREAM)

# ============================================================================
# Profiler State Variables
# ============================================================================
set(SPYRE_PROFILER_ENABLED FALSE)
set(SPYRE_PROFILER_MODE "NONE")

# ============================================================================
# Profiler Dependency Detection and Configuration
# ============================================================================
if(USE_SPYRE_PROFILER)
    message(STATUS "========================================")
    message(STATUS "Spyre Profiler: Dependency Detection")
    message(STATUS "========================================")

    # ------------------------------------------------------------------------
    # Step 1: Find libAIUpti (required for all profiling modes)
    # ------------------------------------------------------------------------
    # libAIUpti provides the low-level profiling API for IBM Spyre hardware.
    # It must be installed via the Spyre SDK.
    #
    # Set SPYRE_SDK_PATH environment variable to the SDK installation directory:
    #   export SPYRE_SDK_PATH=/path/to/spyre-sdk
    find_library(AIUPTI_LIBRARY
        NAMES AIUpti
        HINTS
            $ENV{SPYRE_SDK_PATH}/lib
            $ENV{SPYRE_SDK_PATH}
        PATH_SUFFIXES lib64 lib
        DOC "Path to libAIUpti.so from Spyre SDK"
    )

    if(NOT AIUPTI_LIBRARY)
        message(FATAL_ERROR
            "libAIUpti not found. Spyre profiler requires the Spyre SDK.\n"
            "Please install the Spyre SDK and set SPYRE_SDK_PATH:\n"
            "  export SPYRE_SDK_PATH=/path/to/spyre-sdk\n"
            "  cmake -DUSE_SPYRE_PROFILER=ON .\n"
        )
    endif()

    message(STATUS "Found libAIUpti: ${AIUPTI_LIBRARY}")

    # ------------------------------------------------------------------------
    # Step 2: Find libKineto (mode-dependent)
    # ------------------------------------------------------------------------
    # Kineto provides the PyTorch profiling integration layer.
    # Two valid paths exist with different build logic:
    #
    # Path 1 (WHEEL): kineto-spyre wheel with AiuptiActivityApi
    #   - Pre-built by IBM with Spyre support
    #   - Typically installed in site-packages or conda lib
    #
    # Path 2 (UPSTREAM): PyTorch's kineto built with USE_AIUPTI=ON
    #   - Available in PyTorch 2.10+ after pytorch/kineto#928
    #   - Located in PyTorch installation lib directory

    set(KINETO_LIBRARY "")

    # Try WHEEL path first (if AUTO or WHEEL mode)
    if(SPYRE_KINETO_MODE STREQUAL "AUTO" OR SPYRE_KINETO_MODE STREQUAL "WHEEL")
        message(STATUS "Searching for kineto-spyre wheel...")

        # Common wheel installation locations
        find_library(KINETO_LIBRARY
            NAMES kineto
            HINTS
                /usr/local/lib
                /opt/conda/lib
                ${Python_SITEARCH}/torch/lib
                $ENV{VIRTUAL_ENV}/lib/python*/site-packages/torch/lib
            NO_DEFAULT_PATH
            DOC "Path to libkineto.so from kineto-spyre wheel"
        )

        if(KINETO_LIBRARY)
            set(SPYRE_PROFILER_MODE "WHEEL")
            set(SPYRE_PROFILER_ENABLED TRUE)
            message(STATUS "Found kineto-spyre wheel: ${KINETO_LIBRARY}")
        else()
            message(STATUS "kineto-spyre wheel not found")
        endif()
    endif()

    # Try UPSTREAM path (if AUTO and wheel not found, or UPSTREAM mode)
    if(NOT SPYRE_PROFILER_ENABLED AND
       (SPYRE_KINETO_MODE STREQUAL "AUTO" OR SPYRE_KINETO_MODE STREQUAL "UPSTREAM"))
        message(STATUS "Searching for upstream PyTorch kineto...")

        # PyTorch installation locations
        find_library(KINETO_LIBRARY
            NAMES kineto
            HINTS
                ${TORCH_INSTALL_PREFIX}/lib
                ${Python_SITEARCH}/torch/lib
            NO_DEFAULT_PATH
            DOC "Path to libkineto.so from PyTorch installation"
        )

        if(KINETO_LIBRARY)
            set(SPYRE_PROFILER_MODE "UPSTREAM")
            set(SPYRE_PROFILER_ENABLED TRUE)
            message(STATUS "Found upstream PyTorch kineto: ${KINETO_LIBRARY}")
        else()
            message(STATUS "Upstream PyTorch kineto not found")
        endif()
    endif()

    # ------------------------------------------------------------------------
    # Step 3: Error if no valid Kineto found
    # ------------------------------------------------------------------------
    if(NOT SPYRE_PROFILER_ENABLED)
        message(FATAL_ERROR
            "USE_SPYRE_PROFILER=ON but no valid Kineto library found.\n"
            "\n"
            "Two installation paths are supported:\n"
            "\n"
            "1. WHEEL path (current, will be deprecated):\n"
            "   Download the kineto-spyre wheel from GitHub Releases:\n"
            "   https://github.com/IBM/kineto-spyre/releases\n"
            "\n"
            "   Match the wheel to your PyTorch version:\n"
            "   - PyTorch 2.9.x: torch_kineto-2.9.1+aiu.kineto.1.1-cp310-cp310-linux_x86_64.whl\n"
            "   - PyTorch 2.10.x: torch_kineto-2.10.0+aiu.kineto.1.2-cp310-cp310-linux_x86_64.whl\n"
            "\n"
            "   Install with pip:\n"
            "   pip install torch_kineto-2.9.1+aiu.kineto.1.1-cp310-cp310-linux_x86_64.whl\n"
            "\n"
            "   Then build torch-spyre:\n"
            "   cmake -DUSE_SPYRE_PROFILER=ON -DSPYRE_KINETO_MODE=WHEEL .\n"
            "\n"
            "2. UPSTREAM path (PyTorch 2.10+, recommended for future):\n"
            "   Build PyTorch from source with Kineto and AIUpti support:\n"
            "     cd pytorch\n"
            "     USE_KINETO=ON USE_AIUPTI=ON python setup.py install\n"
            "\n"
            "   Then build torch-spyre:\n"
            "     cmake -DUSE_SPYRE_PROFILER=ON -DSPYRE_KINETO_MODE=UPSTREAM .\n"
            "\n"
            "For AUTO mode (default), the build system tries WHEEL first,\n"
            "then falls back to UPSTREAM.\n"
            "\n"
            "See BUILD.md for detailed installation instructions.\n"
        )
    endif()

    # ------------------------------------------------------------------------
    # Step 4: Print configuration summary
    # ------------------------------------------------------------------------
    message(STATUS "========================================")
    if(SPYRE_PROFILER_MODE STREQUAL "WHEEL")
        message(STATUS "Spyre profiler: ENABLED — using kineto-spyre wheel")
        message(STATUS "  libAIUpti: ${AIUPTI_LIBRARY}")
        message(STATUS "  libKineto: ${KINETO_LIBRARY} (wheel)")
    elseif(SPYRE_PROFILER_MODE STREQUAL "UPSTREAM")
        message(STATUS "Spyre profiler: ENABLED — using upstream PyTorch kineto")
        message(STATUS "  libAIUpti: ${AIUPTI_LIBRARY}")
        message(STATUS "  libKineto: ${KINETO_LIBRARY} (upstream)")
    endif()
    message(STATUS "========================================")

else()
    # Profiler disabled
    message(STATUS "========================================")
    message(STATUS "Spyre profiler: DISABLED")
    message(STATUS "========================================")
endif()

# ============================================================================
# Add Profiler Subdirectory (if enabled)
# ============================================================================
# The profiler subdirectory contains all profiling C++ sources.
# It is only added to the build when USE_SPYRE_PROFILER=ON and
# dependencies are successfully found.
if(USE_SPYRE_PROFILER AND SPYRE_PROFILER_ENABLED)
    add_subdirectory(profiler)

    # Export variables for parent scope (if this CMakeLists.txt is included)
    set(SPYRE_PROFILER_LIBRARIES ${AIUPTI_LIBRARY} ${KINETO_LIBRARY} PARENT_SCOPE)
    set(SPYRE_PROFILER_ENABLED ${SPYRE_PROFILER_ENABLED} PARENT_SCOPE)
endif()