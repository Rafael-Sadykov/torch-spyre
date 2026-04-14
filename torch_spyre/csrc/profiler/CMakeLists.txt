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

# ============================================================================
# Torch-Spyre Profiler C++ Library
# ============================================================================
# This CMakeLists.txt builds the profiling infrastructure for torch-spyre.
# It is only included when USE_SPYRE_PROFILER=ON in the parent CMakeLists.txt.
#
# All sources in this directory must be guarded with:
#   #ifdef USE_SPYRE_PROFILER
#   // profiler code
#   #endif
#
# This ensures clean builds when profiling is disabled.

# Early return if profiler is disabled (defensive check)
if(NOT USE_SPYRE_PROFILER)
    message(WARNING "profiler/CMakeLists.txt included but USE_SPYRE_PROFILER=OFF")
    return()
endif()

message(STATUS "Configuring torch_spyre profiler library...")

# ============================================================================
# Collect Profiler Sources
# ============================================================================
# Gather all .cpp files in this directory.
# As the profiler grows, sources will include:
#   - spyre_profiler.cpp       (main profiler interface)
#   - aiupti_wrapper.cpp       (libAIUpti integration)
#   - kineto_integration.cpp   (Kineto activity API)
#   - profiler_config.cpp      (configuration management)
file(GLOB PROFILER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

if(NOT PROFILER_SOURCES)
    message(STATUS "No profiler sources found yet (expected during initial setup)")
    # Create a dummy source file to allow the library to be created
    # This will be removed once real profiler sources are added
    set(PROFILER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/profiler_stub.cpp")
    if(NOT EXISTS "${PROFILER_SOURCES}")
        file(WRITE "${PROFILER_SOURCES}"
            "// Temporary stub - will be replaced by real profiler sources\n"
            "#ifdef USE_SPYRE_PROFILER\n"
            "namespace torch_spyre { namespace profiler {\n"
            "void profiler_stub() {}\n"
            "}} // namespace torch_spyre::profiler\n"
            "#endif\n"
        )
    endif()
endif()

message(STATUS "Profiler sources: ${PROFILER_SOURCES}")

# ============================================================================
# Create Profiler Object Library
# ============================================================================
# Use OBJECT library to compile profiler sources separately.
# The parent build system will link these objects into torch_spyre._C
add_library(torch_spyre_profiler OBJECT ${PROFILER_SOURCES})

# ============================================================================
# Include Directories
# ============================================================================
# Profiler needs access to:
#   - Parent csrc directory (for torch_spyre headers)
#   - PyTorch headers (for ATen, c10, etc.)
#   - Kineto headers (for profiling API)
#   - AIUpti headers (for Spyre profiling API)
target_include_directories(torch_spyre_profiler PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..  # torch_spyre/csrc
    ${TORCH_INCLUDE_DIRS}            # PyTorch headers
)

# Add AIUpti include directory if SPYRE_SDK_PATH is set
if(DEFINED ENV{SPYRE_SDK_PATH})
    target_include_directories(torch_spyre_profiler PRIVATE
        $ENV{SPYRE_SDK_PATH}/include
    )
endif()

# ============================================================================
# Compile Definitions
# ============================================================================
# Define USE_SPYRE_PROFILER for all profiler sources.
# This allows conditional compilation of profiling code.
target_compile_definitions(torch_spyre_profiler PRIVATE
    USE_SPYRE_PROFILER
)

# Add mode-specific definitions
if(SPYRE_PROFILER_MODE STREQUAL "WHEEL")
    target_compile_definitions(torch_spyre_profiler PRIVATE
        SPYRE_PROFILER_USE_WHEEL
    )
elseif(SPYRE_PROFILER_MODE STREQUAL "UPSTREAM")
    target_compile_definitions(torch_spyre_profiler PRIVATE
        SPYRE_PROFILER_USE_UPSTREAM
    )
endif()

# ============================================================================
# Compile Options
# ============================================================================
# Match the parent project's C++ standard and warning flags
target_compile_options(torch_spyre_profiler PRIVATE
    -std=c++17
    -Wall
    -Wno-deprecated
)

# Add debug symbols (always, for profiling analysis)
target_compile_options(torch_spyre_profiler PRIVATE -g)

message(STATUS "torch_spyre_profiler library configured successfully")

# ============================================================================
# Installation Notes
# ============================================================================
# The profiler object library is not installed separately.
# Its objects are linked into the main torch_spyre._C extension module
# by the parent build system (setup.py or root CMakeLists.txt).
#
# To use the profiler in torch_spyre._C:
#   1. In setup.py, add profiler objects to the extension sources
#   2. Link against ${AIUPTI_LIBRARY} and ${KINETO_LIBRARY}
#   3. Ensure USE_SPYRE_PROFILER is defined for the extension