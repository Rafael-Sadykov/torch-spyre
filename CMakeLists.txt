# Copyright The Torch-Spyre Authors.
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

cmake_minimum_required(VERSION 3.18)
project(torch_spyre LANGUAGES CXX)

option(USE_SPYRE_PROFILER "Enable Spyre profiler support via libAIUpti + Kineto" OFF)

set(SPYRE_KINETO_MODE "AUTO" CACHE STRING "Kineto dependency mode: AUTO, WHEEL, UPSTREAM")
set_property(CACHE SPYRE_KINETO_MODE PROPERTY STRINGS AUTO WHEEL UPSTREAM)

if(USE_SPYRE_PROFILER)
  message(STATUS "Spyre profiler: ENABLED")

  # Find libAIUpti (always required for profiler)
  find_library(AIUPTI_LIBRARY NAMES aiupti
    HINTS $ENV{SPYRE_SDK_PATH}/lib $ENV{LIBAIUPTI_INSTALL_DIR}/lib
    PATHS /opt/spyre/lib /usr/local/lib
    REQUIRED)
  message(STATUS "Found libAIUpti: ${AIUPTI_LIBRARY}")

  # Kineto mode resolution
  if(SPYRE_KINETO_MODE STREQUAL "AUTO")
    # Prefer wheel if kineto-spyre is present
    find_library(KINETO_LIBRARY NAMES kineto_spyre kineto
      HINTS $ENV{PYTHON_SITE_PACKAGES}/torch/lib)
    if(KINETO_LIBRARY)
      set(SPYRE_KINETO_MODE "WHEEL")
    else()
      set(SPYRE_KINETO_MODE "UPSTREAM")
    endif()
  endif()

  if(SPYRE_KINETO_MODE STREQUAL "WHEEL")
    find_library(KINETO_LIBRARY NAMES kineto_spyre
      HINTS $ENV{PYTHON_SITE_PACKAGES}/torch/lib $ENV{SPYRE_SDK_PATH}/lib)
    message(STATUS "Spyre profiler: ENABLED — using kineto-spyre wheel")
  else()
    find_library(KINETO_LIBRARY NAMES kineto
      HINTS $ENV{TORCH_INSTALL_PREFIX}/lib $ENV{PYTHON_SITE_PACKAGES}/torch/lib)
    message(STATUS "Spyre profiler: ENABLED — using upstream PyTorch kineto")
  endif()

  if(NOT KINETO_LIBRARY)
    message(FATAL_ERROR
      "USE_SPYRE_PROFILER=ON but Kineto library not found.\n"
      "WHEEL mode: pip install torch-2.9.1+aiu.kineto.1.1 --extra-index-url ...\n"
      "See: https://github.com/torch-spyre/torch-spyre/releases")
  endif()

  add_library(spyre_profiler INTERFACE)
  target_link_libraries(spyre_profiler INTERFACE ${AIUPTI_LIBRARY} ${KINETO_LIBRARY})
  target_compile_definitions(spyre_profiler INTERFACE USE_SPYRE_PROFILER)
else()
  message(STATUS "Spyre profiler: DISABLED")
endif()

# Subdirectory for profiler (only built when enabled)
add_subdirectory(torch_spyre/csrc/profiler)
