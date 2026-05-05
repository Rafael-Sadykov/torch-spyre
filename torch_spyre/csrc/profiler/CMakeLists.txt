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

if(NOT USE_SPYRE_PROFILER)
  return()
endif()

file(GLOB PROFILER_SOURCES "*.cpp")

add_library(spyre_profiler_objects OBJECT ${PROFILER_SOURCES})

target_include_directories(spyre_profiler_objects PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/torch_spyre/csrc
)

target_link_libraries(spyre_profiler_objects PRIVATE spyre_profiler)

# These objects will be linked into _C extension via setup.py
set(SPYRE_PROFILER_OBJECTS $<TARGET_OBJECTS:spyre_profiler_objects>
    PARENT_SCOPE)

