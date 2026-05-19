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
  message(STATUS "Spyre profiler: using SPYRE_KINETO_MODE=${SPYRE_KINETO_MODE}")
else()
  message(STATUS "Spyre profiler: DISABLED")
endif()
