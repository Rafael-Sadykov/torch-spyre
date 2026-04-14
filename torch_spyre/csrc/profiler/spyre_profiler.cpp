// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ============================================================================
// Spyre Profiler - Main Interface
// ============================================================================
// This file provides the main profiling interface for torch-spyre.
// It integrates libAIUpti (Spyre hardware profiling) with Kineto
// (PyTorch profiling framework).
//
// All code in this file is guarded with USE_SPYRE_PROFILER to ensure
// clean builds when profiling is disabled.

#ifdef USE_SPYRE_PROFILER

#include <iostream>
#include <string>
#include <memory>

// PyTorch headers
#include <c10/util/Exception.h>
#include <torch/csrc/profiler/api.h>

// Kineto headers (path depends on WHEEL vs UPSTREAM mode)
#ifdef SPYRE_PROFILER_USE_WHEEL
// kineto-spyre wheel provides headers in a specific location
#include <ActivityProfilerInterface.h>
#elif defined(SPYRE_PROFILER_USE_UPSTREAM)
// Upstream PyTorch kineto headers
#include <libkineto.h>
#else
// Fallback for AUTO mode - try upstream path
#include <libkineto.h>
#endif

// AIUpti headers (from Spyre SDK)
// Note: Actual header name may vary - update when SDK is available
// #include <aiupti.h>

namespace torch_spyre {
namespace profiler {

// ============================================================================
// Profiler State
// ============================================================================
// Global profiler state management
class SpyreProfiler {
public:
    static SpyreProfiler& instance() {
        static SpyreProfiler profiler;
        return profiler;
    }

    bool is_initialized() const { return initialized_; }
    bool is_enabled() const { return enabled_; }

    void initialize() {
        if (initialized_) {
            return;
        }

        std::cout << "[torch_spyre.profiler] Initializing Spyre profiler..." << std::endl;

        // TODO: Initialize libAIUpti
        // aiupti_status_t status = aiuptiInit();
        // TORCH_CHECK(status == AIUPTI_SUCCESS, "Failed to initialize libAIUpti");

        // TODO: Register with Kineto
        // Register Spyre as a profiling backend with Kineto
        // This allows PyTorch's profiler to collect Spyre hardware metrics

        initialized_ = true;
        std::cout << "[torch_spyre.profiler] Spyre profiler initialized successfully" << std::endl;

#ifdef SPYRE_PROFILER_USE_WHEEL
        std::cout << "[torch_spyre.profiler] Using kineto-spyre wheel" << std::endl;
#elif defined(SPYRE_PROFILER_USE_UPSTREAM)
        std::cout << "[torch_spyre.profiler] Using upstream PyTorch kineto" << std::endl;
#endif
    }

    void enable() {
        TORCH_CHECK(initialized_, "Profiler must be initialized before enabling");
        if (enabled_) {
            return;
        }

        std::cout << "[torch_spyre.profiler] Enabling Spyre profiler..." << std::endl;

        // TODO: Enable AIUpti profiling
        // aiupti_status_t status = aiuptiEnable();
        // TORCH_CHECK(status == AIUPTI_SUCCESS, "Failed to enable libAIUpti");

        enabled_ = true;
        std::cout << "[torch_spyre.profiler] Spyre profiler enabled" << std::endl;
    }

    void disable() {
        if (!enabled_) {
            return;
        }

        std::cout << "[torch_spyre.profiler] Disabling Spyre profiler..." << std::endl;

        // TODO: Disable AIUpti profiling
        // aiupti_status_t status = aiuptiDisable();
        // TORCH_CHECK(status == AIUPTI_SUCCESS, "Failed to disable libAIUpti");

        enabled_ = false;
        std::cout << "[torch_spyre.profiler] Spyre profiler disabled" << std::endl;
    }

    void finalize() {
        if (!initialized_) {
            return;
        }

        if (enabled_) {
            disable();
        }

        std::cout << "[torch_spyre.profiler] Finalizing Spyre profiler..." << std::endl;

        // TODO: Finalize libAIUpti
        // aiupti_status_t status = aiuptiFinalize();
        // TORCH_CHECK(status == AIUPTI_SUCCESS, "Failed to finalize libAIUpti");

        initialized_ = false;
        std::cout << "[torch_spyre.profiler] Spyre profiler finalized" << std::endl;
    }

    ~SpyreProfiler() {
        finalize();
    }

private:
    SpyreProfiler() : initialized_(false), enabled_(false) {}
    SpyreProfiler(const SpyreProfiler&) = delete;
    SpyreProfiler& operator=(const SpyreProfiler&) = delete;

    bool initialized_;
    bool enabled_;
};

// ============================================================================
// Public API Functions
// ============================================================================
// These functions are called from Python via pybind11 bindings

void initialize_profiler() {
    SpyreProfiler::instance().initialize();
}

void enable_profiler() {
    SpyreProfiler::instance().enable();
}

void disable_profiler() {
    SpyreProfiler::instance().disable();
}

void finalize_profiler() {
    SpyreProfiler::instance().finalize();
}

bool is_profiler_initialized() {
    return SpyreProfiler::instance().is_initialized();
}

bool is_profiler_enabled() {
    return SpyreProfiler::instance().is_enabled();
}

// ============================================================================
// Profiler Configuration
// ============================================================================
// Configuration management for profiler settings

struct ProfilerConfig {
    bool collect_kernel_metrics = true;
    bool collect_memory_metrics = true;
    bool collect_communication_metrics = false;
    int sampling_interval_us = 1000;  // 1ms default
};

static ProfilerConfig g_profiler_config;

void set_profiler_config(const ProfilerConfig& config) {
    g_profiler_config = config;
    std::cout << "[torch_spyre.profiler] Configuration updated:" << std::endl;
    std::cout << "  collect_kernel_metrics: " << config.collect_kernel_metrics << std::endl;
    std::cout << "  collect_memory_metrics: " << config.collect_memory_metrics << std::endl;
    std::cout << "  collect_communication_metrics: " << config.collect_communication_metrics << std::endl;
    std::cout << "  sampling_interval_us: " << config.sampling_interval_us << std::endl;
}

const ProfilerConfig& get_profiler_config() {
    return g_profiler_config;
}

} // namespace profiler
} // namespace torch_spyre

#endif // USE_SPYRE_PROFILER

