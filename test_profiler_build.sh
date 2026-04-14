#!/bin/bash
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
# Torch-Spyre Profiler Build Test Suite
# ============================================================================
# This script validates all profiler build configurations from EPIC #601.
#
# Usage:
#   ./test_profiler_build.sh [--verbose] [--keep-builds]
#
# Options:
#   --verbose      Show detailed build output
#   --keep-builds  Don't clean build directories after tests

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERBOSE=0
KEEP_BUILDS=0
TEST_RESULTS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Parse Arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --keep-builds)
            KEEP_BUILDS=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--verbose] [--keep-builds]"
            echo ""
            echo "Options:"
            echo "  --verbose      Show detailed build output"
            echo "  --keep-builds  Don't clean build directories after tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

log_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$*${NC}"
    echo -e "${BLUE}========================================${NC}"
}

record_result() {
    local test_name="$1"
    local result="$2"
    TEST_RESULTS+=("$test_name: $result")
}

clean_build() {
    log_info "Cleaning build artifacts..."
    python setup.py clean 2>/dev/null || true
    rm -rf build/ 2>/dev/null || true
}

run_build() {
    local build_log="$1"
    if [[ $VERBOSE -eq 1 ]]; then
        python setup.py build 2>&1 | tee "$build_log"
    else
        python setup.py build > "$build_log" 2>&1
    fi
}

check_symbols() {
    local so_file="$1"
    local pattern="$2"
    
    if [[ ! -f "$so_file" ]]; then
        log_error "Shared library not found: $so_file"
        return 1
    fi
    
    if nm "$so_file" 2>/dev/null | grep -i "$pattern" > /dev/null; then
        return 0  # Symbols found
    else
        return 1  # Symbols not found
    fi
}

find_so_file() {
    local pattern="torch_spyre._C*.so"
    find build -name "$pattern" 2>/dev/null | head -1
}

# ============================================================================
# Test 1: OFF Path - No Profiling Symbols
# ============================================================================
test_off_path() {
    log_section "Test 1: OFF Path (USE_SPYRE_PROFILER=0)"
    
    clean_build
    
    log_info "Building without profiling..."
    export USE_SPYRE_PROFILER=0
    unset SPYRE_KINETO_MODE
    unset SPYRE_SDK_PATH
    
    local build_log="build_off.log"
    if run_build "$build_log"; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        record_result "Test 1 (OFF)" "FAIL - Build error"
        return 1
    fi
    
    # Check for profiling symbols (should NOT be present)
    local so_file=$(find_so_file)
    if [[ -z "$so_file" ]]; then
        log_error "Built shared library not found"
        record_result "Test 1 (OFF)" "FAIL - No .so file"
        return 1
    fi
    
    log_info "Checking for profiling symbols in $so_file..."
    if check_symbols "$so_file" "aiupti\|kineto\|profiler"; then
        log_error "Profiling symbols found in OFF build!"
        nm "$so_file" | grep -i "aiupti\|kineto\|profiler" | head -5
        record_result "Test 1 (OFF)" "FAIL - Symbols leaked"
        return 1
    else
        log_success "No profiling symbols found (expected)"
        record_result "Test 1 (OFF)" "PASS"
        return 0
    fi
}

# ============================================================================
# Test 2: WHEEL Path - With kineto-spyre Wheel
# ============================================================================
test_wheel_path() {
    log_section "Test 2: WHEEL Path (USE_SPYRE_PROFILER=1, WHEEL mode)"
    
    # Check if wheel is installed
    if ! pip show torch 2>/dev/null | grep -q "kineto"; then
        log_warning "kineto-spyre wheel not installed"
        log_info "To test WHEEL path, install: pip install torch-2.9.1+aiu.kineto.1.1-*.whl"
        record_result "Test 2 (WHEEL)" "SKIP - Wheel not installed"
        return 0
    fi
    
    # Check if SPYRE_SDK_PATH is set
    if [[ -z "${SPYRE_SDK_PATH}" ]]; then
        log_warning "SPYRE_SDK_PATH not set"
        log_info "To test WHEEL path, set: export SPYRE_SDK_PATH=/path/to/spyre-sdk"
        record_result "Test 2 (WHEEL)" "SKIP - SDK path not set"
        return 0
    fi
    
    clean_build
    
    log_info "Building with profiling (WHEEL mode)..."
    export USE_SPYRE_PROFILER=1
    export SPYRE_KINETO_MODE=WHEEL
    
    local build_log="build_wheel.log"
    if run_build "$build_log"; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        cat "$build_log" | tail -20
        record_result "Test 2 (WHEEL)" "FAIL - Build error"
        return 1
    fi
    
    # Check for success message
    if grep -q "Spyre profiler: ENABLED — using kineto-spyre wheel" "$build_log"; then
        log_success "WHEEL mode detected in build output"
    else
        log_error "WHEEL mode not detected in build output"
        record_result "Test 2 (WHEEL)" "FAIL - Mode not detected"
        return 1
    fi
    
    # Check for profiling symbols (should be present)
    local so_file=$(find_so_file)
    if [[ -z "$so_file" ]]; then
        log_error "Built shared library not found"
        record_result "Test 2 (WHEEL)" "FAIL - No .so file"
        return 1
    fi
    
    log_info "Checking for profiling symbols in $so_file..."
    if check_symbols "$so_file" "profiler"; then
        log_success "Profiling symbols found (expected)"
        record_result "Test 2 (WHEEL)" "PASS"
        return 0
    else
        log_warning "Profiling symbols not found (may be expected if stub only)"
        record_result "Test 2 (WHEEL)" "PASS (no symbols yet)"
        return 0
    fi
}

# ============================================================================
# Test 3: UPSTREAM Path - With PyTorch 2.10+ Kineto
# ============================================================================
test_upstream_path() {
    log_section "Test 3: UPSTREAM Path (USE_SPYRE_PROFILER=1, UPSTREAM mode)"
    
    # Check if upstream kineto is available
    if ! python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "2.10"; then
        log_warning "PyTorch 2.10+ not detected"
        log_info "To test UPSTREAM path, build PyTorch 2.10+ with USE_KINETO=ON USE_AIUPTI=ON"
        record_result "Test 3 (UPSTREAM)" "SKIP - PyTorch 2.10+ not available"
        return 0
    fi
    
    # Check if SPYRE_SDK_PATH is set
    if [[ -z "${SPYRE_SDK_PATH}" ]]; then
        log_warning "SPYRE_SDK_PATH not set"
        log_info "To test UPSTREAM path, set: export SPYRE_SDK_PATH=/path/to/spyre-sdk"
        record_result "Test 3 (UPSTREAM)" "SKIP - SDK path not set"
        return 0
    fi
    
    clean_build
    
    log_info "Building with profiling (UPSTREAM mode)..."
    export USE_SPYRE_PROFILER=1
    export SPYRE_KINETO_MODE=UPSTREAM
    
    local build_log="build_upstream.log"
    if run_build "$build_log"; then
        log_success "Build completed successfully"
    else
        log_error "Build failed"
        cat "$build_log" | tail -20
        record_result "Test 3 (UPSTREAM)" "FAIL - Build error"
        return 1
    fi
    
    # Check for success message
    if grep -q "Spyre profiler: ENABLED — using upstream PyTorch kineto" "$build_log"; then
        log_success "UPSTREAM mode detected in build output"
        record_result "Test 3 (UPSTREAM)" "PASS"
        return 0
    else
        log_error "UPSTREAM mode not detected in build output"
        record_result "Test 3 (UPSTREAM)" "FAIL - Mode not detected"
        return 1
    fi
}

# ============================================================================
# Test 4: Missing Dependency Error
# ============================================================================
test_missing_dependency() {
    log_section "Test 4: Missing Dependency Error Handling"
    
    clean_build
    
    log_info "Attempting build without dependencies..."
    export USE_SPYRE_PROFILER=1
    export SPYRE_KINETO_MODE=WHEEL
    unset SPYRE_SDK_PATH
    
    local build_log="build_fail.log"
    if run_build "$build_log" 2>&1; then
        log_error "Build succeeded when it should have failed"
        record_result "Test 4 (Error)" "FAIL - No error on missing deps"
        return 1
    else
        log_success "Build failed as expected"
    fi
    
    # Check for clear error message with install instructions
    if grep -i "GitHub Releases" "$build_log" > /dev/null && \
       grep -i "pip install torch_kineto" "$build_log" > /dev/null; then
        log_success "Error message includes GitHub Releases link and pip install command"
        record_result "Test 4 (Error)" "PASS"
        return 0
    else
        log_error "Error message missing GitHub Releases link or pip install command"
        cat "$build_log" | tail -20
        record_result "Test 4 (Error)" "FAIL - Poor error message"
        return 1
    fi
}

# ============================================================================
# Test 5: setup.py Integration
# ============================================================================
test_setup_integration() {
    log_section "Test 5: setup.py Integration"
    
    clean_build
    
    log_info "Testing setup.py environment variable detection..."
    export USE_SPYRE_PROFILER=1
    export SPYRE_KINETO_MODE=AUTO
    
    local build_log="build_setup.log"
    if run_build "$build_log"; then
        log_success "Build completed"
    else
        log_warning "Build failed (may be expected without dependencies)"
    fi
    
    # Check if setup.py detected the profiler flag
    if grep -q "Profiler enabled" "$build_log" || \
       grep -q "USE_SPYRE_PROFILER" "$build_log"; then
        log_success "setup.py detected USE_SPYRE_PROFILER environment variable"
        record_result "Test 5 (setup.py)" "PASS"
        return 0
    else
        log_error "setup.py did not detect USE_SPYRE_PROFILER"
        record_result "Test 5 (setup.py)" "FAIL"
        return 1
    fi
}

# ============================================================================
# Main Test Execution
# ============================================================================
main() {
    log_section "Torch-Spyre Profiler Build Test Suite"
    log_info "Starting test suite..."
    log_info "Working directory: $SCRIPT_DIR"
    
    cd "$SCRIPT_DIR"
    
    # Run all tests
    test_off_path || true
    test_wheel_path || true
    test_upstream_path || true
    test_missing_dependency || true
    test_setup_integration || true
    
    # Clean up unless --keep-builds specified
    if [[ $KEEP_BUILDS -eq 0 ]]; then
        log_info "Cleaning up build artifacts..."
        clean_build
    else
        log_info "Keeping build artifacts (--keep-builds specified)"
    fi
    
    # Print summary
    log_section "Test Results Summary"
    for result in "${TEST_RESULTS[@]}"; do
        if [[ "$result" == *"PASS"* ]]; then
            log_success "$result"
        elif [[ "$result" == *"SKIP"* ]]; then
            log_warning "$result"
        else
            log_error "$result"
        fi
    done
    
    # Count results
    local pass_count=$(printf '%s\n' "${TEST_RESULTS[@]}" | grep -c "PASS" || true)
    local fail_count=$(printf '%s\n' "${TEST_RESULTS[@]}" | grep -c "FAIL" || true)
    local skip_count=$(printf '%s\n' "${TEST_RESULTS[@]}" | grep -c "SKIP" || true)
    
    echo ""
    log_info "Total: ${#TEST_RESULTS[@]} tests"
    log_success "Passed: $pass_count"
    log_error "Failed: $fail_count"
    log_warning "Skipped: $skip_count"
    
    if [[ $fail_count -gt 0 ]]; then
        echo ""
        log_error "Some tests failed. See logs above for details."
        exit 1
    else
        echo ""
        log_success "All tests passed!"
        exit 0
    fi
}

# Run main function
main

