# Building torch-spyre with Profiling Support

This document describes how to build torch-spyre with the optional **Spyre Profiler** infrastructure (Issue #927).

## Quick Start

### Build without profiling (default)

```bash
python setup.py build_ext --inplace
# or
pip install -e .
```

### Build with profiling enabled

```bash
export USE_SPYRE_PROFILER=1
export SPYRE_KINETO_MODE=AUTO          # AUTO, WHEEL, or UPSTREAM
export SPYRE_SDK_PATH=/path/to/spyre-sdk

python setup.py build_ext --inplace
# or
pip install -e .
```

## Profiler Feature Flag

| Variable                | Values                        | Description |
|-------------------------|-------------------------------|-------------|
| `USE_SPYRE_PROFILER`    | `0` (default), `1`            | Enable/disable profiler |
| `SPYRE_KINETO_MODE`     | `AUTO` (default), `WHEEL`, `UPSTREAM` | Kineto dependency mode |
| `SPYRE_SDK_PATH`        | path to SDK                   | Location of libAIUpti |

- **OFF**: No profiler code is compiled (zero overhead).
- **ON**: Compiles profiler sources, links `libaiupti` + `libkineto`, and defines `USE_SPYRE_PROFILER`, `HAS_AIUPTI`, `USE_KINETO`.

## Kineto Dependency Modes

| Mode       | Description                                      | Recommended For |
|------------|--------------------------------------------------|-----------------|
| `AUTO`     | Prefers wheel, falls back to upstream            | Most users |
| `WHEEL`    | Uses pre-built `kineto-spyre` wheel              | Current development |
| `UPSTREAM` | Uses official PyTorch Kineto (`USE_AIUPTI=ON`)   | PyTorch 2.11+ |

## Build Instructions

### Using setup.py (Recommended)

```bash
export USE_SPYRE_PROFILER=1
export SPYRE_KINETO_MODE=AUTO
export SPYRE_SDK_PATH=/opt/spyre-sdk

python setup.py build_ext --inplace
```

### Using CMake (Advanced)

```bash
mkdir -p build && cd build
cmake .. \
  -DUSE_SPYRE_PROFILER=ON \
  -DSPYRE_KINETO_MODE=AUTO \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build . -j$(nproc)
```

## Testing the Build

```bash
# 1. Disabled (clean build)
USE_SPYRE_PROFILER=0 python setup.py build_ext --inplace --dry-run

# 2. Enabled
USE_SPYRE_PROFILER=1 python setup.py build_ext --inplace
```

Expected output when enabled:
```
Spyre profiler: ENABLED — using kineto-spyre wheel
# or
Spyre profiler: ENABLED — using upstream PyTorch kineto
```

## Troubleshooting

**libAIUpti not found**
```bash
export SPYRE_SDK_PATH=/opt/spyre-sdk
ls $SPYRE_SDK_PATH/lib/libaiupti.so
```

**No valid Kineto library found**
- WHEEL: Install matching `kineto-spyre` wheel from GitHub Releases
- UPSTREAM: Build PyTorch with `USE_KINETO=ON USE_AIUPTI=ON`

**Flex API errors in spyre_stream.cpp**
These are expected. The runtime headers have changed. Update will be done in a follow-up PR.

## Future Work

- Add `#ifdef USE_SPYRE_PROFILER` guards to all profiler C++ files
- Add Tekton CI matrix entry for `USE_SPYRE_PROFILER=1`
- Update documentation after `kineto-spyre` is published to PyPI

---

**Related Issues**
- Fixes #927
- Part of EPIC #601 (Foundational Profiling Infrastructure)

