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

import os
import shutil
from pathlib import Path
from typing import cast

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault(
    "SEN_COMMON_HEADERS", str(Path(__file__).resolve().parent.parent / "flex")
)


from setuptools import Command, setup

PATH_NAME = "torch_spyre"
PACKAGE_NAME = "torch_spyre"
DISTRIBUTED_PACKAGE_NAME = "spyre_ccl"


def get_torch_spyre_version() -> str:
    version_ns: dict[str, object] = {}
    with open(f"{PATH_NAME}/version.py") as f:
        exec(f.read(), version_ns)
        version = cast(str, version_ns["__version__"])
    return version


version = get_torch_spyre_version()


def env_flag(name: str, default: str = "0") -> bool:
    value = os.environ.get(name, default)
    return value.strip().lower() in {"1", "true", "on", "yes"}


# === Issue #927: Spyre Profiler Support ===
USE_SPYRE_PROFILER = env_flag("USE_SPYRE_PROFILER", "0")
SPYRE_KINETO_MODE = os.environ.get("SPYRE_KINETO_MODE", "AUTO").strip().upper()


ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / PATH_NAME / "csrc"
BUILD_DIR = ROOT_DIR / "build"
DISTRIBUTED_SRC_DIR = CSRC_DIR / "distributed"


def maybe_download_nlohmann_json():
    import urllib.request

    NLOHMANN_URL = "https://raw.githubusercontent.com/nlohmann/json/v3.11.2/single_include/nlohmann/json.hpp"
    SHARED_PATH = Path(
        os.environ.get("SHARED_DEPS_DIR", ROOT_DIR / PATH_NAME / "csrc" / "external")
    )
    NLOHMANN_INC_DIR = SHARED_PATH / "nlohmann" / "include"
    NLOHMANN_DIR = NLOHMANN_INC_DIR / "nlohmann"

    NLOHMANN_HEADER = os.path.join(NLOHMANN_DIR, "json.hpp")
    if not os.path.exists(NLOHMANN_HEADER):
        os.makedirs(NLOHMANN_DIR, exist_ok=True)
        print("Downloading nlohmann/json.hpp...")
        urllib.request.urlretrieve(NLOHMANN_URL, NLOHMANN_HEADER)
    return NLOHMANN_INC_DIR


INCLUDE_DIRS = [CSRC_DIR]
# Add profiler headers when enabled
if USE_SPYRE_PROFILER:
    INCLUDE_DIRS.append(CSRC_DIR / "profiler")


LIBRARY_DIRS = []
INCLUDE_DIRS += [maybe_download_nlohmann_json()]

cmake_include_path = os.environ.get("CMAKE_INCLUDE_PATH", "")
extra_include_dirs = cmake_include_path.split(":") if cmake_include_path else []
INCLUDE_DIRS += [Path(p) for p in extra_include_dirs if p]

cmake_library_path = os.environ.get("CMAKE_LIBRARY_PATH", "")
extra_library_dirs = cmake_library_path.split(":") if cmake_library_path else []
LIBRARY_DIRS += [Path(p) for p in extra_library_dirs if p]

if "RUNTIME_INSTALL_DIR" in os.environ:
    RUNTIME_DIR = Path(os.environ["RUNTIME_INSTALL_DIR"])
    SENLIB_DIR = Path(os.environ["SENLIB_INSTALL_DIR"])
    DEEPTOOLS_DIR = Path(os.environ["DEEPTOOLS_INSTALL_DIR"])
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include",
        RUNTIME_DIR / "include" / "concurrentqueue" / "moodycamel",
        SENLIB_DIR / "include",
        DEEPTOOLS_DIR / "include",
    ]
    LIBRARY_DIRS += [RUNTIME_DIR / "lib"]

# The USE_SPYRE_CCL environment variable can be used to build torch-spyre
# without support for Multi-Spyre. This is for developers only.
# If set to '0' then Multi-Spyre support is disabled.
# Otherwise (default) Multi-Spyre support is enabled.
use_spyre_ccl = os.environ.get("USE_SPYRE_CCL", "1") != "0"

if not use_spyre_ccl:
    print("=" * 80)
    print("WARNING: Multi-Spyre support has been disabled")
    print("=" * 80)
else:
    if "SPYRE_COMMS_INSTALL_DIR" in os.environ:
        SPYRE_COMMS_DIR = Path(os.environ["SPYRE_COMMS_INSTALL_DIR"])
        if not SPYRE_COMMS_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR directory does not exist: {SPYRE_COMMS_DIR}"
            )
        SPYRE_COMMS_INCLUDE_DIR = SPYRE_COMMS_DIR / "include"
        if not SPYRE_COMMS_INCLUDE_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR include directory does not exist: {SPYRE_COMMS_INCLUDE_DIR}"
            )
        SPYRE_COMMS_LIB_DIR = SPYRE_COMMS_DIR / "lib"
        if not SPYRE_COMMS_LIB_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR lib directory does not exist: {SPYRE_COMMS_LIB_DIR}"
            )
        INCLUDE_DIRS += [
            SPYRE_COMMS_INCLUDE_DIR,
        ]
        LIBRARY_DIRS += [SPYRE_COMMS_LIB_DIR]
    else:
        raise RuntimeError(
            "SPYRE_COMMS_INSTALL_DIR not set. "
            "Set USE_SPYRE_CCL=0 to build without Multi-Spyre support, "
            "or set the SPYRE_COMMS_INSTALL_DIR to the Spyre Comms install directory."
        )

INCLUDE_DIRS += [os.environ["SEN_COMMON_HEADERS"]]

use_new_system = os.environ.get("NEW_SYSTEM_SETUP", "0") == "1"

if use_new_system:
    LIBRARIES = ["flex"]
else:
    LIBRARIES = ["sendnn", "sendnn_interface", "flex"]

if use_spyre_ccl:
    LIBRARIES.append("spyre_comms")

# === Kineto / Profiler Libraries (for wheel installation) ===
if USE_SPYRE_PROFILER:
    # When using kineto-spyre wheel
    if SPYRE_KINETO_MODE in ("WHEEL", "AUTO"):
        LIBRARIES.extend(["aiupti", "kineto"])
    # When using upstream PyTorch Kineto
    elif SPYRE_KINETO_MODE == "UPSTREAM":
        LIBRARIES.append("kineto")

# === Validate that Kineto libraries are available when profiler is enabled ===
if USE_SPYRE_PROFILER:
    import sysconfig

    # Build list of possible library directories
    possible_lib_dirs = list(LIBRARY_DIRS)

    # Add common locations where pip/uv installs libraries
    venv_lib = Path(sysconfig.get_path("purelib")).parent / "lib"
    if venv_lib.exists():
        possible_lib_dirs.append(venv_lib)

    # Also check torch installation directory (sometimes libraries are there)
    try:
        import torch
        torch_lib_dir = Path(torch.__file__).parent / "lib"
        if torch_lib_dir.exists():
            possible_lib_dirs.append(torch_lib_dir)
    except Exception:
        pass

    # Check if any of the required libraries exist
    kineto_found = any(
        (lib_dir / "libkineto.so").exists() or (lib_dir / "libaiupti.so").exists()
        for lib_dir in possible_lib_dirs
    )

    if not kineto_found:
        raise RuntimeError(
            "\n"
            "ERROR: USE_SPYRE_PROFILER=1 is enabled, but libkineto / libaiupti were not found.\n\n"
            "Please install the matching kineto-spyre wheel:\n\n"
            "    uv pip install --no-deps --force-reinstall \\\n"
            "        https://github.com/IBM/kineto-spyre/releases/download/"
            "torch-2.11.0.aiu.kineto.1.1.2/"
            "torch-2.11.0+aiu.kineto.1.1.2-cp312-cp312-linux_x86_64.whl\n\n"
            "Find matching wheels for other PyTorch versions here:\n"
            "    https://github.com/IBM/kineto-spyre/releases\n"
        )

if USE_SPYRE_PROFILER:
    print("✓ Spyre Profiler support enabled (USE_SPYRE_PROFILER=1)")


NO_OPT_BUILD = os.environ.get("TORCH_SPYRE_DEBUG", "0") == "1"
EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Wno-deprecated", "-std=c++20"]

if NO_OPT_BUILD:
    EXTRA_CXX_FLAGS += ["-O0"]


class clean(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path in (ROOT_DIR / PATH_NAME).glob("**/*.so"):
            path.unlink()
        if BUILD_DIR.exists():
            shutil.rmtree(str(BUILD_DIR), ignore_errors=True)


if __name__ == "__main__":
    import sys

    is_meta = any(
        cmd in sys.argv for cmd in ["dist_info", "egg_info", "install_egg_info"]
    )

    if is_meta:
        setup(
            name=PACKAGE_NAME,
            version=version,
            entry_points={"torch.backends": ["torch_spyre = torch_spyre:_autoload"]},
        )
    else:
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        sources = list(CSRC_DIR.glob("*.cpp"))
        # Add profiler sources when USE_SPYRE_PROFILER=1
        if USE_SPYRE_PROFILER:
            PROFILER_SRC_DIR = CSRC_DIR / "profiler"
            if PROFILER_SRC_DIR.exists():
                sources += list(PROFILER_SRC_DIR.glob("*.cpp"))

        distributed_sources = (
            list(DISTRIBUTED_SRC_DIR.glob("*.cpp")) if use_spyre_ccl else []
        )

        hooks_only_files = {"spyre_hooks.cpp"}
        shared_files = {"spyre_device_enum.cpp", "logging.cpp"}

        hooks_src_paths = [
            p.relative_to(ROOT_DIR).as_posix()
            for p in sources
            if p.name in hooks_only_files | shared_files
        ]
        core_src_paths = [
            p.relative_to(ROOT_DIR).as_posix()
            for p in sources
            if p.name not in hooks_only_files
        ]
        distributed_src_paths = [
            p.relative_to(ROOT_DIR).as_posix() for p in sorted(distributed_sources)
        ]

        # Build define_macros list conditionally
        base_define_macros = [
            ("PACKAGE_NAME", f'"{PACKAGE_NAME}"'),
            ("SPYRE_DEBUG_ENV", '"TORCH_SPYRE_DEBUG"'),
            ("SPYRE_DOWNCAST_ENV", '"TORCH_SPYRE_DOWNCAST_WARN"'),
            ("EAGER_MODE_ENV", '"EAGER_MODE"'),
            ("BOOST_ALL_DYN_LINK", None),  # avoid static link to boost
        ]
        if use_spyre_ccl:
            base_define_macros.append(("USE_SPYRE_CCL", None))
        if use_new_system:
            base_define_macros.append(("USE_FLEX_NAMESPACE", None))

        # profiler_define = [("USE_SPYRE_PROFILER", None)] if USE_SPYRE_PROFILER else []
        profiler_define = []
        if USE_SPYRE_PROFILER:
            profiler_define = [
                ("USE_SPYRE_PROFILER", None),
                ("HAS_AIUPTI", None),
                ("USE_KINETO", None),
            ]

        ext_modules = [
            CppExtension(
                name=f"{PACKAGE_NAME}._C",
                sources=core_src_paths + distributed_src_paths,
                include_dirs=[str(p) for p in INCLUDE_DIRS],
                library_dirs=[str(p) for p in LIBRARY_DIRS],
                libraries=LIBRARIES,
                extra_compile_args={"cxx": EXTRA_CXX_FLAGS},
                define_macros=[
                    ("PACKAGE_NAME", f'"{PACKAGE_NAME}"'),
                    ("MODULE_NAME", f'"{PACKAGE_NAME}._C"'),
                    ("SPYRE_DEBUG_ENV", '"TORCH_SPYRE_DEBUG"'),
                    ("SPYRE_DOWNCAST_ENV", '"TORCH_SPYRE_DOWNCAST_WARN"'),
                    ("EAGER_MODE_ENV", '"EAGER_MODE"'),
                    ("BOOST_ALL_DYN_LINK", None),
                    *profiler_define,
                    ("FMT_HEADER_ONLY", None),
                ],
            ),
            CppExtension(
                name=f"{PACKAGE_NAME}._hooks",
                sources=hooks_src_paths,
                include_dirs=[str(p) for p in INCLUDE_DIRS],
                library_dirs=[str(p) for p in LIBRARY_DIRS],
                libraries=LIBRARIES,
                extra_compile_args={"cxx": EXTRA_CXX_FLAGS},
                define_macros=[
                    ("PACKAGE_NAME", f'"{PACKAGE_NAME}"'),
                    ("MODULE_NAME", f'"{PACKAGE_NAME}._hooks"'),
                    ("SPYRE_DEBUG_ENV", '"TORCH_SPYRE_DEBUG"'),
                    ("SPYRE_DOWNCAST_ENV", '"TORCH_SPYRE_DOWNCAST_WARN"'),
                    ("EAGER_MODE_ENV", '"EAGER_MODE"'),
                    ("BOOST_ALL_DYN_LINK", None),
                ],
            ),
        ]

        _BuildExtension = BuildExtension.with_options(
            no_python_abi_suffix=True, verbose=True
        )

        class PermanentBuildExtension(BuildExtension):
            def finalize_options(self):
                super().finalize_options()
                self.build_temp = str(BUILD_DIR)

            def build_extension(self, ext):
                # Use a per-extension subdirectory so each gets its own build.ninja
                original_build_temp = self.build_temp
                self.build_temp = os.path.join(original_build_temp, ext.name)
                os.makedirs(self.build_temp, exist_ok=True)
                try:
                    super().build_extension(ext)
                finally:
                    self.build_temp = original_build_temp

        setup(
            name=PACKAGE_NAME,
            version=version,
            ext_modules=ext_modules,
            cmdclass={
                "build_ext": PermanentBuildExtension,
                "clean": clean,
            },
            entry_points={"torch.backends": ["torch_spyre = torch_spyre:_autoload"]},
        )
