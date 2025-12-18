# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

import os
import re
import subprocess
import sys
import platform

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Set Python_EXECUTABLE to help CMake find the correct Python
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DZENITH_BUILD_PYTHON=ON",
            "-DZENITH_BUILD_TESTS=OFF",
        ]

        # CUDA Detection logic
        # 1. Explicit env var
        enable_cuda = os.environ.get("ZENITH_ENABLE_CUDA", "")
        if enable_cuda == "1" or enable_cuda.lower() == "on":
            cmake_args.append("-DZENITH_ENABLE_CUDA=ON")
        elif enable_cuda == "0" or enable_cuda.lower() == "off":
            cmake_args.append("-DZENITH_ENABLE_CUDA=OFF")
        else:
            # 2. Auto-detect nvcc
            try:
                subprocess.check_output(["nvcc", "--version"])
                print("Found nvcc, enabling CUDA support")
                cmake_args.append("-DZENITH_ENABLE_CUDA=ON")
            except (OSError, subprocess.CalledProcessError):
                print("nvcc not found, disabling CUDA support")
                cmake_args.append("-DZENITH_ENABLE_CUDA=OFF")

        build_args = []
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        print(f"Building extension {ext.name} with CMake args: {cmake_args}")
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    ext_modules=[CMakeExtension("zenith._zenith_core")],
    cmdclass={"build_ext": CMakeBuild},
)
