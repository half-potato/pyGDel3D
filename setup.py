# setup.py
import os
import pathlib
import subprocess
import sys
import shutil

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import torch.utils

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(pathlib.Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        cfg = "Debug" if self.debug else "Release"
        cmake_prefix_path = torch.utils.cmake_prefix_path

        cmake_configure_cmd = [
            "cmake",
            "-S", ext.sourcedir,
            "-B", self.build_temp,
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        
        cmake_build_cmd = ["cmake", "--build", self.build_temp, "--config", cfg]

        print("Configuring CMake...")
        subprocess.run(cmake_configure_cmd, check=True)
        print("Building extension with CMake...")
        subprocess.run(cmake_build_cmd, check=True)

        # --- THIS IS THE ONLY LINE THAT CHANGED ---
        # Recursively search for the compiled library file (`**/*.so`)
        compiled_lib_path = list(pathlib.Path(self.build_temp).glob("**/*.so"))
        
        if not compiled_lib_path:
            raise RuntimeError("Could not find compiled .so file after build!")
        
        compiled_lib_path = compiled_lib_path[0]
        dest_path = pathlib.Path(self.get_ext_fullpath(ext.name))
        
        print(f"Found compiled library: {compiled_lib_path}")
        print(f"Moving to destination: {dest_path}")
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(compiled_lib_path), str(dest_path))

setup(
    ext_modules=[CMakeExtension("gdel3d._internal")],
    cmdclass={"build_ext": CMakeBuild},
)
