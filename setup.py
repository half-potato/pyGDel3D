import os
import sys
import glob
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# --- 1. File Discovery ---
# Automatically find all C++ and CUDA source files in your src directory
source_dir = "src/gdel3d"
cpp_sources = glob.glob(os.path.join(source_dir, "**/*.cpp"), recursive=True)
cuda_sources = glob.glob(os.path.join(source_dir, "**/*.cu"), recursive=True)
all_sources = cpp_sources + cuda_sources

# --- 2. Base Compiler Arguments ---
cxx_args = ["-O3", "-std=c++17"]
nvcc_args = ["-O3", "--use_fast_math", "-std=c++17"]

# --- 3. Implement CMakeLists.txt Features ---

# a) ABI Matching (replicates add_definitions(-D_GLIBCXX_USE_CXX11_ABI=...))
abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"
cxx_args.append(abi_flag)
# nvcc's host compiler will also use this flag
nvcc_args.append(abi_flag) 

# b) F32 / F64 Precision Flag (replicates option(USE_SINGLE_PRECISION ...))
if os.environ.get("USE_SINGLE_PRECISION", "0") == "1":
    print("Building with single precision (32-bit)")
    fp_flag = "-DREAL_TYPE_FP32"
    cxx_args.append(fp_flag)
    nvcc_args.append(fp_flag)
else:
    print("Building with double precision (64-bit)")


# --- 4. GPU Architecture Detection (from your example) ---
# Default fallback architectures
fallback_archs = ["-gencode=arch=compute_89,code=sm_89"]

if torch.cuda.is_available():
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        print(f"Detected GPU sm_{major}{minor}, using build flag: {arch_flag}")
        nvcc_args.append(arch_flag)
    except Exception as e:
        print(f"Failed to detect GPU, falling back to default archs. Error: {e}")
        nvcc_args.extend(fallback_archs)
else:
    print("CUDA not available, falling back to default archs.")
    nvcc_args.extend(fallback_archs)


# --- 5. The Main Setup Call ---
setup(
    name="gdel3d",
    packages=['gdel3d'],
    package_dir={'': 'src'}, # Important for src-layout
    ext_modules=[
        CUDAExtension(
            name="gdel3d._internal", # The name of the compiled module
            sources=all_sources,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
