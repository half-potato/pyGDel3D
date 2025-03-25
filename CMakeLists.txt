cmake_minimum_required(VERSION 3.24)

project(gflip3d LANGUAGES CUDA CXX)
# link_directories(BEFORE ${CUDAToolkit_LIBRARY_DIR})
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})


enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
option(USE_SINGLE_PRECISION "Use single precision floating point" OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

# Ensure ABI matches PyTorch exactly
execute_process(
    COMMAND python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})

# Automatically get Torch's path
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

# Find PyTorch
find_package(Torch REQUIRED)

execute_process(
    COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'include')"
    OUTPUT_VARIABLE TORCH_INCLUDE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND python -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent / 'lib' / 'libtorch_python.so')"
    OUTPUT_VARIABLE TORCH_PYTHON_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Ensure Python interpreter and dev components match PyTorch's Python
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Include directories explicitly
include_directories(
    ${CMAKE_SOURCE_DIR}/GDelFlipping/Main
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_EXTENSION_INCLUDE} # PyTorch’s own PyBind11!
    ${Python3_INCLUDE_DIRS}
)
set(SOURCES
    GDelFlipping/src/DelaunayChecker.cpp
    GDelFlipping/src/InputCreator.cpp
    GDelFlipping/src/RandGen.cpp
    GDelFlipping/src/gDel3D/CPU/Splaying.cpp
    GDelFlipping/src/gDel3D/CPU/Star.cpp
    GDelFlipping/src/gDel3D/CPU/predicates.cpp
    GDelFlipping/src/gDel3D/CPU/PredWrapper.cpp
)

set(CUDA_SOURCES
    GDelFlipping/src/gDel3D/GpuDelaunay.cu
    GDelFlipping/src/gDel3D/GPU/KerDivision.cu
    GDelFlipping/src/gDel3D/GPU/KerPredicates.cu
    GDelFlipping/src/gDel3D/GPU/ThrustWrapper.cu
)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

# Create executable
add_executable(${PROJECT_NAME}
    GDelFlipping/src/Demo.cpp
    ${SOURCES}
    ${CUDA_SOURCES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${Python3_LIBRARIES})

# PyBind11 module using PyTorch’s built-in headers
add_library(gdel3d.so MODULE
    GDelFlipping/src/bindings.cpp
    ${SOURCES}
    ${CUDA_SOURCES}
)
target_link_libraries(gdel3d.so PRIVATE ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${TORCH_PYTHON_LIB})

# Ensure proper naming (required by Python)
set_target_properties(gdel3d.so PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")

# Configure precision option
if(USE_SINGLE_PRECISION)
    target_compile_definitions(${PROJECT_NAME} PRIVATE REAL_TYPE_FP32)
    target_compile_definitions(gdel3d PRIVATE REAL_TYPE_FP32)
    message(STATUS "Building with single precision (32-bit) floating point")
else()
    message(STATUS "Building with double precision (64-bit) floating point")
endif()

# Optimization and debug flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

