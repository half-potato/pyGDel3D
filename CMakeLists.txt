cmake_minimum_required(VERSION 3.16)
project(gflip3d CUDA CXX)

# Find CUDA toolkit which includes Thrust
find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(USE_SINGLE_PRECISION "Use single precision floating point" OFF)


set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
include_directories(${CONDA_PREFIX}/include)

# Find PyTorch
find_package(Torch REQUIRED)

# Add pybind11
add_subdirectory(pybind11)


# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/GDelFlipping/Main
    ${TORCH_INCLUDE_DIRS}
)

set(SOURCES
    GDelFlipping/src/DelaunayChecker.cpp
    GDelFlipping/src/InputCreator.cpp
    GDelFlipping/src/RandGen.cpp
    GDelFlipping/src/gDel3D/GpuDelaunay.cu
    GDelFlipping/src/gDel3D/CPU/predicates.cpp
    GDelFlipping/src/gDel3D/CPU/PredWrapper.cpp
    GDelFlipping/src/gDel3D/CPU/Splaying.cpp
    GDelFlipping/src/gDel3D/CPU/Star.cpp
    GDelFlipping/src/gDel3D/GPU/KerDivision.cu
    GDelFlipping/src/gDel3D/GPU/KerPredicates.cu
    GDelFlipping/src/gDel3D/GPU/ThrustWrapper.cu
)

# Create executable
add_executable(${PROJECT_NAME}
    GDelFlipping/src/Demo.cpp
    ${SOURCES}
)

pybind11_add_module(gdel3d 
    GDelFlipping/src/bindings.cpp
    ${SOURCES}
)
target_link_libraries(gdel3d PRIVATE ${TORCH_LIBRARIES})


# Configure precision
if(USE_SINGLE_PRECISION)
    target_compile_definitions(${PROJECT_NAME} PRIVATE REAL_TYPE_FP32)
    target_compile_definitions(gdel3d PRIVATE REAL_TYPE_FP32)
    message(STATUS "Building with single precision (32-bit) floating point")
else()
    message(STATUS "Building with double precision (64-bit) floating point")
endif()

# Set optimization level for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
