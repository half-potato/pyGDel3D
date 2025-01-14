cmake_minimum_required(VERSION 3.14)
project(dtet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/CMake"
  ${CMAKE_MODULE_PATH}
  )

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Release")
include_directories(${CONDA_PREFIX}/include)
include_directories(/usr/include)

# Find PyTorch
find_package(Torch REQUIRED)

# Add pybind11
add_subdirectory(pybind11)

# Add CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Find CGAL and its components
find_package(CGAL REQUIRED COMPONENTS Core)
if(NOT CGAL_FOUND)
    message(FATAL_ERROR "CGAL not found!")
endif()

find_package( TBB )
include(CGAL_TBB_support) # Includes a module from CGAL

# Add NumCpp
find_package(NumCpp)

set(SOURCES
    src/main.cpp
)

# Create the extension module
pybind11_add_module(dtet ${SOURCES})

# Link against CGAL and its dependencies
target_link_libraries(dtet PRIVATE 
    CGAL::CGAL
    CGAL::CGAL_Core
    ${CGAL_LIBRARIES}
    ${CGAL_3RD_PARTY_LIBRARIES}
    TBB::tbb
    CGAL::TBB_support
)

# Include directories
target_include_directories(dtet PRIVATE
    ${CGAL_INCLUDE_DIRS}
    ${NUMCPP_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Print CGAL directories for debugging
message(STATUS "CGAL_INCLUDE_DIRS: ${CGAL_INCLUDE_DIRS}")
message(STATUS "CGAL_LIBRARIES: ${CGAL_LIBRARIES}")
