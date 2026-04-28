cmake_minimum_required(VERSION 3.29)

project(LibTritonCore)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(MLIR REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(
  Torch CONFIG REQUIRED PATHS ${Python3_SITELIB}/torch/share/cmake/Torch
  ${Python3_SITEARCH}/torch/share/cmake/Torch)
find_package(CUDAToolkit REQUIRED)
find_package(
  tvm_ffi CONFIG REQUIRED PATHS ${Python3_SITELIB}/tvm_ffi/share/cmake/tvm_ffi
  ${Python3_SITEARCH}/tvm_ffi/share/cmake/tvm_ffi)

set(LIBTRITON_CORE_ENABLE_PYTHON_BINDINGS
    ON
    CACHE BOOL "Enable libtriton-core Python bindings")
set(TORCH_MLIR_OUT_OF_TREE_BUILD
    ON
    CACHE BOOL "Force torch-mlir out-of-tree mode in nested builds" FORCE)
set(TORCH_MLIR_PYTHON_PACKAGE_PREFIX
    "libtriton._C.torch_mlir."
    CACHE STRING "Torch-mlir python package prefix" FORCE)
set(TORCH_MLIR_PYTHON_PACKAGES_DIR
    "${CMAKE_BINARY_DIR}/python_packages"
    CACHE PATH "Torch-mlir python package output dir" FORCE)

FetchContent_Declare(
  torch-mlir
  GIT_REPOSITORY https://github.com/llvm/torch-mlir.git
  GIT_TAG 68213efa76ab3944445b0a0d6073efe7b7b3d039
  GIT_SUBMODULES "externals/stablehlo")
FetchContent_MakeAvailable(torch-mlir)

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${torch-mlir_SOURCE_DIR}/include)
include_directories(${torch-mlir_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(LIBTRITON_CORE_ENABLE_PYTHON_BINDINGS)
  find_package(Python REQUIRED COMPONENTS Interpreter Development)
  find_package(nanobind CONFIG REQUIRED PATHS ${Python_SITELIB}/nanobind/cmake
               ${Python_SITEARCH}/nanobind/cmake)
  include(${MLIR_CMAKE_DIR}/AddMLIRPython.cmake)
  add_subdirectory(python)
endif()

add_subdirectory(test)
