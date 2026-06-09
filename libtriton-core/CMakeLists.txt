cmake_minimum_required(VERSION 3.29)

project(LibTritonCore)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(MLIR REQUIRED CONFIG)
find_package(nanobind CONFIG REQUIRED PATHS ${Python3_SITELIB}/nanobind/cmake
             ${Python3_SITEARCH}/nanobind/cmake)
find_package(
  Torch CONFIG REQUIRED PATHS ${Python3_SITELIB}/torch/share/cmake/Torch
  ${Python3_SITEARCH}/torch/share/cmake/Torch)
find_package(
  tvm_ffi CONFIG REQUIRED PATHS ${Python3_SITELIB}/tvm_ffi/share/cmake/tvm_ffi
  ${Python3_SITEARCH}/tvm_ffi/share/cmake/tvm_ffi)

set(TORCH_MLIR_OUT_OF_TREE_BUILD
    ON
    CACHE BOOL "Force torch-mlir out-of-tree mode in nested builds" FORCE)
set(TORCH_MLIR_PYTHON_PACKAGE_PREFIX
    "libtriton._C.torch_mlir."
    CACHE STRING "Torch-mlir python package prefix" FORCE)
set(TORCH_MLIR_PYTHON_PACKAGES_DIR
    "${CMAKE_BINARY_DIR}/python_packages"
    CACHE PATH "Torch-mlir python package output dir" FORCE)

set(TORCH_MLIR_ENABLE_STABLEHLO
    OFF
    CACHE BOOL "Disable StableHLO in torch-mlir" FORCE)
set(TORCH_MLIR_ENABLE_TOSA
    OFF
    CACHE BOOL "Disable TOSA in torch-mlir" FORCE)

FetchContent_Declare(
  torch-mlir
  GIT_REPOSITORY https://github.com/llvm/torch-mlir.git
  GIT_TAG 68213efa76ab3944445b0a0d6073efe7b7b3d039)
FetchContent_MakeAvailable(torch-mlir)

unset(TORCH_MLIR_ENABLE_STABLEHLO)
unset(TORCH_MLIR_ENABLE_TOSA)

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIRPython.cmake)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${tvm_ffi_INCLUDE_DIR})
include_directories(${tvm_ffi_DLPACK_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(${torch-mlir_SOURCE_DIR}/include)
include_directories(${torch-mlir_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
