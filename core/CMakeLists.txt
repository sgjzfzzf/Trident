cmake_minimum_required(VERSION 3.29)

project(TridentCore)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE TRIDENT_NANOBIND_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TRIDENT_TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(
  COMMAND
    "${Python3_EXECUTABLE}" -c
    "import pathlib, tvm_ffi; print(pathlib.Path(tvm_ffi.__file__).parent / 'share' / 'cmake')"
  OUTPUT_VARIABLE TRIDENT_TVM_FFI_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(MLIR REQUIRED CONFIG)
set(Caffe2_DIR
    "${TRIDENT_TORCH_CMAKE_PREFIX}/Caffe2"
    CACHE PATH "PyTorch Caffe2 CMake package directory" FORCE)
set(nanobind_DIR
    "${TRIDENT_NANOBIND_CMAKE_DIR}"
    CACHE PATH "nanobind CMake package directory" FORCE)
set(Torch_DIR
    "${TRIDENT_TORCH_CMAKE_PREFIX}/Torch"
    CACHE PATH "PyTorch CMake package directory" FORCE)
set(tvm_ffi_DIR
    "${TRIDENT_TVM_FFI_CMAKE_PREFIX}/tvm_ffi"
    CACHE PATH "tvm_ffi CMake package directory" FORCE)
find_package(nanobind CONFIG REQUIRED)
find_package(Torch CONFIG REQUIRED)
find_package(tvm_ffi CONFIG REQUIRED)

set(TORCH_MLIR_OUT_OF_TREE_BUILD
    ON
    CACHE BOOL "Force torch-mlir out-of-tree mode in nested builds" FORCE)
set(TORCH_MLIR_PYTHON_PACKAGE_PREFIX
    "trident.core.torch_mlir."
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
  GIT_TAG 68213efa76ab3944445b0a0d6073efe7b7b3d039
  GIT_SUBMODULES "")
FetchContent_MakeAvailable(torch-mlir)

unset(TORCH_MLIR_ENABLE_STABLEHLO)
unset(TORCH_MLIR_ENABLE_TOSA)

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIRPython.cmake)

include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(SYSTEM ${tvm_ffi_INCLUDE_DIR})
include_directories(SYSTEM ${tvm_ffi_DLPACK_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(SYSTEM ${torch-mlir_SOURCE_DIR}/include)
include_directories(SYSTEM ${torch-mlir_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
