# ===----------------------------------------------------------------------===//
#
# Part of the Trident project, under the MIT License. SPDX-License-Identifier:
# MIT
#
# ===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.29)

project(TridentFFI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
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
set(Caffe2_DIR
    "${TRIDENT_TORCH_CMAKE_PREFIX}/Caffe2"
    CACHE PATH "PyTorch Caffe2 CMake package directory" FORCE)
set(Torch_DIR
    "${TRIDENT_TORCH_CMAKE_PREFIX}/Torch"
    CACHE PATH "PyTorch CMake package directory" FORCE)
set(tvm_ffi_DIR
    "${TRIDENT_TVM_FFI_CMAKE_PREFIX}/tvm_ffi"
    CACHE PATH "tvm_ffi CMake package directory" FORCE)
find_package(Torch CONFIG REQUIRED)
find_package(tvm_ffi CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${tvm_ffi_INCLUDE_DIR})
include_directories(${tvm_ffi_DLPACK_INCLUDE_DIR})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})

add_subdirectory(include)
add_subdirectory(lib)
