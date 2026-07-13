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

find_package(
  tvm_ffi CONFIG REQUIRED PATHS ${Python3_SITELIB}/tvm_ffi/share/cmake/tvm_ffi
  ${Python3_SITEARCH}/tvm_ffi/share/cmake/tvm_ffi)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${tvm_ffi_INCLUDE_DIR})
include_directories(${tvm_ffi_DLPACK_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(python)
add_subdirectory(lib)
