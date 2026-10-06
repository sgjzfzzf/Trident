cmake_minimum_required(VERSION 3.29)

project(TridentCore)

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

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(HandleLLVMOptions)
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_DISABLE_PRECOMPILE_HEADERS ON)

include(${LLVM_CMAKE_DIR}/AddLLVM.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIR.cmake)
include(${MLIR_CMAKE_DIR}/AddMLIRPython.cmake)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include_directories(SYSTEM ${CUDAToolkit_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
include_directories(SYSTEM ${tvm_ffi_INCLUDE_DIR})
include_directories(SYSTEM ${tvm_ffi_DLPACK_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

add_subdirectory(python)
add_subdirectory(test)
add_subdirectory(tools)
