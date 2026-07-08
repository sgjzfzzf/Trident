cmake_minimum_required(VERSION 3.29)

project(Trident)

include(ExternalProject)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(Python_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

ExternalProject_Add(
  llvm-project
  GIT_REPOSITORY https://github.com/llvm/llvm-project.git
  GIT_TAG 977355be38d10c62c3d5498a64618ec926348252
  SOURCE_SUBDIR llvm
  INSTALL_DIR ${CMAKE_BINARY_DIR}/install/llvm-project
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
             -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}
             -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install/llvm-project
             -DCMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}
             -DLLVM_ENABLE_LLD=${LLVM_ENABLE_LLD}
             -DLLVM_ENABLE_PROJECTS=mlir
             -DLLVM_INCLUDE_UTILS=ON
             -DLLVM_INSTALL_UTILS=ON
             -DMLIR_ENABLE_BINDINGS_PYTHON=ON
             -DMLIR_ENABLE_CUDA_RUNNER=ON
             -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
             -DPython_EXECUTABLE=${Python_EXECUTABLE}
             -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
             -DPython_FIND_VIRTUALENV=ONLY
             -DPython3_FIND_VIRTUALENV=ONLY)

ExternalProject_Add(
  trident-core
  SOURCE_DIR ${CMAKE_SOURCE_DIR}/trident-core
  BINARY_DIR ${CMAKE_BINARY_DIR}/trident-core
  INSTALL_DIR ${CMAKE_BINARY_DIR}/install/trident-core
  DEPENDS llvm-project
  CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
             -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
             -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
             -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
             -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
             -DCMAKE_EXPORT_COMPILE_COMMANDS=${CMAKE_EXPORT_COMPILE_COMMANDS}
             -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install/trident-core
             -DCMAKE_LINKER_TYPE=${CMAKE_LINKER_TYPE}
             -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
             -DLLVM_DIR=${CMAKE_BINARY_DIR}/install/llvm-project/lib/cmake/llvm
             -DLLVM_ENABLE_LLD=${LLVM_ENABLE_LLD}
             -DMLIR_DIR=${CMAKE_BINARY_DIR}/install/llvm-project/lib/cmake/mlir
             -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
             -DPython_EXECUTABLE=${Python_EXECUTABLE}
             -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
             -DPython_FIND_VIRTUALENV=ONLY
             -DPython3_FIND_VIRTUALENV=ONLY)

install(
  DIRECTORY
    ${CMAKE_BINARY_DIR}/install/trident-core/python_packages/trident-core/trident_core
  DESTINATION .)
