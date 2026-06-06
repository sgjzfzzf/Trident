set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(LIBTRITON_CORE_PYTHON_PACKAGE_NAME "libtriton-core")
set(LIBTRITON_CORE_MLIR_PYTHON_PACKAGE_PREFIX "libtriton._C.libtriton_core.")
set(LIBTRITON_CORE_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtriton_core")
set(LIBTRITON_CORE_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python_packages")
set(LIBTRITON_CORE_PYTHON_INSTALL_ROOT
    "python_packages/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core")
set(LIBTRITON_CORE_PYTHON_OUTPUT_ROOT
    "${LIBTRITON_CORE_PYTHON_PACKAGES_DIR}/${LIBTRITON_CORE_PYTHON_PACKAGE_NAME}/libtriton_core"
)
file(RELATIVE_PATH LIBTRITON_CORE_TORCH_MLIR_MODULE_SOURCE
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${torch-mlir_SOURCE_DIR}/python/TorchMLIRModule.cpp")

include_directories(${torch-mlir_SOURCE_DIR}/python)
include_directories(${torch-mlir_SOURCE_DIR}/include)
include_directories(${torch-mlir_BINARY_DIR}/include)

add_compile_definitions(
  "MLIR_PYTHON_PACKAGE_PREFIX=${LIBTRITON_CORE_MLIR_PYTHON_PACKAGE_PREFIX}")

set(LIBTRITON_CORE_MLIR_SOURCE_COMPONENTS
    MLIRPythonSources MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything)

declare_mlir_python_sources(LibTritonCorePythonExtensions)

declare_mlir_python_extension(
  LibTritonCorePythonExtensions.Torch
  MODULE_NAME
  _torchMlir
  ADD_TO_PARENT
  LibTritonCorePythonExtensions
  SOURCES
  ${LIBTRITON_CORE_TORCH_MLIR_MODULE_SOURCE}
  EMBED_CAPI_LINK_LIBS
  TorchMLIRCAPI
  PRIVATE_LINK_LIBS
  LLVMSupport)

declare_mlir_python_extension(
  LibTritonCorePythonExtensions.CoreRegister
  MODULE_NAME
  _libtritonCore
  ADD_TO_PARENT
  LibTritonCorePythonExtensions
  SOURCES
  RegistrationModules.cc
  EMBED_CAPI_LINK_LIBS
  LibTritonCoreCAPI
  PRIVATE_LINK_LIBS
  LLVMSupport)

list(APPEND LIBTRITON_CORE_MLIR_SOURCE_COMPONENTS LibTritonCorePythonExtensions)

add_mlir_python_common_capi_library(
  LibTritonCoreMLIRAggregateCAPI
  INSTALL_COMPONENT
  LibTritonCorePythonModules
  INSTALL_DESTINATION
  ${LIBTRITON_CORE_PYTHON_INSTALL_ROOT}/_mlir_libs
  OUTPUT_DIRECTORY
  "${LIBTRITON_CORE_PYTHON_OUTPUT_ROOT}/_mlir_libs"
  RELATIVE_INSTALL_ROOT
  ".."
  DECLARED_SOURCES
  ${LIBTRITON_CORE_MLIR_SOURCE_COMPONENTS})

add_mlir_python_modules(
  LibTritonCoreMLIRPythonModules
  ROOT_PREFIX
  "${LIBTRITON_CORE_PYTHON_OUTPUT_ROOT}"
  INSTALL_PREFIX
  "${LIBTRITON_CORE_PYTHON_INSTALL_ROOT}"
  DECLARED_SOURCES
  ${LIBTRITON_CORE_MLIR_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
  LibTritonCoreMLIRAggregateCAPI)

declare_mlir_python_sources(LibTritonCorePythonSources)

declare_mlir_python_sources(
  LibTritonCorePythonSources.Core
  ROOT_DIR
  "${LIBTRITON_CORE_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT
  LibTritonCorePythonSources
  SOURCES
  __init__.py
  capi_utils.py
  dialects/aoti.py
  dialects/tvm_ffi.py)

declare_mlir_python_sources(
  LibTritonCorePythonSources.TorchMlirCompat
  ROOT_DIR
  "${torch-mlir_SOURCE_DIR}/python/torch_mlir"
  ADD_TO_PARENT
  LibTritonCorePythonSources
  SOURCES
  compiler_utils.py
  dialects/torch/__init__.py
  fx.py)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  LibTritonCorePythonSources.Core
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE
  dialects/AOTInductorOpsPyBind.td
  DIALECT_NAME
  aoti)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  LibTritonCorePythonSources.Core
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE
  dialects/TVMFFIOpsPyBind.td
  DIALECT_NAME
  tvm_ffi)


declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  LibTritonCorePythonSources.Core
  ROOT_DIR
  "${torch-mlir_SOURCE_DIR}/python/torch_mlir"
  TD_FILE
  dialects/TorchBinding.td
  DIALECT_NAME
  torch)

add_mlir_python_sources_target(
  LibTritonCorePythonSourcesTarget
  INSTALL_COMPONENT
  LibTritonCorePythonModules
  INSTALL_DIR
  ${LIBTRITON_CORE_PYTHON_INSTALL_ROOT}
  OUTPUT_DIRECTORY
  ${LIBTRITON_CORE_PYTHON_OUTPUT_ROOT}
  SOURCES_TARGETS
  LibTritonCorePythonSources.Core
  LibTritonCorePythonSources.Core.aoti.ops_gen
  LibTritonCorePythonSources.Core.tvm_ffi.ops_gen
  LibTritonCorePythonSources.Core.torch.ops_gen
  LibTritonCorePythonSources.TorchMlirCompat)

build_nanobind_lib(
  INSTALL_COMPONENT
  LibTritonCorePythonModules
  INSTALL_DESTINATION
  ${LIBTRITON_CORE_PYTHON_INSTALL_ROOT}
  OUTPUT_DIRECTORY
  ${LIBTRITON_CORE_PYTHON_OUTPUT_ROOT}
  MLIR_BINDINGS_PYTHON_NB_DOMAIN
  libtriton_core)

install(TARGETS LibTritonCoreRuntime
        LIBRARY DESTINATION ${LIBTRITON_CORE_PYTHON_INSTALL_ROOT}/_runtime_libs
                COMPONENT LibTritonCorePythonModules)

get_filename_component(_mlir_lib_dir "${MLIR_DIR}/../.." ABSOLUTE)
install(
  FILES "${_mlir_lib_dir}/libmlir_cuda_runtime.so"
  DESTINATION ${LIBTRITON_CORE_PYTHON_INSTALL_ROOT}/_runtime_libs
  COMPONENT LibTritonCorePythonModules
  OPTIONAL)

add_custom_target(
  LibTritonCorePythonModules ALL
  DEPENDS LibTritonCoreRuntime LibTritonCoreMLIRPythonModules
          LibTritonCorePythonSourcesTarget ${NB_LIBRARY_TARGET_NAME})

add_dependencies(LibTritonCorePythonModules TorchMLIRCAPI)

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-LibTritonCorePythonModules DEPENDS LibTritonCorePythonModules
    COMPONENT LibTritonCorePythonModules)
endif()
