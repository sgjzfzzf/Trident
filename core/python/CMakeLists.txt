set(CMAKE_PLATFORM_NO_VERSIONED_SONAME ON)

set(TRIDENT_CORE_PYTHON_INSTALL_ROOT "core")
set(TRIDENT_CORE_PYTHON_OUTPUT_ROOT "${CMAKE_BINARY_DIR}/python_packages/core")

file(RELATIVE_PATH TRIDENT_CORE_TORCH_MLIR_MODULE_SOURCE
     "${CMAKE_CURRENT_SOURCE_DIR}"
     "${torch-mlir_SOURCE_DIR}/python/TorchMLIRModule.cpp")

include_directories(${torch-mlir_SOURCE_DIR}/python)
include_directories(${torch-mlir_SOURCE_DIR}/include)
include_directories(${torch-mlir_BINARY_DIR}/include)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=trident.core.")

set(TRIDENT_CORE_MLIR_SOURCE_COMPONENTS
    MLIRPythonSources MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything)

declare_mlir_python_sources(TridentCorePythonExtensions)

declare_mlir_python_extension(
  TridentCorePythonExtensions.Torch
  MODULE_NAME
  _torchMlir
  ADD_TO_PARENT
  TridentCorePythonExtensions
  SOURCES
  ${TRIDENT_CORE_TORCH_MLIR_MODULE_SOURCE}
  EMBED_CAPI_LINK_LIBS
  TorchMLIRCAPI
  PRIVATE_LINK_LIBS
  LLVMSupport)

declare_mlir_python_extension(
  TridentCorePythonExtensions.CoreRegister
  MODULE_NAME
  _trident
  ADD_TO_PARENT
  TridentCorePythonExtensions
  SOURCES
  RegistrationModules.cc
  EMBED_CAPI_LINK_LIBS
  TridentCoreCAPI
  PRIVATE_LINK_LIBS
  LLVMSupport)

list(APPEND TRIDENT_CORE_MLIR_SOURCE_COMPONENTS TridentCorePythonExtensions)

add_mlir_python_common_capi_library(
  TridentCoreMLIRAggregateCAPI
  INSTALL_COMPONENT
  TridentCorePythonModules
  INSTALL_DESTINATION
  ${TRIDENT_CORE_PYTHON_INSTALL_ROOT}/_mlir_libs
  OUTPUT_DIRECTORY
  "${TRIDENT_CORE_PYTHON_OUTPUT_ROOT}/_mlir_libs"
  RELATIVE_INSTALL_ROOT
  ".."
  DECLARED_SOURCES
  ${TRIDENT_CORE_MLIR_SOURCE_COMPONENTS})

add_mlir_python_modules(
  TridentCoreMLIRPythonModules
  ROOT_PREFIX
  "${TRIDENT_CORE_PYTHON_OUTPUT_ROOT}"
  INSTALL_PREFIX
  "${TRIDENT_CORE_PYTHON_INSTALL_ROOT}"
  DECLARED_SOURCES
  ${TRIDENT_CORE_MLIR_SOURCE_COMPONENTS}
  COMMON_CAPI_LINK_LIBS
  TridentCoreMLIRAggregateCAPI)

declare_mlir_python_sources(TridentCorePythonSources)

declare_mlir_python_sources(
  TridentCorePythonSources.Core
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/trident_core"
  ADD_TO_PARENT
  TridentCorePythonSources
  SOURCES
  __init__.py
  dialects/torchext.py
  dialects/tvm_ffi.py)

declare_mlir_python_sources(
  TridentCorePythonSources.TorchMlirCompat
  ROOT_DIR
  "${torch-mlir_SOURCE_DIR}/python/torch_mlir"
  ADD_TO_PARENT
  TridentCorePythonSources
  SOURCES
  compiler_utils.py
  dialects/torch/__init__.py
  extras/fx_decomp_util.py
  extras/fx_importer.py
  fx.py)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  TridentCorePythonSources.Core
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE
  dialects/TorchExtOpsPyBind.td
  DIALECT_NAME
  torchext)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  TridentCorePythonSources.Core
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE
  dialects/TVMFFIOpsPyBind.td
  DIALECT_NAME
  tvm_ffi)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  TridentCorePythonSources.Core
  ROOT_DIR
  "${torch-mlir_SOURCE_DIR}/python/torch_mlir"
  TD_FILE
  dialects/TorchBinding.td
  DIALECT_NAME
  torch)

add_mlir_python_sources_target(
  TridentCorePythonSourcesTarget
  INSTALL_COMPONENT
  TridentCorePythonModules
  INSTALL_DIR
  ${TRIDENT_CORE_PYTHON_INSTALL_ROOT}
  OUTPUT_DIRECTORY
  ${TRIDENT_CORE_PYTHON_OUTPUT_ROOT}
  SOURCES_TARGETS
  TridentCorePythonSources.Core
  TridentCorePythonSources.Core.torchext.ops_gen
  TridentCorePythonSources.Core.tvm_ffi.ops_gen
  TridentCorePythonSources.Core.torch.ops_gen
  TridentCorePythonSources.TorchMlirCompat)

build_nanobind_lib(
  INSTALL_COMPONENT
  TridentCorePythonModules
  INSTALL_DESTINATION
  ${TRIDENT_CORE_PYTHON_INSTALL_ROOT}
  OUTPUT_DIRECTORY
  ${TRIDENT_CORE_PYTHON_OUTPUT_ROOT}
  MLIR_BINDINGS_PYTHON_NB_DOMAIN
  trident_core)

install(TARGETS TridentCoreRuntime
        LIBRARY DESTINATION ${TRIDENT_CORE_PYTHON_INSTALL_ROOT}/_runtime_libs
                COMPONENT TridentCorePythonModules)

file(GLOB _mlir_runtime_libs "${MLIR_DIR}/../../*_runtime.so*")
install(
  FILES ${_mlir_runtime_libs}
  DESTINATION ${TRIDENT_CORE_PYTHON_INSTALL_ROOT}/_runtime_libs
  COMPONENT TridentCorePythonModules)

add_custom_target(
  TridentCorePythonModules ALL
  DEPENDS TridentCoreRuntime TridentCoreMLIRPythonModules
          TridentCorePythonSourcesTarget ${NB_LIBRARY_TARGET_NAME})

add_dependencies(TridentCorePythonModules TorchMLIRCAPI)
