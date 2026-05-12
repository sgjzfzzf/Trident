// TVMFFIDialect.cc - TVMFFI Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TVMFFI.td / TVMFFITypes.td via ODS-generated
// .cpp.inc files.

#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

namespace libtriton::tvm_ffi {

namespace {

bool isSupportedToType(mlir::Type type) {
  return mlir::isa<libtriton::tvm_ffi::AnyType,
                   libtriton::tvm_ffi::ObjectHandleType,
                   libtriton::dlpack::DLTensorType, mlir::LLVM::LLVMPointerType,
                   mlir::Float64Type, mlir::IntegerType>(type);
}

bool isSupportedToConversionPair(mlir::Type inputType, mlir::Type outputType) {
  const bool inputIsAny = mlir::isa<libtriton::tvm_ffi::AnyType>(inputType);
  const bool outputIsAny = mlir::isa<libtriton::tvm_ffi::AnyType>(outputType);
  return (inputIsAny && outputIsAny) ||
         (inputIsAny &&
          mlir::isa<mlir::IntegerType, mlir::Float64Type,
                    mlir::LLVM::LLVMPointerType,
                    libtriton::tvm_ffi::ObjectHandleType,
                    libtriton::dlpack::DLTensorType>(outputType)) ||
         (outputIsAny &&
          mlir::isa<mlir::IntegerType, mlir::Float64Type,
                    mlir::LLVM::LLVMPointerType,
                    libtriton::tvm_ffi::ObjectHandleType>(inputType));
}

} // namespace

void TVMFFIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"
      >();
}

mlir::LogicalResult ToOp::verify() {
  const mlir::Type inputType = getInput().getType();
  const mlir::Type outputType = getOutput().getType();
  if (!isSupportedToType(inputType)) {
    return emitOpError() << "unsupported input type for tvm_ffi.to: "
                         << inputType;
  } else if (!isSupportedToType(outputType)) {
    return emitOpError() << "unsupported output type for tvm_ffi.to: "
                         << outputType;
  } else if (!isSupportedToConversionPair(inputType, outputType)) {
    return emitOpError() << "unsupported tvm_ffi.to conversion from "
                         << inputType << " to " << outputType;
  } else {
    return mlir::success();
  }
}

} // namespace libtriton::tvm_ffi
