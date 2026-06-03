// TVMFFIDialect.cc - TVMFFI Dialect registration and initialization.
//
// This file implements the dialect `initialize()` method, which registers all
// ops and types defined in TVMFFI.td / TVMFFITypes.td via ODS-generated
// .cpp.inc files.

#include <cstddef>
#include <cstdint>

#include "libtriton-core/Conversion/TVMFFIToLLVM/ToRules.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIOps.h"
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"

#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFIDialect.cpp.inc"

#define GET_OP_CLASSES
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFI.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "libtriton-core/Dialect/TVMFFI/IR/TVMFFITypes.cpp.inc"

namespace libtriton::tvm_ffi {

namespace {

bool isSupportedAsType(mlir::Type type) {
  return mlir::isa<AnyType, mlir::LLVM::LLVMStructType>(type);
}

bool isSupportedAsConversionPair(mlir::Type inputType, mlir::Type outputType) {
  return (mlir::isa<AnyType>(inputType) &&
          mlir::isa<mlir::LLVM::LLVMStructType>(outputType)) ||
         (mlir::isa<AnyType>(outputType) &&
          mlir::isa<mlir::LLVM::LLVMStructType>(inputType));
}

bool isSupportedEnvTensorDType(mlir::Type type) {
  return mlir::isa<mlir::IntegerType>(type) || type.isF16() || type.isF32() ||
         type.isF64() || type.isBF16();
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
  if (!to::ToRuleSet::supports(inputType, outputType)) {
    return emitOpError() << "unsupported tvm_ffi.to conversion from "
                         << inputType << " to " << outputType;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult AsOp::verify() {
  const mlir::Type inputType = getInput().getType();
  const mlir::Type outputType = getOutput().getType();
  if (!isSupportedAsType(inputType)) {
    return emitOpError() << "unsupported input type for tvm_ffi.as: "
                         << inputType;
  } else if (!isSupportedAsType(outputType)) {
    return emitOpError() << "unsupported output type for tvm_ffi.as: "
                         << outputType;
  } else if (!isSupportedAsConversionPair(inputType, outputType)) {
    return emitOpError() << "unsupported tvm_ffi.as conversion from "
                         << inputType << " to " << outputType;
  } else {
    return mlir::success();
  }
}

mlir::LogicalResult EnvTensorAllocOp::verify() {
  mlir::Type dtype = getDtype();
  if (!isSupportedEnvTensorDType(dtype)) {
    return emitOpError() << "unsupported dtype for tvm_ffi.env_tensor_alloc: "
                         << dtype;
  }

  llvm::ArrayRef<int64_t> shape = getShape();
  if (shape.empty()) {
    return emitOpError() << "shape must be non-empty";
  }

  constexpr int64_t kI32Max = std::numeric_limits<int32_t>::max();
  if (shape.size() > kI32Max) {
    return emitOpError() << "rank exceeds i32 range";
  }

  for (int64_t dim : shape) {
    if (dim < 0) {
      return emitOpError() << "shape entries must be non-negative";
    }
  }

  return mlir::success();
}

} // namespace libtriton::tvm_ffi
