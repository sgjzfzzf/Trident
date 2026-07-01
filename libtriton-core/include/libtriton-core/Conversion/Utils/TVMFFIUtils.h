//===----------------------------------------------------------------------===//
//
// Part of the LibTriton project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBTRITON_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
#define LIBTRITON_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_

#include "libtriton-core/Conversion/Utils/GlobalString.h"
#include "libtriton-core/Conversion/Utils/TVMFFICAPIDescriptors.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace libtriton::conversion::utils {

/// Call a TVM FFI global function by name.
///
/// Encapsulates the common pattern:
///   TVMFFIFunctionGetGlobal → TVMFFIFunctionCall → TVMFFIObjectDecRef
///
/// 1. Creates a TVMFFIByteArray with the function name (via global string).
/// 2. Obtains the function handle via TVMFFIFunctionGetGlobal.
/// 3. Copies each pre-built TVMFFIAny slot into a contiguous args array.
/// 4. Calls TVMFFIFunctionCall with the packed arguments.
/// 5. Releases the function handle via TVMFFIObjectDecRef.
/// 6. Extracts and returns the result i64 value.
///
/// \param builder   The op builder (insertion point must be valid).
/// \param loc       Source location for generated ops.
/// \param moduleOp  Parent module (for declaring LLVM function symbols).
/// \param funcName  The TVM FFI function name (e.g. "ffi.Array").
/// \param args      Pre-built TVMFFIAny* slots (each a !llvm.ptr to an
///                  alloca'd {i32, i32, i64}).
/// \return A pointer to the result TVMFFIAny slot (!llvm.ptr to {i32,i32,i64})
///         on the stack, or failure.  The caller extracts the desired field
///         (e.g. field[2] for v_int64/v_obj).
mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         llvm::ArrayRef<mlir::Value> args);

/// Call a TVM FFI global function with a pre-built contiguous args array
/// and a runtime-determined number of arguments.
///
/// Same pattern (GetGlobal → Call → DecRef) but the caller provides
/// the args array already populated instead of individual slots, and
/// \p numArgs is a runtime i32 Value instead of a compile-time constant.
///
/// \param argsArray A !llvm.ptr to a contiguous array of TVMFFIAny elements.
/// \param numArgs   A runtime i32 value specifying how many elements to pass.
/// \return A pointer to the result TVMFFIAny slot (!llvm.ptr to {i32,i32,i64})
///         on the stack, or failure.
mlir::FailureOr<mlir::Value>
callTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ModuleOp moduleOp, llvm::StringRef funcName,
                         mlir::Value argsArray, mlir::Value numArgs);

} // namespace libtriton::conversion::utils

#endif // LIBTRITON_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
