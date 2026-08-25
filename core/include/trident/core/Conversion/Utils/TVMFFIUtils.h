//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace trident::conversion::utils {

/// Allocate a TVMFFIAny slot containing an integer scalar.
mlir::Value buildIntAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                            int64_t value);

/// Allocate a TVMFFIAny slot containing a runtime i32 integer.
mlir::Value buildIntAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value);

/// Allocate a TVMFFIAny slot containing a DLPack dtype value.
mlir::Value buildDTypeAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                              int64_t code, int64_t bits, int64_t lanes = 1);

/// Allocate a TVMFFIAny slot containing an opaque pointer.
mlir::Value buildOpaquePtrAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value pointer);

/// Load an integer value from a TVMFFIAny result slot.
mlir::Value loadIntFromAnySlot(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value slot);

/// Get an owned TVM FFI global function handle by name.
mlir::FailureOr<mlir::Value> getTVMFFIGlobalFunction(mlir::OpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::ModuleOp moduleOp,
                                                     llvm::StringRef funcName);

/// Call a TVM FFI function handle.
mlir::LogicalResult
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   llvm::ArrayRef<mlir::Value> args, mlir::Value resultSlot);

mlir::LogicalResult
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   mlir::Value argsArray, mlir::Value numArgs,
                   mlir::Value resultSlot);

/// Call a TVM FFI global function by name.
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
/// Same pattern (GetGlobal -> Call -> DecRef) but the caller provides
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

} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
