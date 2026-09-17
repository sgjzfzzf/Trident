//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.

// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
#define TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>
#include <tuple>

namespace trident::conversion::utils {

/// Get an owned TVM FFI global function handle by name.
///
/// The name is resolved once per module: the handle and the status reported
/// by `TVMFFIFunctionGetGlobal` are cached in module-level globals filled by a
/// generated constructor, and a matching destructor drops that reference on
/// unload. Call sites load the cached values instead of repeating the lookup,
/// and retain the handle so the caller still owns the returned reference.
///
/// \return The cached function handle and the i32 status of the load-time
///         lookup (zero on success).
mlir::FailureOr<std::tuple<mlir::Value, mlir::Value>>
getTVMFFIGlobalFunction(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::ModuleOp moduleOp, llvm::StringRef funcName);

/// Call a borrowed TVM FFI function handle.
mlir::FailureOr<mlir::Value>
callTVMFFIFunction(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::ModuleOp moduleOp, mlir::Value funcHandle,
                   llvm::ArrayRef<mlir::Value> args, mlir::Value resultSlot);

mlir::FailureOr<mlir::Value>
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
} // namespace trident::conversion::utils

#endif // TRIDENT_CORE_CONVERSION_UTILS_TVMFFIUTILS_H_
