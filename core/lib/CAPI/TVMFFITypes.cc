//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFITypes.h"
#include "trident/core/Dialect/TVMFFI/IR/TVMFFITypes.h"
#include <cstdint>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Support/LLVM.h>

#define DEFINE_SIMPLE_TYPE(Name)                                               \
  MlirType mlirTVMFFI##Name##TypeGet(MlirContext context) {                    \
    return wrap(trident::tvm_ffi::Name##Type::get(unwrap(context)));           \
  }                                                                            \
  MlirStringRef mlirTVMFFI##Name##TypeGetName(void) {                          \
    return wrap(trident::tvm_ffi::Name##Type::name);                           \
  }                                                                            \
  MlirTypeID mlirTVMFFI##Name##TypeGetTypeID(void) {                           \
    return wrap(trident::tvm_ffi::Name##Type::getTypeID());                    \
  }                                                                            \
  bool mlirTypeIsATVMFFI##Name##Type(MlirType type) {                          \
    return mlir::isa<trident::tvm_ffi::Name##Type>(unwrap(type));              \
  }

DEFINE_SIMPLE_TYPE(Any)
DEFINE_SIMPLE_TYPE(Array)
DEFINE_SIMPLE_TYPE(Bool)
DEFINE_SIMPLE_TYPE(DType)
DEFINE_SIMPLE_TYPE(Device)
DEFINE_SIMPLE_TYPE(Exception)
DEFINE_SIMPLE_TYPE(Float)
DEFINE_SIMPLE_TYPE(Function)
DEFINE_SIMPLE_TYPE(Int)
DEFINE_SIMPLE_TYPE(None)
DEFINE_SIMPLE_TYPE(Object)
DEFINE_SIMPLE_TYPE(RawStr)
DEFINE_SIMPLE_TYPE(SmallStr)
DEFINE_SIMPLE_TYPE(Str)
DEFINE_SIMPLE_TYPE(Tensor)

#undef DEFINE_SIMPLE_TYPE

MlirType mlirTVMFFIUnionTypeGetChecked(MlirLocation location, intptr_t numTypes,
                                       const MlirType *types) {
  const mlir::Location loc = unwrap(location);
  llvm::SmallVector<mlir::Type> members;
  members.reserve(numTypes);
  for (intptr_t index = 0; index < numTypes; ++index) {
    const mlir::Type member = unwrap(types[index]);
    if (member.getContext() != loc.getContext()) {
      mlir::emitError(loc)
          << "union members must belong to the construction context";
      return MlirType{nullptr};
    }
    members.push_back(member);
  }
  return wrap(trident::tvm_ffi::UnionType::getChecked(
      loc, loc.getContext(), llvm::ArrayRef<mlir::Type>(members)));
}

MlirStringRef mlirTVMFFIUnionTypeGetName(void) {
  return wrap(trident::tvm_ffi::UnionType::name);
}

intptr_t mlirTVMFFIUnionTypeGetNumTypes(MlirType type) {
  return static_cast<intptr_t>(
      mlir::cast<trident::tvm_ffi::UnionType>(unwrap(type)).getTypes().size());
}

MlirType mlirTVMFFIUnionTypeGetType(MlirType type, intptr_t index) {
  return wrap(
      mlir::cast<trident::tvm_ffi::UnionType>(unwrap(type)).getTypes()[index]);
}

MlirTypeID mlirTVMFFIUnionTypeGetTypeID(void) {
  return wrap(trident::tvm_ffi::UnionType::getTypeID());
}

bool mlirTypeIsATVMFFIUnionType(MlirType type) {
  return mlir::isa<trident::tvm_ffi::UnionType>(unwrap(type));
}
