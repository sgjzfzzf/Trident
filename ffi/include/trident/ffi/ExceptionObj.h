//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// ExceptionObj / Exception — a tvm::ffi::Object for composable error handling
// in FFI runtime operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_FFI_EXCEPTIONOBJ_H_
#define TRIDENT_FFI_EXCEPTIONOBJ_H_

#include <string>
#include <tvm/ffi/object.h>
#include <tvm/ffi/type_traits.h>

namespace trident::runtime {

//===----------------------------------------------------------------------===//
// ExceptionObj — the ref-counted heap object
//===----------------------------------------------------------------------===//

class ExceptionObj : public tvm::ffi::Object {
public:
  ExceptionObj(const std::string &);
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("trident.ffi.Exception", ExceptionObj,
                                    tvm::ffi::Object);
  static constexpr TVMFFISEqHashKind _type_s_eq_hash_kind =
      kTVMFFISEqHashKindUniqueInstance;
  const std::string &GetKind() const &;

private:
  std::string kind_;
};

} // namespace trident::runtime

namespace tvm::ffi {
template <>
struct TypeTraits<trident::runtime::ExceptionObj> : public TypeTraitsBase {};
} // namespace tvm::ffi

#endif // TRIDENT_FFI_EXCEPTIONOBJ_H_
