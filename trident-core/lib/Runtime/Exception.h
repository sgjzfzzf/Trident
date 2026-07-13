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
// Design:
//   ExceptionObj (heap Object)  ──ref-counted──  Exception (value-type handle)
//     | - bool is_ok_
//     | - Any  data_   (success value when is_ok_, Error when !is_ok_)
//
// Usage:
//   Exception m = Exception::ok(someAny);
//   Exception m = Exception::error(someError);
//   if (m.is_ok()) { handle(m.value()); }
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_RUNTIME_EXCEPTION_H_
#define TRIDENT_CORE_RUNTIME_EXCEPTION_H_

#include "tvm/ffi/object.h"
#include "tvm/ffi/string.h"
#include "tvm/ffi/type_traits.h"

namespace trident::runtime {

class ExceptionObj;
class Exception;

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
  friend Exception;

private:
  std::string kind_;
};

//===----------------------------------------------------------------------===//
// Exception — the ObjectRef handle (user-facing)
//===----------------------------------------------------------------------===//

class Exception : public tvm::ffi::ObjectRef {
public:
  Exception(const std::string &);
  Exception(tvm::ffi::ObjectPtr<ExceptionObj> data);
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(Exception, tvm::ffi::ObjectRef,
                                                ExceptionObj);
  const std::string &GetKind() const &;
};

} // namespace trident::runtime

namespace tvm::ffi {
template <>
struct tvm::ffi::TypeTraits<trident::runtime::ExceptionObj>
    : public TypeTraitsBase {};
} // namespace tvm::ffi

#endif // TRIDENT_CORE_RUNTIME_EXCEPTION_H_
