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

#ifndef TRIDENT_FFI_EXCEPTION_H_
#define TRIDENT_FFI_EXCEPTION_H_

#include "trident/ffi/ExceptionObj.h"
#include <string>
#include <tvm/ffi/object.h>

namespace trident::runtime {

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

#endif // TRIDENT_FFI_EXCEPTION_H_
