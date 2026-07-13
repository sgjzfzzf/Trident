//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// ExceptionObj / Exception — tvm::ffi::Object implementation.
//
//===----------------------------------------------------------------------===//

#include "Exception.h"
#include "tvm/ffi/reflection/registry.h"

namespace trident::runtime {

ExceptionObj::ExceptionObj(const std::string &kind) : kind_(kind) {}

Exception::Exception(tvm::ffi::ObjectPtr<ExceptionObj> data)
    : tvm::ffi::ObjectRef(std::move(data)) {}

Exception::Exception(const std::string &kind)
    : tvm::ffi::ObjectRef(tvm::ffi::make_object<ExceptionObj>(kind)) {}

const std::string &Exception::GetKind() const & {
  return static_cast<ExceptionObj *>(data_.get())->kind_;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("trident.ffi.Exception",
           [](const tvm::ffi::String &kind) -> Exception {
             return Exception(kind);
           })
      .def("trident.ffi.GetExceptionIndex", []() -> int32_t {
        return tvm::ffi::TypeToRuntimeTypeIndex<Exception>::v();
      });
}

} // namespace trident::runtime
