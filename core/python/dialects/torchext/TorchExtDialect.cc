//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "TorchExtDialect.h"

#include "trident/core/Dialect/TorchExt/IR/TorchExtTypes.h"
#include <mlir/Bindings/Python/IRCore.h>
#include <mlir/Bindings/Python/Nanobind.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h> // NOLINT(misc-include-cleaner)
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;

using namespace mlir::python::nanobind_adaptors;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torchext {

bool isDType(MlirType type) {
  return ::mlir::isa<::trident::torchext::DTypeType>(::unwrap(type));
}

MlirTypeID getDTypeTypeID() {
  return ::wrap(::trident::torchext::DTypeType::getTypeID());
}

struct DTypeType : PyConcreteType<DTypeType> {
  static constexpr IsAFunctionTy isaFunction = isDType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction = getDTypeTypeID;
  static constexpr const char *pyClassName = "DTypeType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return DTypeType(context->getRef(),
                           ::wrap(::trident::torchext::DTypeType::get(
                               ::unwrap(context.get()->get()))));
        },
        nb::arg("context").none() = nb::none());
  }
};

void bindTorchExtTypes(nb::module_ &module) { DTypeType::bind(module); }

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torchext
