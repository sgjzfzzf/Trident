//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "trident-c/core/TVMFFITypes.h"
#include <cstdint>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>
#include <mlir/Bindings/Python/Diagnostics.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h> // IWYU pragma: keep
#include <vector>

namespace nb = nanobind;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi {

template <typename DerivedTy> struct SimpleType : PyConcreteType<DerivedTy> {
  using Base = PyConcreteType<DerivedTy>;
  using GetFunctionTy = MlirType (*)(MlirContext);
  using Base::Base;

  static void bindDerived(typename Base::ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](DefaultingPyMlirContext context) -> DerivedTy {
          return DerivedTy(context->getRef(),
                           DerivedTy::getFunction(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
    typeClass.def_static(
        "isinstance",
        [](const PyType &type) -> bool { return DerivedTy::isaFunction(type); },
        nb::arg("other_type"));
  }
};

#define DEFINE_SIMPLE_TYPE(Name)                                               \
  struct Name##Type final : SimpleType<Name##Type> {                           \
    using SimpleType<Name##Type>::SimpleType;                                  \
    static constexpr GetFunctionTy getFunction = mlirTVMFFI##Name##TypeGet;    \
    static constexpr GetTypeIDFunctionTy getTypeIdFunction =                   \
        mlirTVMFFI##Name##TypeGetTypeID;                                       \
    static constexpr IsAFunctionTy isaFunction =                               \
        mlirTypeIsATVMFFI##Name##Type;                                         \
    static inline const MlirStringRef name = mlirTVMFFI##Name##TypeGetName();  \
    static constexpr const char *pyClassName = #Name "Type";                   \
  };

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

struct UnionType final : PyConcreteType<UnionType> {
  using Base::Base;
  static constexpr IsAFunctionTy isaFunction = mlirTypeIsATVMFFIUnionType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirTVMFFIUnionTypeGetTypeID;
  static constexpr const char *pyClassName = "UnionType";
  static inline const MlirStringRef name = mlirTVMFFIUnionTypeGetName();

  static void bindDerived(ClassTy &typeClass) {
    typeClass.def_static(
        "get",
        [](const std::vector<PyType> &types,
           DefaultingPyMlirContext context) -> UnionType {
          std::vector<MlirType> members(types.begin(), types.end());
          CollectDiagnosticsToStringScope diagnostics(context.get()->get());
          const MlirType type = mlirTVMFFIUnionTypeGetChecked(
              mlirLocationUnknownGet(context.get()->get()),
              static_cast<intptr_t>(members.size()), members.data());
          if (mlirTypeIsNull(type)) {
            throw nb::value_error(diagnostics.takeMessage().c_str());
          }
          return UnionType(context->getRef(), type);
        },
        nb::arg("types"), nb::kw_only(),
        nb::arg("context").none() = nb::none());
    typeClass.def_static(
        "isinstance",
        [](const PyType &type) -> bool { return isaFunction(type); },
        nb::arg("other_type"));
    typeClass.def_prop_ro("types", [](UnionType &type) -> nb::tuple {
      nb::list members;
      const intptr_t size = mlirTVMFFIUnionTypeGetNumTypes(type);
      for (intptr_t index = 0; index < size; ++index) {
        members.append(
            PyType(type.getContext(), mlirTVMFFIUnionTypeGetType(type, index))
                .maybeDownCast());
      }
      return nb::tuple(members);
    });
  }
};

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi

NB_MODULE(_tvm_ffi, module) {
  module.doc() = "TVM FFI dialect type bindings";
  using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::tvm_ffi;
  AnyType::bind(module);
  ArrayType::bind(module);
  BoolType::bind(module);
  DTypeType::bind(module);
  DeviceType::bind(module);
  ExceptionType::bind(module);
  FloatType::bind(module);
  FunctionType::bind(module);
  IntType::bind(module);
  NoneType::bind(module);
  ObjectType::bind(module);
  RawStrType::bind(module);
  SmallStrType::bind(module);
  StrType::bind(module);
  TensorType::bind(module);
  UnionType::bind(module);
}
