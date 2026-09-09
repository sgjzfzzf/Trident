//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "TorchDialect.h"

#include <cstddef>
#include <cstdint>
#include <mlir-c/IR.h>
#include <mlir/Bindings/Python/IRCore.h>
#include <mlir/Bindings/Python/NanobindAdaptors.h> // NOLINT(misc-include-cleaner)
#include <nanobind/nanobind.h>
#include <torch-mlir-c/TorchTypes.h>
#include <vector>

namespace nb = nanobind;

using namespace mlir::python::nanobind_adaptors;

namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torch {

struct BoolType : PyConcreteType<BoolType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchBool;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchBoolTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchBoolType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return BoolType(context->getRef(),
                          torchMlirTorchBoolTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct DeviceType : PyConcreteType<DeviceType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchDevice;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchDeviceTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchDeviceType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return DeviceType(context->getRef(),
                            torchMlirTorchDeviceTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct FloatType : PyConcreteType<FloatType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchFloat;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchFloatTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchFloatType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return FloatType(context->getRef(),
                           torchMlirTorchFloatTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct GeneratorType : PyConcreteType<GeneratorType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchGenerator;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchGeneratorTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchGeneratorType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return GeneratorType(
              context->getRef(),
              torchMlirTorchGeneratorTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct IntType : PyConcreteType<IntType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchInt;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchIntTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchIntType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return IntType(context->getRef(),
                         torchMlirTorchIntTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct NonValueTensorType : PyConcreteType<NonValueTensorType> {
  static constexpr IsAFunctionTy isaFunction =
      torchMlirTypeIsATorchNonValueTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchNonValueTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchNonValueTensorType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return NonValueTensorType(
              context->getRef(),
              torchMlirTorchNonValueTensorTypeGetWithLeastStaticInformation(
                  context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct ListType : PyConcreteType<ListType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchList;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchListTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchListType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType containedType) {
          return ListType(containedType.getContext(),
                          torchMlirTorchListTypeGet(containedType));
        },
        nb::arg("contained_type"));
    c.def_prop_ro(
        "contained_type",
        [](ListType &self) -> nb::typed<nb::object, PyType> {
          return PyType(self.getContext(),
                        torchMlirTorchListTypeGetContainedType(self))
              .maybeDownCast();
        },
        "Returns the contained type of the list type.");
  }
};

struct TupleType : PyConcreteType<TupleType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchTuple;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchTupleTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchTupleType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::vector<PyType> &types, DefaultingPyMlirContext context) {
          std::vector<MlirType> containedTypes;
          containedTypes.reserve(types.size());
          for (const PyType &type : types) {
            containedTypes.push_back(type);
          }
          return TupleType(
              context->getRef(),
              torchMlirTorchTupleTypeGet(context.get()->get(),
                                         static_cast<intptr_t>(types.size()),
                                         containedTypes.data()));
        },
        nb::arg("types"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "types",
        [](TupleType &self) -> nb::tuple {
          const size_t numTypes = torchMlirTorchTupleTypeGetNumTypes(self);
          nb::list types;
          for (size_t pos = 0; pos < numTypes; ++pos) {
            types.append(
                PyType(self.getContext(), torchMlirTorchTupleTypeGetType(
                                              self, static_cast<intptr_t>(pos)))
                    .maybeDownCast());
          }
          return nb::tuple(types);
        },
        "Returns the types contained in the tuple type.");
  }
};

struct UnionType : PyConcreteType<UnionType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchUnion;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchUnionTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchUnionType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](const std::vector<PyType> &types,
           DefaultingPyMlirContext context) -> UnionType {
          std::vector<MlirType> containedTypes;
          containedTypes.reserve(types.size());
          for (const PyType &type : types) {
            containedTypes.push_back(type);
          }
          return UnionType(
              context->getRef(),
              torchMlirTorchUnionTypeGet(context.get()->get(),
                                         static_cast<intptr_t>(types.size()),
                                         containedTypes.data()));
        },
        nb::arg("types"), nb::arg("context").none() = nb::none());
    c.def_prop_ro(
        "types",
        [](UnionType &self) -> nb::tuple {
          const size_t numTypes = torchMlirTorchUnionTypeGetNumTypes(self);
          nb::list types;
          for (size_t pos = 0; pos < numTypes; ++pos) {
            types.append(
                PyType(self.getContext(), torchMlirTorchUnionTypeGetType(
                                              self, static_cast<intptr_t>(pos)))
                    .maybeDownCast());
          }
          return nb::tuple(types);
        },
        "Returns the types contained in the union type.");
  }
};

struct ValueTensorType : PyConcreteType<ValueTensorType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchValueTensor;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchValueTensorTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchValueTensorType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return ValueTensorType(
              context->getRef(),
              torchMlirTorchValueTensorTypeGetWithLeastStaticInformation(
                  context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

struct AnyType : PyConcreteType<AnyType> {
  static constexpr IsAFunctionTy isaFunction = torchMlirTypeIsATorchAny;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      torchMlirTorchAnyTypeGetTypeID;
  static constexpr const char *pyClassName = "TorchAnyType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          return AnyType(context->getRef(),
                         torchMlirTorchAnyTypeGet(context.get()->get()));
        },
        nb::arg("context").none() = nb::none());
  }
};

void bindTorchTypes(nb::module_ &module) {
  AnyType::bind(module);
  BoolType::bind(module);
  DeviceType::bind(module);
  FloatType::bind(module);
  GeneratorType::bind(module);
  IntType::bind(module);
  ListType::bind(module);
  NonValueTensorType::bind(module);
  TupleType::bind(module);
  UnionType::bind(module);
  ValueTensorType::bind(module);
}

} // namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::torch
