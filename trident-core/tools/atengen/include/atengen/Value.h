//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Type-driven (not value-driven) conversion between TVMFFIAny and IValue.
//
// Design
// ------
//   Kind<K>            lifts a TypeKind enum constant into the type domain.
//   Contain<K, S>      wraps Kind<K> with SubTypes<S...> for compound kinds.
//   SubTypes<T...>     describes element type(s) of a Contain.
//
// Simple types dispatch directly on Kind<X>:
//   buildValue<Kind<TensorType>>(any)
//
// Compound types dispatch on Contain<...>:
//   buildValue<Contain<Kind<OptionalType>, SubTypes<Kind<TensorType>>>>(any)
//   buildValue<Contain<Kind<ListType>, SubTypes<Kind<IntType>>>>(any)
//
// Nesting is expressed by Contain inside SubTypes:
//   buildValue<Contain<Kind<OptionalType>,
//              SubTypes<Contain<Kind<ListType>,
//                       SubTypes<Kind<TensorType>>>>>>(any)
//
//===----------------------------------------------------------------------===//

#ifndef ATENGEN_VALUE_H_
#define ATENGEN_VALUE_H_

#include "ATen/DLConvertor.h"
#include "ATen/core/ivalue.h"
#include "tvm/ffi/any.h"
#include "tvm/ffi/c_api.h"
#include "tvm/ffi/container/array.h"
#include "tvm/ffi/container/map.h"
#include "tvm/ffi/container/tensor.h"
#include "tvm/ffi/string.h"

#include <stdexcept>
#include <utility>

namespace atengen {

//===----------------------------------------------------------------------===//
// Metaprogramming helpers
//===----------------------------------------------------------------------===//

/// Lift a c10::TypeKind constant into the type domain.
template <c10::TypeKind K> struct Kind {};

/// Describes the element type(s) of a Contain.
template <typename...> struct SubTypes {};

/// Wraps a Kind with SubTypes describing element(s) for compound types.
template <typename K, typename S> struct Contain {};

//===----------------------------------------------------------------------===//
// Value<T> — purely type-driven dispatch (single type parameter)
//===----------------------------------------------------------------------===//

/// Primary template — never instantiated directly.
template <typename T> struct Value;

//===----------------------------------------------------------------------===//
// Simple types
//===----------------------------------------------------------------------===//

template <> struct Value<Kind<c10::TypeKind::TensorType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    tvm::ffi::Tensor tensor =
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any).cast<tvm::ffi::Tensor>();
    return c10::IValue(at::fromDLPack(tensor.ToDLPack()));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    at::Tensor tensor = val.toTensor();
    DLManagedTensor *dl_managed = at::toDLPack(tensor);
    TVMFFIAny result;
    tvm::ffi::TypeTraits<tvm::ffi::Tensor>::MoveToAny(
        tvm::ffi::Tensor::FromDLPack(dl_managed), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::IntType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(any.v_int64);
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<int64_t>::CopyToAnyView(val.toInt(), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::FloatType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(any.v_float64);
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<double>::CopyToAnyView(val.toDouble(), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::BoolType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(static_cast<bool>(any.v_int64));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny any;
    any.type_index = kTVMFFIBool;
    any.zero_padding = 0;
    any.v_int64 = val.toBool();
    return any;
  }
};

template <> struct Value<Kind<c10::TypeKind::ScalarTypeType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(static_cast<c10::ScalarType>(any.v_int64));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<int64_t>::CopyToAnyView(
        static_cast<int64_t>(val.toScalarType()), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::LayoutType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(static_cast<c10::Layout>(any.v_int64));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<int64_t>::CopyToAnyView(
        static_cast<int64_t>(val.toLayout()), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::MemoryFormatType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(static_cast<c10::MemoryFormat>(any.v_int64));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<int64_t>::CopyToAnyView(
        static_cast<int64_t>(val.toMemoryFormat()), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::DeviceObjType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    DLDevice dl = any.v_device;
    return c10::IValue(at::dlDeviceToTorchDevice(dl.device_type, dl.device_id));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<DLDevice>::CopyToAnyView(
        at::torchDeviceToDLDevice(val.toDevice()), &result);
    return result;
  }
};

template <> struct Value<Kind<c10::TypeKind::NumberType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    switch (any.type_index) {
    case kTVMFFIInt:
      return c10::IValue(c10::Scalar(any.v_int64));
    case kTVMFFIFloat:
      return c10::IValue(c10::Scalar(any.v_float64));
    case kTVMFFIBool:
      return c10::IValue(c10::Scalar(static_cast<bool>(any.v_int64)));
    default:
      return c10::IValue(c10::Scalar(any.v_int64));
    }
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    c10::Scalar s = val.toScalar();
    if (s.isBoolean()) {
      TVMFFIAny any;
      any.type_index = kTVMFFIBool;
      any.zero_padding = 0;
      any.v_int64 = s.toBool();
      return any;
    } else if (s.isIntegral(false)) {
      TVMFFIAny result;
      tvm::ffi::TypeTraits<int64_t>::CopyToAnyView(s.to<int64_t>(), &result);
      return result;
    } else {
      TVMFFIAny result;
      tvm::ffi::TypeTraits<double>::CopyToAnyView(s.to<double>(), &result);
      return result;
    }
  }
};

template <> struct Value<Kind<c10::TypeKind::StringType>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(std::string(
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any).cast<tvm::ffi::String>()));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<tvm::ffi::String>::MoveToAny(
        tvm::ffi::String(val.toStringRef()), &result);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// Compound — DictType
//===----------------------------------------------------------------------===//

template <typename KeyKind, typename ValKind>
struct Value<
    Contain<Kind<c10::TypeKind::DictType>, SubTypes<KeyKind, ValKind>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    auto map = tvm::ffi::AnyView::CopyFromTVMFFIAny(any)
                   .cast<tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any>>();
    c10::impl::GenericDict result(c10::StringType::get(),
                                  c10::StringType::get());
    for (auto &kv : map) {
      result.insert(
          Value<KeyKind>::build(tvm::ffi::AnyView(kv.first).CopyToTVMFFIAny()),
          Value<ValKind>::build(
              tvm::ffi::AnyView(kv.second).CopyToTVMFFIAny()));
    }
    return c10::IValue(std::move(result));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    auto dict = val.toGenericDict();
    tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any> map;
    for (auto &kv : dict) {
      map.Set(tvm::ffi::AnyView::CopyFromTVMFFIAny(
                  Value<KeyKind>::resolve(kv.key())),
              tvm::ffi::AnyView::CopyFromTVMFFIAny(
                  Value<ValKind>::resolve(kv.value())));
    }
    TVMFFIAny result;
    tvm::ffi::TypeTraits<
        tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any>>::MoveToAny(std::move(map),
                                                                &result);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// Remaining stubs for full ATen schema coverage
//===----------------------------------------------------------------------===//

#define ATENGEN_UNIMPL(TYPE_NAME)                                              \
  template <> struct Value<Kind<c10::TypeKind::TYPE_NAME>> {                   \
    static inline c10::IValue build(const TVMFFIAny &) {                       \
      throw std::runtime_error(#TYPE_NAME " build not yet implemented");       \
    }                                                                          \
    static inline TVMFFIAny resolve(const c10::IValue &) {                     \
      throw std::runtime_error(#TYPE_NAME " resolve not yet implemented");     \
    }                                                                          \
  }

ATENGEN_UNIMPL(AnyType);
ATENGEN_UNIMPL(StreamObjType);
ATENGEN_UNIMPL(StorageType);
ATENGEN_UNIMPL(CapsuleType);
ATENGEN_UNIMPL(GeneratorType);
ATENGEN_UNIMPL(ComplexType);
ATENGEN_UNIMPL(VarType);
ATENGEN_UNIMPL(QSchemeType);
ATENGEN_UNIMPL(SymBoolType);
ATENGEN_UNIMPL(AnyEnumType);
ATENGEN_UNIMPL(NoneType);

#undef ATENGEN_UNIMPL

template <typename ElemKind>
struct Value<Contain<Kind<c10::TypeKind::RRefType>, SubTypes<ElemKind>>> {
  static inline c10::IValue build(const TVMFFIAny &) {
    throw std::runtime_error("RRefType build not yet implemented");
  }
  static inline TVMFFIAny resolve(const c10::IValue &) {
    throw std::runtime_error("RRefType resolve not yet implemented");
  }
};

template <typename ElemKind>
struct Value<Contain<Kind<c10::TypeKind::ClassType>, SubTypes<ElemKind>>> {
  static inline c10::IValue build(const TVMFFIAny &) {
    throw std::runtime_error("ClassType build not yet implemented");
  }
  static inline TVMFFIAny resolve(const c10::IValue &) {
    throw std::runtime_error("ClassType resolve not yet implemented");
  }
};

//===----------------------------------------------------------------------===//
// Compound — FutureType  (raw pointer stub — Future ops are rarely exported)
//===----------------------------------------------------------------------===//

template <typename ElemKind>
struct Value<Contain<Kind<c10::TypeKind::FutureType>, SubTypes<ElemKind>>> {
  static inline c10::IValue build(const TVMFFIAny &) {
    throw std::runtime_error("FutureType build not yet implemented");
  }
  static inline TVMFFIAny resolve(const c10::IValue &) {
    throw std::runtime_error("FutureType resolve not yet implemented");
  }
};

//===----------------------------------------------------------------------===//
// Compound — OptionalType  (single partial spec covers simple + nested)
//===----------------------------------------------------------------------===//

template <typename ElemKind>
struct Value<Contain<Kind<c10::TypeKind::OptionalType>, SubTypes<ElemKind>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return any.type_index == kTVMFFINone ? c10::IValue()
                                         : Value<ElemKind>::build(any);
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    return val.isNone() ? TVMFFIAny{.type_index = kTVMFFINone,
                                    .zero_padding = 0,
                                    .v_int64 = 0}
                        : Value<ElemKind>::resolve(val);
  }
};

//===----------------------------------------------------------------------===//
// Compound — TupleType  (variadic for multiple returns)
//===----------------------------------------------------------------------===//

template <typename... ElemKinds>
struct Value<Contain<Kind<c10::TypeKind::TupleType>, SubTypes<ElemKinds...>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    tvm::ffi::Array<tvm::ffi::Any> array =
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any)
            .cast<tvm::ffi::Array<tvm::ffi::Any>>();
    return buildTuple(array, std::index_sequence_for<ElemKinds...>{});
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    const c10::ivalue::TupleElements &elements = val.toTupleRef().elements();
    return resolveTuple(elements, std::index_sequence_for<ElemKinds...>{});
  }

private:
  template <size_t... Is>
  static c10::IValue buildTuple(const tvm::ffi::Array<tvm::ffi::Any> &array,
                                std::index_sequence<Is...>) {
    return c10::IValue(c10::ivalue::Tuple::create(Value<ElemKinds>::build(
        tvm::ffi::AnyView(array[Is]).CopyToTVMFFIAny())...));
  }
  template <size_t... Is>
  static TVMFFIAny resolveTuple(const c10::ivalue::TupleElements &elements,
                                std::index_sequence<Is...>) {
    tvm::ffi::Array<tvm::ffi::Any> array;
    (array.push_back(tvm::ffi::AnyView::CopyFromTVMFFIAny(
         Value<ElemKinds>::resolve(elements[Is]))),
     ...);
    TVMFFIAny result;
    tvm::ffi::TypeTraits<tvm::ffi::Array<tvm::ffi::Any>>::MoveToAny(
        std::move(array), &result);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// Compound — ListType  (single partial spec covers simple + nested)
//===----------------------------------------------------------------------===//

template <typename ElemKind>
struct Value<Contain<Kind<c10::TypeKind::ListType>, SubTypes<ElemKind>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    tvm::ffi::Array<tvm::ffi::Any> array =
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any)
            .cast<tvm::ffi::Array<tvm::ffi::Any>>();
    c10::impl::GenericList result(c10::StringType::get());
    for (const tvm::ffi::Any &elem : array) {
      result.push_back(
          Value<ElemKind>::build(tvm::ffi::AnyView(elem).CopyToTVMFFIAny()));
    }
    return c10::IValue(std::move(result));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    c10::ArrayRef<c10::IValue> list = val.toListRef();
    tvm::ffi::Array<tvm::ffi::Any> array;
    for (const c10::IValue &elem : list) {
      array.push_back(
          tvm::ffi::AnyView::CopyFromTVMFFIAny(Value<ElemKind>::resolve(elem)));
    }
    TVMFFIAny result;
    tvm::ffi::TypeTraits<tvm::ffi::Array<tvm::ffi::Any>>::MoveToAny(
        std::move(array), &result);
    return result;
  }
};

//===----------------------------------------------------------------------===//
// Public API — thin wrappers (single type parameter)
//===----------------------------------------------------------------------===//

/// Build: TVMFFIAny → IValue (type parameter)
template <typename T> inline c10::IValue buildValue(const TVMFFIAny &any) {
  return Value<T>::build(any);
}

/// Build overload: accepts c10::TypeKind directly, wraps in Kind<K>
template <c10::TypeKind K> inline c10::IValue buildValue(const TVMFFIAny &any) {
  return buildValue<Kind<K>>(any);
}

/// Resolve: IValue → TVMFFIAny (type parameter)
template <typename T> inline TVMFFIAny resolveValue(const c10::IValue &val) {
  return Value<T>::resolve(val);
}

/// Resolve overload: accepts c10::TypeKind directly, wraps in Kind<K>
template <c10::TypeKind K>
inline TVMFFIAny resolveValue(const c10::IValue &val) {
  return resolveValue<Kind<K>>(val);
}

} // namespace atengen

#endif // ATENGEN_VALUE_H_
