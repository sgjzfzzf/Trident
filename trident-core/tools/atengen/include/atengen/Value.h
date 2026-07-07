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
//   Contain<Ty, S>     wraps a c10 type class with SubTypes<S...> for compound
//   types. SubTypes<T...>     describes element type(s) of a Contain.
//
// Simple types dispatch directly on the c10 type class:
//   buildValue<c10::TensorType>(any)
//
// Compound types dispatch on Contain<...>:
//   buildValue<Contain<c10::OptionalType, SubTypes<c10::TensorType>>>(any)
//   buildValue<Contain<c10::ListType, SubTypes<c10::IntType>>>(any)
//
// Nesting is expressed by Contain inside SubTypes:
//   buildValue<Contain<c10::OptionalType,
//              SubTypes<Contain<c10::ListType,
//                       SubTypes<c10::TensorType>>>>>(any)
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

#include <utility>

namespace atengen {

//===----------------------------------------------------------------------===//
// Metaprogramming helpers
//===----------------------------------------------------------------------===//

/// Describes the element type(s) of a Contain.
template <typename...> struct SubTypes {};

/// Wraps a c10 type class with SubTypes describing element(s) for compound
/// types.
template <typename Ty, typename S> struct Contain {};

//===----------------------------------------------------------------------===//
// TypePtrOf — maps type-level type class/Contain to c10::TypePtr
//===----------------------------------------------------------------------===//

/// Primary template — most simple types use ::get().
template <typename T> struct TypePtrOf {
  static c10::TypePtr get() { return T::get(); }
};

/// Default for compound types: types with ::create(...).
template <typename Ty, typename... ElemTys>
struct TypePtrOf<Contain<Ty, SubTypes<ElemTys...>>> {
  static c10::TypePtr get() { return Ty::create(TypePtrOf<ElemTys>::get()...); }
};

// --- Compound type specializations ---

/// AnyClassType: uses ::get() instead of ::create.
template <typename... ElemTys>
struct TypePtrOf<Contain<c10::AnyClassType, SubTypes<ElemTys...>>> {
  static c10::TypePtr get() { return c10::AnyClassType::get(); }
};

/// TupleType: variadic args wrapped in std::vector.
template <typename... ElemTys>
struct TypePtrOf<Contain<c10::TupleType, SubTypes<ElemTys...>>> {
  static c10::TypePtr get() {
    return c10::TupleType::create({TypePtrOf<ElemTys>::get()...});
  }
};

/// VarType uses ::create instead of ::get.
template <> struct TypePtrOf<c10::VarType> {
  static c10::TypePtr get() { return c10::VarType::create(""); }
};

//===----------------------------------------------------------------------===//
// Value<T> — purely type-driven dispatch (single type parameter)
//===----------------------------------------------------------------------===//

/// Primary template — never instantiated directly.
template <typename T> struct Value;

//===----------------------------------------------------------------------===//
// ATENGEN_UNIMPL — stub helper for unimplemented types
//===----------------------------------------------------------------------===//

#define ATENGEN_UNIMPL(TYPE_NAME)                                              \
  template <> struct Value<c10::TYPE_NAME> {                                   \
    static inline c10::IValue build(const TVMFFIAny &) {                       \
      throw std::runtime_error(#TYPE_NAME " build not yet implemented");       \
    }                                                                          \
    static inline TVMFFIAny resolve(const c10::IValue &) {                     \
      throw std::runtime_error(#TYPE_NAME " resolve not yet implemented");     \
    }                                                                          \
  }

//===----------------------------------------------------------------------===//
// Value specializations — ordered by c10::TypeKind
//===----------------------------------------------------------------------===//

ATENGEN_UNIMPL(AnyType);
ATENGEN_UNIMPL(AnyEnumType);

// --- TensorType ---
template <> struct Value<c10::TensorType> {
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

ATENGEN_UNIMPL(StorageType);

// --- TupleType (variadic for multiple returns) ---
template <typename... ElemTys>
struct Value<Contain<c10::TupleType, SubTypes<ElemTys...>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    tvm::ffi::Array<tvm::ffi::Any> array =
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any)
            .cast<tvm::ffi::Array<tvm::ffi::Any>>();
    return buildTuple(array, std::index_sequence_for<ElemTys...>{});
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    const c10::ivalue::TupleElements &elements = val.toTupleRef().elements();
    return resolveTuple(elements, std::index_sequence_for<ElemTys...>{});
  }

private:
  template <size_t... Is>
  static c10::IValue buildTuple(const tvm::ffi::Array<tvm::ffi::Any> &array,
                                std::index_sequence<Is...>) {
    return c10::IValue(c10::ivalue::Tuple::create(Value<ElemTys>::build(
        tvm::ffi::AnyView(array[Is]).CopyToTVMFFIAny())...));
  }
  template <size_t... Is>
  static TVMFFIAny resolveTuple(const c10::ivalue::TupleElements &elements,
                                std::index_sequence<Is...>) {
    tvm::ffi::Array<tvm::ffi::Any> array;
    (array.emplace_back(tvm::ffi::AnyView::CopyFromTVMFFIAny(
         Value<ElemTys>::resolve(elements[Is]))),
     ...);
    TVMFFIAny result;
    tvm::ffi::TypeTraits<tvm::ffi::Array<tvm::ffi::Any>>::MoveToAny(
        std::move(array), &result);
    return result;
  }
};

// --- ListType ---
template <typename ElemTy>
struct Value<Contain<c10::ListType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    tvm::ffi::Array<tvm::ffi::Any> array =
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any)
            .cast<tvm::ffi::Array<tvm::ffi::Any>>();
    c10::impl::GenericList result(TypePtrOf<ElemTy>::get());
    for (const tvm::ffi::Any &elem : array) {
      result.emplace_back(
          Value<ElemTy>::build(tvm::ffi::AnyView(elem).CopyToTVMFFIAny()));
    }
    return c10::IValue(std::move(result));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    c10::ArrayRef<c10::IValue> list = val.toListRef();
    tvm::ffi::Array<tvm::ffi::Any> array;
    for (const c10::IValue &elem : list) {
      array.emplace_back(
          tvm::ffi::AnyView::CopyFromTVMFFIAny(Value<ElemTy>::resolve(elem)));
    }
    TVMFFIAny result;
    tvm::ffi::TypeTraits<tvm::ffi::Array<tvm::ffi::Any>>::MoveToAny(
        std::move(array), &result);
    return result;
  }
};

// --- DictType ---
template <typename KeyTy, typename ValTy>
struct Value<Contain<c10::DictType, SubTypes<KeyTy, ValTy>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any> map =
        tvm::ffi::AnyView::CopyFromTVMFFIAny(any)
            .cast<tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any>>();
    c10::impl::GenericDict result(TypePtrOf<KeyTy>::get(),
                                  TypePtrOf<ValTy>::get());
    for (auto &[k, v] : map) {
      result.insert(
          Value<KeyTy>::build(tvm::ffi::AnyView(k).CopyToTVMFFIAny()),
          Value<ValTy>::build(tvm::ffi::AnyView(v).CopyToTVMFFIAny()));
    }
    return c10::IValue(std::move(result));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    c10::impl::GenericDict dict = val.toGenericDict();
    tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any> map;
    for (auto &kv : dict) {
      map.Set(
          tvm::ffi::AnyView::CopyFromTVMFFIAny(Value<KeyTy>::resolve(kv.key())),
          tvm::ffi::AnyView::CopyFromTVMFFIAny(
              Value<ValTy>::resolve(kv.value())));
    }
    TVMFFIAny result;
    tvm::ffi::TypeTraits<
        tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any>>::MoveToAny(std::move(map),
                                                                &result);
    return result;
  }
};

// --- NumberType ---
template <> struct Value<c10::NumberType> {
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
      return TVMFFIAny{
          .type_index = kTVMFFIBool, .zero_padding = 0, .v_int64 = s.toBool()};
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

// --- FloatType ---
template <> struct Value<c10::FloatType> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(any.v_float64);
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<double>::CopyToAnyView(val.toDouble(), &result);
    return result;
  }
};

ATENGEN_UNIMPL(ComplexType);

// --- FutureType ---
template <typename ElemTy>
struct Value<Contain<c10::FutureType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const TVMFFIAny &) {
    throw std::runtime_error("FutureType build not yet implemented");
  }
  static inline TVMFFIAny resolve(const c10::IValue &) {
    throw std::runtime_error("FutureType resolve not yet implemented");
  }
};

// --- RRefType ---
template <typename ElemTy>
struct Value<Contain<c10::RRefType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const TVMFFIAny &) {
    throw std::runtime_error("RRefType build not yet implemented");
  }
  static inline TVMFFIAny resolve(const c10::IValue &) {
    throw std::runtime_error("RRefType resolve not yet implemented");
  }
};

// --- IntType ---
template <> struct Value<c10::IntType> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(any.v_int64);
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    TVMFFIAny result;
    tvm::ffi::TypeTraits<int64_t>::CopyToAnyView(val.toInt(), &result);
    return result;
  }
};

ATENGEN_UNIMPL(NoneType);

// --- StringType ---
template <> struct Value<c10::StringType> {
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

ATENGEN_UNIMPL(GeneratorType);

// --- BoolType ---
template <> struct Value<c10::BoolType> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return c10::IValue(static_cast<bool>(any.v_int64));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    return TVMFFIAny{
        .type_index = kTVMFFIBool, .zero_padding = 0, .v_int64 = val.toBool()};
  }
};

// --- OptionalType ---
template <typename ElemTy>
struct Value<Contain<c10::OptionalType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const TVMFFIAny &any) {
    return any.type_index == kTVMFFINone
               ? c10::IValue()
               : c10::IValue(Value<ElemTy>::build(any));
  }
  static inline TVMFFIAny resolve(const c10::IValue &val) {
    return val.isNone() ? TVMFFIAny{.type_index = kTVMFFINone,
                                    .zero_padding = 0,
                                    .v_int64 = 0}
                        : Value<ElemTy>::resolve(val);
  }
};

ATENGEN_UNIMPL(VarType);

// --- DeviceObjType ---
template <> struct Value<c10::DeviceObjType> {
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

ATENGEN_UNIMPL(StreamObjType);
ATENGEN_UNIMPL(CapsuleType);
ATENGEN_UNIMPL(QSchemeType);

// --- ScalarTypeType ---
template <> struct Value<c10::ScalarTypeType> {
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

// --- LayoutType ---
template <> struct Value<c10::LayoutType> {
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

// --- MemoryFormatType ---
template <> struct Value<c10::MemoryFormatType> {
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

// --- AnyClassType ---
template <typename ElemTy>
struct Value<Contain<c10::AnyClassType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const TVMFFIAny &) {
    throw std::runtime_error("ClassType build not yet implemented");
  }
  static inline TVMFFIAny resolve(const c10::IValue &) {
    throw std::runtime_error("ClassType resolve not yet implemented");
  }
};

ATENGEN_UNIMPL(SymBoolType);

#undef ATENGEN_UNIMPL

//===----------------------------------------------------------------------===//
// Public API — thin wrappers (single type parameter)
//===----------------------------------------------------------------------===//

/// Build: TVMFFIAny → IValue (type parameter).
template <typename T> inline c10::IValue buildValue(const TVMFFIAny &any) {
  return Value<T>::build(any);
}

/// Resolve: IValue → TVMFFIAny (type parameter).
template <typename T> inline TVMFFIAny resolveValue(const c10::IValue &val) {
  return Value<T>::resolve(val);
}

} // namespace atengen

#endif // ATENGEN_VALUE_H_
