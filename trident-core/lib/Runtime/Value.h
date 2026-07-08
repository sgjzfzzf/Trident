//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Type-driven (not value-driven) conversion between AnyView/Any and IValue.
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

#ifndef TRIDENT_CORE_RUNTIME_VALUE_H_
#define TRIDENT_CORE_RUNTIME_VALUE_H_

#include "ATen/DLConvertor.h"
#include "ATen/core/ivalue.h"
#include "tvm/ffi/any.h"
#include "tvm/ffi/container/array.h"
#include "tvm/ffi/container/map.h"
#include "tvm/ffi/container/tensor.h"

#include <tvm/ffi/type_traits.h>
#include <utility>

namespace trident::runtime {

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
// TRIDENT_RUNTIME_UNIMPL — stub helper for unimplemented types
//===----------------------------------------------------------------------===//

#define TRIDENT_RUNTIME_UNIMPL(TYPE_NAME)                                      \
  template <> struct Value<c10::TYPE_NAME> {                                   \
    static inline c10::IValue build(const tvm::ffi::AnyView &) {               \
      throw std::runtime_error(#TYPE_NAME " build not yet implemented");       \
    }                                                                          \
    static inline tvm::ffi::Any resolve(c10::IValue &&) {                      \
      throw std::runtime_error(#TYPE_NAME " resolve not yet implemented");     \
    }                                                                          \
  }

//===----------------------------------------------------------------------===//
// Value specializations — ordered by c10::TypeKind
//===----------------------------------------------------------------------===//

TRIDENT_RUNTIME_UNIMPL(AnyType);
TRIDENT_RUNTIME_UNIMPL(AnyEnumType);

// --- TensorType ---
template <> struct Value<c10::TensorType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return at::fromDLPack(any.cast<tvm::ffi::Tensor>().ToDLPack());
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return tvm::ffi::Tensor::FromDLPack(
        at::toDLPack(std::move(val).toTensor()));
  }
};

TRIDENT_RUNTIME_UNIMPL(StorageType);

// --- TupleType (variadic for multiple returns) ---
template <typename... ElemTys>
struct Value<Contain<c10::TupleType, SubTypes<ElemTys...>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return buildTuple(any.cast<tvm::ffi::Array<tvm::ffi::Any>>(),
                      std::index_sequence_for<ElemTys...>{});
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return resolveTuple(std::move(*std::move(val).toTuple()).elements(),
                        std::index_sequence_for<ElemTys...>{});
  }

private:
  template <size_t... Is>
  static c10::IValue buildTuple(const tvm::ffi::Array<tvm::ffi::Any> &array,
                                std::index_sequence<Is...>) {
    return c10::IValue(c10::ivalue::Tuple::create(
        Value<ElemTys>::build(tvm::ffi::AnyView(array[Is]))...));
  }
  template <size_t... Is>
  static tvm::ffi::Any resolveTuple(c10::ivalue::TupleElements &&elements,
                                    std::index_sequence<Is...>) {
    tvm::ffi::Array<tvm::ffi::Any> array;
    (array.emplace_back(Value<ElemTys>::resolve(std::move(elements[Is]))), ...);
    return array;
  }
};

// --- ListType ---
template <typename ElemTy>
struct Value<Contain<c10::ListType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    tvm::ffi::Array<tvm::ffi::Any> array =
        any.cast<tvm::ffi::Array<tvm::ffi::Any>>();
    c10::impl::GenericList result(TypePtrOf<ElemTy>::get());
    for (const tvm::ffi::Any &elem : array) {
      result.emplace_back(Value<ElemTy>::build(tvm::ffi::AnyView(elem)));
    }
    return result;
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    c10::List<c10::IValue> list = std::move(val).toList();
    tvm::ffi::Array<tvm::ffi::Any> array;
    for (c10::IValue elem : list) {
      array.emplace_back(Value<ElemTy>::resolve(std::move(elem)));
    }
    return array;
  }
};

// --- DictType ---
template <typename KeyTy, typename ValTy>
struct Value<Contain<c10::DictType, SubTypes<KeyTy, ValTy>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any> map =
        any.cast<tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any>>();
    c10::impl::GenericDict result(TypePtrOf<KeyTy>::get(),
                                  TypePtrOf<ValTy>::get());
    for (auto &[k, v] : map) {
      result.insert(Value<KeyTy>::build(tvm::ffi::AnyView(k)),
                    Value<ValTy>::build(tvm::ffi::AnyView(v)));
    }
    return result;
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    c10::impl::GenericDict dict = val.toGenericDict();
    tvm::ffi::Map<tvm::ffi::Any, tvm::ffi::Any> map;
    for (auto &kv : dict) {
      map.Set(Value<KeyTy>::resolve(c10::IValue(kv.key())),
              Value<ValTy>::resolve(c10::IValue(kv.value())));
    }
    return map;
  }
};

// --- NumberType ---
template <> struct Value<c10::NumberType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    if (std::optional<int64_t> i = any.as<int64_t>()) {
      return *i;
    } else if (std::optional<double> f = any.as<double>()) {
      return *f;
    } else if (std::optional<bool> b = any.as<bool>()) {
      return *b;
    } else {
      return any.cast<int64_t>();
    }
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    c10::Scalar s = val.toScalar();
    if (s.isBoolean()) {
      return static_cast<bool>(s.toBool());
    } else if (s.isIntegral(false)) {
      return s.to<int64_t>();
    } else {
      return s.to<double>();
    }
  }
};

// --- FloatType ---
template <> struct Value<c10::FloatType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return any.cast<double>();
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return val.toDouble();
  }
};

TRIDENT_RUNTIME_UNIMPL(ComplexType);

// --- FutureType ---
template <typename ElemTy>
struct Value<Contain<c10::FutureType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &) {
    throw std::runtime_error("FutureType build not yet implemented");
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&) {
    throw std::runtime_error("FutureType resolve not yet implemented");
  }
};

// --- RRefType ---
template <typename ElemTy>
struct Value<Contain<c10::RRefType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &) {
    throw std::runtime_error("RRefType build not yet implemented");
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&) {
    throw std::runtime_error("RRefType resolve not yet implemented");
  }
};

// --- IntType ---
template <> struct Value<c10::IntType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return any.cast<int64_t>();
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) { return val.toInt(); }
};

TRIDENT_RUNTIME_UNIMPL(NoneType);

// --- StringType ---
template <> struct Value<c10::StringType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return any.cast<std::string>();
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return val.toStringRef();
  }
};

TRIDENT_RUNTIME_UNIMPL(GeneratorType);

// --- BoolType ---
template <> struct Value<c10::BoolType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return any.cast<bool>();
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return val.toBool();
  }
};

// --- OptionalType ---
template <typename ElemTy>
struct Value<Contain<c10::OptionalType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return any.type_index() == kTVMFFINone ? c10::IValue()
                                           : Value<ElemTy>::build(any);
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return val.isNone() ? tvm::ffi::Any()
                        : Value<ElemTy>::resolve(std::move(val));
  }
};

TRIDENT_RUNTIME_UNIMPL(VarType);

// --- DeviceObjType ---
template <> struct Value<c10::DeviceObjType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    const DLDevice dl = any.cast<DLDevice>();
    return at::dlDeviceToTorchDevice(dl.device_type, dl.device_id);
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return at::torchDeviceToDLDevice(val.toDevice());
  }
};

TRIDENT_RUNTIME_UNIMPL(StreamObjType);
TRIDENT_RUNTIME_UNIMPL(CapsuleType);
TRIDENT_RUNTIME_UNIMPL(QSchemeType);

// --- ScalarTypeType ---
template <> struct Value<c10::ScalarTypeType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return static_cast<c10::ScalarType>(any.cast<int64_t>());
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return static_cast<int64_t>(val.toScalarType());
  }
};

// --- LayoutType ---
template <> struct Value<c10::LayoutType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return any.cast<int64_t>();
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return static_cast<int64_t>(val.toLayout());
  }
};

// --- MemoryFormatType ---
template <> struct Value<c10::MemoryFormatType> {
  static inline c10::IValue build(const tvm::ffi::AnyView &any) {
    return static_cast<c10::MemoryFormat>(any.cast<int64_t>());
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&val) {
    return static_cast<int64_t>(val.toMemoryFormat());
  }
};

// --- AnyClassType ---
template <typename ElemTy>
struct Value<Contain<c10::AnyClassType, SubTypes<ElemTy>>> {
  static inline c10::IValue build(const tvm::ffi::AnyView &) {
    throw std::runtime_error("ClassType build not yet implemented");
  }
  static inline tvm::ffi::Any resolve(c10::IValue &&) {
    throw std::runtime_error("ClassType resolve not yet implemented");
  }
};

TRIDENT_RUNTIME_UNIMPL(SymBoolType);

#undef TRIDENT_RUNTIME_UNIMPL

//===----------------------------------------------------------------------===//
// Public API — thin wrappers (single type parameter)
//===----------------------------------------------------------------------===//

/// Build: AnyView -> IValue (type parameter).
template <typename T>
inline c10::IValue buildValue(const tvm::ffi::AnyView &any) {
  return Value<T>::build(any);
}

/// Resolve: IValue -> Any (type parameter).
/// Takes ownership of the IValue via rvalue reference.
template <typename T> inline tvm::ffi::Any resolveValue(c10::IValue &&val) {
  return Value<T>::resolve(std::move(val));
}

} // namespace trident::runtime

#endif // TRIDENT_CORE_RUNTIME_VALUE_H_
