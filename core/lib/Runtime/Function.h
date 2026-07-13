//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Function<Ret, Args<...>> — type-driven callBoxed wrapper.
//
// Uses trident::runtime::buildValue<T> / resolveValue<T> from Value.h to
// convert AnyView ↔ IValue, then dispatches via c10::OperatorHandle::callBoxed.
//
// Usage:
//   Function<c10::TensorType, Args<c10::TensorType, c10::NumberType>>
//       ::call(op, args, &rv);
//
//   Function<Contain<c10::OptionalType, SubTypes<c10::TensorType>>,
//            Args<c10::TensorType, c10::NumberType>>
//       ::call(op, args, &rv);
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_RUNTIME_FUNCTION_H_
#define TRIDENT_CORE_RUNTIME_FUNCTION_H_

#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/stack.h"
#include "Value.h"
#include "tvm/ffi/container/tuple.h"
#include "tvm/ffi/function.h"

#include <utility>

namespace trident::runtime {

/// Parameter type list for Function.
template <typename... ArgKinds> struct Args {};

/// Primary template (declaration only — partial specialization below).
template <typename Ret, typename ArgList> struct Function;

namespace detail {

/// Primary: single return — pop one IValue and resolve.
template <typename Ret> struct ResolveReturn {
  static tvm::ffi::Any resolve(torch::jit::Stack &stack) {
    return resolveValue<Ret>(torch::jit::pop(stack));
  }
};

/// Specialization: Tuple return — callBoxed pushes each element individually.
template <typename... RetKinds>
struct ResolveReturn<Contain<c10::TupleType, SubTypes<RetKinds...>>> {
  static tvm::ffi::Any resolve(torch::jit::Stack &stack) {
    return resolveTuple(stack, std::index_sequence_for<RetKinds...>{});
  }

private:
  template <size_t... Is>
  static tvm::ffi::Any resolveTuple(torch::jit::Stack &stack,
                                    std::index_sequence<Is...>) {
    return tvm::ffi::Tuple(Value<RetKinds>::resolve(std::move(stack[Is]))...);
  }
};

} // namespace detail

/// Type-driven callBoxed wrapper.
template <typename Ret, typename... ArgKinds>
struct Function<Ret, Args<ArgKinds...>> {
  static void call(c10::OperatorHandle op, tvm::ffi::PackedArgs args,
                   tvm::ffi::Any *rv) {
    torch::jit::Stack stack;
    pushStack(stack, args, std::index_sequence_for<ArgKinds...>{});
    op.callBoxed(stack);
    *rv = detail::ResolveReturn<Ret>::resolve(stack);
  }

private:
  template <size_t... Is>
  static void pushStack(torch::jit::Stack &stack, tvm::ffi::PackedArgs args,
                        std::index_sequence<Is...>) {
    (torch::jit::push(stack, buildValue<ArgKinds>(args[Is])), ...);
  }
};

} // namespace trident::runtime

#endif // TRIDENT_CORE_RUNTIME_FUNCTION_H_
