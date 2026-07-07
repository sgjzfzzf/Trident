//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Function<Ret, Args<...>> — type-driven callBoxed wrapper.
//
// Uses atengen::buildValue<T> / resolveValue<T> from Value.h to convert
// TVMFFIAny ↔ IValue, then dispatches via c10::OperatorHandle::callBoxed.
//
// Usage:
//   Function<c10::TensorType, Args<c10::TensorType, c10::NumberType>>
//       ::call(op, args, n, &result);
//
//   Function<Contain<c10::OptionalType, SubTypes<c10::TensorType>>,
//            Args<c10::TensorType, c10::NumberType>>
//       ::call(op, args, n, &result);
//
//===----------------------------------------------------------------------===//

#ifndef ATENGEN_FUNCTION_H_
#define ATENGEN_FUNCTION_H_

#include "ATen/core/dispatch/Dispatcher.h"
#include "ATen/core/stack.h"
#include "atengen/Value.h"

#include <utility>

namespace atengen {

/// Parameter type list for Function.
template <typename... ArgKinds> struct Args {};

/// Callable wrapper around callBoxed with type-driven IValue conversion.
template <typename Ret, typename ArgList> struct Function;

/// Partial specialization: Ret + variadic parameter types.
template <typename Ret, typename... ArgKinds>
struct Function<Ret, Args<ArgKinds...>> {
  static int32_t call(c10::OperatorHandle op, TVMFFIAny *args, int32_t num_args,
                      TVMFFIAny *result) {
    if (num_args != sizeof...(ArgKinds)) {
      return -1;
    }
    torch::jit::Stack stack;
    pushStack(stack, args, std::index_sequence_for<ArgKinds...>{});
    op.callBoxed(stack);
    *result = resolveValue<Ret>(torch::jit::pop(stack));
    return 0;
  }

private:
  template <size_t... Is>
  static void pushStack(torch::jit::Stack &stack, TVMFFIAny *args,
                        std::index_sequence<Is...>) {
    (torch::jit::push(stack, buildValue<ArgKinds>(args[Is])), ...);
  }
};

} // namespace atengen

#endif // ATENGEN_FUNCTION_H_
