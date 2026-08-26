//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_CONVERSION_TVMFFITOFUNC_TVMFFITOFUNC_H_
#define TRIDENT_CORE_CONVERSION_TVMFFITOFUNC_TVMFFITOFUNC_H_

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace trident::tvm_ffi {

#define GEN_PASS_DECL_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION_CONVERTTVMFFITOFUNC
#include "trident/core/Conversion/Passes.h.inc"

} // namespace trident::tvm_ffi

#endif // TRIDENT_CORE_CONVERSION_TVMFFITOFUNC_TVMFFITOFUNC_H_
