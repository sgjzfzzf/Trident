//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRIDENT_CORE_DIALECT_ARITHEXT_IR_ARITHEXTBASE_TD_
#define TRIDENT_CORE_DIALECT_ARITHEXT_IR_ARITHEXTBASE_TD_

include "mlir/IR/DialectBase.td"

def ArithExt_Dialect : Dialect {
  let name = "arithex";
  let cppNamespace = "::trident::arithex";
  let summary = "ArithExt dialect";
  let description = [{
    The ArithExt dialect provides arithmetic-adjacent control-flow
    operations that are not part of the standard Arith dialect.
  }];
}

#endif // TRIDENT_CORE_DIALECT_ARITHEXT_IR_ARITHEXTBASE_TD_
