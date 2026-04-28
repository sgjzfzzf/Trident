#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTBASE_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTBASE_TD_

include "mlir/IR/DialectBase.td"

def TorchExt_Dialect : Dialect {
  let name = "torch_ext";
  let cppNamespace = "::libtriton::torch_ext";
  let useDefaultTypePrinterParser = 0;
  let summary = "Torch extension dialect";
  let description = [{The TorchExt dialect models runtime -
                      facing Torch kernel launch semantics and related ABI -
                      level constructs.}];
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTBASE_TD_
