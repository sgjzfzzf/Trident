#ifndef LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
#define LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "torch-mlir/Dialect/Torch/IR/TorchTypes.td"

include "libtriton-core/Dialect/TorchExt/IR/TorchExtBase.td"

class TorchExt_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchExt_Dialect, mnemonic, traits>;

def TorchExt_CallDispatcherOp : TorchExt_Op<"aoti.CallDispatcher"> {
  let summary = "Call a PyTorch operator through the AOTInductor dispatcher";
  // clang-format off
  let description = [{
    Invokes a PyTorch operator by name via `aoti_torch_call_dispatcher()`.
    The op_name and overload_name strings identify the target operator.
    Inputs and outputs are passed as type-erased StableIValue slots, modelled
    here as variadic AnyTorchType values.

    Corresponds to:
      aoti_torch_call_dispatcher(opName, overloadName, stack)
  }];
  // clang-format on

  let arguments = (ins
    StrAttr:$op_name,
    StrAttr:$overload_name,
    Variadic<AnyTorchType>:$inputs
  );
  let results = (outs Variadic<AnyTorchType> : $results);

  let assemblyFormat = "$op_name `:` $overload_name `(` $inputs `)` attr-dict "
                       "`:` functional-type($inputs, $results)";
}

//===----------------------------------------------------------------------===//
// Stable list operations — map 1:1 to torch/csrc/stable/c/shim.h
//===----------------------------------------------------------------------===//

def TorchExt_ListDeleteListOp : TorchExt_Op<"aoti.ListDeleteList"> {
  let summary = "Delete a StableList via torch_delete_list()";
  // clang-format off
  let description = [{
    Deletes the underlying StableList referenced by the handle.

    Corresponds to:
      torch_delete_list(list_handle)
  }];
  // clang-format on

  let arguments = (ins AnyTorchListType : $list);
  let results = (outs);

  let assemblyFormat = "$list attr-dict `:` type($list)";
}

#endif // LIBTRITON_CORE_DIALECT_TORCHEXT_IR_TORCHEXTOPS_TD_
