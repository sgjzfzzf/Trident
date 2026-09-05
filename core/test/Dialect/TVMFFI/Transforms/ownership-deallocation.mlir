//===----------------------------------------------------------------------===//
//
// Part of the Trident project, under the MIT License.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: trident-core-opt %s -ownership-deallocation | FileCheck %s

module {
  func.func private @make_tensor() -> !tvm_ffi.tensor

  // Each conditional edge retains exactly the value needed by that path and
  // releases the owned value created in the entry block.
  // CHECK-LABEL: func.func @diamond(
  // CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1, %[[BORROWED:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
  // CHECK: %[[OWNED:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
  // CHECK-NEXT: cf.cond_br %[[COND]], [[OWNED_EDGE:\^bb[0-9]+]], [[BORROWED_EDGE:\^bb[0-9]+]]
  // CHECK: [[OWNED_EDGE]]:
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OWNED]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OWNED]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[OWNED_PATH:\^bb[0-9]+]]
  // CHECK: [[OWNED_PATH]]:
  // CHECK-NEXT: cf.br [[OWNED_TRANSFER:\^bb[0-9]+]](%[[OWNED]] : !tvm_ffi.tensor)
  // CHECK: [[BORROWED_EDGE]]:
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[BORROWED]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OWNED]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[BORROWED_PATH:\^bb[0-9]+]]
  // CHECK: [[BORROWED_PATH]]:
  // CHECK-NEXT: cf.br [[BORROWED_TRANSFER:\^bb[0-9]+]](%[[BORROWED]] : !tvm_ffi.tensor)
  // CHECK: [[OWNED_TRANSFER]](%[[OWNED_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[OWNED]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[OWNED]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[MERGE:\^bb[0-9]+]](%[[OWNED_ARG]] : !tvm_ffi.tensor)
  // CHECK: [[BORROWED_TRANSFER]](%[[BORROWED_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[BORROWED]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[BORROWED]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[MERGE]](%[[BORROWED_ARG]] : !tvm_ffi.tensor)
  // CHECK: [[MERGE]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: return %[[RESULT]] : !tvm_ffi.tensor
  func.func @diamond(%cond: i1, %borrowed: !tvm_ffi.tensor)
      -> !tvm_ffi.tensor {
    %owned = func.call @make_tensor() : () -> !tvm_ffi.tensor
    cf.cond_br %cond, ^owned_path, ^borrowed_path

  ^owned_path:
    cf.br ^merge(%owned : !tvm_ffi.tensor)

  ^borrowed_path:
    cf.br ^merge(%borrowed : !tvm_ffi.tensor)

  ^merge(%result: !tvm_ffi.tensor):
    return %result : !tvm_ffi.tensor
  }

  // A backedge follows the same edge-local transfer protocol as a forward
  // edge, so every loop iteration has one incoming and one outgoing credit.
  // CHECK-LABEL: func.func @loop(
  // CHECK-SAME: %[[LOOP_COND:[a-zA-Z0-9_]+]]: i1, %[[LOOP_INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
  // CHECK-NEXT: cf.br [[ENTRY_EDGE:\^bb[0-9]+]](%[[LOOP_INPUT]] : !tvm_ffi.tensor)
  // CHECK: [[ENTRY_EDGE]](%[[ENTRY_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[LOOP_INPUT]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[HEADER:\^bb[0-9]+]](%[[ENTRY_ARG]] : !tvm_ffi.tensor)
  // CHECK: [[BACKEDGE:\^bb[0-9]+]](%[[BACK_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ITER:[a-zA-Z0-9_]+]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ITER]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[HEADER]](%[[BACK_ARG]] : !tvm_ffi.tensor)
  // CHECK: [[HEADER]](%[[ITER]]: !tvm_ffi.tensor):
  // CHECK-NEXT: cf.cond_br %[[LOOP_COND]], [[BACKEDGE]](%[[ITER]] : !tvm_ffi.tensor), [[EXIT_EDGE:\^bb[0-9]+]](%[[ITER]] : !tvm_ffi.tensor)
  // CHECK: [[EXIT_EDGE]](%[[EXIT_VALUE:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ITER]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ITER]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[EXIT:\^bb[0-9]+]](%[[EXIT_VALUE]] : !tvm_ffi.tensor)
  // CHECK: [[EXIT]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: return %[[RESULT]] : !tvm_ffi.tensor
  func.func @loop(%cond: i1, %input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
    cf.br ^header(%input : !tvm_ffi.tensor)

  ^header(%value: !tvm_ffi.tensor):
    cf.cond_br %cond, ^header(%value : !tvm_ffi.tensor),
                         ^exit(%value : !tvm_ffi.tensor)

  ^exit(%result: !tvm_ffi.tensor):
    return %result : !tvm_ffi.tensor
  }

  // Forwarding an object and keeping its original SSA value live represent
  // two ownership credits, even when both values alias the same object.
  // CHECK-LABEL: func.func @forwarded_and_live_through(
  // CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
  // CHECK-NEXT: cf.br [[ENTRY_EDGE:\^bb[0-9]+]](%[[INPUT]] : !tvm_ffi.tensor)
  // CHECK: [[ENTRY_EDGE]](%[[FORWARDED:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[NEXT:\^bb[0-9]+]](%[[FORWARDED]] : !tvm_ffi.tensor)
  // CHECK: [[NEXT]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[INPUT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[INPUT]] : !tvm_ffi.tensor
  // CHECK-NEXT: return %[[INPUT]] : !tvm_ffi.tensor
  func.func @forwarded_and_live_through(%input: !tvm_ffi.tensor)
      -> !tvm_ffi.tensor {
    cf.br ^next(%input : !tvm_ffi.tensor)

  ^next(%forwarded: !tvm_ffi.tensor):
    return %input : !tvm_ffi.tensor
  }

  // Generic multi-successor BranchOpInterface operations use one trampoline
  // per successor, including multiple edges that target the same merge block.
  // CHECK-LABEL: func.func @switch(
  // CHECK-SAME: %[[SELECTOR:[a-zA-Z0-9_]+]]: i32, %[[DEFAULT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor, %[[CASE:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor)
  // CHECK: cf.switch %[[SELECTOR]] : i32, [
  // CHECK: default: [[DEFAULT_EDGE:\^bb[0-9]+]](%[[DEFAULT]] : !tvm_ffi.tensor),
  // CHECK: 1: [[CASE_EDGE:\^bb[0-9]+]](%[[CASE]] : !tvm_ffi.tensor)
  // CHECK: [[DEFAULT_EDGE]](%[[DEFAULT_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[DEFAULT]] : !tvm_ffi.tensor
  // CHECK: [[CASE_EDGE]](%[[CASE_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[CASE]] : !tvm_ffi.tensor
  func.func @switch(%selector: i32, %default: !tvm_ffi.tensor,
                    %case: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
    cf.switch %selector : i32, [
      default: ^merge(%default : !tvm_ffi.tensor),
      1: ^merge(%case : !tvm_ffi.tensor)
    ]

  ^merge(%result: !tvm_ffi.tensor):
    return %result : !tvm_ffi.tensor
  }

  // Alias casts do not create a new credit. Each branch transfers its owned
  // tensor through the cast and releases the original credit on the edge.
  // CHECK-LABEL: func.func @cast_control_flow_return(
  // CHECK-SAME: %[[COND:[a-zA-Z0-9_]+]]: i1) -> !tvm_ffi.any
  // CHECK-NEXT: cf.cond_br %[[COND]], [[THEN_ENTRY:\^bb[0-9]+]], [[ELSE_ENTRY:\^bb[0-9]+]]
  // CHECK: [[THEN_ENTRY]]:
  // CHECK-NEXT: cf.br [[THEN_PATH:\^bb[0-9]+]]
  // CHECK: [[THEN_PATH]]:
  // CHECK-NEXT: %[[THEN_VALUE:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
  // CHECK-NEXT: %[[THEN_CAST:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[THEN_VALUE]] : !tvm_ffi.tensor -> !tvm_ffi.any
  // CHECK-NEXT: cf.br [[THEN_EDGE:\^bb[0-9]+]](%[[THEN_CAST]] : !tvm_ffi.any)
  // CHECK: [[ELSE_ENTRY]]:
  // CHECK-NEXT: cf.br [[ELSE_PATH:\^bb[0-9]+]]
  // CHECK: [[ELSE_PATH]]:
  // CHECK-NEXT: %[[ELSE_VALUE:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
  // CHECK-NEXT: %[[ELSE_CAST:[a-zA-Z0-9_]+]] = tvm_ffi.cast %[[ELSE_VALUE]] : !tvm_ffi.tensor -> !tvm_ffi.any
  // CHECK-NEXT: cf.br [[ELSE_EDGE:\^bb[0-9]+]](%[[ELSE_CAST]] : !tvm_ffi.any)
  // CHECK: [[THEN_EDGE]](%[[THEN_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.any):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[THEN_CAST]] : !tvm_ffi.any
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[THEN_VALUE]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[MERGE:\^bb[0-9]+]](%[[THEN_ARG]] : !tvm_ffi.any)
  // CHECK: [[ELSE_EDGE]](%[[ELSE_ARG:[a-zA-Z0-9_]+]]: !tvm_ffi.any):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[ELSE_CAST]] : !tvm_ffi.any
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[ELSE_VALUE]] : !tvm_ffi.tensor
  // CHECK-NEXT: cf.br [[MERGE]](%[[ELSE_ARG]] : !tvm_ffi.any)
  // CHECK: [[MERGE]](%[[RESULT:[a-zA-Z0-9_]+]]: !tvm_ffi.any):
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]] : !tvm_ffi.any
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]] : !tvm_ffi.any
  // CHECK-NEXT: return %[[RESULT]] : !tvm_ffi.any
  func.func @cast_control_flow_return(%cond: i1) -> !tvm_ffi.any {
    cf.cond_br %cond, ^then, ^else

  ^then:
    %then_value = func.call @make_tensor() : () -> !tvm_ffi.tensor
    %then_cast = tvm_ffi.cast %then_value : !tvm_ffi.tensor -> !tvm_ffi.any
    cf.br ^merge(%then_cast : !tvm_ffi.any)

  ^else:
    %else_value = func.call @make_tensor() : () -> !tvm_ffi.tensor
    %else_cast = tvm_ffi.cast %else_value : !tvm_ffi.tensor -> !tvm_ffi.any
    cf.br ^merge(%else_cast : !tvm_ffi.any)

  ^merge(%result: !tvm_ffi.any):
    return %result : !tvm_ffi.any
  }

  // Existing explicit reference operations participate in the same ledger.
  // CHECK-LABEL: func.func @explicit_balance()
  // CHECK: %[[BALANCED:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[BALANCED]] : !tvm_ffi.tensor
  // CHECK-NEXT: return
  func.func @explicit_balance() {
    %value = func.call @make_tensor() : () -> !tvm_ffi.tensor
    tvm_ffi.ObjectDecRef %value : !tvm_ffi.tensor
    return
  }

  // An explicit increment adds another credit, so both references are
  // released at the function exit.
  // CHECK-LABEL: func.func @explicit_increment()
  // CHECK: %[[RETAINED:[a-zA-Z0-9_]+]] = call @make_tensor() : () -> !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RETAINED]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RETAINED]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RETAINED]] : !tvm_ffi.tensor
  // CHECK-NEXT: return
  func.func @explicit_increment() {
    %value = func.call @make_tensor() : () -> !tvm_ffi.tensor
    tvm_ffi.ObjectIncRef %value : !tvm_ffi.tensor
    return
  }

  // FunctionCall consumes the owned global-function handle, so no additional
  // cleanup is emitted for that handle.
  // CHECK-LABEL: func.func @consumed_function_handle()
  // CHECK: %[[HANDLE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "test.make_tensor" : !tvm_ffi.function
  // CHECK-NEXT: %[[VALUE:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[HANDLE]]() : () -> !tvm_ffi.tensor
  // CHECK-NOT: tvm_ffi.ObjectDecRef %[[HANDLE]]
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[VALUE]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[VALUE]] : !tvm_ffi.tensor
  // CHECK-NEXT: return %[[VALUE]] : !tvm_ffi.tensor
  func.func @consumed_function_handle() -> !tvm_ffi.tensor {
    %function = tvm_ffi.FunctionGetGlobal "test.make_tensor"
        : !tvm_ffi.function
    %value = tvm_ffi.FunctionCall %function() : () -> !tvm_ffi.tensor
    return %value : !tvm_ffi.tensor
  }

  // TVMFFI functions are FunctionOpInterface roots and follow the same return
  // transfer rule as func.func.
  // CHECK-LABEL: tvm_ffi.func @ffi_return(
  // CHECK-SAME: %[[INPUT:[a-zA-Z0-9_]+]]: !tvm_ffi.tensor) -> !tvm_ffi.tensor
  // CHECK: %[[FUNCTION:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionGetGlobal "trident.aten.t" : !tvm_ffi.function
  // CHECK-NEXT: %[[RESULT:[a-zA-Z0-9_]+]] = tvm_ffi.FunctionCall %[[FUNCTION]](%[[INPUT]]) : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectIncRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.ObjectDecRef %[[RESULT]] : !tvm_ffi.tensor
  // CHECK-NEXT: tvm_ffi.return %[[RESULT]] : !tvm_ffi.tensor
  tvm_ffi.func @ffi_return(%input: !tvm_ffi.tensor) -> !tvm_ffi.tensor {
    %function = tvm_ffi.FunctionGetGlobal "trident.aten.t"
        : !tvm_ffi.function
    %result = tvm_ffi.FunctionCall %function(%input)
        : (!tvm_ffi.tensor) -> !tvm_ffi.tensor
    tvm_ffi.return %result : !tvm_ffi.tensor
  }
}
