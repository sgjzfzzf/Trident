// RUN: libtriton-core-opt %s --convert-to-llvm -split-input-file | FileCheck %s

// Globals and function declarations at module level (order may vary).
// CHECK-DAG:  llvm.mlir.global internal constant @__libtriton_constant_overload_
// CHECK-DAG:  llvm.mlir.global internal constant @"__libtriton_constant_op_aten::empty_like"
// CHECK-DAG:  llvm.func @aoti_torch_call_dispatcher

// CHECK-LABEL: llvm.func @aten_empty_like(
// CHECK-SAME:    %[[ARG0:.*]]: !llvm.ptr) -> !llvm.ptr {
// Allocate slot array (max(num_operands=6, num_results=1) = 6 slots).
// CHECK:         %[[N_SLOTS:.*]] = llvm.mlir.constant(6 : i64) : i64
// CHECK-NEXT:    %[[SLOTS:.*]] = llvm.alloca %[[N_SLOTS]] x i64 : (i64) -> !llvm.ptr
// Convert TVMFFIObjectHandle -> AtenTensorHandle -> i64 via mLibTritonTVMFFIObjectToTensor, then store at slot[0].
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor(%[[ARG0]],
// CHECK:         %[[ATEN_HDL:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[ARG_I64:.*]] = llvm.ptrtoint %[[ATEN_HDL]] : !llvm.ptr to i64
// CHECK:         llvm.getelementptr %[[SLOTS]][0]
// CHECK:         llvm.store %[[ARG_I64]],
// Store zero (lowered from torch.constant.none / torch.constant.bool false)
// at slots[1..5] for the 5 trailing none/bool arguments of empty_like.
// CHECK:         %[[CZ:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:         llvm.getelementptr %[[SLOTS]][1]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][2]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][3]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][4]
// CHECK:         llvm.store {{%.*}},
// CHECK:         llvm.getelementptr %[[SLOTS]][5]
// CHECK:         llvm.store {{%.*}},
// Call dispatcher with (op_name, overload_name, slot_array).
// CHECK:         %[[OPNAME:.*]] = llvm.mlir.addressof @"__libtriton_constant_op_aten::empty_like"
// CHECK:         %[[OVERLOAD:.*]] = llvm.mlir.addressof @__libtriton_constant_overload_
// CHECK:         %[[RET:.*]] = llvm.call @aoti_torch_call_dispatcher(%[[OPNAME]], %[[OVERLOAD]], %[[SLOTS]])
// CHECK-SAME:      : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
// Load result from slot[0] and convert i64 -> AtenTensorHandle -> TVMFFIObjectHandle.
// CHECK:         %[[RES_GEP:.*]] = llvm.getelementptr %[[SLOTS]][0]
// CHECK-NEXT:    %[[RES_VAL:.*]] = llvm.load %[[RES_GEP]] : !llvm.ptr -> i64
// CHECK-NEXT:    %[[ATEN_PTR:.*]] = llvm.inttoptr %[[RES_VAL]] : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTensorToTVMFFIObject(%[[ATEN_PTR]],
// CHECK:         %[[RES_PTR:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.return %[[RES_PTR]] : !llvm.ptr

module {
func.func @aten_empty_like(%arg0: !torch.vtensor<[200,200,26],f64>) -> !torch.vtensor<[200,200,26],f64> {
  %none = torch.constant.none
  %false = torch.constant.bool false
  %0 = torch.aten.empty_like %arg0, %none, %none, %none, %false, %none : !torch.vtensor<[200,200,26],f64>, !torch.none, !torch.none, !torch.none, !torch.bool, !torch.none -> !torch.vtensor<[200,200,26],f64>
  return %0 : !torch.vtensor<[200,200,26],f64>
}
}

// -----

// CHECK-LABEL: llvm.func @list_construct(
// CHECK-SAME:    %[[A:.*]]: i64, %[[B:.*]]: i64) -> !llvm.ptr {
// CHECK:      %[[N:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-NEXT: %[[ARR:.*]] = llvm.alloca %[[N]] x !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.getelementptr %[[ARR]][0]
// Fill TVMFFIAny[0] with kTVMFFIInt=1, v_int64=%A
// CHECK:      llvm.getelementptr {{%.*}}[0, 0]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 1]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 2]
// CHECK:      llvm.store %[[A]], {{%.*}} : i64, !llvm.ptr
// CHECK:      llvm.getelementptr %[[ARR]][1]
// Fill TVMFFIAny[1] with kTVMFFIInt=1, v_int64=%B
// CHECK:      llvm.getelementptr {{%.*}}[0, 0]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 1]
// CHECK:      llvm.store {{%.*}}, {{%.*}} : i32, !llvm.ptr
// CHECK:      llvm.getelementptr {{%.*}}[0, 2]
// CHECK:      llvm.store %[[B]], {{%.*}} : i64, !llvm.ptr
// Build slot pointers.
// CHECK:      llvm.getelementptr %[[ARR]][0]
// CHECK:      llvm.getelementptr %[[ARR]][1]
// TVMFFIFunctionGetGlobal("ffi.Array", &func)
// CHECK:      llvm.mlir.addressof @__libtriton_constant_ffi.Array_ffi.Array
// CHECK:      llvm.call @TVMFFIFunctionGetGlobal
// CHECK:      llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// Copy slots to contiguous array.
// CHECK:      llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.store {{%.*}}, {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// CHECK:      llvm.load {{%.*}} : !llvm.ptr -> !llvm.struct<(i32, i32, i64)>
// CHECK:      llvm.store {{%.*}}, {{%.*}} : !llvm.struct<(i32, i32, i64)>, !llvm.ptr
// TVMFFIFunctionCall + TVMFFIObjectDecRef
// CHECK:      llvm.call @TVMFFIFunctionCall
// CHECK:      llvm.call @TVMFFIObjectDecRef
// Extract v_obj, inttoptr, return.
// CHECK:      llvm.getelementptr {{%.*}}[0, 2]
// CHECK:      %[[VOBJ:.*]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK:      %[[RET:.*]] = llvm.inttoptr %[[VOBJ]] : i64 to !llvm.ptr
// CHECK:      llvm.return %[[RET]] : !llvm.ptr

module {
func.func @list_construct(%arg0: !torch.int, %arg1: !torch.int) -> !torch.list<int> {
  %0 = torch.prim.ListConstruct %arg0, %arg1 : (!torch.int, !torch.int) -> !torch.list<int>
  return %0 : !torch.list<int>
}
}

// CHECK-LABEL: llvm.func @list_delete_list(
// CHECK-SAME:    %[[LIST:.*]]: !llvm.ptr) {
// CHECK-NEXT:    llvm.call @TVMFFIObjectDecRef(%[[LIST]]) : (!llvm.ptr) -> i32
// CHECK-NEXT:    llvm.return

module {
func.func @list_delete_list(%list: !torch.list<int>) {
  torchext.aoti.ListDeleteList %list : !torch.list<int>
  return
}
}

// -----

// Test OptionalHandler: optional tensor input (bias) boxed via malloc.
// Note: function signatures no longer use !torch.optional; the bias is
// passed as a plain !torch.vtensor when present. The c10 schema still
// reports OptionalType, so the Schema dispatch code boxes the value.
// CHECK-DAG:  llvm.mlir.global internal constant @"__libtriton_constant_op_aten::linear"
// CHECK-DAG:  llvm.mlir.global internal constant @__libtriton_constant_overload_
// CHECK-DAG:  llvm.func @aoti_torch_call_dispatcher

// CHECK-LABEL: llvm.func @aten_linear_optional(
// CHECK-SAME:    %[[A0:.*]]: !llvm.ptr, %[[A1:.*]]: !llvm.ptr, %[[A2:.*]]: !llvm.ptr) -> !llvm.ptr {
// Allocate slot array (max(num_operands=3, num_results=1) = 3 slots).
// CHECK:         %[[N_SLOTS:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-NEXT:    %[[SLOTS:.*]] = llvm.alloca %[[N_SLOTS]] x i64 : (i64) -> !llvm.ptr
// Slot 0 (arg0): unpack TVMFFIObjectHandle -> AtenTensorHandle via mLibTritonTVMFFIObjectToTensor, then PtrToInt + store.
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor(%[[A0]],
// CHECK:         %[[A0_ATEN:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[A0_I64:.*]] = llvm.ptrtoint %[[A0_ATEN]] : !llvm.ptr to i64
// CHECK:         llvm.getelementptr %[[SLOTS]][0]
// CHECK:         llvm.store %[[A0_I64]],
// Slot 1 (arg1): unpack TVMFFIObjectHandle -> AtenTensorHandle via mLibTritonTVMFFIObjectToTensor, then PtrToInt + store.
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor(%[[A1]],
// CHECK:         %[[A1_ATEN:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[A1_I64:.*]] = llvm.ptrtoint %[[A1_ATEN]] : !llvm.ptr to i64
// CHECK:         llvm.getelementptr %[[SLOTS]][1]
// CHECK:         llvm.store %[[A1_I64]],
// Slot 2 (bias: optional tensor input → unpack TVMFFIObjectHandle → AtenTensorHandle,
// then box via malloc, flowing through CondBr + merge for the OptionalType encoding).
// The cond_br terminator sits in ^bb0 (before the then/else blocks).
// CHECK:         llvm.cond_br
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTVMFFIObjectToTensor(%[[A2]],
// CHECK:         %[[A2_ATEN:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK:         %[[A2_I64:.*]] = llvm.ptrtoint %[[A2_ATEN]] : !llvm.ptr to i64
// CHECK-COUNT-2: llvm.br
// CHECK:         llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK:         llvm.getelementptr %[[SLOTS]][2]
// CHECK:         llvm.store {{%.*}}, {{%.*}} : i64, !llvm.ptr
// Call dispatcher with (op_name, overload_name, slot_array).
// CHECK:         llvm.call @aoti_torch_call_dispatcher
// Load result from slot[0] and convert i64 → !llvm.ptr.
// CHECK:         %[[RES_VAL:.*]] = llvm.load {{%.*}} : !llvm.ptr -> i64
// CHECK-NEXT:    %[[ATEN_PTR:.*]] = llvm.inttoptr %[[RES_VAL]] : i64 to !llvm.ptr
// CHECK:         llvm.alloca {{%.*}} x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK:         llvm.call @mLibTritonTensorToTVMFFIObject(%[[ATEN_PTR]],
// CHECK:         %[[RES_PTR:.*]] = llvm.load {{%.*}} : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:    llvm.return %[[RES_PTR]] : !llvm.ptr

module {
func.func @aten_linear_optional(%arg0: !torch.vtensor<[2,3],f32>, %arg1: !torch.vtensor<[4,3],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,4],f32> {
  %0 = torch.aten.linear %arg0, %arg1, %arg2 : !torch.vtensor<[2,3],f32>, !torch.vtensor<[4,3],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}
}
