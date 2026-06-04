from functools import cached_property, reduce
from typing import Any, Final, Generator, List, Optional, Tuple, Union

import tvm_ffi

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import (
    arith,
    dlpack,
    func,
    llvm,
    tvm_ffi as tvm_ffi_d,
)


class Signature(object):
    """Describes how to marshal arguments/results across the TVM-FFI boundary."""

    kTVMFFITensor: Final[int] = tvm_ffi.core.TypeSchema.from_annotation(
        tvm_ffi.Tensor
    ).origin_type_index

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def __init__(
        self,
        outputs_schema: Union[tvm_ffi.core.TypeSchema, List[tvm_ffi.core.TypeSchema]],
        inputs_schema: List[tvm_ffi.core.TypeSchema],
        ctx: Optional[ir.Context] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.outputs_schema: Final[
            Union[tvm_ffi.core.TypeSchema, List[tvm_ffi.core.TypeSchema]]
        ] = outputs_schema
        self.inputs_schema: Final[List[tvm_ffi.core.TypeSchema]] = inputs_schema
        self.ctx: Final[Optional[ir.Context]] = ctx

    # ── Derived properties ─────────────────────────────────────────────────

    @cached_property
    def has_multiple_outputs(self) -> bool:
        return isinstance(self.outputs_schema, list)

    @cached_property
    def outputs_schema_list(self) -> List[tvm_ffi.core.TypeSchema]:
        """Return a flattened list of output schemas.

        If *self.outputs_schema* is already a list, returns it directly;
        otherwise returns a singleton list containing it.
        """
        if self.has_multiple_outputs:
            return self.outputs_schema
        else:
            return [self.outputs_schema]

    # ── Public API ─────────────────────────────────────────────────────────

    def call(
        self,
        callee_op: func.FuncOp,
        packed_args_ptr: ir.Value,
        packed_result_ptr: ir.Value,
    ) -> ir.Value:
        """Emit IR to call *callee_op* through the TVM-FFI boundary.

        Unpacks user arguments from *packed_args_ptr*, allocates out-param
        memrefs, calls the callee, boxes every result into ``!tvm_ffi.any``,
        stores the (possibly array-packed) result into *packed_result_ptr*,
        and returns the ``i32`` status code.

        Out-params are identified by the ``bufferize.result`` attribute
        on *callee_op*'s block arguments (set by
        ``buffer-results-to-out-params{add-result-attr}``).
        """
        callee_name: str = callee_op.sym_name.value
        callee_ty: ir.FunctionType = ir.FunctionType(callee_op.type)

        # -- Cached types ---------------------------------------------------
        i32_type: ir.Type = ir.IntegerType.get_signless(32)
        ptr_type: ir.Type = llvm.PointerType.get()
        any_type: ir.Type = ir.Type.parse("!tvm_ffi.any")
        dltensor_ty: ir.Type = ir.Type.parse("!dlpack.tensor")
        object_handle_type: ir.Type = ir.Type.parse("!tvm_ffi.object_handle")
        managed_tensor_type: ir.Type = ir.Type.parse("!dlpack.managed_tensor")

        # -- Classify inputs via bufferize.result attribute ------------------
        def _partition(
            acc: Tuple[List[ir.Type], List[ir.Type]],
            pair: Tuple[ir.Type, ir.DictAttr],
        ) -> Tuple[List[ir.Type], List[ir.Type]]:
            lhs, rhs = acc
            ty, attr = pair
            if "bufferize.result" in attr:
                return lhs, rhs + [ty]
            else:
                return lhs + [ty], rhs

        call_input_tys, out_param_tys = reduce(
            _partition,
            zip(callee_ty.inputs, callee_op.arg_attrs),
            ([], []),
        )

        # -- Classify outputs ------------------------------------------------
        tensor_schemas: List[tvm_ffi.core.TypeSchema] = [
            s for s in self.outputs_schema_list if self._is_tensor_schema(s)
        ]
        scalar_schemas: List[tvm_ffi.core.TypeSchema] = [
            s for s in self.outputs_schema_list if not self._is_tensor_schema(s)
        ]

        # -- Validate arity --------------------------------------------------
        assert all(map(lambda ty: isinstance(ty, ir.MemRefType), out_param_tys)), (
            f"out-param must be memref: {callee_name}: {callee_ty}"
        )

        # -- 1. Unpack user inputs from packed_args --------------------------
        call_args: List[ir.Value] = [
            self._unpack_input(packed_args_ptr, idx, schema, ty, dltensor_ty)
            for idx, (schema, ty) in enumerate(zip(self.inputs_schema, call_input_tys))
        ]

        # -- 2. Allocate out-param memrefs -----------------------------------
        pairs: List[Tuple[ir.Value, ir.Value]] = [
            self._alloc_out_param(
                ty, ptr_type, object_handle_type, managed_tensor_type, dltensor_ty
            )
            for ty in out_param_tys
        ]
        allocs: List[ir.Value] = [h for h, _ in pairs]
        out_memrefs: List[ir.Value] = [o for _, o in pairs]

        # -- 3. Call the callee ----------------------------------------------
        call_op: func.CallOp = func.CallOp(
            callee_ty.results, callee_name, call_args + out_memrefs
        )

        # -- 4. Map results back to schema order -----------------------------
        def _collect() -> Generator[ir.Value, None, None]:
            """Interleave *allocs* and *call_op.results* in schema order.

            The first *n_out_params* tensor schemas draw from *allocs*
            (pre-allocated handles); all remaining schemas draw from
            *call_op.results*.
            """
            allocs_iter = iter(allocs)
            results_iter = iter(call_op.results)
            for schema in self.outputs_schema_list:
                if self._is_tensor_schema(schema):
                    yield next(allocs_iter)
                else:
                    yield next(results_iter)

        collected: List[ir.Value] = list(_collect())

        # -- 5. Box results and store to packed_result_ptr -------------------
        boxed: List[ir.Value] = [tvm_ffi_d.to(any_type, v) for v in collected]
        if self.has_multiple_outputs:
            result: ir.Value = tvm_ffi_d.call(any_type, "ffi.Array", boxed)
        else:
            [result] = boxed
        tvm_ffi_d.store(result, packed_result_ptr)
        return arith.constant(i32_type, 0)

    # ── Private helpers ────────────────────────────────────────────────────

    def _unpack_input(
        self,
        packed_args_ptr: ir.Value,
        index: int,
        schema: tvm_ffi.core.TypeSchema,
        target_ty: ir.Type,
        dltensor_ty: ir.Type,
    ) -> ir.Value:
        val: ir.Value = self._emit_unpack_any_arg(packed_args_ptr, index)
        if self._is_tensor_schema(schema) and isinstance(target_ty, ir.MemRefType):
            return dlpack.to_memref(target_ty, tvm_ffi_d.to(dltensor_ty, val))
        return tvm_ffi_d.to(target_ty, val)

    @staticmethod
    def _is_tensor_schema(schema: tvm_ffi.core.TypeSchema) -> bool:
        return schema.origin_type_index == Signature.kTVMFFITensor

    @staticmethod
    def _emit_unpack_any_arg(
        packed_args_ptr: ir.Value,
        index: int,
    ) -> ir.Value:
        i32_type: ir.Type = ir.IntegerType.get_signless(32)
        i64_type: ir.Type = ir.IntegerType.get_signless(64)
        ptr_type: ir.Type = llvm.PointerType.get()
        any_llvm_type: ir.Type = llvm.StructType.get_literal(
            [i32_type, i32_type, i64_type]
        )
        any_type: ir.Type = ir.Type.parse("!tvm_ffi.any")
        arg_slot_ptr: ir.Value = llvm.GEPOp(
            ptr_type,
            packed_args_ptr,
            [],
            ir.DenseI32ArrayAttr.get([index]),
            any_llvm_type,
            None,
        ).result
        return tvm_ffi_d.load(any_type, arg_slot_ptr)

    @staticmethod
    def _alloc_out_param(
        out_param_ty: ir.MemRefType,
        ptr_type: ir.Type,
        object_handle_type: ir.Type,
        managed_tensor_type: ir.Type,
        dltensor_ty: ir.Type,
    ) -> Tuple[ir.Value, ir.Value]:
        """Allocate an out-param memref.  Returns ``(handle, memref_operand)``."""
        handle: ir.Value = tvm_ffi_d.env_tensor_alloc(
            object_handle_type,
            out_param_ty.element_type,
            out_param_ty.shape,
        )
        opaque: ir.Value = tvm_ffi_d.get_opaque_ptr(ptr_type, handle)
        managed: ir.Value = tvm_ffi_d.load(managed_tensor_type, opaque)
        dl_t: ir.Value = dlpack.view(dltensor_ty, managed)
        operand: ir.Value = dlpack.to_memref(out_param_ty, dl_t)
        return handle, operand
