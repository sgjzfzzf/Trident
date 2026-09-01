# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import operator
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any, ClassVar, Final, Self, TypeAlias, cast

import torch
import triton
from torch._ops import OpOverloadPacket

from trident.core import ir
from trident.core.dialects import (
    arith,
    gpu,
    torchext,
)
from trident.core.dialects import (
    torch as torch_d,
)
from trident.core.dialects import (
    tvm_ffi as tvm_ffi_d,
)
from trident.core.extras import fx_importer
from trident.core.extras.fx_importer import GraphNodeImporter

KernelScalar: TypeAlias = None | bool | int | float | str | torch.dtype | torch.device
KernelValue: TypeAlias = (
    torch.Tensor
    | torch.SymInt
    | torch.SymFloat
    | torch.SymBool
    | KernelScalar
    | tuple["KernelValue", ...]
    | list["KernelValue"]
    | dict[str, "KernelValue"]
)
KernelArgument: TypeAlias = torch.fx.Node | KernelValue


class GraphNodeImporterTritonHopPatchState:
    refcount: int = 0
    original_attrs: ClassVar[dict[str, Any]] = {}
    original_scalar_type_map: ClassVar[dict[type[Any], str] | None] = None
    original_builtin_ops: ClassVar[dict[str, OpOverloadPacket] | None] = None
    _lock: Final[threading.RLock] = threading.RLock()

    @staticmethod
    def _symbolic_builtin_ops() -> dict[str, OpOverloadPacket]:
        return {
            "or_": torch.ops.aten.__or__,
            "pow": torch.ops.aten.pow,
            "rshift": torch.ops.aten.__rshift__,
        }

    @staticmethod
    def _resolve_triton_binder_value(
        name: str,
        value: KernelArgument,
        path: str = "",
    ) -> KernelValue:
        location = f"{name}{path}"
        if isinstance(value, torch.fx.Node):
            assert "val" in value.meta, (
                f"Triton binder argument {location} node {value.name} "
                "does not contain metadata value"
            )
            metadata_value = value.meta["val"]
            if isinstance(metadata_value, torch.Tensor):
                return triton.MockTensor(dtype=metadata_value.dtype)
            return GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                location, metadata_value
            )

        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                location, cast(KernelValue, value.node.hint)
            )

        if isinstance(value, tuple):
            items = tuple(
                GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                    name, item, f"{path}[{index}]"
                )
                for index, item in enumerate(value)
            )
            if hasattr(value, "_fields"):
                return type(value)(*items)
            return items

        if isinstance(value, list):
            return [
                GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                    name, item, f"{path}[{index}]"
                )
                for index, item in enumerate(value)
            ]

        if isinstance(value, dict):
            return {
                key: GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
                    name, item, f"{path}[{key!r}]"
                )
                for key, item in value.items()
            }

        return value

    @staticmethod
    def _import_kernel_value(
        importer: GraphNodeImporter,
        loc: ir.Location,
        value: KernelArgument,
    ) -> ir.Value | None:
        i64_type = ir.IntegerType.get_signless(64)
        if isinstance(value, torch.fx.Node):
            arithmetic_ops = {
                operator.add: arith.addi,
                operator.sub: arith.subi,
                operator.mul: arith.muli,
                operator.floordiv: arith.floordivsi,
                operator.or_: arith.ori,
                operator.rshift: arith.shrsi,
            }
            if value.target == torch.ops.aten.sym_size.int:
                [tensor, index] = value.args
                tensor = importer._import_argument(loc, tensor)
                index = arith.constant(i64_type, index, loc=loc)
                return tvm_ffi_d.tensor_size(i64_type, tensor, index, loc=loc)
            if value.target == torch.ops.aten.sym_stride.int:
                [tensor, index] = value.args
                tensor = importer._import_argument(loc, tensor)
                index = arith.constant(i64_type, index, loc=loc)
                return tvm_ffi_d.tensor_stride(i64_type, tensor, index, loc=loc)
            if value.target == operator.pow:
                [base, exponent] = value.args
                base = GraphNodeImporterTritonHopPatchState._import_kernel_value(
                    importer, loc, base
                )
                assert isinstance(exponent, int) and exponent >= 0
                result = arith.constant(i64_type, 1, loc=loc)
                for _ in range(exponent):
                    result = arith.muli(result, base, loc=loc)
                return result
            if value.target in arithmetic_ops:
                [lhs, rhs] = value.args
                lhs = GraphNodeImporterTritonHopPatchState._import_kernel_value(
                    importer, loc, lhs
                )
                rhs = GraphNodeImporterTritonHopPatchState._import_kernel_value(
                    importer, loc, rhs
                )
                return arithmetic_ops[value.target](lhs, rhs, loc=loc)
            value = value.meta.get("val")
        if isinstance(value, torch.SymInt):
            value = ast.literal_eval(value)
        if isinstance(value, int):
            return arith.constant(i64_type, value, loc=loc)
        return None

    @classmethod
    def apply(cls) -> None:
        with cls._lock:
            if cls.refcount > 0:
                cls.refcount += 1
            else:
                importer_patches: tuple[Callable[..., None], ...] = (
                    _import_hop_triton_kernel_wrapper_functional,
                    _import_hop_triton_kernel_wrapper_mutation,
                )
                for importer_patch in importer_patches:
                    attr_name: str = importer_patch.__name__
                    if hasattr(GraphNodeImporter, attr_name):
                        cls.original_attrs[attr_name] = getattr(
                            GraphNodeImporter, attr_name
                        )
                    setattr(GraphNodeImporter, attr_name, importer_patch)
                cls.original_scalar_type_map = (
                    fx_importer.SCALAR_TYPE_TO_TORCH_MLIR_TYPE
                )
                fx_importer.SCALAR_TYPE_TO_TORCH_MLIR_TYPE = {
                    **cls.original_scalar_type_map,
                    torch.dtype: "!torch.int",
                }
                builtin_ops = cls._symbolic_builtin_ops()
                cls.original_builtin_ops = {
                    name: fx_importer.PY_BUILTIN_TO_TORCH_OP[name]
                    for name in builtin_ops
                    if name in fx_importer.PY_BUILTIN_TO_TORCH_OP
                }
                fx_importer.PY_BUILTIN_TO_TORCH_OP.update(builtin_ops)
                cls.refcount = 1

    @classmethod
    def restore(cls) -> None:
        with cls._lock:
            if cls.refcount == 0:
                return
            cls.refcount -= 1
            if cls.refcount > 0:
                return
            importer_patches: tuple[Callable[..., None], ...] = (
                _import_hop_triton_kernel_wrapper_functional,
                _import_hop_triton_kernel_wrapper_mutation,
            )
            for importer_patch in importer_patches:
                attr_name: str = importer_patch.__name__
                if hasattr(GraphNodeImporter, attr_name):
                    delattr(GraphNodeImporter, attr_name)
                if attr_name in cls.original_attrs:
                    setattr(
                        GraphNodeImporter,
                        attr_name,
                        cls.original_attrs.pop(attr_name),
                    )
            assert cls.original_scalar_type_map is not None
            fx_importer.SCALAR_TYPE_TO_TORCH_MLIR_TYPE = cls.original_scalar_type_map
            cls.original_scalar_type_map = None
            assert cls.original_builtin_ops is not None
            builtin_ops = cls._symbolic_builtin_ops()
            for name in builtin_ops:
                if name in cls.original_builtin_ops:
                    fx_importer.PY_BUILTIN_TO_TORCH_OP[name] = cls.original_builtin_ops[
                        name
                    ]
                else:
                    fx_importer.PY_BUILTIN_TO_TORCH_OP.pop(name, None)
            cls.original_builtin_ops = None

    def __enter__(self) -> Self:
        self.apply()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.restore()


def _import_hop_triton_kernel_wrapper(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: Any,
) -> None:
    knodes = cast(dict[str, KernelArgument], node.kwargs["kwargs"])
    kvalues: dict[str, ir.Value] = {
        name: self._import_argument(loc, value) for name, value in knodes.items()
    }
    output_names: list[str] = node.kwargs.get("tensors_to_clone", [])
    constant_args_idx: Final[int] = node.kwargs["constant_args_idx"]
    constant_args = cast(
        dict[str, KernelValue],
        torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_constant_args(
            constant_args_idx
        ),
    )
    kernel_idx: Final[int] = node.kwargs["kernel_idx"]
    function: triton.KernelInterface = (
        torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
            kernel_idx
        )
    )
    device = triton.runtime.driver.active.get_current_device()
    configs: list[triton.Config] = getattr(function, "configs", [])
    best_config: triton.Config | None = getattr(function, "best_config", None)
    while not isinstance(function, triton.JITFunction):
        function = function.fn
    kernel_cache, kernel_key_cache, _, _, binder = function.device_caches[device]
    values = {**knodes, **constant_args}
    binder_args = {
        parameter.name: GraphNodeImporterTritonHopPatchState._resolve_triton_binder_value(
            parameter.name, values[parameter.name]
        )
        for parameter in function.params
        if parameter.name in values
    }
    _, specialization, options = binder(
        **binder_args,
        **({} if best_config is None else best_config.all_kwargs()),
        debug=binder_args.get("debug", function.debug) or triton.knobs.runtime.debug,
        instrumentation_mode=triton.knobs.compilation.instrumentation_mode,
    )
    key: str = triton.runtime.jit.compute_cache_key(
        kernel_key_cache, specialization, options
    )
    kernel: triton.compiler.CompiledKernel = kernel_cache.get(key)
    assert kernel is not None, f"failed to get compiled Triton kernel for {node.name}"
    runtime_parameters: list[tuple[str, str]] = [
        (arg_name, triton_type)
        for arg_name, triton_type in kernel.src.signature.items()
        if triton_type != "constexpr"
    ]
    call_arguments: dict[str, ir.Value] = {}
    for name, triton_type in runtime_parameters:
        if name in kvalues:
            call_arguments[name] = kvalues[name]
        elif name in constant_args:
            value = constant_args[name]
            with loc:
                if triton_type.startswith("*") and value is None:
                    call_arguments[name] = self._make_null_ptr()
                elif triton_type in (
                    "i1",
                    "u1",
                    "i8",
                    "u8",
                    "i16",
                    "u16",
                    "i32",
                    "u32",
                    "i64",
                    "u64",
                ):
                    const_val = torch_d.constant_int(value)
                    target = ir.IntegerType.get_signless(
                        ast.literal_eval(triton_type[1:])
                    )
                    call_arguments[name] = torchext.get(target, const_val)
                elif triton_type == "fp32":
                    const_val = torch_d.constant_float(value)
                    target = ir.F32Type.get()
                    call_arguments[name] = torchext.get(target, const_val)
                elif triton_type == "fp64":
                    const_val = torch_d.constant_float(value)
                    target = ir.F64Type.get()
                    call_arguments[name] = torchext.get(target, const_val)
                else:
                    raise RuntimeError(
                        f"unsupported constant argument type: {triton_type}"
                    )
        else:
            raise RuntimeError(
                f"missing runtime argument for {name} of type {triton_type}"
            )
    operands: list[ir.Value] = [call_arguments[name] for name, _ in runtime_parameters]
    grids: list[tuple[int, int, int]] = node.kwargs["grid"]
    if len(configs) > 0 and best_config is not None:
        i: Final[int] = configs.index(best_config)
        grid: tuple[int, int, int] = grids[i]
    else:
        [grid] = grids
    binary_name: Final[str] = f"_{node.name}"
    module_op = self.fx_importer.module.operation
    module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
    if all(
        not isinstance(op, gpu.BinaryOp) or op.sym_name != binary_name
        for op in self.fx_importer.module.body.operations
    ):
        cubin = kernel.asm["cubin"]
        cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
        gpu_object = ir.Attribute.parse(
            f'#gpu.object<#nvvm.target<chip = "sm_{kernel.metadata.target.arch}">, '
            f'"{cubin_mlir}">'
        )
        with ir.InsertionPoint(self.fx_importer.module.body):
            gpu.binary(
                binary_name,
                ir.ArrayAttr.get([gpu_object]),
                offloading_handler=ir.Attribute.parse("#gpu.select_object"),
                loc=loc,
            ).attributes["sym_visibility"] = ir.StringAttr.get("private")
    i64_type = ir.IntegerType.get_signless(64)
    i32_type = ir.IntegerType.get_signless(32)
    grid_x, grid_y, grid_z = grid

    torchext.trident_kernel_launch(
        ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
        GraphNodeImporterTritonHopPatchState._import_kernel_value(self, loc, grid_x),
        GraphNodeImporterTritonHopPatchState._import_kernel_value(self, loc, grid_y),
        GraphNodeImporterTritonHopPatchState._import_kernel_value(self, loc, grid_z),
        arith.constant(
            i64_type,
            kernel.metadata.num_warps * kernel.metadata.warp_size,
            loc=loc,
        ),
        arith.constant(i64_type, 1, loc=loc),
        arith.constant(i64_type, 1, loc=loc),
        operands,
        dynamic_shared_memory_size=arith.constant(
            i32_type, kernel.metadata.shared, loc=loc
        ),
        loc=loc,
    )

    self._multi_result_nodes.add(node)

    for output_name in output_names:
        if output_name in call_arguments:
            self.bind_node_value(node, call_arguments[output_name], output_name)


def _import_hop_triton_kernel_wrapper_functional(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: Any,
) -> None:
    _import_hop_triton_kernel_wrapper(self, loc, node, hop)


def _import_hop_triton_kernel_wrapper_mutation(
    self: GraphNodeImporter,
    loc: ir.Location,
    node: torch.fx.Node,
    hop: Any,
) -> None:
    _import_hop_triton_kernel_wrapper(self, loc, node, hop)


def apply_patch() -> GraphNodeImporterTritonHopPatchState:
    return GraphNodeImporterTritonHopPatchState()
