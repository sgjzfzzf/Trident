# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import ast
import random
import threading
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Final

import torch
import triton

from trident._C.trident_core import ir
from trident._C.trident_core.dialects import (
    arith,
    gpu,
    llvm,
    torch as torch_d,
    torchext,
)
from trident._C.trident_core.extras.fx_importer import GraphNodeImporter


class GraphNodeImporterTritonHopPatchState:
    refcount: int = 0
    original_attrs: Dict[str, Any] = {}
    _lock: Final[threading.RLock] = threading.RLock()

    @classmethod
    def apply(cls) -> None:
        with cls._lock:
            if cls.refcount > 0:
                cls.refcount += 1
            else:
                attr_name: Final[str] = (
                    _import_hop_triton_kernel_wrapper_mutation.__name__
                )
                if hasattr(GraphNodeImporter, attr_name):
                    cls.original_attrs[attr_name] = getattr(
                        GraphNodeImporter, attr_name
                    )
                GraphNodeImporter._import_hop_triton_kernel_wrapper_mutation = (
                    _import_hop_triton_kernel_wrapper_mutation
                )
                cls.refcount = 1

    @classmethod
    def restore(cls) -> None:
        with cls._lock:
            if cls.refcount == 0:
                return
            cls.refcount -= 1
            if cls.refcount > 0:
                return
            attr_name: Final[str] = _import_hop_triton_kernel_wrapper_mutation.__name__
            if hasattr(GraphNodeImporter, attr_name):
                delattr(GraphNodeImporter, attr_name)
            if attr_name in cls.original_attrs:
                setattr(GraphNodeImporter, attr_name, cls.original_attrs.pop(attr_name))

    def __enter__(self) -> GraphNodeImporterTritonHopPatchState:
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.restore()


def _import_hop_triton_kernel_wrapper_mutation(
    self: GraphNodeImporter,
    loc: Any,
    node: torch.fx.Node,
    hop: Any,
) -> None:
    knodes: Dict[str, Any] = node.kwargs["kwargs"]
    ktensors: Dict[str, torch.Tensor] = {
        name: triton.MockTensor(dtype=value.meta["val"].dtype)
        for name, value in knodes.items()
    }
    kvalues: Dict[str, ir.Value] = {
        name: self._import_argument(loc, value) for name, value in knodes.items()
    }
    output_names: List[str] = node.kwargs.get("tensors_to_clone", [])
    constant_args_idx: Final[int] = node.kwargs["constant_args_idx"]
    constant_args: Dict[str, Any] = (
        torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_constant_args(
            constant_args_idx
        )
    )
    kernel_idx: Final[int] = node.kwargs["kernel_idx"]
    function: triton.KernelInterface = (
        torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
            kernel_idx
        )
    )
    device = triton.runtime.driver.active.get_current_device()
    configs: List[triton.Config] = getattr(function, "configs", [])
    best_config: Optional[triton.Config] = getattr(function, "best_config", None)
    while not isinstance(function, triton.JITFunction):
        function = function.fn
    kernel_cache, kernel_key_cache, _, _, binder = function.device_caches[device]
    args: Dict[str, Any] = {**ktensors, **constant_args}
    _, specialization, options = binder(
        *[],
        **args,
        **({} if best_config is None else best_config.all_kwargs()),
        debug=args.get("debug", function.debug) or triton.knobs.runtime.debug,
        instrumentation_mode=triton.knobs.compilation.instrumentation_mode,
    )
    key: str = triton.runtime.jit.compute_cache_key(
        kernel_key_cache, specialization, options
    )
    kernel: triton.compiler.CompiledKernel = kernel_cache.get(key)
    assert kernel is not None, f"failed to get compiled Triton kernel for {node.name}"
    runtime_parameters: List[Tuple[str, str]] = [
        (arg_name, triton_type)
        for arg_name, triton_type in kernel.src.signature.items()
        if triton_type != "constexpr"
    ]
    call_arguments: Dict[str, ir.Value] = {}
    for name, triton_type in runtime_parameters:
        if value := kvalues.get(name):
            call_arguments[name] = value
        elif value := constant_args.get(name):
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
                    const_val = torch_d.ConstantIntOp(value)
                    target = ir.IntegerType.get_signless(
                        ast.literal_eval(triton_type[1:])
                    )
                    call_arguments[name] = torchext.CastOp(target, const_val)
                elif triton_type == "fp32":
                    const_val = torch_d.ConstantFloatOp(value)
                    target = ir.F32Type.get()
                    call_arguments[name] = torchext.CastOp(target, const_val)
                elif triton_type == "fp64":
                    const_val = torch_d.ConstantFloatOp(value)
                    target = ir.F64Type.get()
                    call_arguments[name] = torchext.CastOp(target, const_val)
                else:
                    raise RuntimeError(
                        f"unsupported constant argument type: {triton_type}"
                    )
        else:
            raise RuntimeError(
                f"missing runtime argument for {name} of type {triton_type}"
            )
    operands: List[ir.Value] = [call_arguments[name] for name, _ in runtime_parameters]
    grids: List[Tuple[int, int, int]] = node.kwargs["grid"]
    if len(configs) > 0 and best_config is not None:
        i: Final[int] = configs.index(best_config)
        grid: Tuple[int, int, int] = grids[i]
    else:
        [grid] = grids
    binary_name: Final[str] = f"_{node.name}_{random.randint(0, 1 << 32)}"
    module_op = self.fx_importer.module.operation
    module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
    if all(
        not isinstance(op, gpu.BinaryOp) or op.sym_name != binary_name
        for op in self.fx_importer.module.body.operations
    ):
        cubin = kernel.asm["cubin"]
        cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
        gpu_object = ir.Attribute.parse(
            '#gpu.object<#nvvm.target<chip = "sm_{arch}">, "{cubin}">'.format(
                arch=kernel.metadata.target.arch,
                cubin=cubin_mlir,
            )
        )
        with ir.InsertionPoint(self.fx_importer.module.body):
            gpu.binary(
                ir.StringAttr.get(binary_name),
                ir.ArrayAttr.get([gpu_object]),
                offloading_handler=ir.Attribute.parse("#gpu.select_object"),
                loc=loc,
            )
    i64_type = ir.IntegerType.get_signless(64)
    i32_type = ir.IntegerType.get_signless(32)
    grid_x, grid_y, grid_z = grid
    torchext.TridentKernelLaunchOp(
        ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
        arith.constant(i64_type, grid_x, loc=loc),
        arith.constant(i64_type, grid_y, loc=loc),
        arith.constant(i64_type, grid_z, loc=loc),
        arith.constant(
            i64_type,
            kernel.metadata.num_warps * kernel.metadata.warp_size,
            loc=loc,
        ),
        arith.constant(i64_type, 1, loc=loc),
        arith.constant(i64_type, 1, loc=loc),
        operands,
        dynamicSharedMemorySize=arith.constant(
            i32_type, kernel.metadata.shared, loc=loc
        ),
        loc=loc,
    )

    self._multi_result_nodes.add(node)

    for output_name in output_names:
        if output_name in call_arguments:
            self.bind_node_value(node, call_arguments[output_name], output_name)


def apply_patch() -> GraphNodeImporterTritonHopPatchState:
    return GraphNodeImporterTritonHopPatchState()
