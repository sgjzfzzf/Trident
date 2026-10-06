# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Import compiled Triton kernels using existing launch/specialization ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, TypeAlias, cast

import torch
import triton

from trident.core import ir
from trident.core.dialects import arith, gpu, tvm_ffi

from .ir_utils import box, checked_call, constant, ffi_type, native, native_constant

if TYPE_CHECKING:
    from .fx_importer import TridentFxImporter


KernelValue: TypeAlias = Any
KernelArgument: TypeAlias = Any


def _resolve_triton_binder_value(
    name: str,
    value: KernelArgument,
) -> KernelValue:
    if isinstance(value, torch.fx.Node):
        assert "val" in value.meta, (
            f"Triton binder argument {name} node {value.name} "
            "does not contain metadata value"
        )
        metadata_value = value.meta["val"]
        if isinstance(metadata_value, torch.Tensor):
            return triton.MockTensor(dtype=metadata_value.dtype)
        return _resolve_triton_binder_value(name, metadata_value)

    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return _resolve_triton_binder_value(name, cast(KernelValue, value.node.hint))

    if isinstance(value, tuple):
        items = tuple(_resolve_triton_binder_value(name, item) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return items

    if isinstance(value, list):
        return [_resolve_triton_binder_value(name, item) for item in value]

    if isinstance(value, dict):
        return {
            key: _resolve_triton_binder_value(name, item) for key, item in value.items()
        }

    return value


def import_kernel(
    self: TridentFxImporter,
    loc: ir.Location,
    node: torch.fx.Node,
) -> None:
    knodes = cast(dict[str, KernelArgument], node.kwargs["kwargs"])
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
        parameter.name: _resolve_triton_binder_value(
            parameter.name, values[parameter.name]
        )
        for parameter in function.params
        if parameter.name in values
    }
    config_kwargs = {} if best_config is None else best_config.all_kwargs()
    runtime_kwargs = {
        "debug": binder_args.get("debug", function.debug) or triton.knobs.runtime.debug,
        "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
    }
    bound_args, specialization, options = binder(
        **binder_args, **config_kwargs, **runtime_kwargs
    )
    key: str = triton.runtime.jit.compute_cache_key(
        kernel_key_cache, specialization, options
    )
    kernel: triton.compiler.CompiledKernel | None = kernel_cache.get(key)
    assert kernel is not None, f"failed to get compiled Triton kernel for {node.name}"
    operands: dict[str, tuple[ir.Attribute, ir.Value]] = {}

    def import_constant(value: KernelValue, name: str) -> ir.Value:
        return box(constant(value))

    def constant_value_attribute(value: Any, name: str) -> ir.Attribute | None:
        if isinstance(value, tuple):
            return ir.ArrayAttr.get(
                [
                    constant_value_attribute(element, f"{name}[{index}]")
                    for index, element in enumerate(value)
                ]
            )
        if isinstance(value, bool):
            return ir.BoolAttr.get(value)
        if isinstance(value, int):
            assert -(1 << 63) <= value < (1 << 63), (
                f"constexpr specialization {name!r} does not fit in i64: {value}"
            )
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        if isinstance(value, float):
            return ir.FloatAttr.get_f64(value)
        if isinstance(value, str):
            return ir.StringAttr.get(value)

    def constant_specialization(value: Any, name: str) -> ir.Attribute:
        value_attr = constant_value_attribute(value, name)
        return ir.Attribute.parse(
            f"#tvm_ffi.constant_specialization<value = {value_attr}>"
        )

    for parameter, (triton_type, specialization_descriptor), compiled_parameter in zip(
        function.params,
        specialization,
        kernel.src.signature.items(),
        strict=True,
    ):
        name = parameter.name
        compiled_name, compiled_type = compiled_parameter
        assert (compiled_name, compiled_type) == (name, triton_type), (
            f"compiled Triton signature does not match its source parameter order: expected {(name, triton_type)!r}, got {(compiled_name, compiled_type)!r}"
        )

        if triton_type == "constexpr":
            if name in knodes:
                operand = box(self.argument(knodes[name]))
            else:
                operand = import_constant(bound_args[name], name)
            specialization_attr = constant_specialization(
                specialization_descriptor, name
            )
        else:
            if triton_type.startswith("*"):
                native_type = "!llvm.ptr"
            elif triton_type in {
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
            }:
                native_type = f"i{triton_type[1:]}"
            elif triton_type in {"fp32", "fp64"}:
                native_type = f"f{triton_type[2:]}"
            else:
                raise RuntimeError(f"unsupported Triton argument type: {triton_type}")

            if name in knodes:
                operand = box(self.argument(knodes[name]))
            elif name in bound_args:
                operand = import_constant(bound_args[name], name)
            else:
                raise RuntimeError(
                    f"missing runtime argument for {name} of type {triton_type}"
                )

            specialization_attr = ir.Attribute.parse(
                "#tvm_ffi.variable_specialization<"
                f"kind = {native_type}"
                f"{', divisibility = 16' if specialization_descriptor == 'D' else ''}>"
            )

        assert operand is not None
        operands[name] = (specialization_attr, operand)
    for name in output_names:
        specialization_attr, operand = operands[name]
        clone = checked_call(
            "trident.aten.clone", [operand, constant(None)], ffi_type("tensor")
        )
        operands[name] = (specialization_attr, clone)

    grids: list[tuple[int, int, int]] = node.kwargs["grid"]
    if len(configs) > 0 and best_config is not None:
        i: Final[int] = configs.index(best_config)
        grid: tuple[int, int, int] = grids[i]
    else:
        [grid] = grids
    # Each imported specialization gets a unique symbol namespace from the
    # graph module.  FX node names are unique within an imported graph, so the
    # pair is sufficient to keep binaries distinct after module merging.
    cubin = kernel.asm["cubin"]
    binary_name: Final[str] = f"_trident_s{self.specialization_id}_{node.name}"
    module_op = self.module.operation
    module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
    if all(
        not isinstance(op, gpu.BinaryOp) or op.sym_name != binary_name
        for op in self.module.body.operations
    ):
        cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
        gpu_object = ir.Attribute.parse(
            f'#gpu.object<#nvvm.target<chip = "sm_{kernel.metadata.target.arch}">, '
            f'"{cubin_mlir}">'
        )
        with ir.InsertionPoint(self.module.body):
            gpu.binary(
                binary_name,
                ir.ArrayAttr.get([gpu_object]),
                offloading_handler=ir.Attribute.parse("#gpu.select_object"),
                loc=loc,
            )
    i32_type = ir.IntegerType.get_signless(32)
    grid_x, grid_y, grid_z = grid

    def import_launch_value(value: KernelArgument) -> ir.Value:
        return native(self.argument(value))

    def import_launch_constant(value: int) -> ir.Value:
        return native_constant(value)

    tvm_ffi.kernel_launch(
        ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}"),
        import_launch_value(grid_x),
        import_launch_value(grid_y),
        import_launch_value(grid_z),
        import_launch_constant(kernel.metadata.num_warps * kernel.metadata.warp_size),
        import_launch_constant(1),
        import_launch_constant(1),
        [operand for _, operand in operands.values()],
        ir.ArrayAttr.get([specialization for specialization, _ in operands.values()]),
        dynamic_shared_memory_size=arith.constant(
            i32_type, kernel.metadata.shared, loc=loc
        ),
        loc=loc,
    )
    self.values[node] = {name: operands[name][1] for name in output_names}
