# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Import FX graphs and compiled Triton kernels into semantic MLIR.

Scalars stay native while computing and are boxed at FFI boundaries.
ATen call order, tensor aliases, and mutations are preserved.
"""

from __future__ import annotations

import operator
from functools import cached_property, reduce
from typing import Any, TypeAlias, cast

import sympy
import torch
import triton
import tvm_ffi
from torch.export.graph_signature import ConstantArgument, InputKind, OutputKind

from trident.core import ir
from trident.core.dialects import arith, cf, func, gpu, scf
from trident.core.dialects import tvm_ffi as tvm_ffi_d

from .ir import (
    array,
    array_get,
    binary,
    box,
    checked_call,
    constant,
    is_binary_operation,
    native,
    native_constant,
    value_type,
)

ImportedValue: TypeAlias = (
    ir.Value
    | tuple[ir.Value, ...]
    | list[ir.Value]
    | dict[str, ir.Value]
    | torch.fx.GraphModule
)


class TridentFxImporter:
    def __init__(self, *, context: ir.Context, specialization_id: int):
        self.context = context
        self.specialization_id = specialization_id
        with context, ir.Location.unknown(context):
            self.module = ir.Module.create()
        self.values: dict[torch.fx.Node, ImportedValue] = {}
        self.symbols: dict[Any, ir.Value] = {}

    def _aten(self, node: torch.fx.Node) -> ir.Value | tuple[ir.Value, ...]:
        schema = self._select_aten_schema(node)
        name = schema.name.removeprefix("aten::")
        overload = schema.overload_name
        if (
            operation := {
                "dim": tvm_ffi_d.tensor_dim,
                "sym_size": tvm_ffi_d.tensor_size,
                "sym_storage_offset": tvm_ffi_d.tensor_storage_offset,
                "sym_stride": tvm_ffi_d.tensor_stride,
            }.get(name)
        ) is not None:
            arguments = [native(self.argument(argument)) for argument in node.args]
            return operation(*arguments)
        if name == "sym_numel":
            (tensor_node,) = node.args
            tensor = self.argument(tensor_node)
            return reduce(
                arith.muli,
                (
                    tvm_ffi_d.tensor_size(tensor, native_constant(index))
                    for index in range(tensor_node.meta["val"].dim())
                ),
                native_constant(1),
            )
        if name == "_assert_scalar":
            cf.AssertOp(
                native(self.argument(node.args[0])),
                node.args[1] if len(node.args) > 1 else "FX assertion failed",
            )
            return constant(None)
        arguments = [
            self._aten_argument(node, index, parameter)
            for index, parameter in enumerate(schema.arguments)
        ]
        metadata = node.meta.get("val")
        result_type = value_type(metadata)
        registry_name = f"trident.aten.{name}" + (f".{overload}" if overload else "")
        result = checked_call(registry_name, arguments, result_type)
        if len(schema.returns) > 1:
            return tuple(
                array_get(result, index, value_type(element))
                for index, element in enumerate(metadata)
            )
        return result

    def _aten_argument(
        self, node: torch.fx.Node, index: int, parameter: torch.Argument
    ) -> ir.Value:
        if index < len(node.args):
            argument = node.args[index]
        elif parameter.name in node.kwargs:
            argument = node.kwargs[parameter.name]
        elif parameter.has_default_value():
            argument = parameter.default_value
        else:
            raise ValueError(f"missing argument {parameter.name} for {node.target}")
        if isinstance(argument, (torch.layout, torch.memory_format)):
            argument = self._convert_torch_enum(argument)
        imported = self.argument(argument)
        if isinstance(argument, torch.dtype):
            return checked_call(
                "trident.runtime.tvm_ffi_to_torch_type",
                [imported],
                tvm_ffi_d.IntType.get(),
            )
        return imported

    def _bind(self, node: torch.fx.Node, value: ImportedValue) -> None:
        self.values[node] = value
        metadata = node.meta.get("val")
        if isinstance(
            metadata, (torch.SymInt, torch.SymFloat, torch.SymBool)
        ) and isinstance(value, ir.Value):
            self.symbols[metadata.node.expr] = native(value)
        if isinstance(metadata, torch.Tensor) and isinstance(value, ir.Value):
            # Bind bare input shape symbols to runtime metadata. Composite
            # expressions are evaluated from their symbols, never from hints.
            for index, size in enumerate(metadata.shape):
                if (
                    isinstance(size, torch.SymInt)
                    and size.node.expr.is_Symbol
                    and size.node.expr not in self.symbols
                ):
                    self.symbols[size.node.expr] = tvm_ffi_d.tensor_size(
                        value, native_constant(index)
                    )

    def _cond(self, node: torch.fx.Node) -> tuple[ir.Value, ...]:
        predicate, true_graph, false_graph, operands = node.args
        metadata = node.meta["val"]
        result_types = [value_type(element) for element in metadata]
        conditional = scf.IfOp(
            native(self.argument(predicate)), result_types, has_else=True
        )
        for block, graph in (
            (conditional.then_block, true_graph),
            (conditional.else_block, false_graph),
        ):
            child = self.values[graph]
            assert isinstance(child, torch.fx.GraphModule)
            with ir.InsertionPoint(block):
                importer = TridentFxImporter(
                    context=self.context, specialization_id=self.specialization_id
                )
                importer.module = self.module
                placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
                for placeholder, operand in zip(placeholders, operands, strict=True):
                    importer._bind(placeholder, self.argument(operand))
                importer._import_nodes(child.graph, child)
                output = next(n for n in child.graph.nodes if n.op == "output")
                scf.YieldOp([box(importer.argument(value)) for value in output.args[0]])
        return tuple(conditional.results)

    @staticmethod
    def _constant_value_attribute(value: Any, name: str) -> ir.Attribute:
        if isinstance(value, tuple):
            return ir.ArrayAttr.get(
                [
                    TridentFxImporter._constant_value_attribute(
                        element, f"{name}[{index}]"
                    )
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
        raise NotImplementedError(
            f"unsupported constexpr specialization {name!r}: {value!r}"
        )

    @cached_property
    def _convert_torch_enum(self) -> tvm_ffi.Function:
        return tvm_ffi.get_global_func("trident.runtime.torch_enum_to_int")

    def _import_binary(
        self,
        kernel: triton.compiler.CompiledKernel,
        node: torch.fx.Node,
        loc: ir.Location,
    ) -> ir.Attribute:
        # Each imported specialization gets a unique symbol namespace from the
        # graph module.  FX node names are unique within an imported graph, so the
        # pair is sufficient to keep binaries distinct after module merging.
        cubin = kernel.asm["cubin"]
        binary_name = f"_trident_s{self.specialization_id}_{node.name}"
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
        return ir.Attribute.parse(f"@{binary_name}::@{kernel.metadata.name}")

    def _import_call(self, node: torch.fx.Node) -> ImportedValue:
        target = {torch.sym_max: max, torch.sym_min: min}.get(node.target, node.target)
        if target is operator.getitem:
            aggregate, index = node.args
            imported = self.values[aggregate]
            assert isinstance(imported, (ir.Value, tuple, list, dict))
            return (
                array_get(imported, index, value_type(node.meta["val"]))
                if isinstance(imported, ir.Value)
                else imported[index]
            )
        if is_binary_operation(target):
            lhs, rhs = (self.argument(argument) for argument in node.args)
            return binary(target, lhs, rhs)
        if target is torch.sym_sum:
            arguments = (
                node.args[0]
                if len(node.args) == 1 and isinstance(node.args[0], (tuple, list))
                else node.args
            )
            return reduce(
                arith.addi,
                (native(self.argument(arg)) for arg in arguments),
                native_constant(0),
            )
        if target is operator.pow:
            base, exponent = node.args
            if not isinstance(exponent, int) or exponent < 0:
                raise NotImplementedError(
                    "symbolic integer power requires a nonnegative constant exponent"
                )
            if exponent == 0:
                return native_constant(1)
            return self._integer_power(self.argument(base), exponent)
        if target in (operator.neg, operator.not_, torch.sym_not):
            (argument,) = node.args
            value = native(self.argument(argument))
            return (
                arith.xori(value, native_constant(True))
                if target is not operator.neg
                else binary(operator.sub, native_constant(0), value)
            )
        if isinstance(target, torch._ops.OpOverload):
            return self._aten(node)
        if isinstance(target, torch._ops.HigherOrderOperator):
            if target.__name__ in (
                "triton_kernel_wrapper_functional",
                "triton_kernel_wrapper_mutation",
            ):
                return self._import_kernel(ir.Location.current, node)
            if target.__name__ == "cond":
                return self._cond(node)
            raise NotImplementedError(
                f"unsupported higher-order FX operation: {target}"
            )
        raise NotImplementedError(f"unsupported FX call: {target}")

    def _import_kernel(
        self, loc: ir.Location, node: torch.fx.Node
    ) -> dict[str, ir.Value]:
        kernel, function, bound_arguments, specialization, grid = (
            self._resolve_triton_kernel(node)
        )
        arguments = cast(dict[str, Any], node.kwargs["kwargs"])
        output_names: list[str] = node.kwargs.get("tensors_to_clone", [])
        operands: dict[str, tuple[ir.Attribute, ir.Value]] = {}
        for parameter, (triton_type, descriptor), (compiled_name, compiled_type) in zip(
            function.params,
            specialization,
            kernel.src.signature.items(),
            strict=True,
        ):
            name = parameter.name
            assert (compiled_name, compiled_type) == (name, triton_type), (
                f"compiled Triton signature does not match its source parameter order: expected {(name, triton_type)!r}, got {(compiled_name, compiled_type)!r}"
            )
            specialization_attr = self._triton_specialization(
                triton_type, descriptor, name
            )
            if name in arguments:
                operand = box(self.argument(arguments[name]))
            elif name in bound_arguments:
                operand = box(constant(bound_arguments[name]))
            else:
                raise RuntimeError(
                    f"missing runtime argument for {name} of type {triton_type}"
                )
            operands[name] = (specialization_attr, operand)
        for name in output_names:
            specialization_attr, operand = operands[name]
            clone = checked_call(
                "trident.aten.clone",
                [operand, constant(None)],
                tvm_ffi_d.TensorType.get(),
            )
            operands[name] = (specialization_attr, clone)

        launch_grid = [native(self.argument(value)) for value in grid]
        launch_target = self._import_binary(kernel, node, loc)
        tvm_ffi_d.kernel_launch(
            launch_target,
            *launch_grid,
            native_constant(kernel.metadata.num_warps * kernel.metadata.warp_size),
            native_constant(1),
            native_constant(1),
            [operand for _, operand in operands.values()],
            ir.ArrayAttr.get([attribute for attribute, _ in operands.values()]),
            dynamic_shared_memory_size=arith.constant(
                ir.IntegerType.get_signless(32), kernel.metadata.shared, loc=loc
            ),
            loc=loc,
        )
        return {name: operands[name][1] for name in output_names}

    def _import_nodes(
        self, graph: torch.fx.Graph, graph_module: torch.fx.GraphModule
    ) -> None:
        for node in graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            if node.op == "get_attr":
                value = graph_module
                for attribute in f"{node.target}".split("."):
                    value = getattr(value, attribute)
                self.values[node] = (
                    value
                    if isinstance(value, torch.fx.GraphModule)
                    else constant(value)
                )
                continue
            if node.op != "call_function":
                raise NotImplementedError(f"unsupported FX node: {node.format_node()}")
            self._bind(node, self._import_call(node))

    @staticmethod
    def _integer_power(base: ir.Value, exponent: int) -> ir.Value:
        base = native(base)
        result = native_constant(1)
        for _ in range(exponent):
            result = arith.muli(result, base)
        return result

    @staticmethod
    def _resolve_triton_binder_value(
        name: str,
        value: Any,
    ) -> Any:
        if isinstance(value, torch.fx.Node):
            assert "val" in value.meta, (
                f"Triton binder argument {name} node {value.name} "
                "does not contain metadata value"
            )
            return (
                triton.MockTensor(dtype=metadata_value.dtype)
                if isinstance(metadata_value := value.meta["val"], torch.Tensor)
                else TridentFxImporter._resolve_triton_binder_value(
                    name, metadata_value
                )
            )
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return TridentFxImporter._resolve_triton_binder_value(name, value.node.hint)
        elif isinstance(value, tuple):
            items = tuple(
                TridentFxImporter._resolve_triton_binder_value(name, item)
                for item in value
            )
            return type(value)(*items) if hasattr(value, "_fields") else items
        elif isinstance(value, list):
            return [
                TridentFxImporter._resolve_triton_binder_value(name, item)
                for item in value
            ]
        elif isinstance(value, dict):
            return {
                key: TridentFxImporter._resolve_triton_binder_value(name, item)
                for key, item in value.items()
            }
        else:
            return value

    @staticmethod
    def _resolve_triton_kernel(
        node: torch.fx.Node,
    ) -> tuple[
        triton.compiler.CompiledKernel,
        triton.JITFunction,
        dict[str, Any],
        list[tuple[str, Any]],
        tuple[Any, Any, Any],
    ]:
        knodes = cast(dict[str, Any], node.kwargs["kwargs"])
        constant_args_idx: int = node.kwargs["constant_args_idx"]
        constant_args = cast(
            dict[str, Any],
            torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_constant_args(
                constant_args_idx
            ),
        )
        kernel_idx: int = node.kwargs["kernel_idx"]
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
            parameter.name: TridentFxImporter._resolve_triton_binder_value(
                parameter.name, values[parameter.name]
            )
            for parameter in function.params
            if parameter.name in values
        }
        config_kwargs = {} if best_config is None else best_config.all_kwargs()
        runtime_kwargs = {
            "debug": binder_args.get("debug", function.debug)
            or triton.knobs.runtime.debug,
            "instrumentation_mode": triton.knobs.compilation.instrumentation_mode,
        }
        bound_args, specialization, options = binder(
            **binder_args, **config_kwargs, **runtime_kwargs
        )
        key: str = triton.runtime.jit.compute_cache_key(
            kernel_key_cache, specialization, options
        )
        kernel: triton.compiler.CompiledKernel | None = kernel_cache.get(key)
        assert kernel is not None, (
            f"failed to get compiled Triton kernel for {node.name}"
        )
        grids = node.kwargs["grid"]
        if configs and best_config is not None:
            grid = grids[configs.index(best_config)]
        else:
            [grid] = grids
        return kernel, function, bound_args, specialization, grid

    @staticmethod
    def _select_aten_schema(node: torch.fx.Node) -> torch.FunctionSchema:
        target = node.target
        assert isinstance(target, torch._ops.OpOverload)
        schema = target._schema
        # FX may record the Tensor overload with a Python scalar operand.
        # Python dispatch accepts this shorthand; the boxed C++ schema does not.
        scalar_arguments = [
            index
            for index, argument in enumerate(node.args)
            if index < len(schema.arguments)
            and schema.arguments[index].type.kind() == "TensorType"
            and isinstance(
                argument.meta.get("val")
                if isinstance(argument, torch.fx.Node)
                else argument,
                (bool, int, float, torch.SymInt, torch.SymFloat, torch.SymBool),
            )
        ]
        if scalar_arguments and "Scalar" in target.overloadpacket.overloads():
            candidate = target.overloadpacket.Scalar._schema
            if all(
                index < len(candidate.arguments)
                and candidate.arguments[index].name == schema.arguments[index].name
                and candidate.arguments[index].type.kind()
                in ("NumberType", "IntType", "FloatType", "BoolType")
                for index in scalar_arguments
            ):
                schema = candidate
        return schema

    @staticmethod
    def _triton_specialization(
        triton_type: str, descriptor: Any, name: str
    ) -> ir.Attribute:
        if triton_type == "constexpr":
            value = TridentFxImporter._constant_value_attribute(descriptor, name)
            return ir.Attribute.parse(
                f"#tvm_ffi.constant_specialization<value = {value}>"
            )
        if triton_type.startswith("*"):
            native_type = "!llvm.ptr"
        elif triton_type in {
            "i1",
            "i16",
            "i32",
            "i64",
            "i8",
            "u1",
            "u16",
            "u32",
            "u64",
            "u8",
        }:
            native_type = f"i{triton_type[1:]}"
        elif triton_type in {"fp32", "fp64"}:
            native_type = f"f{triton_type[2:]}"
        else:
            raise RuntimeError(f"unsupported Triton argument type: {triton_type}")
        divisibility = ", divisibility = 16" if descriptor == "D" else ""
        return ir.Attribute.parse(
            f"#tvm_ffi.variable_specialization<kind = {native_type}{divisibility}>"
        )

    def argument(self, value: Any) -> ir.Value:
        if isinstance(value, torch.fx.Node):
            value = self.values[value]
        if isinstance(value, ir.Value):
            return value
        elif isinstance(value, (torch.fx.GraphModule, dict)):
            raise NotImplementedError(f"aggregate node requires getitem: {value}")
        elif isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return self.symbolic(value.node.expr)
        elif isinstance(value, (tuple, list)):
            return array([self.argument(element) for element in value])
        else:
            return constant(value)

    def import_program(
        self, program: torch.export.ExportedProgram, *, func_name: str
    ) -> func.FuncOp:
        producers = {node.name: node for node in program.graph.nodes}
        inputs = [
            spec
            for spec in program.graph_signature.input_specs
            if spec.kind == InputKind.USER_INPUT
            and not isinstance(spec.arg, ConstantArgument)
        ]
        outputs = [
            spec
            for spec in program.graph_signature.output_specs
            if spec.kind == OutputKind.USER_OUTPUT
        ]
        with (
            self.context,
            ir.Location.unknown(self.context),
            ir.InsertionPoint(self.module.body),
        ):
            input_types = [
                value_type(producers[spec.arg.name].meta["val"]) for spec in inputs
            ]
            result_types = [
                value_type(
                    spec.arg.value
                    if isinstance(spec.arg, ConstantArgument)
                    else producers[spec.arg.name].meta["val"]
                )
                for spec in outputs
            ]
            function = func.FuncOp(
                func_name,
                ir.FunctionType.get(input_types, result_types),
                visibility="private",
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                arguments = iter(block.arguments)
                for spec in program.graph_signature.input_specs:
                    if isinstance(spec.arg, ConstantArgument):
                        value = constant(spec.arg.value)
                    elif spec.kind == InputKind.USER_INPUT:
                        value = next(arguments)
                    else:
                        state = (
                            program.state_dict
                            if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
                            else program.constants
                        )
                        value = constant(state[spec.target])
                    self._bind(producers[spec.arg.name], value)
                self._import_nodes(program.graph, program.graph_module)
                results = [
                    box(
                        constant(spec.arg.value)
                        if isinstance(spec.arg, ConstantArgument)
                        else self.argument(producers[spec.arg.name])
                    )
                    for spec in outputs
                ]
                results = [
                    value if value.type == type_ else tvm_ffi_d.cast(type_, value)
                    for value, type_ in zip(results, result_types, strict=True)
                ]
                func.ReturnOp(results)
        return function

    def symbolic(self, expression: Any) -> ir.Value:
        if expression in self.symbols:
            return self.symbols[expression]
        elif expression.is_Integer:
            return native_constant(operator.index(expression))
        elif expression.is_Float:
            return native_constant(float(expression))
        operations = {
            sympy.Add: operator.add,
            sympy.Max: max,
            sympy.Min: min,
            sympy.Mul: operator.mul,
        }
        if expression.func in operations:
            return reduce(
                lambda lhs, rhs: binary(operations[expression.func], lhs, rhs),
                (self.symbolic(argument) for argument in expression.args),
            )
        elif expression.func.__name__ in ("FloorDiv", "Mod", "PythonMod"):
            lhs, rhs = (self.symbolic(argument) for argument in expression.args)
            operation = (
                operator.floordiv
                if expression.func.__name__ == "FloorDiv"
                else operator.mod
            )
            return binary(operation, lhs, rhs)
        elif expression.func is sympy.Pow:
            base, exponent = expression.args
            if exponent.is_Integer and exponent >= 0:
                return self._integer_power(
                    self.symbolic(base), operator.index(exponent)
                )
        raise NotImplementedError(
            f"unbound symbolic FX expression: {expression}; refusing to use a concrete hint"
        )
