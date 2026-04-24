from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Final

import torch
import triton

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, func, gpu, llvm
from libtriton._C.libtriton_core.extras.fx_importer import FxImporter, GraphNodeImporter


class TritonGraphNodeImporter(GraphNodeImporter):
    """GraphNodeImporter subclass that handles triton higher-order ops natively."""

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def _runtime_parameters(
        kernel: triton.compiler.CompiledKernel,
    ) -> List[Tuple[str, str]]:
        return [
            (arg_name, triton_type)
            for arg_name, triton_type in kernel.src.signature.items()
            if triton_type != "constexpr"
        ]

    @staticmethod
    def _add_gpu_binary(
        loc: Any,
        module: ir.Module,
        binary_name: str,
        kernel: triton.compiler.CompiledKernel,
    ) -> None:
        module_op = module.operation
        module_op.attributes["gpu.container_module"] = ir.UnitAttr.get()
        cubin = kernel.asm["cubin"]
        cubin_mlir: str = "".join(f"\\{byte:02X}" for byte in cubin)
        gpu_object = ir.Attribute.parse(
            '#gpu.object<#nvvm.target<chip = "sm_{arch}">, "{cubin}">'.format(
                arch=kernel.metadata.target.arch,
                cubin=cubin_mlir,
            )
        )
        with loc:
            module.body.append(
                ir.Operation.create(
                    "gpu.binary",
                    attributes={
                        "sym_name": ir.StringAttr.get(binary_name),
                        "objects": ir.ArrayAttr.get([gpu_object]),
                        "offloadingHandler": ir.Attribute.parse("#gpu.select_object"),
                    },
                )
            )

    @staticmethod
    def _triton_type_to_ir_type(triton_type: str) -> ir.Type:
        if triton_type.startswith("*"):
            return ir.Type.parse("!llvm.ptr")
        elif triton_type == "bf16":
            return ir.BF16Type.get()
        elif triton_type == "fp16":
            return ir.F16Type.get()
        elif triton_type == "fp32":
            return ir.F32Type.get()
        elif triton_type == "fp64":
            return ir.F64Type.get()
        elif triton_type.startswith(("i", "u")):
            return ir.IntegerType.get_signless(int(triton_type[1:]))
        else:
            raise ValueError(f"unsupported Triton runtime type: {triton_type}")

    @staticmethod
    def _cast_value(result_type: ir.Type, value: ir.Value, loc: Any) -> ir.Value:
        if value.type == result_type:
            return value
        else:
            return ir.Operation.create(
                "builtin.unrealized_conversion_cast",
                results=[result_type],
                operands=[value],
                loc=loc,
            ).results[0]

    @staticmethod
    def _make_null_ptr() -> ir.Value:
        return llvm.mlir_zero(ir.Type.parse("!llvm.ptr"))

    def _materialize_runtime_argument(
        self,
        loc: Any,
        name: str,
        triton_type: str,
        imported_values: Dict[str, ir.Value],
        constant_args: Dict[str, Any],
    ) -> ir.Value:
        if value := imported_values.get(name):
            return value
        elif value := constant_args.get(name):
            target_type = self._triton_type_to_ir_type(triton_type)
            with loc:
                if triton_type.startswith("*") and value is None:
                    return self._make_null_ptr()
                elif triton_type.startswith(("i", "u")):
                    return arith.constant(target_type, int(value))
                elif triton_type in ("fp16", "bf16", "fp32", "fp64"):
                    return arith.constant(target_type, float(value))
                else:
                    raise ValueError(
                        f"unsupported Triton runtime constant for {name}: {value}"
                    )
        else:
            raise ValueError(f"missing Triton runtime argument: {name}")

    def _build_wrapper_function(
        self,
        loc: Any,
        fname: str,
        binary_name: str,
        kernel: triton.compiler.CompiledKernel,
        runtime_parameters: List[Tuple[str, str]],
        function_arguments: List[Tuple[str, ir.Type]],
        grid: Tuple[int, int, int],
        output_names: List[str],
    ) -> ir.FunctionType:
        function_arguments_by_name: Dict[str, ir.Type] = dict(function_arguments)
        input_names: List[str] = [arg_name for arg_name, _ in runtime_parameters]
        input_types: List[ir.Type] = [
            function_arguments_by_name[arg_name] for arg_name in input_names
        ]
        result_types: List[ir.Type] = [
            function_arguments_by_name[arg_name]
            for arg_name in output_names
            if arg_name in function_arguments_by_name
        ]
        fty = ir.FunctionType.get(input_types, result_types)
        with loc:
            func_op = func.FuncOp(
                fname,
                fty,
                visibility="private",
                ip=self.fx_importer._m_ip,
            )
            entry_block = ir.Block.create_at_start(func_op.body, fty.inputs)
            with ir.InsertionPoint(entry_block):
                index_type = ir.IndexType.get()
                i32_type = ir.IntegerType.get_signless(32)
                grid_x, grid_y, grid_z = grid
                grid_x = arith.constant(index_type, grid_x)
                grid_y = arith.constant(index_type, grid_y)
                grid_z = arith.constant(index_type, grid_z)
                block_x = arith.constant(
                    index_type,
                    kernel.metadata.num_warps * kernel.metadata.warp_size,
                )
                block_y = arith.constant(index_type, 1)
                block_z = arith.constant(index_type, 1)
                dynamic_smem = arith.constant(i32_type, kernel.metadata.shared)
                entry_arguments: Dict[str, ir.Value] = dict(
                    zip(input_names, entry_block.arguments)
                )
                kernel_operands: List[ir.Value] = [
                    self._cast_value(
                        self._triton_type_to_ir_type(triton_type),
                        entry_arguments[arg_name],
                        loc,
                    )
                    for arg_name, triton_type in runtime_parameters
                ]
                kernel_operands.extend(
                    [
                        self._make_null_ptr(),
                        self._make_null_ptr(),
                    ]
                )
                gpu.LaunchFuncOp(
                    kernel=[binary_name, kernel.metadata.name],
                    grid_size=(grid_x, grid_y, grid_z),
                    block_size=(block_x, block_y, block_z),
                    kernel_operands=kernel_operands,
                    dynamic_shared_memory_size=dynamic_smem,
                )
                func.ReturnOp(
                    [
                        entry_arguments[arg_name]
                        for arg_name in output_names
                        if arg_name in entry_arguments
                    ]
                )
        return fty

    def _import_hop_triton_kernel_wrapper_functional(
        self,
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
        function: triton.JITFunction = (
            torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
                kernel_idx
            )
        )
        kernel: triton.compiler.CompiledKernel = self.get_compiled_kernel(
            function,
            [],
            ktensors | constant_args,
        )
        if kernel is None:
            raise ValueError(f"failed to get compiled Triton kernel for {node.name}")
        runtime_parameters: List[Tuple[str, str]] = self._runtime_parameters(kernel)
        call_arguments: Dict[str, ir.Value] = {
            name: self._materialize_runtime_argument(
                loc,
                name,
                triton_type,
                kvalues,
                constant_args,
            )
            for name, triton_type in runtime_parameters
        }
        function_arguments: List[Tuple[str, ir.Type]] = [
            (name, call_arguments[name].type) for name, _ in runtime_parameters
        ]
        operands: List[ir.Value] = [
            call_arguments[name] for name, _ in runtime_parameters
        ]
        (grid,) = node.kwargs["grid"]
        fname: Final[str] = f"__triton_{node.name}"
        binary_name: Final[str] = f"{fname}_bin"
        self._add_gpu_binary(loc, self.fx_importer.module, binary_name, kernel)
        fty = self._build_wrapper_function(
            loc,
            fname,
            binary_name,
            kernel,
            runtime_parameters,
            function_arguments,
            grid,
            output_names,
        )

        call: ir.Operation = ir.Operation.create(
            "func.call",
            attributes={"callee": ir.FlatSymbolRefAttr.get(fname)},
            results=list(fty.results),
            operands=operands,
            loc=loc,
        )

        self._multi_result_nodes.add(node)

        for output_name, result in zip(output_names, call.results):
            self.bind_node_value(node, result, output_name)

    @staticmethod
    def get_compiled_kernel(
        function: triton.JITFunction, args: List[Any], kwargs: Dict[str, Any]
    ) -> Optional[triton.compiler.CompiledKernel]:
        device = triton.runtime.driver.active.get_current_device()
        kernel_cache, kernel_key_cache, _, _, binder = function.device_caches[device]
        _, specialization, options = binder(
            *args,
            **kwargs,
            debug=kwargs.get("debug", function.debug) or triton.knobs.runtime.debug,
            instrumentation_mode=triton.knobs.compilation.instrumentation_mode,
        )
        key: str = triton.runtime.jit.compute_cache_key(
            kernel_key_cache, specialization, options
        )
        return kernel_cache.get(key, None)


class TritonFxImporter(FxImporter):
    """FxImporter subclass that uses TritonGraphNodeImporter for triton op support."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def import_stateless_graph(
        self,
        g: torch.fx.Graph,
        *,
        func_name: str = "main",
        func_visibility: Optional[str] = None,
        import_symbolic_shape_expressions: bool = False,
    ) -> Any:
        """Override to inject TritonGraphNodeImporter."""
        ftype, loc = self._graph_to_function_meta(g)
        with loc:
            func_op = func.FuncOp(
                func_name,
                ftype,
                ip=self._m_ip,
                visibility=func_visibility,
            )
            entry_block = ir.Block.create_at_start(func_op.body, ftype.inputs)
        node_importer = TritonGraphNodeImporter(
            self,
            self._c,
            self._cc,
            entry_block,
        )
        node_importer.import_nodes(
            g.nodes,
            import_symbolic_shape_expressions=import_symbolic_shape_expressions,
        )
        self.symbol_table.insert(func_op)
        return func_op
