# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Final, Self, TypeAlias, override

import torch
import torch.utils._pytree as pytree
from torch.export.graph_signature import (
    InputKind,
    SymIntArgument,
    TensorArgument,
)

from trident.core import ir
from trident.core.dialects import tvm_ffi as tvm_ffi_d

InputPath: TypeAlias = Sequence[int | str]


class _InputNode(ABC):
    """A table-owned input node with a cached region-local value."""

    __slots__ = (
        "type",
        "value",
    )

    def __init__(
        self,
        type: ir.Type,
        value: ir.Value | None = None,
    ) -> None:
        self.type: Final[ir.Type] = type
        self.value: ir.Value | None = value

    @abstractmethod
    def resolve(self, path: InputPath) -> ir.Value | None: ...

    @abstractmethod
    def flatten(self) -> list[ir.Value]: ...


class _InputLeafNode(_InputNode):
    """A terminal input node."""

    __slots__ = ()

    @override
    def resolve(self, path: InputPath) -> ir.Value | None:
        if path:
            return None
        return self.value

    @override
    def flatten(self) -> list[ir.Value]:
        return [] if self.value is None else [self.value]


class _InputNonLeafNode(_InputNode):
    """A container input node which materializes and caches its children."""

    __slots__ = ("children",)

    def __init__(
        self,
        type: ir.Type,
        children: Sequence[_InputNode | None],
        value: ir.Value | None = None,
    ) -> None:
        super().__init__(type, value)
        self.children: Final[list[_InputNode | None]] = [*children]

    @property
    def _children_iter(self) -> Iterator[tuple[int, _InputNode]]:
        return (
            (index, child)
            for index, child in enumerate(self.children)
            if child is not None
        )

    def _find_child(self, key: int | str) -> tuple[int, _InputNode] | None:
        return next(
            ((index, child) for index, child in self._children_iter if index == key),
            None,
        )

    def _get_child(self, index: int, child: _InputNode) -> _InputNode | None:
        if child.value is None:
            parent = self.value
            if parent is None:
                return None
            child.value = tvm_ffi_d.array_get_item(
                child.type,
                parent,
                tvm_ffi_d.constant_int(
                    ir.Type.parse("!tvm_ffi.int", context=parent.context),
                    ir.IntegerAttr.get(
                        ir.IntegerType.get_signless(64, parent.context), index
                    ),
                ),
                element_type=child.type,
            )
        return child

    def child(self, key: int | str) -> _InputNode | None:
        """Return the table-owned child selected by *key*."""
        result = self._find_child(key)
        if result is None:
            return None
        _, child = result
        return child

    @override
    def resolve(self, path: InputPath) -> ir.Value | None:
        if not path:
            return self.value
        step, *remaining = path
        result = self._find_child(step)
        if result is None:
            return None
        index, child = result
        child = self._get_child(index, child)
        return None if child is None else child.resolve(remaining)

    @override
    def flatten(self) -> list[ir.Value]:
        values: list[ir.Value] = []
        for index, child in self._children_iter:
            child = self._get_child(index, child)
            if child is not None:
                values.extend(child.flatten())
        return values


class InputTable:
    """Table-owned input nodes addressable through Python pytree paths."""

    def __init__(
        self,
        nodes: Mapping[str, _InputNode],
        input_names: Sequence[str],
    ) -> None:
        self._nodes: Final[dict[str, _InputNode]] = {**nodes}
        self._input_names: Final[list[str]] = [*input_names]
        assert all(name in self._nodes for name in self._input_names), (
            "exported input names must be present in the input table"
        )

    def __getitem__(self, path: InputPath) -> ir.Value | None:
        assert path, "input paths must not be empty"
        name, *remaining = path
        assert isinstance(name, str), "input paths must start with a parameter name"
        node = self._nodes.get(name)
        if node is None:
            return None
        return node.resolve(remaining)

    def flatten_inputs(self) -> list[ir.Value]:
        """Return all exported input leaves in graph-signature order."""
        return [
            value for name in self._input_names for value in self._nodes[name].flatten()
        ]


class InputNodeBuilder:
    """Static recipe for recursively building table-owned input nodes."""

    __slots__ = (
        "children",
        "type",
    )

    def __init__(
        self,
        type: ir.Type,
        children: Sequence[InputNodeBuilder | None] | None = None,
    ) -> None:
        self.children: Final[list[InputNodeBuilder | None] | None] = (
            None if children is None else [*children]
        )
        self.type: Final[ir.Type] = type

    def build(self, value: ir.Value | None = None) -> _InputNode:
        """Recursively build table-owned nodes with an optional root value."""
        if self.children is None:
            return _InputLeafNode(self.type, value)
        return _InputNonLeafNode(
            self.type,
            [None if child is None else child.build() for child in self.children],
            value,
        )


class InputTableBuilder:
    """Build the wrapper input schema and bind region-local IR operands."""

    def __init__(
        self,
        entries: Sequence[tuple[str, InputNodeBuilder]],
        input_names: Sequence[str],
    ) -> None:
        self._entries: Final[list[tuple[str, InputNodeBuilder]]] = [*entries]
        self._input_names: Final[list[str]] = [*input_names]
        self.input_types: Final[list[ir.Type]] = [
            builder.type for _, builder in self._entries
        ]

    @classmethod
    def get(
        cls,
        exported_program: torch.export.ExportedProgram,
        signature: inspect.Signature,
        bound_arguments: Mapping[str, Any],
        main_input_types: Sequence[ir.Type],
        value_type: Callable[[Any], ir.Type],
        context: ir.Context,
    ) -> Self:
        array_type = ir.Type.parse("!tvm_ffi.array", context=context)
        dtype_type = ir.Type.parse("!torchext.dtype", context=context)
        input_specs = exported_program.graph_signature.input_specs
        exported_input_values = pytree.tree_leaves(exported_program.example_inputs)
        assert len(input_specs) == len(exported_input_values), (
            "ExportedProgram input specs do not match flattened exported inputs: "
            f"got {len(input_specs)} specs and "
            f"{len(exported_input_values)} values"
        )

        main_input_count = sum(
            input_spec.kind == InputKind.USER_INPUT
            and isinstance(input_spec.arg, (TensorArgument, SymIntArgument))
            for input_spec in input_specs
        )
        assert main_input_count == len(main_input_types), (
            "ExportedProgram graph inputs do not match imported MLIR inputs: "
            f"got {main_input_count} graph inputs and "
            f"{len(main_input_types)} MLIR inputs"
        )

        in_spec = exported_program.call_spec.in_spec
        assert isinstance(in_spec, pytree.TreeSpec), (
            "trident.jit requires a pytree TreeSpec for exported inputs; "
            f"got {in_spec!r}"
        )
        root_children = in_spec.children()
        assert in_spec.type is tuple and len(root_children) == 2, (
            f"unexpected input spec root (expected tuple TreeSpec, got {in_spec})"
        )
        [args_spec, kwargs_spec] = root_children
        assert isinstance(args_spec, pytree.TreeSpec), (
            f"expected args tree spec, got {type(args_spec)!r}"
        )
        assert isinstance(kwargs_spec, pytree.TreeSpec), (
            f"expected kwargs tree spec, got {type(kwargs_spec)!r}"
        )

        signature_names = [*signature.parameters]
        positional_names = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        args_children = args_spec.children()
        kwargs_names = [*kwargs_spec.context]
        kwargs_children = kwargs_spec.children()
        assert len(kwargs_names) == len(kwargs_children), (
            "ExportedProgram keyword inputs do not match the call spec: "
            f"got {len(kwargs_children)} inputs and {len(kwargs_names)} names"
        )
        assert len(args_children) <= len(positional_names), (
            "ExportedProgram positional inputs exceed the positional parameters: "
            f"got {len(args_children)} inputs and "
            f"{len(positional_names)} parameters"
        )
        args_names = [name for name, _ in zip(positional_names, args_children)]
        provided_names = [*args_names, *kwargs_names]
        assert len(provided_names) == len(set(provided_names)), (
            "ExportedProgram input trees map multiple inputs to the same "
            f"function parameter: {provided_names!r}"
        )
        assert all(name in signature_names for name in provided_names), (
            "ExportedProgram input trees contain unknown function parameters: "
            f"got {provided_names!r}, expected names from {signature_names!r}"
        )
        leaf_iter = zip(input_specs, exported_input_values)
        main_input_type_iter = iter(main_input_types)

        def build_node(node: pytree.TreeSpec, name: str) -> InputNodeBuilder:
            if node.is_leaf():
                input_spec, value = next(leaf_iter)
                type = (
                    next(main_input_type_iter)
                    if input_spec.kind == InputKind.USER_INPUT
                    and isinstance(input_spec.arg, (TensorArgument, SymIntArgument))
                    else value_type(value)
                )
                return InputNodeBuilder(type)
            else:
                assert node.type is not dict, (
                    f"dict parameters (path step {name!r}) are not yet "
                    "supported by trident.jit"
                )
                children = [build_node(child, name) for child in node.children()]
                assert not children or not all(
                    child.type == dtype_type for child in children
                ), "containers of multiple torch.dtype values are not supported"
                return InputNodeBuilder(array_type, children=children)

        provided_entries = {
            name: build_node(child, name)
            for name, child in [
                *zip(args_names, args_children),
                *zip(kwargs_names, kwargs_children),
            ]
        }
        entries = [
            provided_entries[name]
            if name in provided_entries
            else InputNodeBuilder(
                array_type
                if isinstance(bound_arguments[name], (list, tuple))
                else value_type(bound_arguments[name])
            )
            for name in signature_names
        ]
        input_names = [*args_names, *kwargs_names]
        return cls(
            [*zip(signature_names, entries)],
            input_names,
        )

    def build(self, operands: Sequence[ir.Value]) -> InputTable:
        """Bind *operands* to a fresh table for the current IR region."""
        assert len(operands) == len(self._entries), (
            "input operand count does not match the wrapper signature: "
            f"got {len(operands)}, expected {len(self._entries)}"
        )
        nodes = {
            name: builder.build(operand)
            for (name, builder), operand in zip(self._entries, operands)
        }
        return InputTable(nodes, self._input_names)
