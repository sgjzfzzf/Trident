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
from trident.core.dialects import torch as torch_d
from trident.core.dialects import torchext

InputPath: TypeAlias = Sequence[int | str]
InputValue: TypeAlias = Any


pytree.register_pytree_node(
    torch.device,
    lambda value: ([], value),
    lambda _, value: value,
    flatten_with_keys_fn=lambda value: ([], value),
    serialized_type_name="trident.torch.device",
    to_dumpable_context=str,
    from_dumpable_context=torch.device,
)
pytree.register_pytree_node(
    torch.dtype,
    lambda value: ([], value),
    lambda _, value: value,
    flatten_with_keys_fn=lambda value: ([], value),
    serialized_type_name="trident.torch.dtype",
    to_dumpable_context=str,
    from_dumpable_context=lambda value: getattr(torch, value.removeprefix("torch.")),
)


class _InputNode(ABC):
    """A table-owned input node with a cached region-local value."""

    __slots__ = ("type", "value")

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
            if isinstance(parent.type, torch_d.TorchListType):
                unpacked = torch_d.prim_ListUnpack(
                    [node.type for _, node in self._children_iter], parent
                )
                results = (
                    [unpacked] if isinstance(unpacked, ir.Value) else list(unpacked)
                )
                for (_, node), value in zip(self._children_iter, results, strict=True):
                    node.value = value
            elif isinstance(parent.type, torch_d.TorchTupleType):
                unpacked = torch_d.prim_TupleUnpack(
                    [node.type for _, node in self._children_iter], parent
                )
                results = (
                    [unpacked] if isinstance(unpacked, ir.Value) else list(unpacked)
                )
                for (_, node), value in zip(self._children_iter, results, strict=True):
                    node.value = value
            else:
                raise TypeError(
                    f"unsupported guard container type: {parent.type}; expected a Torch list or tuple"
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

    __slots__ = ("children", "type")

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
        bound_arguments: Mapping[str, InputValue],
        main_input_types: Sequence[ir.Type],
        value_type: Callable[[InputValue], ir.Type],
        context: ir.Context,
    ) -> Self:
        def container_type(
            node_type: type, element_types: Sequence[ir.Type]
        ) -> ir.Type:
            if node_type is tuple:
                return torch_d.TorchTupleType.get(element_types, context=context)
            assert node_type is list
            [element_type] = {*element_types}
            return torch_d.TorchListType.get(element_type)

        input_specs = exported_program.graph_signature.input_specs
        exported_input_values = pytree.tree_leaves(exported_program.example_inputs)
        assert len(input_specs) == len(exported_input_values), (
            f"ExportedProgram input specs do not match flattened exported inputs: got {len(input_specs)} specs and {len(exported_input_values)} values"
        )

        main_input_count = sum(
            input_spec.kind == InputKind.USER_INPUT
            and isinstance(input_spec.arg, (TensorArgument, SymIntArgument))
            for input_spec in input_specs
        )
        assert main_input_count == len(main_input_types), (
            f"ExportedProgram graph inputs do not match imported MLIR inputs: got {main_input_count} graph inputs and {len(main_input_types)} MLIR inputs"
        )

        in_spec = exported_program.call_spec.in_spec
        assert isinstance(in_spec, pytree.TreeSpec), (
            f"trident.jit requires a pytree TreeSpec for exported inputs; got {in_spec!r}"
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
            f"ExportedProgram keyword inputs do not match the call spec: got {len(kwargs_children)} inputs and {len(kwargs_names)} names"
        )
        assert len(args_children) <= len(positional_names), (
            f"ExportedProgram positional inputs exceed the positional parameters: got {len(args_children)} inputs and {len(positional_names)} parameters"
        )
        args_names = [name for name, _ in zip(positional_names, args_children)]
        provided_names = [*args_names, *kwargs_names]
        assert len(provided_names) == len(set(provided_names)), (
            f"ExportedProgram input trees map multiple inputs to the same function parameter: {provided_names!r}"
        )
        assert all(name in signature_names for name in provided_names), (
            f"ExportedProgram input trees contain unknown function parameters: got {provided_names!r}, expected names from {signature_names!r}"
        )
        leaf_iter = zip(input_specs, exported_input_values)
        main_input_type_iter = iter(main_input_types)

        def build_node(node: pytree.TreeSpec, name: str) -> InputNodeBuilder:
            if (
                node.type in (torch.device, torch.dtype)
                and not node.children()
                and isinstance(value := node.context, (torch.device, torch.dtype))
            ):
                return InputNodeBuilder(value_type(value), children=[])
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
                    f"dict parameters (path step {name!r}) are not yet supported by trident.jit"
                )
                children = [build_node(child, name) for child in node.children()]
                assert not children or not all(
                    isinstance(child.type, torchext.DTypeType) for child in children
                ), "containers of multiple torch.dtype values are not supported"
                return InputNodeBuilder(
                    container_type(node.type, [child.type for child in children]),
                    children=children,
                )

        provided_entries = {
            name: build_node(child, name)
            for name, child in [
                *zip(args_names, args_children),
                *zip(kwargs_names, kwargs_children),
            ]
        }

        def build_value(value: InputValue) -> InputNodeBuilder:
            if isinstance(value, (list, tuple)):
                children = [build_value(child) for child in value]
                return InputNodeBuilder(
                    container_type(type(value), [child.type for child in children]),
                    children=children,
                )
            return InputNodeBuilder(value_type(value))

        entries = [
            provided_entries[name]
            if name in provided_entries
            else build_value(bound_arguments[name])
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
            f"input operand count does not match the wrapper signature: got {len(operands)}, expected {len(self._entries)}"
        )
        nodes = {
            name: builder.build(operand)
            for (name, builder), operand in zip(self._entries, operands)
        }
        return InputTable(nodes, self._input_names)
