# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Final, List, Sequence

from trident._C.trident_core import ir

from .guard import Guard


class Guards(object):
    """Represents a normalized set of guard code snippets."""

    def __init__(self, codes: Sequence[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guards: Final[List[Guard]] = [
            g for code in codes if (g := Guard.parse(code)) is not None
        ]

    def __hash__(self) -> int:
        return hash(self.__class__) ^ hash(tuple(self.guards))

    def build(
        self,
        names: Sequence[str],
        context: ir.Context,
    ) -> ir.ArrayAttr:
        """Build arg_attrs for a function-like operation.

        Returns an ``ir.ArrayAttr`` of ``ir.DictAttr`` entries, one per
        parameter, with ``tvm_ffi.guard`` set to an ``ArrayAttr`` of all
        matching guard attributes (or empty dict for parameters with no
        guards).
        """
        with context:
            guards_by_index: Dict[int, List[ir.Attribute]] = defaultdict(list)
            for guard in self.guards:
                if (index := names.index(guard.variable)) is not None and (
                    attr := guard.to_attribute(context)
                ) is not None:
                    guards_by_index[index].append(attr)

            arg_attrs: List[ir.DictAttr] = [ir.DictAttr.get({}) for _ in names]
            for index, attrs in guards_by_index.items():
                arg_attrs[index] = ir.DictAttr.get(
                    {"tvm_ffi.guard": ir.ArrayAttr.get(attrs)}
                )

            return ir.ArrayAttr.get(arg_attrs)
