# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from typing import Final

import torch._guards
from torch._guards import GuardSource

from trident.core import ir
from trident.core.dialects import arithext
from trident.input import InputTable

from .codes import GuardBuilder
from .kinds import Guard

_CAPTURE_GUARD_SOURCES: Final[frozenset[GuardSource]] = frozenset(
    {
        GuardSource.CONSTANT,
        GuardSource.GLOBAL,
        GuardSource.GLOBAL_FSDP_MODULE,
        GuardSource.GLOBAL_SPECIALIZED_NN_MODULE,
        GuardSource.GLOBAL_UNSPECIALIZED_BUILTIN_NN_MODULE,
        GuardSource.GLOBAL_UNSPECIALIZED_NN_MODULE,
    }
)


class Guards:
    """Build a short-circuiting IR chain from a Torch ``GuardsSet``."""

    def __init__(self, guards: Iterable[torch._guards.Guard]) -> None:
        self.guards: Final[list[torch._guards.Guard]] = list(guards)

    def build(
        self,
        table_factory: Callable[[], InputTable],
        context: ir.Context,
    ) -> ir.Value:
        codes: list[GuardBuilder] = []
        for guard in self.guards:
            # Export has already resolved captured globals/defaults into the
            # graph. They are not wrapper ABI arguments, so there is no runtime
            # value from which Trident could build a semantic check.
            if guard.source not in _CAPTURE_GUARD_SOURCES:
                parsed = Guard.parse(guard)
                assert parsed is not None, (
                    "unsupported Dynamo guard: "
                    f"{guard.create_fn_name()} {guard.name!r} "
                    f"{guard.code_list!r}"
                )
                codes.extend(parsed.codes)

        unique: dict[Hashable, GuardBuilder] = {code.key: code for code in codes}
        ordered: list[GuardBuilder] = sorted(
            unique.values(),
            key=lambda code: (
                # Lower priorities run first; depth orders checks within each
                # priority, with nested structure checks running outside-in.
                code.priority,
                code.depth,
            ),
        )

        i1 = ir.IntegerType.get_signless(1, context)
        and_then = arithext.AndThenOp(i1, len(ordered))
        for region, code in zip(and_then.regions_, ordered):
            block = ir.Block.create_at_start(region, [])
            with ir.InsertionPoint(block):
                table = table_factory()
                result = code.build(table, context)
                arithext.and_then_yield(result)
        return and_then.result
