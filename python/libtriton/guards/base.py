from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Final, Iterator, List, Optional, Sequence, Set, Type
from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith, scf


class Guard(object):
    """Base class for guard definitions with opt-in subclass registration."""

    _registry: Set[Type[Guard]] = set()

    def __hash__(self) -> int:
        return hash(self.__class__)

    @staticmethod
    def parse(code: str) -> Guard:
        for cls in Guard._registry:
            if guard := cls._parse(code):
                return guard
        from .uncheck import UnCheckGuard

        return UnCheckGuard()

    @classmethod
    @abstractmethod
    def _parse(cls, code: str) -> Optional[Guard]: ...

    @abstractmethod
    def build_ir(
        self,
        symbol_table: Dict[str, ir.Value],
        *,
        context: Optional[ir.Context] = None,
        loc: Optional[ir.Location] = None,
    ) -> ir.Operation:
        return arith.constant(
            ir.IntegerType.get_signless(1, context=context), 1, loc=loc
        )


class Guards(object):
    """Represents a normalized set of guard code snippets."""

    def __init__(self, codes: Sequence[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guards: Final[List[Guard]] = [Guard.parse(code) for code in codes]

    def __hash__(self) -> int:
        return hash(self.__class__) ^ hash(tuple(self.guards))

    def build_ir(
        self,
        symbol_table: Dict[str, ir.Value],
        *,
        context: Optional[ir.Context] = None,
        loc: Optional[ir.Location] = None,
    ) -> ir.Operation:
        i1_type: ir.Type = ir.IntegerType.get_signless(1, context=context)
        false_const: ir.Value = arith.constant(i1_type, 0, loc=loc)
        true_const: ir.Value = arith.constant(i1_type, 1, loc=loc)

        def build_chain(giter: Iterator[Guard]) -> ir.Value:
            guard: Optional[Guard] = next(giter, None)
            if guard is None:
                return true_const
            guard_cond: ir.Value = guard.build_ir(
                symbol_table, context=context, loc=loc
            )
            if_op: ir.Value = scf.IfOp(guard_cond, [i1_type], has_else=True, loc=loc)
            with ir.InsertionPoint(if_op.then_block):
                next_val: ir.Value = build_chain(giter)
                scf.YieldOp([next_val])
            with ir.InsertionPoint(if_op.else_block):
                scf.YieldOp([false_const])
            [result_value] = if_op.results
            return result_value

        return build_chain(iter(self.guards))
