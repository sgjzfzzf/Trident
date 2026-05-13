from __future__ import annotations

# from abc import abstractmethod
from functools import reduce
from typing import (
    Any,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Type,
)

from libtriton._C.libtriton_core import ir
from libtriton._C.libtriton_core.dialects import arith


class Guard(object):
    """Base class for guard definitions with opt-in subclass registration."""

    _registry: Set[Type[Guard]] = set()

    def __init__(self, code: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.code: Final[str] = code

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)
        Guard._registry = {*Guard._registry, cls}

    def __hash__(self) -> int:
        return hash(Guard) ^ hash(self.code)

    @classmethod
    # @abstractmethod
    def parse(cls, code: str) -> Guard:
        return Guard(code)

    # @abstractmethod
    def build_ir(
        self, context: Optional[ir.Context] = None, loc: Optional[ir.Location] = None
    ) -> ir.Operation:
        return arith.constant(ir.Type.parse("i1"), 1, loc=loc)


class Guards(object):
    """Represents a normalized set of guard code snippets."""

    def __init__(self, codes: Sequence[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guards: Final[List[Guard]] = [Guard.parse(code) for code in codes]

    def __hash__(self) -> int:
        return reduce(lambda acc, guard: acc ^ hash(guard), self.guards, hash(Guards))

    def build_ir(
        self, context: Optional[ir.Context] = None, loc: Optional[ir.Location] = None
    ) -> ir.Operation:
        return reduce(
            lambda cond, guard: arith.andi(
                cond,
                guard.build_ir(context=context, loc=loc),
            ),
            self.guards,
            arith.constant(ir.Type.parse("i1"), 1),
        )


class GuardParser(object):
    """Minimal parser that preserves stable guard keys for backend dispatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def parse(codes: Iterable[str]) -> Iterable[str]:
        return filter(None, codes)

    def parse_guards(self, guards_set: Any) -> Guards:
        """Collect guard code snippets from torch._guards.GuardsSet."""
        codes: List[str] = list(
            self.parse(
                code for entry in guards_set.inner for code in entry.code_list or []
            )
        )
        return Guards(codes)
