from __future__ import annotations

from abc import abstractmethod
from typing import Any, Final, List, Optional, Sequence, Set, Type


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
        return UnCheckGuard()

    @classmethod
    @abstractmethod
    def _parse(cls, code: str) -> Optional[Guard]: ...


class Guards(object):
    """Represents a normalized set of guard code snippets."""

    def __init__(self, codes: Sequence[str], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guards: Final[List[Guard]] = [Guard.parse(code) for code in codes]

    def __hash__(self) -> int:
        return hash(self.__class__) ^ hash(tuple(self.guards))


# Late import to avoid circular dependency
from .uncheck import UnCheckGuard  # noqa: E402, F811
