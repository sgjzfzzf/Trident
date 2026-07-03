# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Final, Optional, Set, Type

from trident._C.trident_core import ir


class Guard(object):
    """Base class for guard definitions with opt-in subclass registration."""

    _regex_int: str = r"(\d+)"
    _regex_variable: str = r"L\['([a-zA-Z_][a-zA-Z0-9_]*)'\]"
    _registry: Set[Type[Guard]] = set()

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable

    def __hash__(self) -> int:
        return hash(self.__class__)

    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)
        Guard._registry = {*Guard._registry, cls}

    @staticmethod
    def parse(code: str) -> Optional[Guard]:
        for cls in Guard._registry:
            if guard := cls._parse(code):
                return guard
        return None

    @classmethod
    @abstractmethod
    def _parse(cls, code: str) -> Optional[Guard]: ...

    @abstractmethod
    def to_attribute(self, context: ir.Context) -> Optional[ir.Attribute]:
        """Generate the MLIR guard attribute for this guard.

        Returns None for guards that don't produce an attribute.
        """
        return None
