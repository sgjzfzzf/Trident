# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import ClassVar, Final, Self

from trident.core import ir
from trident.input import InputTable

from ..ast import GuardBuildFn
from ..local import Local


@dataclass(frozen=True)
class GuardBuilder:
    """Metadata and delayed IR builder for one parsed guard code."""

    code: GuardCode
    build_fn: GuardBuildFn

    def build(self, tree: InputTable, context: ir.Context) -> ir.Value:
        return self.build_fn(tree, context)

    @property
    def depth(self) -> int:
        return self.code.depth

    @property
    def key(self) -> Hashable:
        return self.code.key

    @property
    def priority(self) -> int:
        return self.code.priority

    @property
    def text(self) -> str:
        return self.code.text


class GuardCode:
    """Parsed representation of one expression from a Guard code list."""

    # Lower priorities are built first.  Priority zero is reserved for
    # structure checks, whose source depth is used as a secondary key.
    priority: ClassVar[int] = 1

    def __init__(self, text: str, source: Local | None) -> None:
        self.text: Final[str] = text
        self.source: Final[Local | None] = source

    @abstractmethod
    def build(
        self,
        tree: InputTable,
        context: ir.Context,
    ) -> ir.Value:
        raise NotImplementedError(
            f"guard code {type(self).__name__} cannot be lowered: {self.text!r}"
        )

    @property
    def depth(self) -> int:
        return 0

    @property
    def key(self) -> Hashable:
        return (type(self), self.source, self.text)

    @classmethod
    @abstractmethod
    def parse(
        cls,
        text: str,
        source: Local | None,
    ) -> Self | None: ...

    def to_builder(self) -> GuardBuilder:
        return GuardBuilder(
            code=self,
            build_fn=self.build,
        )
