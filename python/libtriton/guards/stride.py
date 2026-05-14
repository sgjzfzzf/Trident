from __future__ import annotations

import re
from typing import Any, Final, Optional
from typing_extensions import override

from .check import CheckGuard


class StrideGuard(CheckGuard):
    _regex_pattern: re.Pattern = re.compile(
        rf"{CheckGuard._regex_variable}\.stride\(\)\[{CheckGuard._regex_int}\] == {CheckGuard._regex_int}"
    )

    def __init__(
        self, variable: str, index: int, expected: int, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable
        self.index: Final[int] = index
        self.expected: Final[int] = expected

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.index, self.expected))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[StrideGuard]:
        if match := cls._regex_pattern.match(code):
            variable, index, expected = match.groups()
            return StrideGuard(variable, int(index), int(expected))
        else:
            return None
