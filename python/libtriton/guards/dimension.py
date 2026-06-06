from __future__ import annotations

import re
from typing import Any, Final, Optional
from typing_extensions import override

from .check import CheckGuard


class DimensionGuard(CheckGuard):
    _regex_pattern: re.Pattern = re.compile(
        rf"{CheckGuard._regex_variable}\.ndimension\(\) == {CheckGuard._regex_int}"
    )

    def __init__(self, variable: str, expected: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable
        self.expected: Final[int] = expected

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.variable, self.expected))

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[DimensionGuard]:
        if match := cls._regex_pattern.match(code):
            variable, expected = match.groups()
            return DimensionGuard(variable, int(expected))
        else:
            return None
