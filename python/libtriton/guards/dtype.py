from __future__ import annotations

import re
from typing import Any, Final, Optional
from typing_extensions import override

from .check import CheckGuard


class DTypeGuard(CheckGuard):
    _regex_pattern: re.Pattern = re.compile(
        rf"str\({CheckGuard._regex_variable}\.dtype\) == 'torch\.float32'"
    )

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.variable)

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[DTypeGuard]:
        if match := cls._regex_pattern.match(code):
            [variable] = match.groups()
            return DTypeGuard(variable)
        else:
            return None
