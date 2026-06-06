from __future__ import annotations

import re
from typing import Any, Final, Optional
from typing_extensions import override

from .check import CheckGuard


class TensorTypeGuard(CheckGuard):
    _regex_pattern: re.Pattern = re.compile(
        rf"___check_type_id\({CheckGuard._regex_variable},\s*{CheckGuard._regex_int}\),\s*type=<class 'torch\.Tensor'>"
    )

    def __init__(self, variable: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.variable: Final[str] = variable

    @override
    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.variable)

    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[TensorTypeGuard]:
        if match := cls._regex_pattern.match(code):
            variable, _ = match.groups()
            return TensorTypeGuard(variable)
        else:
            return None
