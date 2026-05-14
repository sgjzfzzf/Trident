from __future__ import annotations

from typing import Any
from typing_extensions import override

from .base import Guard


class CheckGuard(Guard):
    _regex_int: str = r"(\d+)"
    _regex_variable: str = r"L\['([a-zA-Z_][a-zA-Z0-9_]*)'\]"

    @override
    def __init_subclass__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)
        Guard._registry = {*Guard._registry, cls}
