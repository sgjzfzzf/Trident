from __future__ import annotations

from typing_extensions import override

from .base import Guard


class UnCheckGuard(Guard):
    @override
    def __hash__(self) -> int:
        return super().__hash__()
