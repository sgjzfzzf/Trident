from __future__ import annotations

from typing import Any, Optional
from typing_extensions import override

from .base import Guard


class UnCheckGuard(Guard):
    @classmethod
    @override
    def _parse(cls, code: str) -> Optional[Guard]:
        return UnCheckGuard()
