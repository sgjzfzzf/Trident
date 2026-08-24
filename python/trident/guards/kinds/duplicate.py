# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from ..codes.expression import ExpressionCode
from ..local import Local
from .base import Guard


class DuplicateInputGuard(Guard):
    """Require the two named wrapper inputs to have object identity."""

    create_fn_name = "DUPLICATE_INPUT"
    code_types = (ExpressionCode,)

    @classmethod
    def validate_code_list(cls, texts: list[str]) -> bool:
        return len(texts) == 1

    @classmethod
    def parse_codes(
        cls,
        texts: list[str],
        source: Local | None,
    ) -> list[ExpressionCode] | None:
        if source is None:
            return None
        return super().parse_codes(texts, source)  # type: ignore[return-value]
