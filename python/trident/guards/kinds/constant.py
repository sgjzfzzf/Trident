# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from ..codes import ConstantCode
from .base import Guard


class ConstantMatchGuard(Guard):
    create_fn_name = "CONSTANT_MATCH"
    code_types = (ConstantCode,)

    @classmethod
    def validate_code_list(
        cls,
        texts: tuple[str, ...],
    ) -> bool:
        return len(texts) == 1
