# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Iterable

import torch._guards

from .collection import Guards


def parse_guards(guards: Iterable[torch._guards.Guard]) -> Guards:
    return Guards(guards)


__all__ = [
    "parse_guards",
]
