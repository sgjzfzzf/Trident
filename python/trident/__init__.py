# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from .compile import compile, jit


__all__ = [
    "compile",
    "jit",
]
