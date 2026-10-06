# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from . import execution_engine, ir, passmanager, rewrite
from ._mlir_libs._trident import (
    register_all_dialects,
    register_all_passes,
)

__all__ = [
    "execution_engine",
    "ir",
    "passmanager",
    "register_all_dialects",
    "register_all_passes",
    "rewrite",
]
