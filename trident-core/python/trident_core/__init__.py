# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from ._mlir_libs._tridentCore import register_all_dialects, register_all_passes
from . import capi_utils, compiler_utils, execution_engine, fx, ir, passmanager, rewrite

__all__ = [
    "capi_utils",
    "compiler_utils",
    "execution_engine",
    "fx",
    "ir",
    "passmanager",
    "register_all_dialects",
    "register_all_passes",
    "rewrite",
]
