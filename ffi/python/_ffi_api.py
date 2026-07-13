# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""FFI API bindings for trident.ffi.

This file contains ``tvm-ffi-stubgen`` directive blocks that are filled in
at build time by running ``tvm-ffi-stubgen`` against ``libTridentFFI.so``.
Do **not** manually edit the sections between ``tvm-ffi-stubgen(begin)`` and
``tvm-ffi-stubgen(end)`` markers — they are regenerated on every build.

.. code-block:: bash

    tvm-ffi-stubgen <dir> --dlls <path/to/libTridentFFI.so>
"""

# Imports required by the stubgen-generated code below.
# Hard-coded rather than using an ``import-section`` block because stubgen
# automatically writes ``from __future__ import annotations`` there, which
# would appear after non-future statements and cause a SyntaxError.
# isort: off
from __future__ import annotations
from tvm_ffi import (
    Object as _Object,
    init_ffi_api as _FFI_INIT_FUNC,  # noqa: F401
    register_object as _FFI_REG_OBJ,
)
from typing import TYPE_CHECKING  # noqa: F401
# isort: on

# tvm-ffi-stubgen(import-object): ffi.Object;False;_Object
# tvm-ffi-stubgen(import-object): tvm_ffi.register_object;False;_FFI_REG_OBJ

# tvm-ffi-stubgen(begin): global/trident.ffi
# tvm-ffi-stubgen(end)


@_FFI_REG_OBJ("trident.ffi.Exception")
class Exception(_Object):
    """FFI binding for `trident.ffi.Exception`."""

    # tvm-ffi-stubgen(begin): object/trident.ffi.Exception
    # tvm-ffi-stubgen(end)


__all__ = [
    # tvm-ffi-stubgen(begin): __all__
    # tvm-ffi-stubgen(end)
]
