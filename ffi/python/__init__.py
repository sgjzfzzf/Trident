# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

"""Trident FFI runtime — Exception types and FFI utilities."""

import tvm_ffi

# Preload libTridentFFI.so so that tvm_ffi can discover its registered types.
# Uses :func:`tvm_ffi.libinfo.load_lib_ctypes` which locates the shared
# library via the distribution RECORD.
tvm_ffi.libinfo.load_lib_ctypes("trident", "TridentFFI", "RTLD_GLOBAL")

# Import auto-generated FFI API bindings (produced by tvm-ffi-stubgen).
# ``_ffi_api.py`` is generated at build time from libTridentFFI.so and
# installed alongside this ``__init__.py``.
from ._ffi_api import Exception

__all__ = [
    "Exception",
]
