# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import tvm_ffi


@tvm_ffi.register_error
class GuardMatchException(Exception):
    pass
