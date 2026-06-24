from __future__ import annotations

import tvm_ffi


@tvm_ffi.register_error
class GuardMatchException(Exception):
    pass
