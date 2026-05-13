from __future__ import annotations

import tvm_ffi


@tvm_ffi.register_error("GuardMatchException")
class GuardMatchException(Exception):
    pass
