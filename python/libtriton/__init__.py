from __future__ import annotations

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from .backend import (  # noqa: E402
    triton_graph_backend,
)

__all__ = [
    "triton_graph_backend",
]
