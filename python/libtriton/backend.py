from __future__ import annotations
from typing import Any, Callable, Final

import torch


class LibTritonGraphModule(object):
    """Compiles a torch.fx.GraphModule via torch.export and calls gm directly."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Callable[..., Any]] = fn
        self.gm: torch.fx.GraphModule | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.gm is None:
            return self.compile(*args, **kwargs)
        return self.gm(*args, **kwargs)

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        gm, _ = torch._dynamo.export(
            self.fn, aten_graph=True, assume_static_by_default=True
        )(*args)
        self.gm = gm
        return self.gm(*args, **kwargs)
