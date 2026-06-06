from __future__ import annotations

from typing import Any, Final

import torch


class Function(object):
    """A compiled function variant."""

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.gm: Final[torch.fx.GraphModule] = gm
