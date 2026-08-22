# Part of the Trident project, under the MIT License.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final

from ..codes.base import GuardCode
from ..local import Local

if TYPE_CHECKING:
    import torch._guards


class Guard:
    """Select and compose Code implementations for one Dynamo Guard kind."""

    create_fn_name: ClassVar[str | None] = None
    code_types: ClassVar[tuple[type[GuardCode], ...]] = ()
    _registry: ClassVar[set[type[Guard]]] = set()

    def __init__(
        self,
        source: Local | None,
        codes: tuple[GuardCode, ...],
    ) -> None:
        self.source: Final[Local | None] = source
        self.codes: Final[tuple[GuardCode, ...]] = codes

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        create_fn_name = cls.__dict__.get("create_fn_name")
        if create_fn_name is not None:
            assert isinstance(create_fn_name, str) and create_fn_name, (
                "Guard.create_fn_name must be a non-empty string"
            )
            assert not any(
                registered.create_fn_name == create_fn_name
                for registered in Guard._registry
            ), f"duplicate Guard registration for {create_fn_name!r}"
            Guard._registry.add(cls)

    @classmethod
    def parse(cls, guard: torch._guards.Guard) -> Guard | None:
        create_fn_name = guard.create_fn_name()
        builder = next(
            (
                registered
                for registered in cls._registry
                if registered.create_fn_name == create_fn_name
            ),
            None,
        )
        if builder is None or not builder.validate_guard(guard):
            return None
        texts = tuple(guard.code_list or ())
        if not builder.validate_code_list(texts):
            return None
        source = builder.parse_source(guard)
        codes = builder.parse_codes(texts, source)
        if codes is None:
            return None
        return builder(source, codes)

    @classmethod
    def validate_code_list(
        cls,
        texts: tuple[str, ...],
    ) -> bool:
        """Validate create-function-specific CodeList cardinality."""
        return True

    @classmethod
    def validate_guard(cls, guard: torch._guards.Guard) -> bool:
        return True

    @classmethod
    def parse_source(
        cls,
        guard: torch._guards.Guard,
    ) -> Local | None:
        return Local.parse(guard.name)

    @classmethod
    def parse_codes(
        cls,
        texts: tuple[str, ...],
        source: Local | None,
    ) -> tuple[GuardCode, ...] | None:
        assert cls.create_fn_name is not None
        parsed_codes: list[GuardCode] = []
        for text in texts:
            matches = tuple(
                parsed
                for code_type in cls.code_types
                if (parsed := code_type.parse(text, source)) is not None
            )
            if len(matches) != 1:
                return None
            parsed_codes.append(matches[0])
        return tuple(parsed_codes)
