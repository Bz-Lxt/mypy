from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from mypy.checker_shared import TypeCheckerSharedApi

# This is global mutable state. Don't add anything here unless there's a very
# good reason.


class FunctionWorkLimitExhausted(BaseException):
    """Signals that checking the current function exceeded its work limit.

    This inherits from BaseException (rather than Exception) because it must
    propagate through generic ``except Exception`` blocks in the checker (e.g.
    the internal error handlers) up to the per-function boundary that reports
    the function as incompletely checked and continues with other functions.
    """

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(f"per-function work limit exhausted: {used} > {limit}")


class TypeCheckerState:
    # Wrap this in a class since it's faster that using a module-level attribute.

    def __init__(self, type_checker: TypeCheckerSharedApi | None) -> None:
        # Value varies by file being processed
        self.type_checker = type_checker

    @contextmanager
    def set(self, value: TypeCheckerSharedApi) -> Iterator[None]:
        saved = self.type_checker
        self.type_checker = value
        try:
            yield
        finally:
            self.type_checker = saved


checker_state: Final = TypeCheckerState(type_checker=None)
