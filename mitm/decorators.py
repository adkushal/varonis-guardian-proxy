"""Decorator pattern: wrap a PromptGate to add cross-cutting behavior."""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from chain import PromptGate
from strategies import BlockDecision, PromptContext

logger = logging.getLogger("prompt_gate")

LogFn = Callable[[str], None]


class PromptGateDecorator(PromptGate):
    """Base Decorator: same PromptGate interface, delegates to a wrapped gate."""

    def __init__(self, wrapped: PromptGate) -> None:
        self._wrapped = wrapped

    @property
    def wrapped(self) -> PromptGate:
        return self._wrapped

    async def handle(self, ctx: PromptContext) -> Optional[BlockDecision]:
        return await self._wrapped.handle(ctx)


class LoggingPromptGate(PromptGateDecorator):
    """Logs gate entry and block/allow outcome around the wrapped gate."""

    def __init__(self, wrapped: PromptGate, log: LogFn | None = None) -> None:
        super().__init__(wrapped)
        self._log = log or logger.info

    async def handle(self, ctx: PromptContext) -> Optional[BlockDecision]:
        rid = ctx.request_id or "-"
        self._log(f"[{rid}] gate start role={ctx.role} len={len(ctx.text)}")
        decision = await self._wrapped.handle(ctx)
        if decision is None:
            self._log(f"[{rid}] gate allow")
        else:
            self._log(
                f"[{rid}] gate block strategy={decision.strategy_name} "
                f"code={decision.code} detail={decision.detail}"
            )
        return decision


class TimingPromptGate(PromptGateDecorator):
    """Records how long the wrapped gate takes (milliseconds)."""

    def __init__(self, wrapped: PromptGate, log: LogFn | None = None) -> None:
        super().__init__(wrapped)
        self._log = log or logger.info

    async def handle(self, ctx: PromptContext) -> Optional[BlockDecision]:
        rid = ctx.request_id or "-"
        started = time.perf_counter()
        decision = await self._wrapped.handle(ctx)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._log(f"[{rid}] gate elapsed_ms={elapsed_ms:.1f}")
        return decision


def decorate_prompt_gate(
    gate: PromptGate,
    *,
    logging_enabled: bool = True,
    timing_enabled: bool = True,
    log: LogFn | None = None,
) -> PromptGate:
    """Compose decorators around a gate. Outermost first: Logging → Timing → gate."""
    decorated = gate
    if timing_enabled:
        decorated = TimingPromptGate(decorated, log=log)
    if logging_enabled:
        decorated = LoggingPromptGate(decorated, log=log)
    return decorated
