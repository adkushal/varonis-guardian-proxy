"""Chain of Responsibility: ordered prompt gate handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from guardian_client import GuardianClient
from strategies import (
    BlockDecision,
    GuardianToxicityStrategy,
    HeuristicBlockStrategy,
    PromptContext,
    PromptEvaluationStrategy,
)


class PromptGate(ABC):
    """Anything that can decide whether a prompt is blocked (chain or decorator)."""

    @abstractmethod
    async def handle(self, ctx: PromptContext) -> Optional[BlockDecision]:
        """Return a BlockDecision to block, or None to allow."""


class PromptHandler(PromptGate):
    """
    Chain of Responsibility node.
    If this handler produces a BlockDecision, the chain stops; otherwise it
    forwards to the next handler.
    """

    def __init__(self) -> None:
        self._next: Optional[PromptHandler] = None

    @property
    def next_handler(self) -> Optional[PromptHandler]:
        return self._next

    def set_next(self, handler: PromptHandler) -> PromptHandler:
        self._next = handler
        return handler

    async def handle(self, ctx: PromptContext) -> Optional[BlockDecision]:
        decision = await self._check(ctx)
        if decision is not None:
            return decision
        if self._next is not None:
            return await self._next.handle(ctx)
        return None

    @abstractmethod
    async def _check(self, ctx: PromptContext) -> Optional[BlockDecision]:
        ...


class StrategyHandler(PromptHandler):
    """Chain node that delegates the check to a PromptEvaluationStrategy."""

    def __init__(self, strategy: PromptEvaluationStrategy) -> None:
        super().__init__()
        self.strategy = strategy

    async def _check(self, ctx: PromptContext) -> Optional[BlockDecision]:
        return await self.strategy.evaluate(ctx)


def build_chain(strategies: Sequence[PromptEvaluationStrategy]) -> PromptHandler:
    """Link strategies into a Chain of Responsibility (order preserved)."""
    if not strategies:
        raise ValueError("build_chain requires at least one strategy")
    handlers = [StrategyHandler(s) for s in strategies]
    head = handlers[0]
    current = head
    for handler in handlers[1:]:
        current = current.set_next(handler)
    return head


def build_default_prompt_chain(guardian: GuardianClient | None = None) -> PromptHandler:
    """Default gate: heuristics first (named reasons), then Guardian toxicity."""
    client = guardian or GuardianClient()
    return build_chain(
        [
            HeuristicBlockStrategy(),
            GuardianToxicityStrategy(client),
        ]
    )
