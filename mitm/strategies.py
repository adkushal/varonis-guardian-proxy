"""Strategy pattern: pluggable prompt evaluation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from guardian_client import GuardianClient
from simple_classifier import REASON_TOXIC, classify_prompt

GUARDIAN_UNAVAILABLE_MESSAGE = (
    "The prompt could not be screened because the safety service is unavailable."
)


@dataclass(frozen=True)
class PromptContext:
    text: str
    role: str = "user"
    request_id: str = ""


@dataclass(frozen=True)
class BlockDecision:
    """Outcome when a strategy decides the prompt must be blocked."""

    message: str
    code: str
    strategy_name: str
    detail: str = ""
    status: int = 403


class PromptEvaluationStrategy(ABC):
    """Strategy interface: evaluate a prompt and optionally return a block decision."""

    name: str = "strategy"

    @abstractmethod
    async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
        """Return a BlockDecision to block, or None to allow this strategy to pass."""


class HeuristicBlockStrategy(PromptEvaluationStrategy):
    """Simple keyword/regex classifier for violence / illegal / sexual content."""

    name = "heuristic"

    async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
        result = classify_prompt(ctx.text)
        if not result.block_message:
            return None
        return BlockDecision(
            message=result.block_message,
            code=f"heuristic_{result.category}",
            strategy_name=self.name,
            detail=result.category or "",
        )


class GuardianToxicityStrategy(PromptEvaluationStrategy):
    """
    Multi-criteria Granite Guardian toxicity check.
    Blocks when any enabled criterion returns yes. If some criteria produced no
    verdict and the client is fail-closed, blocks with 503 instead of allowing.
    """

    name = "guardian"

    def __init__(self, client: GuardianClient | None = None) -> None:
        self.client = client or GuardianClient()

    async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
        results = await self.client.evaluate_text(ctx.text, role=ctx.role)
        risky = [r.criterion_id for r in results if r.risk]
        if risky:
            return BlockDecision(
                message=REASON_TOXIC,
                code="guardian_toxic",
                strategy_name=self.name,
                detail=",".join(risky),
            )
        failed = [r.criterion_id for r in results if r.error]
        if failed and self.client.fail_closed:
            return BlockDecision(
                message=GUARDIAN_UNAVAILABLE_MESSAGE,
                code="guardian_unavailable",
                strategy_name=self.name,
                detail=",".join(failed),
                status=503,
            )
        return None
