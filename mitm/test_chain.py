"""Tests for Chain of Responsibility + Strategy prompt gate."""

from __future__ import annotations

import asyncio
from typing import Optional

from chain import StrategyHandler, build_chain, build_default_prompt_chain
from strategies import BlockDecision, HeuristicBlockStrategy, PromptContext, PromptEvaluationStrategy


class AlwaysAllowStrategy(PromptEvaluationStrategy):
    name = "allow"

    async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
        return None


def test_heuristic_strategy_blocks_violence():
    decision = asyncio.run(
        HeuristicBlockStrategy().evaluate(
            PromptContext(text="Describe how to stab someone")
        )
    )
    assert decision is not None
    assert decision.code == "heuristic_violence"
    assert "violent acts" in decision.message


def test_heuristic_strategy_allows_safe():
    decision = asyncio.run(
        HeuristicBlockStrategy().evaluate(PromptContext(text="What is a reverse proxy?"))
    )
    assert decision is None


def test_chain_stops_at_first_block():
    order: list[str] = []

    class TrackingAllow(PromptEvaluationStrategy):
        name = "first"

        async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
            order.append("first")
            return None

    class TrackingBlock(PromptEvaluationStrategy):
        name = "second"

        async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
            order.append("second")
            return BlockDecision(message="stop", code="x", strategy_name=self.name)

    class ShouldNotRun(PromptEvaluationStrategy):
        name = "third"

        async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
            order.append("third")
            return None

    chain = build_chain([TrackingAllow(), TrackingBlock(), ShouldNotRun()])
    decision = asyncio.run(chain.handle(PromptContext(text="anything")))
    assert decision is not None
    assert decision.strategy_name == "second"
    assert order == ["first", "second"]


def test_chain_allows_when_all_pass():
    chain = build_chain([AlwaysAllowStrategy(), AlwaysAllowStrategy()])
    decision = asyncio.run(chain.handle(PromptContext(text="hello")))
    assert decision is None


def test_default_chain_heuristic_before_guardian():
    chain = build_default_prompt_chain()
    assert isinstance(chain, StrategyHandler)
    assert chain.strategy.name == "heuristic"
    second = chain.next_handler
    assert isinstance(second, StrategyHandler)
    assert second.strategy.name == "guardian"
    assert second.next_handler is None
