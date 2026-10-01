"""Tests for Decorator pattern around the prompt gate."""

from __future__ import annotations

import asyncio
from typing import Optional

from chain import build_chain
from decorators import LoggingPromptGate, TimingPromptGate, decorate_prompt_gate
from strategies import BlockDecision, PromptContext, PromptEvaluationStrategy


class AllowStrategy(PromptEvaluationStrategy):
    name = "allow"

    async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
        return None


class BlockStrategy(PromptEvaluationStrategy):
    name = "block"

    async def evaluate(self, ctx: PromptContext) -> Optional[BlockDecision]:
        return BlockDecision(message="nope", code="x", strategy_name=self.name, detail="d")


def test_logging_decorator_records_allow_and_preserves_result():
    lines: list[str] = []
    gate = LoggingPromptGate(build_chain([AllowStrategy()]), log=lines.append)
    decision = asyncio.run(gate.handle(PromptContext(text="hi", request_id="abc")))
    assert decision is None
    assert any("gate start" in line and "abc" in line for line in lines)
    assert any("gate allow" in line for line in lines)


def test_logging_decorator_records_block():
    lines: list[str] = []
    gate = LoggingPromptGate(build_chain([BlockStrategy()]), log=lines.append)
    decision = asyncio.run(gate.handle(PromptContext(text="bad", request_id="r1")))
    assert decision is not None
    assert decision.code == "x"
    assert any("gate block" in line and "strategy=block" in line for line in lines)


def test_decorate_prompt_gate_composes_logging_outside_timing():
    lines: list[str] = []
    chain = build_chain([AllowStrategy()])
    gate = decorate_prompt_gate(chain, log=lines.append)
    assert isinstance(gate, LoggingPromptGate)
    assert isinstance(gate.wrapped, TimingPromptGate)
    assert gate.wrapped.wrapped is chain

    decision = asyncio.run(gate.handle(PromptContext(text="ok", request_id="t1")))
    assert decision is None
    assert [l.split("] ", 1)[1].split("=")[0] for l in lines] == [
        "gate start role",
        "gate elapsed_ms",
        "gate allow",
    ]


def test_timing_decorator_alone():
    lines: list[str] = []
    gate = TimingPromptGate(build_chain([AllowStrategy()]), log=lines.append)
    asyncio.run(gate.handle(PromptContext(text="ok", request_id="z")))
    assert any("elapsed_ms=" in line for line in lines)
