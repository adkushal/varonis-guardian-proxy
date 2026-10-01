"""
mitmproxy addon: reverse-proxy OpenAI Chat Completions behind NGINX.

Prompt gating uses Chain of Responsibility + Strategy, wrapped by Decorators:
  Logging → Timing → HeuristicBlockStrategy → GuardianToxicityStrategy → allow

- Only POST /v1/chat/completions is accepted; other routes get 404
- Responses are monitored in the background (never blocked)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from mitmproxy import http

from chain import PromptGate, build_default_prompt_chain
from decorators import decorate_prompt_gate
from guardian_client import GuardianClient
from strategies import (
    BlockDecision,
    GuardianToxicityStrategy,
    HeuristicBlockStrategy,
    PromptContext,
)

logger = logging.getLogger("openai_guardian_addon")

CHAT_PATH = "/v1/chat/completions"


def _content_text(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    return parts


def _extract_prompt_text(body: dict[str, Any]) -> str:
    # Every role is screened: system/developer/tool/assistant messages are all
    # client-supplied and would otherwise be a way to smuggle content past the gate.
    parts: list[str] = []
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            parts.extend(_content_text(msg.get("content")))
    return "\n".join(p for p in parts if p).strip()


def _extract_assistant_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for choice in body.get("choices") or []:
        parts.extend(_content_text((choice.get("message") or {}).get("content")))
    return "\n".join(p for p in parts if p).strip()


def _error_response(
    flow: http.HTTPFlow, status: int, message: str, error_type: str, code: str
) -> None:
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "param": None,
        }
    }
    flow.response = http.Response.make(
        status,
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )


def _block_response(flow: http.HTTPFlow, decision: BlockDecision) -> None:
    error_type = "prompt_blocked" if decision.status == 403 else "service_unavailable"
    _error_response(flow, decision.status, decision.message, error_type, decision.code)


def _is_chat_route(flow: http.HTTPFlow) -> bool:
    path = flow.request.path.split("?", 1)[0]
    return path.rstrip("/") == CHAT_PATH


def _enforce_chat_route(flow: http.HTTPFlow) -> bool:
    """Allowlist: anything else would reach OpenAI uninspected. Sets a 404 and returns False otherwise."""
    if flow.request.method.upper() == "POST" and _is_chat_route(flow):
        return True
    logger.warning("rejected unsupported route %s %s", flow.request.method, flow.request.path)
    _error_response(
        flow,
        404,
        f"Only POST {CHAT_PATH} is supported by this proxy",
        "invalid_request_error",
        "unsupported_route",
    )
    return False


def _parse_json_object(flow: http.HTTPFlow) -> dict[str, Any] | None:
    """Return the request body as a dict, or set a 400 response and return None."""
    try:
        body = json.loads(flow.request.get_text() or "{}")
    except json.JSONDecodeError:
        _error_response(flow, 400, "Invalid JSON body", "invalid_request_error", "invalid_json")
        return None
    if not isinstance(body, dict):
        _error_response(flow, 400, "JSON body must be an object", "invalid_request_error", "invalid_json")
        return None
    return body


class OpenAIGuardianAddon:
    def __init__(self, guardian: GuardianClient | None = None) -> None:
        self.guardian = guardian or GuardianClient()
        self.prompt_gate: PromptGate = decorate_prompt_gate(
            build_default_prompt_chain(self.guardian)
        )
        self._monitor_heuristic = HeuristicBlockStrategy()
        self._monitor_guardian = GuardianToxicityStrategy(self.guardian)
        self._background_tasks: set[asyncio.Task] = set()

    def load(self, loader) -> None:  # noqa: ANN001
        logger.info(
            "OpenAIGuardianAddon loaded (criteria=%s, guardian=%s, fail_closed=%s)",
            self.guardian.criteria,
            self.guardian.base_url,
            self.guardian.fail_closed,
        )

    async def done(self) -> None:
        await self.guardian.aclose()

    async def request(self, flow: http.HTTPFlow) -> None:
        if not _enforce_chat_route(flow):
            return
        body = _parse_json_object(flow)
        if body is None:
            return

        # All roles are concatenated into one text, judged with the "user" (prompt) schema.
        ctx = PromptContext(
            text=_extract_prompt_text(body),
            role="user",
            request_id=str(uuid.uuid4())[:8],
        )
        decision = await self.prompt_gate.handle(ctx)
        if decision is not None:
            _block_response(flow, decision)

    async def response(self, flow: http.HTTPFlow) -> None:
        if not _is_chat_route(flow) or flow.response is None:
            return
        if flow.response.status_code != 200:
            return

        try:
            body = json.loads(flow.response.get_text() or "{}")
        except json.JSONDecodeError:
            logger.info("response monitor: non-JSON body (streaming responses are not monitored)")
            return

        assistant_text = _extract_assistant_text(body) if isinstance(body, dict) else ""
        if not assistant_text:
            logger.info("response monitor: empty assistant content")
            return

        # Monitor-only, so don't hold the reply back while Guardian runs.
        task = asyncio.create_task(self._monitor_response(assistant_text))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _monitor_response(self, assistant_text: str) -> None:
        monitor_ctx = PromptContext(text=assistant_text, role="assistant")
        heur = await self._monitor_heuristic.evaluate(monitor_ctx)
        guard = await self._monitor_guardian.evaluate(monitor_ctx)
        logger.info(
            "response monitor: heuristic=%s guardian=%s len=%d",
            heur.detail if heur else None,
            f"{guard.code}:{guard.detail}" if guard else None,
            len(assistant_text),
        )


addons = [OpenAIGuardianAddon()]
