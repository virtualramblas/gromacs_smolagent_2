"""
Mock LLM for SmolAgent 1.26.0 integration tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MockMessage:
    """Minimal message object SmolAgent's CodeAgent can consume."""
    content: str
    tool_calls: None = None


def _detect_code_block_tags() -> tuple[str, str]:
    """
    Detect the code block open/close tags expected by the installed
    CodeAgent version. Falls back to standard markdown if not found.

    Returns:
        (open_tag, close_tag) e.g. ("```py", "```")
    """
    try:
        from smolagents import CodeAgent
        tags = getattr(CodeAgent, "code_block_tags", None)
        if isinstance(tags, (list, tuple)) and len(tags) == 2:
            return str(tags[0]), str(tags[1])
        if isinstance(tags, str):
            return tags, "```"
    except Exception:
        pass
    # Default: standard markdown python block
    return "```py", "```"


# Detect once at module load time
_OPEN_TAG, _CLOSE_TAG = _detect_code_block_tags()


def _wrap_in_code_block(code: str) -> str:
    """
    Wrap a code string in the code block tags expected by CodeAgent.
    Does not double-wrap if tags are already present.
    """
    if _OPEN_TAG in code or "```" in code:
        return code
    return f"{_OPEN_TAG}\n{code}\n{_CLOSE_TAG}"


class ScriptedLLM:
    """
    A mock LLM that returns pre-scripted responses in order.
    Responses are wrapped in the correct code block tags for
    the installed SmolAgent version.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._index     = 0
        self.call_log: list[str] = []

    def __call__(self, messages: list, **kwargs) -> MockMessage:
        if self._index >= len(self._responses):
            content = _wrap_in_code_block(
                'final_answer("No more scripted responses.")'
            )
        else:
            raw     = self._responses[self._index]
            content = _wrap_in_code_block(raw)
            self._index += 1

        self.call_log.append(content)
        return MockMessage(content=content)

    @property
    def responses_exhausted(self) -> bool:
        return self._index >= len(self._responses)

    def reset(self) -> None:
        self._index = 0
        self.call_log.clear()


class CapturingLLM:
    """
    A mock LLM that captures all messages sent to it.
    """

    def __init__(self, default_response: str = 'final_answer("done")'):
        self.default_response   = default_response
        self.received_messages: list[list] = []

    def __call__(self, messages: list, **kwargs) -> MockMessage:
        self.received_messages.append(messages)
        return MockMessage(
            content=_wrap_in_code_block(self.default_response)
        )

    @property
    def last_messages(self) -> list:
        return self.received_messages[-1] if self.received_messages else []

    @property
    def call_count(self) -> int:
        return len(self.received_messages)