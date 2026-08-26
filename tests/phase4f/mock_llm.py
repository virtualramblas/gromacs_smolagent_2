"""
Mock LLM conforming to the smolagents 1.26.0 Model interface.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Detect code block tags
# ---------------------------------------------------------------------------

def _detect_code_block_tags() -> tuple[str, str]:
    """
    Detect the code block open/close tags expected by CodeAgent.
    CodeAgent has no code_block_tags class attribute in 1.26.0,
    so we fall back to the standard markdown python block.
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
    # Default confirmed by smolagents 1.26.0 source
    return "```py", "```"


_OPEN_TAG, _CLOSE_TAG = _detect_code_block_tags()


def _wrap_in_code_block(code: str) -> str:
    """Wrap code in the tags CodeAgent expects for parsing."""
    if _OPEN_TAG in code or "```" in code:
        return code
    return f"{_OPEN_TAG}\n{code}\n{_CLOSE_TAG}"


# ---------------------------------------------------------------------------
# Minimal ChatMessage-compatible return type
# ---------------------------------------------------------------------------

@dataclass
class MockTokenUsage:
    """Minimal token usage object — CodeAgent reads this for monitoring."""
    input_tokens:  int = 10
    output_tokens: int = 10


@dataclass
class MockChatMessage:
    """
    Return type of generate().
    Must have .content, .tool_calls, and .token_usage.
    CodeAgent accesses all three after every model call.
    """
    content:     str
    tool_calls:  None          = None
    token_usage: MockTokenUsage = field(
        default_factory=MockTokenUsage
    )
    role: str = "assistant"


# ---------------------------------------------------------------------------
# ScriptedLLM
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """
    A mock LLM that returns pre-scripted responses in order.

    Implements the smolagents 1.26.0 model interface:
        generate(messages, stop_sequences=None, grammar=None, **kwargs)
            → MockChatMessage

    Does NOT subclass smolagents.Model — subclassing caused __init__
    signature conflicts. Plain duck-typing is sufficient because
    CodeAgent only calls .generate() and reads .last_input_token_count
    / .last_output_token_count on the model object.
    """

    def __init__(self, responses: list[str]):
        self._responses  = list(responses)
        self._index      = 0
        self.call_log:   list[str] = []

        # Attributes CodeAgent reads for token monitoring
        self.last_input_token_count  = 0
        self.last_output_token_count = 0

    def generate(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        grammar: Any = None,
        **kwargs,
    ) -> MockChatMessage:
        """
        Called by CodeAgent._step_stream() on every agent step.
        Returns the next scripted response wrapped in code block tags.
        """
        if self._index >= len(self._responses):
            raw = 'final_answer("No more scripted responses.")'
        else:
            raw          = self._responses[self._index]
            self._index += 1

        content = _wrap_in_code_block(raw)
        self.call_log.append(content)

        # Update token counts
        self.last_input_token_count  = sum(
            len(str(m)) for m in messages
        ) // 4
        self.last_output_token_count = len(content) // 4

        return MockChatMessage(content=content)

    # ------------------------------------------------------------------
    # Compatibility shim — some code paths may still call __call__
    # ------------------------------------------------------------------
    def __call__(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        grammar: Any = None,
        **kwargs,
    ) -> MockChatMessage:
        return self.generate(
            messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            **kwargs,
        )

    @property
    def responses_exhausted(self) -> bool:
        return self._index >= len(self._responses)

    def reset(self) -> None:
        self._index = 0
        self.call_log.clear()


# ---------------------------------------------------------------------------
# CapturingLLM
# ---------------------------------------------------------------------------

class CapturingLLM:
    """
    A mock LLM that captures all messages sent to it.
    Implements the same generate() interface as ScriptedLLM.
    """

    def __init__(self, default_response: str = 'final_answer("done")'):
        self.default_response   = default_response
        self.received_messages: list[list] = []
        self.last_input_token_count  = 0
        self.last_output_token_count = 0

    def generate(
        self,
        messages: list,
        stop_sequences: list[str] | None = None,
        grammar: Any = None,
        **kwargs,
    ) -> MockChatMessage:
        self.received_messages.append(messages)
        content = _wrap_in_code_block(self.default_response)
        self.last_input_token_count  = 10
        self.last_output_token_count = len(content) // 4
        return MockChatMessage(content=content)

    def __call__(self, messages: list, **kwargs) -> MockChatMessage:
        return self.generate(messages, **kwargs)

    @property
    def last_messages(self) -> list:
        return self.received_messages[-1] if self.received_messages else []

    @property
    def call_count(self) -> int:
        return len(self.received_messages)