"""
Mock LLM for SmolAgent integration tests.

Design:
    ScriptedLLM takes a list of responses. Each call to __call__()
    returns the next response in the script. This lets us test
    deterministic agent behaviour without a real LLM.

    Responses are either:
        - A code string the CodeAgent will execute
        - A final_answer() call to terminate the agent loop

SmolAgent's CodeAgent expects the model to return an object with:
    - .content: str  (the raw model output)

The CodeAgent parses code blocks from .content and executes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MockMessage:
    """Minimal message object SmolAgent's CodeAgent can consume."""
    content: str
    tool_calls: None = None


class ScriptedLLM:
    """
    A mock LLM that returns pre-scripted responses in order.

    Usage:
        llm = ScriptedLLM([
            'result = pdb2gmx(pdb_file="eiwit.pdb")\nprint(result)',
            'final_answer("Pipeline complete")',
        ])
        agent = CodeAgent(tools=[...], model=llm)
        agent.run("Run the pipeline")
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._index     = 0
        self.call_log: list[str] = []   # records all calls for assertion

    def __call__(self, messages: list, **kwargs) -> MockMessage:
        if self._index >= len(self._responses):
            # Safety fallback — return final_answer to stop infinite loop
            content = 'final_answer("No more scripted responses.")'
        else:
            content = self._responses[self._index]
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
    A mock LLM that captures all messages sent to it and
    returns a configurable response. Used to inspect what
    the agent sends to the LLM (e.g. tool outputs, state).
    """

    def __init__(self, default_response: str = 'final_answer("done")'):
        self.default_response  = default_response
        self.received_messages: list[list] = []

    def __call__(self, messages: list, **kwargs) -> MockMessage:
        self.received_messages.append(messages)
        return MockMessage(content=self.default_response)

    @property
    def last_messages(self) -> list:
        return self.received_messages[-1] if self.received_messages else []

    @property
    def call_count(self) -> int:
        return len(self.received_messages)