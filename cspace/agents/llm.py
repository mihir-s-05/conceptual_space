"""Chat-model clients for research workers.

OpenAIChat reads OPENAI_API_KEY from the environment (never from files or
arguments). Every call returns token usage so the campaign can charge it to
the inference ledger. ScriptedLLM replays canned responses for offline tests.
"""

import json
import time
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text from the model


@dataclass
class Reply:
    content: str | None
    tool_calls: list
    usage: dict = field(default_factory=dict)
    raw_message: dict = field(default_factory=dict)  # assistant message to append to history


class LLMError(RuntimeError):
    pass


class OpenAIChat:
    def __init__(self, model: str, reasoning_effort: str | None = None, max_completion_tokens: int = 32000,
                 max_retries: int = 4, timeout_s: float = 600.0):
        from openai import OpenAI

        self.client = OpenAI(max_retries=max_retries, timeout=timeout_s)  # key from OPENAI_API_KEY
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens

    def chat(self, messages, tools=None, tool_choice="required") -> Reply:
        kwargs = dict(model=self.model, messages=messages, max_completion_tokens=self.max_completion_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
            kwargs["parallel_tool_calls"] = False
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:  # surface API failures as a single type
            raise LLMError(f"{type(e).__name__}: {e}") from e
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in (msg.tool_calls or [])]
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cdetails = getattr(u, "completion_tokens_details", None)
        usage = dict(
            model=resp.model,
            prompt_tokens=u.prompt_tokens,
            completion_tokens=u.completion_tokens,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(cdetails, "reasoning_tokens", 0) or 0,
            seconds=time.time() - t0,
            finish_reason=resp.choices[0].finish_reason,
        )
        raw = {"role": "assistant", "content": msg.content}
        if calls:
            raw["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
        return Reply(msg.content, calls, usage, raw)


class ScriptedLLM:
    """Offline stand-in. `script` is a callable (messages, tools) -> (tool_name, args_dict)
    or a list of (tool_name, args_dict) consumed in order."""

    def __init__(self, script, model="scripted"):
        self.script = script
        self.model = model
        self.calls = 0

    def chat(self, messages, tools=None, tool_choice="required") -> Reply:
        if callable(self.script):
            name, args = self.script(messages, tools)
        else:
            name, args = self.script[self.calls]
        self.calls += 1
        call = ToolCall(f"call_{self.calls}", name, json.dumps(args))
        raw = {"role": "assistant", "content": None,
               "tool_calls": [{"id": call.id, "type": "function",
                               "function": {"name": name, "arguments": call.arguments}}]}
        usage = dict(model=self.model, prompt_tokens=sum(len(str(m.get("content") or "")) for m in messages) // 4,
                     completion_tokens=len(call.arguments) // 4, cached_tokens=0, reasoning_tokens=0,
                     seconds=0.0, finish_reason="tool_calls")
        return Reply(None, [call], usage, raw)
