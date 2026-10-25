"""OpenAI tool calling, read once for every upstream that translates it.

Crucible stores no tools and keeps nothing between rounds: the caller sends its tools
and the whole conversation each round, runs the tools itself and drives the loop. These
readers check OpenAI's form; each upstream in `upstreams` writes its own form from what
they read, and refuses by name (through the `refuse` it passes) what it cannot carry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .errors import ApiError

Refuse = Callable[[list[str], str], ApiError]

TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# A function declared with no `parameters` takes none, in OpenAI's words; an upstream
# that needs a schema is sent this one, which says exactly that.
NO_PARAMETERS: dict[str, Any] = {"type": "object", "properties": {}}


@dataclass(frozen=True)
class FunctionTool:

    name: str
    description: str | None
    parameters: dict[str, Any] | None
    strict: bool | None


@dataclass(frozen=True)
class ToolChoice:

    # "auto", "none", "required" or "named"
    kind: str
    name: str | None = None

    @property
    def wire(self) -> str:
        if self.kind == "named":
            return f"tool_choice function {self.name!r}"
        return f"tool_choice {self.kind!r}"


@dataclass(frozen=True)
class ToolCall:

    id: str
    name: str
    arguments: dict[str, Any]


def _invalid(field: str, message: str) -> ApiError:
    return ApiError(400, "invalid_request", message, {"field": field})


def read_tools(body: dict[str, Any], refuse: Refuse) -> list[FunctionTool]:
    tools = body.get("tools")
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise _invalid("tools", "tools must be a list of function tools")
    read: list[FunctionTool] = []
    for number, tool in enumerate(tools):
        at = f"tools[{number}]"
        if not isinstance(tool, dict):
            raise _invalid(at, f"{at} must be an object")
        if tool.get("type") != "function":
            raise refuse(
                [f"{at}.type"],
                f"a tool of type {tool.get('type')!r} has no translation here; "
                "function tools cross",
            )
        extra = sorted(set(tool) - {"type", "function"})
        if extra:
            raise refuse([f"{at}.{key}" for key in extra],
                         "a tool crosses as its type and its function")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise _invalid(f"{at}.function", f"{at}.function must be an object")
        extra = sorted(set(function) - {"name", "description", "parameters", "strict"})
        if extra:
            raise refuse(
                [f"{at}.function.{key}" for key in extra],
                "a function crosses as its name, description, parameters and strict",
            )
        name = function.get("name")
        if not isinstance(name, str) or not TOOL_NAME.match(name):
            raise _invalid(
                f"{at}.function.name",
                f"{at}.function.name must be 1-64 letters, digits, `_` or `-`, got "
                f"{name!r}",
            )
        if any(seen.name == name for seen in read):
            raise _invalid(f"{at}.function.name",
                           f"two tools are named {name!r}; a call could not say which")
        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise _invalid(f"{at}.function.description",
                           f"{at}.function.description must be a string")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise _invalid(f"{at}.function.parameters",
                           f"{at}.function.parameters must be a JSON schema object")
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise _invalid(f"{at}.function.strict",
                           f"{at}.function.strict must be a boolean")
        read.append(FunctionTool(name, description, parameters, strict))
    return read


def read_tool_choice(body: dict[str, Any], tools: list[FunctionTool]) -> ToolChoice | None:
    choice = body.get("tool_choice")
    if choice is None:
        return None
    if isinstance(choice, str):
        if choice not in ("auto", "none", "required"):
            raise _invalid(
                "tool_choice",
                "tool_choice must be auto, none, required or "
                f"{{type: function, function: {{name}}}}, got {choice!r}",
            )
        read = ToolChoice(choice)
    else:
        function = choice.get("function") if isinstance(choice, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if (
            not isinstance(choice, dict)
            or choice.get("type") != "function"
            or set(choice) != {"type", "function"}
            or not isinstance(function, dict)
            or set(function) != {"name"}
            or not isinstance(name, str)
        ):
            raise _invalid("tool_choice", "a named tool_choice is exactly "
                           "{type: function, function: {name}}")
        if not any(tool.name == name for tool in tools):
            raise _invalid("tool_choice.function.name",
                           f"tool_choice names {name!r}, which is not one of the tools")
        read = ToolChoice("named", name)
    if not tools and read.kind == "required":
        raise _invalid("tool_choice",
                       "tool_choice 'required' needs tools to call, and the request "
                       "sends none")
    return read


def _tool_call(at: str, call: Any, refuse: Refuse) -> ToolCall:
    if not isinstance(call, dict):
        raise _invalid(at, f"{at} must be an object")
    extra = sorted(set(call) - {"id", "type", "function"})
    if extra:
        raise refuse([f"{at}.{key}" for key in extra],
                     "a tool call crosses as its id, type and function")
    if call.get("type") != "function":
        raise _invalid(f"{at}.type",
                       f"{at}.type must be function, got {call.get('type')!r}")
    call_id = call.get("id")
    if not isinstance(call_id, str) or call_id == "":
        raise _invalid(f"{at}.id", f"{at}.id must be a non-empty string")
    function = call.get("function")
    if not isinstance(function, dict):
        raise _invalid(f"{at}.function", f"{at}.function must be an object")
    extra = sorted(set(function) - {"name", "arguments"})
    if extra:
        raise refuse([f"{at}.function.{key}" for key in extra],
                     "a called function crosses as its name and its arguments")
    name = function.get("name")
    if not isinstance(name, str) or name == "":
        raise _invalid(f"{at}.function.name", f"{at}.function.name must be a string")
    raw = function.get("arguments")
    if not isinstance(raw, str):
        raise _invalid(f"{at}.function.arguments",
                       f"{at}.function.arguments must be a JSON string")
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _invalid(f"{at}.function.arguments",
                       f"{at}.function.arguments is not JSON: {exc}") from None
    if not isinstance(arguments, dict):
        raise _invalid(
            f"{at}.function.arguments",
            f"{at}.function.arguments must be a JSON object, got "
            f"{type(arguments).__name__}",
        )
    return ToolCall(call_id, name, arguments)


def read_tool_calls(where: str, message: dict[str, Any], refuse: Refuse) -> list[ToolCall]:
    if "tool_calls" not in message:
        return []
    calls = message["tool_calls"]
    if message.get("role") != "assistant":
        raise _invalid(f"{where}.tool_calls", "only an assistant message carries tool_calls")
    if not isinstance(calls, list) or not calls:
        raise _invalid(f"{where}.tool_calls", f"{where}.tool_calls must be a non-empty list")
    read = [_tool_call(f"{where}.tool_calls[{n}]", c, refuse) for n, c in enumerate(calls)]
    ids = [call.id for call in read]
    if len(set(ids)) != len(ids):
        raise _invalid(f"{where}.tool_calls", f"{where}.tool_calls repeats an id")
    return read


def read_tool_result(
    where: str, message: dict[str, Any], calls: dict[str, ToolCall], refuse: Refuse
) -> tuple[ToolCall, str | list[str]]:
    """A `tool` message: the call it answers, which an earlier assistant message must
    have made, and its content, as a string or as the texts of its parts."""
    extra = sorted(set(message) - {"role", "content", "tool_call_id"})
    if extra:
        raise refuse([f"{where}.{key}" for key in extra],
                     "a tool result crosses as its tool_call_id and its content")
    call_id = message.get("tool_call_id")
    if not isinstance(call_id, str):
        raise _invalid(f"{where}.tool_call_id", f"{where}.tool_call_id must be a string")
    call = calls.get(call_id)
    if call is None:
        raise _invalid(
            f"{where}.tool_call_id",
            f"{where}.tool_call_id {call_id!r} answers no tool call an earlier "
            "assistant message made",
        )
    content = message.get("content")
    if isinstance(content, str):
        return call, content
    if not isinstance(content, list):
        raise _invalid(f"{where}.content",
                       f"{where}.content must be a string or a list of text parts")
    texts: list[str] = []
    for number, part in enumerate(content):
        at = f"{where}.content[{number}]"
        if (not isinstance(part, dict) or part.get("type") != "text"
                or not isinstance(part.get("text"), str)):
            raise refuse([at], "a tool result crosses as text")
        extra = sorted(set(part) - {"type", "text"})
        if extra:
            raise refuse([f"{at}.{key}" for key in extra],
                         "a text part crosses as its text alone")
        texts.append(part["text"])
    return call, texts


def openai_tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


__all__ = [
    "NO_PARAMETERS",
    "FunctionTool",
    "Refuse",
    "ToolCall",
    "ToolChoice",
    "openai_tool_call",
    "read_tool_calls",
    "read_tool_choice",
    "read_tool_result",
    "read_tools",
]
