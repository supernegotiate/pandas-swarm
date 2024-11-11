# Standard library imports
import base64  # For base64 encoding/decoding
import copy
import inspect
import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd  # For handling pandas DataFrames
from pydantic import BaseModel

# Package/library imports
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

__CTX_VARS_NAME__ = "context_variables"


def debug_print(debug: bool, *args: Any) -> None:
    if debug:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = " ".join(map(str, args))
        print(f"[{timestamp}] {message}")


def serialize_context_value(value: Any) -> Any:
    """Serializes context values, handling DataFrames and base64 content."""
    if isinstance(value, pd.DataFrame):
        return value.to_json(orient="split")
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    return value


def deserialize_context_value(value: Any, target_type: Type) -> Any:
    """Deserializes context values, converting from JSON/base64 to the target type."""
    if target_type == pd.DataFrame and isinstance(value, str):
        return pd.read_json(value, orient="split")
    if target_type is bytes and isinstance(value, str):
        return base64.b64decode(value.encode("utf-8"))
    return value


def merge_fields(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif isinstance(value, dict):
            target.setdefault(key, {})
            merge_fields(target[key], value)
        else:
            target[key] = value


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)
    tool_calls = delta.get("tool_calls")
    if tool_calls:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func: Callable) -> dict:
    """Converts a Python function into a JSON-serializable dictionary."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        pd.DataFrame: "object",
        bytes: "string",
        type(None): "null",
        Any: "any",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {e}")

    parameters = {}
    required = []
    for param in signature.parameters.values():
        annotation = param.annotation if param.annotation != inspect._empty else Any
        param_type = type_map.get(annotation, "string")
        parameters[param.name] = {"type": param_type}
        if param.default == inspect._empty:
            required.append(param.name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


AgentFunction = Callable[..., Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: Union[str, Callable[[dict], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: Optional[str] = None
    parallel_tool_calls: bool = True


class Response(BaseModel):
    messages: List[Dict[str, Any]] = []
    agent: Optional[Agent] = None
    context_variables: Dict[str, Any] = {}


class Result(BaseModel):
    """Encapsulates the possible return values for an agent function."""

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: Dict[str, Any] = {}


class Swarm:
    def __init__(
        self,
        client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client = client or OpenAI(api_key=api_key, base_url=base_url)

    def get_chat_completion(
        self,
        agent: Agent,
        history: List[Dict[str, Any]],
        context_variables: Dict[str, Any],
        model_override: Optional[str],
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = {
            k: serialize_context_value(v) for k, v in context_variables.items()
        }
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params.get("required", []):
                params["required"].remove(__CTX_VARS_NAME__)

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        return self.client.chat.completions.create(**create_params)

    def handle_function_result(self, result: Any, debug: bool) -> Result:
        if isinstance(result, Result):
            return result
        if isinstance(result, Agent):
            return Result(value=json.dumps({"assistant": result.name}), agent=result)
        try:
            return Result(value=str(result))
        except Exception as e:
            error_message = (
                f"Failed to cast response to string: {result}. "
                f"Make sure agent functions return a string or Result object. Error: {e}"
            )
            debug_print(debug, error_message)
            raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: Dict[str, Any],
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response()

        for tool_call in tool_calls:
            name = tool_call.function.name
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(debug, f"Processing tool call: {name} with arguments {args}")
            func = function_map[name]
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = func(**args)
            result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List[Dict[str, Any]],
        context_variables: Dict[str, Any] = {},
        model_override: Optional[str] = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:
            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {"function": {"arguments": "", "name": ""}, "id": "", "type": ""}
                ),
            }

            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta.get("role") == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(message.get("tool_calls", {}).values()) or None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            tool_calls = [
                ChatCompletionMessageToolCall(
                    id=tool_call["id"],
                    function=Function(
                        arguments=tool_call["function"]["arguments"],
                        name=tool_call["function"]["name"],
                    ),
                    type=tool_call["type"],
                )
                for tool_call in message["tool_calls"]
            ]

            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List[Dict[str, Any]],
        context_variables: Dict[str, Any] = {},
        model_override: Optional[str] = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(message.model_dump())

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
