"""OpenClaw Agent — Function-calling agent for MCP tool execution.

A proper agent framework that uses the OpenAI function_calling protocol
to interact with MCP tools. Designed for:

1. High-quality tool-calling behavior (native function_calling, not JSON-in-prompt)
2. Direct SFT-ready trajectory collection (standard messages format)
3. Pluggable into MCP-Bench evaluation pipeline

The key difference from MCP-Bench's built-in TaskExecutor:
- TaskExecutor: puts tool schemas in the prompt, asks LLM to output JSON
- OpenClaw: uses native function_calling protocol → model returns tool_calls objects

This means the trajectories are directly usable for training with frameworks
like LLaMA-Factory, Axolotl, or any system that accepts OpenAI-format conversations.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def mcp_tools_to_openai_tools(mcp_tools: Dict[str, Any]) -> tuple:
    """Convert MCP tool definitions to OpenAI function-calling format.

    MCP uses "ServerName:tool_name" as keys and has input_schema.
    OpenAI uses a flat function name and "parameters".

    Returns:
        (openai_tools, name_mapping) where name_mapping maps
        sanitized_name -> original "ServerName:tool_name"
    """
    openai_tools = []
    name_to_mcp = {}  # sanitized_name -> "ServerName:tool_name"
    mcp_to_name = {}  # "ServerName:tool_name" -> sanitized_name

    for mcp_key, info in mcp_tools.items():
        # OpenAI function names: ^[a-zA-Z0-9_-]+$ , max 64 chars
        sanitized = mcp_key.replace(":", "__").replace(" ", "_")
        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        name_to_mcp[sanitized] = mcp_key
        mcp_to_name[mcp_key] = sanitized

        server = info.get("server", "")
        desc = info.get("description", "")
        full_desc = f"[{server}] {desc}" if server else desc

        tool_def = {
            "type": "function",
            "function": {
                "name": sanitized,
                "description": full_desc[:1024],
                "parameters": info.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        openai_tools.append(tool_def)

    return openai_tools, name_to_mcp, mcp_to_name


class OpenClawAgent:
    """Function-calling agent that produces SFT-ready trajectories.

    Architecture:
        User task
          → System prompt + tools definition
          → LLM returns tool_calls (native function calling)
          → Execute tools via call_tool bridge
          → Feed results back as tool messages
          → Repeat until LLM returns final text response
          → Save complete message history as trajectory

    Args:
        api_key: API key for the LLM provider
        base_url: Base URL for the API endpoint
        model: Model name (e.g., "kimi-k2.5")
        temperature: Temperature for generation
        max_rounds: Maximum number of agent rounds
        max_tool_calls_per_round: Maximum tool calls allowed in a single round
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-k2.5",
        temperature: float = 1.0,
        max_rounds: int = 15,
        max_tool_calls_per_round: int = 20,
    ):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_rounds = max_rounds
        self.max_tool_calls_per_round = max_tool_calls_per_round

        # Rate limiting
        self._last_call_time = 0.0
        self._min_interval = 3.5  # ~17 RPM to stay under RPM 20 limit

    async def _rate_limited_completion(self, **kwargs):
        """Make an API call with rate limiting."""
        now = time.monotonic()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug(f"Rate limit: waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        self._last_call_time = time.monotonic()
        return await self.client.chat.completions.create(**kwargs)

    async def run(
        self,
        task: str,
        mcp_tools: Dict[str, Any],
        call_tool: Callable,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a task using function calling and return trajectory + results.

        Args:
            task: Natural language task description
            mcp_tools: MCP tool definitions (from server_manager.all_tools)
            call_tool: Async function: call_tool(mcp_tool_name, params) -> dict
            system_prompt: Optional custom system prompt

        Returns:
            Dictionary containing:
            - "messages": Complete message history (SFT-ready)
            - "solution": Final text response from the agent
            - "execution_results": Tool call records in MCP-Bench format
            - "total_rounds": Number of rounds executed
            - "reasoning_traces": List of reasoning_content from thinking model
            - "token_usage": Token consumption stats
        """
        openai_tools, name_to_mcp, mcp_to_name = mcp_tools_to_openai_tools(mcp_tools)

        if system_prompt is None:
            system_prompt = (
                "You are a capable AI assistant with access to various tools. "
                "Use the tools to complete the user's task thoroughly and accurately. "
                "Call tools when needed, analyze the results, and provide a comprehensive final answer. "
                "When multiple independent tool calls are needed, request them all at once for efficiency."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        execution_results = []
        reasoning_traces = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        round_num = 0
        solution = ""

        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"=== OpenClaw Round {round_num}/{self.max_rounds} ===")

            try:
                response = await self._rate_limited_completion(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                )
            except Exception as e:
                logger.error(f"LLM call failed in round {round_num}: {e}")
                if "429" in str(e) or "rate" in str(e).lower():
                    logger.info("Rate limited, waiting 30s before retry...")
                    await asyncio.sleep(30)
                    try:
                        response = await self._rate_limited_completion(
                            model=self.model,
                            temperature=self.temperature,
                            messages=messages,
                            tools=openai_tools if openai_tools else None,
                            tool_choice="auto" if openai_tools else None,
                        )
                    except Exception as e2:
                        logger.error(f"Retry also failed: {e2}")
                        break
                else:
                    break

            # Track tokens
            if response.usage:
                total_prompt_tokens += response.usage.prompt_tokens or 0
                total_completion_tokens += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Capture reasoning from thinking models
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                reasoning_traces.append({
                    "round": round_num,
                    "reasoning": reasoning,
                })
                logger.info(f"Reasoning ({len(reasoning)} chars): {reasoning[:150]}...")

            # Build the assistant message for the conversation history
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            # Thinking models (e.g., kimi-k2.5) require reasoning_content
            # in assistant messages for multi-turn conversations
            if reasoning is not None:
                assistant_msg["reasoning_content"] = reasoning
            messages.append(assistant_msg)

            # If no tool calls, this is the final response
            if not msg.tool_calls or finish_reason == "stop":
                solution = msg.content or ""
                logger.info(f"Agent finished after {round_num} rounds. Solution length: {len(solution)}")
                break

            # Execute tool calls
            logger.info(f"Round {round_num}: {len(msg.tool_calls)} tool calls")

            tool_tasks = []
            for tc in msg.tool_calls[:self.max_tool_calls_per_round]:
                sanitized_name = tc.function.name
                mcp_name = name_to_mcp.get(sanitized_name, sanitized_name)

                try:
                    params = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    params = {}
                    logger.warning(f"Failed to parse arguments for {sanitized_name}: {tc.function.arguments}")

                tool_tasks.append((tc.id, sanitized_name, mcp_name, params))

            # Execute all tool calls (concurrently where possible)
            tool_results = await self._execute_tool_calls(
                tool_tasks, call_tool, round_num, execution_results
            )

            # Add tool results as messages
            for tool_call_id, result_content in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result_content,
                })

        return {
            "messages": messages,
            "solution": solution,
            "execution_results": execution_results,
            "total_rounds": round_num,
            "reasoning_traces": reasoning_traces,
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
        }

    async def _execute_tool_calls(
        self,
        tool_tasks: list,
        call_tool: Callable,
        round_num: int,
        execution_results: list,
    ) -> list:
        """Execute tool calls and return results for message history."""

        async def _exec_one(tool_call_id, sanitized_name, mcp_name, params):
            start = time.time()
            try:
                result = await call_tool(mcp_name, params)
                latency = time.time() - start
                success = result.get("success", False)
                result_text = result.get("result", "") or ""
                error_text = result.get("error", "") or ""

                server = mcp_name.split(":")[0] if ":" in mcp_name else "unknown"
                status = "OK" if success else "FAIL"
                logger.info(f"  [{status}] {mcp_name} ({latency:.1f}s)")

                execution_results.append({
                    "tool": mcp_name,
                    "server": server,
                    "parameters": params,
                    "round_num": round_num,
                    "success": success,
                    **({"result": result_text} if success else {"error": error_text}),
                })

                content = result_text if success else f"Error: {error_text}"
                return (tool_call_id, content[:30000])

            except Exception as e:
                logger.error(f"  [ERROR] {mcp_name}: {e}")
                server = mcp_name.split(":")[0] if ":" in mcp_name else "unknown"
                execution_results.append({
                    "tool": mcp_name,
                    "server": server,
                    "parameters": params,
                    "round_num": round_num,
                    "success": False,
                    "error": str(e),
                })
                return (tool_call_id, f"Error: {e}")

        results = await asyncio.gather(
            *[_exec_one(*t) for t in tool_tasks],
            return_exceptions=False,
        )
        return results


def save_trajectory(
    trajectory: Dict[str, Any],
    output_dir: str = "trajectories",
    task_id: str = "",
) -> str:
    """Save trajectory in SFT-ready format.

    The saved format is directly compatible with training frameworks:
    {
        "task_id": "...",
        "model": "kimi-k2.5",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "...", "tool_calls": [...]},
            {"role": "tool", "tool_call_id": "...", "content": "..."},
            ...
            {"role": "assistant", "content": "final answer"}
        ],
        "reasoning_traces": [...],  // thinking model's chain-of-thought
        "metadata": {...}
    }
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"traj_{task_id}_{timestamp}.json" if task_id else f"traj_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)

    logger.info(f"Trajectory saved: {filepath}")
    return filepath
