"""OpenClaw Agent Executor Adapter.

Integrates an external OpenClaw (or similar) agent framework into MCP-Bench's
evaluation pipeline. This adapter:

1. Receives tasks and available MCP tools from MCP-Bench
2. Delegates execution to the OpenClaw agent
3. Collects execution trajectories (for training data)
4. Converts results to MCP-Bench's evaluation format

Trajectory Logging:
    Every execution is logged as a complete trajectory suitable for SFT training.
    Each trajectory captures the full multi-turn interaction including:
    - System context (available tools with schemas)
    - Each planning step (reasoning + tool selection)
    - Each tool call and its result
    - Final solution synthesis

Usage:
    # 1. Implement your OpenClaw agent by subclassing or configuring OpenClawExecutor
    # 2. Use it with BenchmarkRunner:

    runner = BenchmarkRunner(
        executor_factory=lambda llm, sm, cs: OpenClawExecutor(
            server_manager=sm,
            openclaw_agent=my_agent,
            trajectory_dir="trajectories/"
        )
    )
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Protocol, runtime_checkable

from agent.base_executor import BaseAgentExecutor
from mcp_modules.server_manager_persistent import PersistentMultiServerManager as MultiServerManager

logger = logging.getLogger(__name__)


@runtime_checkable
class OpenClawAgent(Protocol):
    """Protocol that your OpenClaw agent must implement.

    This is the minimal interface MCP-Bench needs from your agent.
    Your agent receives a task and a tool-calling function, and returns
    its execution trace.
    """

    async def run(
        self,
        task: str,
        tools: Dict[str, Any],
        call_tool: Any
    ) -> "AgentResult":
        """Run the agent on a task.

        Args:
            task: Natural language task description.
            tools: Available tools dict. Keys are "ServerName:tool_name",
                values contain "name", "server", "description", "input_schema".
            call_tool: An async callable with signature:
                async def call_tool(tool_name: str, parameters: dict) -> dict
                Returns {"success": bool, "result": str} or {"success": bool, "error": str}

        Returns:
            AgentResult with the execution trace.
        """
        ...


class AgentResult:
    """Container for OpenClaw agent execution results.

    Your agent should return an instance of this class (or any object with
    these attributes) from its `run()` method.
    """

    def __init__(
        self,
        solution: str,
        rounds: List["RoundTrace"],
        total_output_tokens: int = 0,
        total_prompt_tokens: int = 0,
    ):
        self.solution = solution
        self.rounds = rounds
        self.total_output_tokens = total_output_tokens
        self.total_prompt_tokens = total_prompt_tokens


class RoundTrace:
    """Trace of a single execution round.

    Each round may contain planning reasoning and one or more tool calls.
    This is the unit of trajectory data captured for training.
    """

    def __init__(
        self,
        round_num: int,
        reasoning: str = "",
        tool_calls: Optional[List["ToolCallTrace"]] = None,
        should_continue: bool = True,
    ):
        self.round_num = round_num
        self.reasoning = reasoning
        self.tool_calls = tool_calls or []
        self.should_continue = should_continue


class ToolCallTrace:
    """Trace of a single tool call within a round."""

    def __init__(
        self,
        tool_name: str,
        server_name: str,
        parameters: Dict[str, Any],
        result: Optional[str] = None,
        error: Optional[str] = None,
        success: bool = True,
        latency_ms: float = 0,
    ):
        self.tool_name = tool_name
        self.server_name = server_name
        self.parameters = parameters
        self.result = result
        self.error = error
        self.success = success
        self.latency_ms = latency_ms


class OpenClawExecutor(BaseAgentExecutor):
    """Adapter that bridges OpenClaw agent to MCP-Bench's evaluation pipeline.

    This executor:
    - Wraps MCP-Bench's server_manager.call_tool as a simple async function
    - Passes it to the OpenClaw agent along with task and tool info
    - Converts the agent's trace into MCP-Bench's result format
    - Saves full trajectories for SFT training data

    Args:
        server_manager: MCP-Bench's server manager (provides tool calling)
        openclaw_agent: Your agent implementing the OpenClawAgent protocol
        trajectory_dir: Directory to save trajectory logs (None to disable)
        save_trajectories: Whether to save trajectories to disk
    """

    def __init__(
        self,
        server_manager: MultiServerManager,
        openclaw_agent: OpenClawAgent,
        trajectory_dir: Optional[str] = "trajectories",
        save_trajectories: bool = True,
    ):
        self.server_manager = server_manager
        self.all_tools = server_manager.all_tools
        self.agent = openclaw_agent
        self.trajectory_dir = trajectory_dir
        self.save_trajectories = save_trajectories

        if save_trajectories and trajectory_dir:
            os.makedirs(trajectory_dir, exist_ok=True)

    async def execute(self, task: str) -> Dict[str, Any]:
        """Execute task via OpenClaw agent, collect trajectory, return MCP-Bench format."""
        logger.info(f"OpenClawExecutor: Starting task execution")
        logger.info(f"Available tools: {len(self.all_tools)}")

        trajectory_log = {
            "task": task,
            "available_tools": {
                name: {
                    "name": info["name"],
                    "server": info["server"],
                    "description": info["description"],
                    "input_schema": info.get("input_schema", {}),
                }
                for name, info in self.all_tools.items()
            },
            "rounds": [],
            "start_time": datetime.now().isoformat(),
        }

        call_log: List[Dict[str, Any]] = []

        async def call_tool_bridge(tool_name: str, parameters: dict) -> dict:
            """Bridge function that wraps server_manager.call_tool for OpenClaw.

            This is the function your agent calls to invoke MCP tools.
            It handles result extraction and error handling, and logs every call.
            """
            start = time.time()
            try:
                result_obj = await self.server_manager.call_tool(tool_name, parameters)

                is_error = hasattr(result_obj, "isError") and result_obj.isError
                result_text = self._extract_text(result_obj)

                server_name = self.all_tools.get(tool_name, {}).get("server", "unknown")
                latency_ms = (time.time() - start) * 1000

                record = {
                    "tool": tool_name,
                    "server": server_name,
                    "parameters": parameters,
                    "success": not is_error,
                    "latency_ms": latency_ms,
                }
                if is_error:
                    record["error"] = result_text
                else:
                    record["result"] = result_text

                call_log.append(record)

                return {
                    "success": not is_error,
                    "result": result_text if not is_error else None,
                    "error": result_text if is_error else None,
                }

            except Exception as e:
                server_name = self.all_tools.get(tool_name, {}).get("server", "unknown")
                latency_ms = (time.time() - start) * 1000
                error_msg = str(e)

                call_log.append({
                    "tool": tool_name,
                    "server": server_name,
                    "parameters": parameters,
                    "error": error_msg,
                    "success": False,
                    "latency_ms": latency_ms,
                })

                return {"success": False, "error": error_msg, "result": None}

        agent_result = await self.agent.run(
            task=task,
            tools=self.all_tools,
            call_tool=call_tool_bridge,
        )

        mcpbench_result = self._convert_to_mcpbench_format(
            agent_result, call_log, task
        )

        trajectory_log["rounds"] = self._build_trajectory_rounds(agent_result)
        trajectory_log["solution"] = agent_result.solution
        trajectory_log["end_time"] = datetime.now().isoformat()
        trajectory_log["total_rounds"] = len(agent_result.rounds)
        trajectory_log["total_tool_calls"] = len(call_log)

        if self.save_trajectories:
            self._save_trajectory(trajectory_log, task)

        return mcpbench_result

    def _convert_to_mcpbench_format(
        self,
        agent_result: AgentResult,
        call_log: List[Dict[str, Any]],
        task: str,
    ) -> Dict[str, Any]:
        """Convert OpenClaw agent output to MCP-Bench evaluator format."""

        execution_results = []
        for i, record in enumerate(call_log):
            entry = {
                "tool": record["tool"],
                "server": record["server"],
                "parameters": record["parameters"],
                "round_num": self._find_round_for_call(agent_result, record, i),
                "success": record["success"],
            }
            if record["success"]:
                entry["result"] = record.get("result", "")
            else:
                entry["error"] = record.get("error", "Unknown error")
            execution_results.append(entry)

        accumulated = self._build_accumulated_information(execution_results)

        return {
            "solution": agent_result.solution,
            "total_rounds": len(agent_result.rounds),
            "execution_results": execution_results,
            "planning_json_compliance": 1.0,
            "accumulated_information": accumulated,
            "accumulated_information_uncompressed": accumulated,
            "available_tools": self.all_tools,
            "total_output_tokens": getattr(agent_result, "total_output_tokens", 0),
            "total_prompt_tokens": getattr(agent_result, "total_prompt_tokens", 0),
            "total_tokens": (
                getattr(agent_result, "total_output_tokens", 0)
                + getattr(agent_result, "total_prompt_tokens", 0)
            ),
        }

    def _find_round_for_call(
        self, agent_result: AgentResult, record: Dict, index: int
    ) -> int:
        """Determine which round a tool call belongs to."""
        if not hasattr(agent_result, "rounds") or not agent_result.rounds:
            return 1

        call_counter = 0
        for rnd in agent_result.rounds:
            for tc in rnd.tool_calls:
                if call_counter == index:
                    return rnd.round_num
                call_counter += 1

        return len(agent_result.rounds)

    def _build_accumulated_information(
        self, execution_results: List[Dict[str, Any]]
    ) -> str:
        """Build accumulated information string from execution results."""
        parts = []
        rounds_seen = set()

        for result in execution_results:
            round_num = result.get("round_num", 1)
            if round_num not in rounds_seen:
                parts.append(f"\n--- Summary of Round {round_num} ---")
                rounds_seen.add(round_num)

            tool = result["tool"]
            server = result.get("server", "unknown")
            params = result.get("parameters", {})
            params_str = f"{params}" if params else "{}"

            if result["success"]:
                content = result.get("result", "")
                parts.append(
                    f"Tool `{tool}` with Parameter {params_str} on {server} "
                    f"succeeded. Result: {content}\n"
                )
            else:
                error = result.get("error", "")
                parts.append(
                    f"Tool `{tool}` with Parameter {params_str} on {server} "
                    f"failed. Error: {error}\n"
                )

        return "\n".join(parts)

    def _build_trajectory_rounds(self, agent_result: AgentResult) -> List[Dict]:
        """Build trajectory round data for training logs."""
        rounds = []
        for rnd in agent_result.rounds:
            round_data = {
                "round_num": rnd.round_num,
                "reasoning": rnd.reasoning,
                "should_continue": rnd.should_continue,
                "tool_calls": [],
            }
            for tc in rnd.tool_calls:
                round_data["tool_calls"].append({
                    "tool_name": tc.tool_name,
                    "server_name": tc.server_name,
                    "parameters": tc.parameters,
                    "result": tc.result,
                    "error": tc.error,
                    "success": tc.success,
                    "latency_ms": tc.latency_ms,
                })
            rounds.append(round_data)
        return rounds

    def _save_trajectory(self, trajectory: Dict, task: str) -> None:
        """Save trajectory to disk as JSONL (one trajectory per line)."""
        if not self.trajectory_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.trajectory_dir, f"trajectory_{timestamp}.json")

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
            logger.info(f"Trajectory saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save trajectory: {e}")

    @staticmethod
    def _extract_text(result) -> str:
        """Extract text from MCP CallToolResult."""
        if hasattr(result, "content") and result.content:
            return "".join(
                item.text for item in result.content if hasattr(item, "text")
            )
        return str(result)


# =============================================================================
# Example: Minimal OpenClaw agent implementation for reference
# =============================================================================

class ExampleOpenClawAgent:
    """Minimal example showing how to implement the OpenClawAgent protocol.

    Replace this with your actual OpenClaw agent. This example demonstrates
    the expected interface and return format.
    """

    def __init__(self, llm_provider):
        """
        Args:
            llm_provider: Your LLM provider (any model with chat completion API).
        """
        self.llm = llm_provider

    async def run(
        self,
        task: str,
        tools: Dict[str, Any],
        call_tool: Any,
    ) -> AgentResult:
        """
        Your agent's main loop. This example shows a simple 1-round execution.

        In a real OpenClaw agent, this would be a multi-round planning loop
        where the LLM decides which tools to call, processes results, and
        iterates until the task is complete.
        """
        rounds = []

        # --- Round 1: Simple example - call the first available tool ---
        round_traces = []
        tool_names = list(tools.keys())

        if tool_names:
            first_tool = tool_names[0]
            result = await call_tool(first_tool, {})

            round_traces.append(ToolCallTrace(
                tool_name=first_tool,
                server_name=tools[first_tool]["server"],
                parameters={},
                result=result.get("result"),
                error=result.get("error"),
                success=result["success"],
            ))

        rounds.append(RoundTrace(
            round_num=1,
            reasoning="Example: calling first available tool",
            tool_calls=round_traces,
            should_continue=False,
        ))

        return AgentResult(
            solution=f"Example solution for: {task}",
            rounds=rounds,
        )
