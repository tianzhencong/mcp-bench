#!/usr/bin/env python3
"""
Entry point for running MCP-Bench with an external OpenClaw agent.

This script demonstrates how to integrate your OpenClaw (or similar) agent
framework into MCP-Bench's evaluation pipeline. It:

1. Loads MCP-Bench tasks
2. Starts MCP servers
3. Routes tasks through your OpenClaw agent (instead of the built-in TaskExecutor)
4. Evaluates results using MCP-Bench's evaluator
5. Saves execution trajectories for SFT training

Quick Start:
    # Step 1: Implement your agent (see YourOpenClawAgent below)
    # Step 2: Run:
    python run_openclaw_benchmark.py --models o4-mini --tasks-file tasks/mcpbench_tasks_single_runner_format.json

Trajectory Output:
    Trajectories are saved to ./trajectories/ as JSON files.
    Each file contains the full execution trace suitable for training.
"""

import sys
import asyncio
import argparse
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.openclaw_executor import (
    OpenClawExecutor,
    AgentResult,
    RoundTrace,
    ToolCallTrace,
    OpenClawAgent,
)
from mcp_modules.server_manager_persistent import PersistentMultiServerManager

logger = logging.getLogger(__name__)


# =============================================================================
# YOUR AGENT IMPLEMENTATION GOES HERE
# =============================================================================

class YourOpenClawAgent:
    """Replace this with your actual OpenClaw agent.

    This is a template showing the interface your agent must implement.
    The `run()` method receives:
      - task: the natural language task
      - tools: dict of available MCP tools with schemas
      - call_tool: async function to invoke MCP tools

    Your agent should:
      1. Analyze the task and available tools
      2. Plan which tools to call (with what parameters)
      3. Call tools via `call_tool(tool_name, parameters)`
      4. Process results, decide if more rounds are needed
      5. Synthesize a final solution
      6. Return AgentResult with the full trace

    IMPORTANT for training data quality:
      - Record your agent's reasoning at each step (in RoundTrace.reasoning)
      - This reasoning is the most valuable part of the trajectory for SFT
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        # Initialize your LLM client here
        # e.g., self.client = openai.AsyncOpenAI(...)

    async def run(self, task, tools, call_tool) -> AgentResult:
        """Your agent's main execution loop.

        This example shows the structure. Replace with your actual agent logic.

        A typical loop:
            for round in range(max_rounds):
                # 1. Build prompt with task + tools + previous results
                # 2. Call your LLM to get a plan
                # 3. Parse the plan to extract tool calls
                # 4. Execute tool calls via call_tool()
                # 5. Record everything in RoundTrace
                # 6. Decide whether to continue
        """
        rounds = []
        all_results = []
        max_rounds = 5

        for round_num in range(1, max_rounds + 1):
            # ----------------------------------------------------------
            # TODO: Replace this block with your actual agent logic
            # ----------------------------------------------------------
            #
            # Your agent should:
            # 1. Build a prompt that includes:
            #    - The task description
            #    - Available tools (names, descriptions, schemas from `tools` dict)
            #    - Results from previous rounds (from `all_results`)
            #
            # 2. Call your LLM:
            #    response = await self.client.chat.completions.create(...)
            #
            # 3. Parse the response to extract:
            #    - reasoning: why the agent chose these tools
            #    - tool_calls: list of (tool_name, parameters) to execute
            #    - should_continue: whether more rounds are needed
            #
            # 4. Execute each tool call:
            #    for tool_name, params in planned_calls:
            #        result = await call_tool(tool_name, params)
            #
            # 5. Record the trace:
            #    round_trace = RoundTrace(round_num, reasoning, tool_call_traces, should_continue)
            #
            # Example placeholder (remove this and add your logic):
            reasoning = f"Round {round_num}: Analyzing task and selecting tools..."
            tool_call_traces = []

            # Example: Call a tool
            # result = await call_tool("Weather Data:get_weather", {"city": "Tokyo"})
            # tool_call_traces.append(ToolCallTrace(
            #     tool_name="Weather Data:get_weather",
            #     server_name="Weather Data",
            #     parameters={"city": "Tokyo"},
            #     result=result.get("result"),
            #     error=result.get("error"),
            #     success=result["success"],
            # ))

            should_continue = False  # Set based on your agent's decision

            rounds.append(RoundTrace(
                round_num=round_num,
                reasoning=reasoning,
                tool_calls=tool_call_traces,
                should_continue=should_continue,
            ))

            if not should_continue:
                break

        solution = "TODO: Your agent's final synthesized answer goes here"

        return AgentResult(
            solution=solution,
            rounds=rounds,
        )


def create_openclaw_executor_factory(
    agent: OpenClawAgent,
    trajectory_dir: str = "trajectories",
    save_trajectories: bool = True,
):
    """Create an executor factory function for BenchmarkRunner.

    Args:
        agent: Your OpenClaw agent instance
        trajectory_dir: Where to save trajectory logs
        save_trajectories: Whether to save trajectories

    Returns:
        A factory function compatible with BenchmarkRunner's executor_factory parameter.
        Signature: (llm_provider, server_manager, concurrent_summarization) -> executor
    """

    def factory(llm_provider, server_manager, concurrent_summarization):
        return OpenClawExecutor(
            server_manager=server_manager,
            openclaw_agent=agent,
            trajectory_dir=trajectory_dir,
            save_trajectories=save_trajectories,
        )

    return factory


async def main():
    from benchmark.runner import BenchmarkRunner, main as original_main, parse_arguments, \
        _parse_and_validate_args, _create_runner_and_get_models, \
        _determine_selected_models, _print_configuration

    args, tasks_file, enable_distraction = _parse_and_validate_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize your agent
    agent = YourOpenClawAgent(model_name="gpt-4o")

    trajectory_dir = getattr(args, "trajectory_dir", "trajectories")
    executor_factory = create_openclaw_executor_factory(
        agent=agent,
        trajectory_dir=trajectory_dir,
        save_trajectories=True,
    )

    runner = BenchmarkRunner(
        tasks_file=tasks_file,
        enable_distraction_servers=enable_distraction,
        distraction_count=args.distraction_count,
        enable_judge_stability=not args.disable_judge_stability,
        filter_problematic_tools=not args.disable_filter_problematic_tools,
        concurrent_summarization=not args.disable_concurrent_summarization,
        use_fuzzy_descriptions=not args.disable_fuzzy,
        executor_factory=executor_factory,
    )

    available_models = list(runner.model_configs.keys())

    if args.list_models:
        print("Available models:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")
        return

    selected_models = _determine_selected_models(args, available_models)
    _print_configuration(selected_models, available_models, runner, args)

    print(f"\n   Executor: OpenClaw (trajectories -> {trajectory_dir}/)")
    print(f"   Agent: {agent.__class__.__name__}")

    try:
        logger.info("Starting OpenClaw benchmark execution...")
        results = await runner.run_benchmark(
            selected_models=selected_models,
            task_limit=None,
        )

        if results:
            from datetime import datetime
            output_file = args.output if args.output else \
                f'openclaw_benchmark_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            import json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")

        logger.info(f"Trajectories saved to {trajectory_dir}/")
        logger.info("Use these trajectories for SFT training of your model.")

    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(main())
