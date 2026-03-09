#!/usr/bin/env python3
"""
Run a single MCP-Bench task with Kimi K2.5.

This tests the full pipeline:
- Connect to Unit Converter MCP server
- Kimi K2.5 plans and executes tool calls
- Evaluate results (using Kimi K2.5 as judge since no Azure)
- Save trajectory
"""

import sys
import asyncio
import json
import logging
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    from llm.factory import LLMFactory
    from agent.executor import TaskExecutor
    from benchmark.runner import ConnectionManager
    from benchmark.evaluator import TaskEvaluator

    # Load the Unit Converter task (simpler subset)
    with open("tasks/mcpbench_tasks_single_runner_format.json") as f:
        data = json.load(f)

    task_data = None
    for group in data["server_tasks"]:
        if group["server_name"] == "Unit Converter":
            task_data = group["tasks"][0]
            break

    task_description = task_data["fuzzy_description"]
    task_id = task_data["task_id"]

    print("=" * 70)
    print(f"Running task: {task_id}")
    print(f"Model: kimi-k2.5 (Moonshot)")
    print("=" * 70)

    # Create LLM provider
    configs = LLMFactory.get_model_configs()
    if "kimi-k2.5" not in configs:
        print("ERROR: KIMI_API_KEY not set")
        sys.exit(1)

    llm_provider = await LLMFactory.create_llm_provider(configs["kimi-k2.5"])

    # Server config - just Unit Converter + Time MCP (no distractions for speed)
    server_configs = [
        {
            "name": "Unit Converter",
            "command": ["python3", "-m", "unit_converter_mcp.server"],
            "env": {},
            "cwd": "mcp_servers/unit-converter-mcp",
        },
        {
            "name": "Time MCP",
            "command": ["python3", "-m", "mcp_server_time"],
            "env": {},
            "cwd": "mcp_servers/time-mcp",
        },
    ]

    print("\nStep 1: Connecting to MCP servers...")
    start_time = time.time()

    async with ConnectionManager(server_configs) as conn_mgr:
        print(f"  Connected! {len(conn_mgr.all_tools)} tools discovered")
        for tool_name in conn_mgr.all_tools:
            print(f"    - {tool_name}")

        # Create executor with built-in TaskExecutor
        print("\nStep 2: Executing task with Kimi K2.5...")
        executor = TaskExecutor(llm_provider, conn_mgr.server_manager, concurrent_summarization=True)

        try:
            result = await asyncio.wait_for(
                executor.execute(task_description),
                timeout=600,
            )
        except asyncio.TimeoutError:
            print("Task timed out after 600 seconds")
            return

        result["available_tools"] = conn_mgr.all_tools
        execution_time = time.time() - start_time

        print(f"\nStep 3: Execution complete in {execution_time:.1f}s")
        print(f"  Rounds: {result['total_rounds']}")
        print(f"  Tool calls: {len(result['execution_results'])}")
        print(f"  Tokens: {result.get('total_tokens', 'N/A')}")

        # Show execution results
        print("\n--- Tool Call Summary ---")
        for er in result["execution_results"]:
            status = "OK" if er["success"] else "FAIL"
            print(f"  [{status}] R{er['round_num']}: {er['tool']}({json.dumps(er['parameters'])[:100]}...)")

        # Show final solution
        print("\n--- Final Solution (first 500 chars) ---")
        print(result.get("solution", "No solution")[:500])

        # Evaluate
        print("\nStep 4: Evaluating with Kimi K2.5 as judge...")
        try:
            evaluator = TaskEvaluator(llm_provider, enable_judge_stability=False)
            evaluation = await evaluator.evaluate(
                task=task_description,
                execution_results=result.get("execution_results", []),
                final_solution=result.get("solution", ""),
                total_rounds=result.get("total_rounds", 0),
                available_tools=result.get("available_tools", {}),
                planning_json_compliance=result.get("planning_json_compliance", 1.0),
                accumulated_information=result.get("accumulated_information_uncompressed")
                or result.get("accumulated_information", ""),
                concrete_task_description=task_data.get("task_description"),
                dependency_analysis=task_data.get("dependency_analysis"),
            )

            if evaluation:
                print("\n--- Evaluation Scores ---")
                print(f"  Task Fulfillment:        {evaluation.get('task_fulfillment', 'N/A')}/10")
                print(f"  Grounding:               {evaluation.get('grounding', 'N/A')}/10")
                print(f"  Tool Appropriateness:    {evaluation.get('tool_appropriateness', 'N/A')}/10")
                print(f"  Parameter Accuracy:      {evaluation.get('parameter_accuracy', 'N/A')}/10")
                print(f"  Dependency Awareness:    {evaluation.get('dependency_awareness', 'N/A')}/10")
                print(f"  Parallelism/Efficiency:  {evaluation.get('parallelism_and_efficiency', 'N/A')}/10")
                print(f"\n  Task Completion Score:   {evaluation.get('task_completion_score', 'N/A'):.1f}/10")
                print(f"  Tool Selection Score:    {evaluation.get('tool_selection_score', 'N/A'):.1f}/10")
                print(f"  Planning Score:          {evaluation.get('planning_effectiveness_and_efficiency_score', 'N/A'):.1f}/10")

                # Schema metrics
                print(f"\n  Schema Compliance:       {evaluation.get('input_schema_compliance', 'N/A')}")
                print(f"  Valid Tool Name Rate:    {evaluation.get('valid_tool_name_rate', 'N/A')}")
                print(f"  Execution Success Rate:  {evaluation.get('execution_success_rate', 'N/A')}")
            else:
                print("  Evaluation failed")
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    output = {
        "task_id": task_id,
        "model": "kimi-k2.5",
        "execution_time": execution_time,
        "total_rounds": result["total_rounds"],
        "total_tool_calls": len(result["execution_results"]),
        "solution": result.get("solution", ""),
        "execution_results": result.get("execution_results", []),
        "evaluation": evaluation if evaluation else None,
        "tokens": {
            "output": result.get("total_output_tokens", 0),
            "prompt": result.get("total_prompt_tokens", 0),
            "total": result.get("total_tokens", 0),
        },
    }

    outfile = f"kimi_result_{task_id}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outfile}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
