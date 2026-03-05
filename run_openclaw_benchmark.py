#!/usr/bin/env python3
"""
Run MCP-Bench tasks with OpenClaw agent (native function calling).

This uses the OpenClaw agent which leverages the LLM's native function_calling
protocol instead of MCP-Bench's built-in JSON-prompt approach. The result is:

1. Better tool-calling behavior (model uses its trained function_calling ability)
2. SFT-ready trajectory data (standard messages format, directly usable for training)
3. Full MCP-Bench evaluation scoring

Usage:
    export KIMI_API_KEY="sk-..."
    export KIMI_BASE_URL="https://api.moonshot.cn/v1"

    # Run a single task (quick test)
    python run_openclaw_benchmark.py --task-id unit_converter_000

    # Run all single-server tasks
    python run_openclaw_benchmark.py --tasks-file tasks/mcpbench_tasks_single_runner_format.json

    # Run with custom trajectory output
    python run_openclaw_benchmark.py --task-id unit_converter_000 --trajectory-dir my_trajectories
"""

import sys
import asyncio
import argparse
import json
import logging
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.openclaw_agent import OpenClawAgent, save_trajectory
from benchmark.runner import ConnectionManager
from benchmark.evaluator import TaskEvaluator
from llm.factory import LLMFactory
from utils.local_server_config import LocalServerConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_task_by_id(tasks_file: str, task_id: str) -> dict:
    """Find a specific task by ID from a tasks file."""
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for group in data.get("server_tasks", []):
        for task in group.get("tasks", []):
            if task.get("task_id") == task_id:
                return {
                    "task": task,
                    "server_name": group.get("server_name", ""),
                    "servers": group.get("servers", [group.get("server_name", "")]),
                }
    raise ValueError(f"Task {task_id} not found in {tasks_file}")


def load_all_tasks(tasks_file: str) -> list:
    """Load all tasks from a tasks file."""
    with open(tasks_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    for group in data.get("server_tasks", []):
        for task in group.get("tasks", []):
            tasks.append({
                "task": task,
                "server_name": group.get("server_name", ""),
                "servers": group.get("servers", [group.get("server_name", "")]),
            })
    return tasks


def build_server_configs(server_names: list, commands_config: dict, local_loader) -> list:
    """Build server configs for ConnectionManager."""
    import shutil
    configs = []
    for name in server_names:
        if name not in commands_config:
            logger.warning(f"Server {name} not found in commands.json, skipping")
            continue
        srv = commands_config[name]
        cmd_parts = srv.get("cmd", "").split()
        if not cmd_parts:
            continue

        # Fix: replace bare "python" with "python3" if python is not available
        if cmd_parts[0] == "python" and not shutil.which("python"):
            cmd_parts[0] = "python3"

        cwd = srv.get("cwd", "")
        if cwd.startswith("../"):
            cwd = f"mcp_servers/{cwd[3:]}"
        env = {}
        for var in srv.get("env", []):
            if var in local_loader.api_keys:
                env[var] = local_loader.api_keys[var]
        config = {"name": name, "command": cmd_parts, "env": env, "cwd": cwd}
        if srv.get("transport") == "http":
            config["transport"] = "http"
            config["port"] = srv.get("port", 3001)
            config["endpoint"] = srv.get("endpoint", "/mcp")
        configs.append(config)
    return configs


async def run_single_task(
    agent: OpenClawAgent,
    task_info: dict,
    commands_config: dict,
    local_loader,
    trajectory_dir: str,
    use_fuzzy: bool = True,
    run_eval: bool = True,
) -> dict:
    """Run a single task with OpenClaw agent."""
    task_data = task_info["task"]
    task_id = task_data["task_id"]
    server_name = task_info["server_name"]
    servers = task_info.get("servers", [server_name])

    if use_fuzzy:
        task_description = task_data.get("fuzzy_description", task_data.get("task_description", ""))
    else:
        task_description = task_data.get("task_description", "")

    print(f"\n{'='*70}")
    print(f"Task: {task_id} | Servers: {servers}")
    print(f"{'='*70}")

    # Add Time MCP as resident server
    all_servers = list(servers) + (["Time MCP"] if "Time MCP" not in servers else [])
    server_configs = build_server_configs(all_servers, commands_config, local_loader)

    if not server_configs:
        logger.error(f"No server configs available for {servers}")
        return {"task_id": task_id, "status": "failed", "error": "No servers"}

    start_time = time.time()

    async with ConnectionManager(server_configs) as conn_mgr:
        if not conn_mgr.all_tools:
            return {"task_id": task_id, "status": "failed", "error": "No tools discovered"}

        logger.info(f"Connected to {len(server_configs)} servers, {len(conn_mgr.all_tools)} tools")

        # Bridge: MCP-Bench's call_tool for OpenClaw
        async def call_tool(tool_name: str, params: dict) -> dict:
            try:
                result_obj = await conn_mgr.server_manager.call_tool(tool_name, params)
                is_error = hasattr(result_obj, "isError") and result_obj.isError
                text = ""
                if hasattr(result_obj, "content") and result_obj.content:
                    text = "".join(item.text for item in result_obj.content if hasattr(item, "text"))
                else:
                    text = str(result_obj)
                return {"success": not is_error, "result": text if not is_error else None, "error": text if is_error else None}
            except Exception as e:
                return {"success": False, "result": None, "error": str(e)}

        # Run agent
        agent_result = await agent.run(
            task=task_description,
            mcp_tools=conn_mgr.all_tools,
            call_tool=call_tool,
        )

        execution_time = time.time() - start_time

        # Build accumulated_information for evaluator
        accumulated = ""
        for er in agent_result["execution_results"]:
            rn = er.get("round_num", 1)
            tool = er["tool"]
            params = er.get("parameters", {})
            server = er.get("server", "unknown")
            if er["success"]:
                accumulated += f"Tool `{tool}` with Parameter {params} on {server} succeeded. Result: {er.get('result','')}\n"
            else:
                accumulated += f"Tool `{tool}` with Parameter {params} on {server} failed. Error: {er.get('error','')}\n"

        # Save trajectory
        traj_data = {
            "task_id": task_id,
            "model": agent.model,
            "messages": agent_result["messages"],
            "reasoning_traces": agent_result["reasoning_traces"],
            "metadata": {
                "server_name": server_name,
                "execution_time": execution_time,
                "total_rounds": agent_result["total_rounds"],
                "total_tool_calls": len(agent_result["execution_results"]),
                "token_usage": agent_result["token_usage"],
            },
        }
        traj_path = save_trajectory(traj_data, trajectory_dir, task_id)

        # Print summary
        n_calls = len(agent_result["execution_results"])
        n_ok = sum(1 for r in agent_result["execution_results"] if r["success"])
        print(f"\n--- Execution Summary ---")
        print(f"  Rounds: {agent_result['total_rounds']}")
        print(f"  Tool calls: {n_calls} ({n_ok} ok, {n_calls - n_ok} failed)")
        print(f"  Tokens: {agent_result['token_usage']}")
        print(f"  Time: {execution_time:.1f}s")
        print(f"  Trajectory: {traj_path}")

        if agent_result["solution"]:
            preview = agent_result["solution"][:300]
            print(f"\n--- Solution Preview ---\n{preview}...")

        # Evaluate
        evaluation = None
        if run_eval:
            print(f"\n--- Evaluating ---")
            try:
                configs = LLMFactory.get_model_configs()
                judge_config = configs.get("kimi-k2.5")
                if judge_config:
                    judge_provider = await LLMFactory.create_llm_provider(judge_config)
                    evaluator = TaskEvaluator(judge_provider, enable_judge_stability=False)
                    evaluation = await evaluator.evaluate(
                        task=task_description,
                        execution_results=agent_result["execution_results"],
                        final_solution=agent_result["solution"],
                        total_rounds=agent_result["total_rounds"],
                        available_tools=conn_mgr.all_tools,
                        planning_json_compliance=1.0,
                        accumulated_information=accumulated,
                        concrete_task_description=task_data.get("task_description"),
                        dependency_analysis=task_data.get("dependency_analysis"),
                    )
                    if evaluation:
                        print(f"  Task Fulfillment:    {evaluation.get('task_fulfillment', 'N/A')}/10")
                        print(f"  Grounding:           {evaluation.get('grounding', 'N/A')}/10")
                        print(f"  Tool Appropriateness:{evaluation.get('tool_appropriateness', 'N/A')}/10")
                        print(f"  Parameter Accuracy:  {evaluation.get('parameter_accuracy', 'N/A')}/10")
                        print(f"  Dependency Awareness:{evaluation.get('dependency_awareness', 'N/A')}/10")
                        print(f"  Parallelism:         {evaluation.get('parallelism_and_efficiency', 'N/A')}/10")
                        print(f"  Schema Compliance:   {evaluation.get('input_schema_compliance', 'N/A')}")
                        print(f"  Exec Success Rate:   {evaluation.get('execution_success_rate', 'N/A')}")
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                import traceback
                traceback.print_exc()

    return {
        "task_id": task_id,
        "status": "completed",
        "execution_time": execution_time,
        "total_rounds": agent_result["total_rounds"],
        "total_tool_calls": n_calls,
        "solution": agent_result["solution"],
        "evaluation": evaluation,
        "trajectory_path": traj_path,
        "token_usage": agent_result["token_usage"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run MCP-Bench with OpenClaw agent")
    parser.add_argument("--task-id", help="Run a specific task by ID")
    parser.add_argument("--tasks-file", default="tasks/mcpbench_tasks_single_runner_format.json")
    parser.add_argument("--trajectory-dir", default="trajectories")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--detailed", action="store_true", help="Use detailed (non-fuzzy) descriptions")
    parser.add_argument("--max-tasks", type=int, help="Limit number of tasks to run")
    parser.add_argument("--model", default="kimi-k2.5")
    parser.add_argument("--max-rounds", type=int, default=15)
    return parser.parse_args()


async def main():
    args = parse_args()

    api_key = os.environ.get("KIMI_API_KEY")
    base_url = os.environ.get("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    if not api_key:
        print("ERROR: KIMI_API_KEY environment variable not set")
        sys.exit(1)

    agent = OpenClawAgent(
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        temperature=1.0,
        max_rounds=args.max_rounds,
    )

    # Load commands.json for server configs
    local_loader = LocalServerConfigLoader()
    with open("mcp_servers/commands.json", "r") as f:
        commands_config = json.load(f)

    results = []

    if args.task_id:
        task_info = load_task_by_id(args.tasks_file, args.task_id)
        result = await run_single_task(
            agent, task_info, commands_config, local_loader,
            args.trajectory_dir, use_fuzzy=not args.detailed,
            run_eval=not args.no_eval,
        )
        results.append(result)
    else:
        tasks = load_all_tasks(args.tasks_file)
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
        print(f"Running {len(tasks)} tasks with OpenClaw agent ({args.model})")
        for i, task_info in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] ", end="")
            result = await run_single_task(
                agent, task_info, commands_config, local_loader,
                args.trajectory_dir, use_fuzzy=not args.detailed,
                run_eval=not args.no_eval,
            )
            results.append(result)

    # Save summary
    outfile = f"openclaw_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"Results: {outfile}")
    print(f"Trajectories: {args.trajectory_dir}/")
    completed = sum(1 for r in results if r["status"] == "completed")
    print(f"Completed: {completed}/{len(results)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
