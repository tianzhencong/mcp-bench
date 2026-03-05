#!/usr/bin/env python3
"""
Demo: Run a single MCP-Bench task through OpenClaw integration layer.

This demonstrates the full flow without needing LLM API keys:
1. Connect to a real MCP server (Unit Converter)
2. Run a mock agent that makes real tool calls
3. Collect trajectory data
4. Show the results in MCP-Bench evaluation format

No API keys required - uses a hardcoded agent plan for demonstration.
"""

import sys
import asyncio
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.openclaw_executor import (
    OpenClawExecutor,
    AgentResult,
    RoundTrace,
    ToolCallTrace,
)
from mcp_modules.server_manager_persistent import PersistentMultiServerManager
from benchmark.runner import ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DemoAgent:
    """A hardcoded agent that demonstrates real MCP tool calls.

    This agent converts 350°F to Celsius, then checks supported units.
    It simulates what a real OpenClaw agent would do, but without needing an LLM.
    """

    async def run(self, task, tools, call_tool) -> AgentResult:
        rounds = []
        logger.info(f"DemoAgent received task: {task[:100]}...")
        logger.info(f"DemoAgent sees {len(tools)} available tools")

        # Log available tools
        for tool_name, tool_info in tools.items():
            logger.info(f"  Tool: {tool_name} - {tool_info['description'][:80]}...")

        # === Round 1: List supported units to understand what's available ===
        round1_calls = []

        logger.info("\n=== Round 1: Discover supported unit types ===")
        r1 = await call_tool("Unit Converter:list_supported_units", {"unit_type": None})
        round1_calls.append(ToolCallTrace(
            tool_name="Unit Converter:list_supported_units",
            server_name="Unit Converter",
            parameters={"unit_type": None},
            result=r1.get("result"),
            error=r1.get("error"),
            success=r1["success"],
        ))

        rounds.append(RoundTrace(
            round_num=1,
            reasoning=(
                "First, I need to understand what unit types and units are supported. "
                "I'll call list_supported_units with unit_type=null to get all types."
            ),
            tool_calls=round1_calls,
            should_continue=True,
        ))

        # === Round 2: Convert temperature and do a batch conversion ===
        round2_calls = []

        logger.info("\n=== Round 2: Convert 350°F to Celsius ===")
        r2 = await call_tool("Unit Converter:convert_temperature", {
            "value": 350,
            "from_unit": "fahrenheit",
            "to_unit": "celsius",
        })
        round2_calls.append(ToolCallTrace(
            tool_name="Unit Converter:convert_temperature",
            server_name="Unit Converter",
            parameters={"value": 350, "from_unit": "fahrenheit", "to_unit": "celsius"},
            result=r2.get("result"),
            error=r2.get("error"),
            success=r2["success"],
        ))

        logger.info("\n=== Round 2: Batch convert pressure and length ===")
        r3 = await call_tool("Unit Converter:convert_batch", {
            "conversions": [
                {
                    "value": 50,
                    "from_unit": "psi",
                    "to_unit": "kilopascals",
                    "conversion_type": "pressure",
                    "request_id": "pressure_001",
                },
                {
                    "value": 10,
                    "from_unit": "feet",
                    "to_unit": "meters",
                    "conversion_type": "length",
                    "request_id": "length_001",
                },
            ]
        })
        round2_calls.append(ToolCallTrace(
            tool_name="Unit Converter:convert_batch",
            server_name="Unit Converter",
            parameters={
                "conversions": [
                    {"value": 50, "from_unit": "psi", "to_unit": "kilopascals",
                     "conversion_type": "pressure", "request_id": "pressure_001"},
                    {"value": 10, "from_unit": "feet", "to_unit": "meters",
                     "conversion_type": "length", "request_id": "length_001"},
                ]
            },
            result=r3.get("result"),
            error=r3.get("error"),
            success=r3["success"],
        ))

        rounds.append(RoundTrace(
            round_num=2,
            reasoning=(
                "Now I'll convert the specific measurements. "
                "350°F to Celsius for temperature check, "
                "and batch convert 50 psi to kPa and 10 ft to meters."
            ),
            tool_calls=round2_calls,
            should_continue=False,
        ))

        # Build solution from actual results
        temp_result = r2.get("result", "conversion failed")
        batch_result = r3.get("result", "batch conversion failed")

        solution = (
            f"Conversion Results:\n"
            f"1. Temperature: 350°F = {temp_result}\n"
            f"2. Batch conversions: {batch_result}\n\n"
            f"The inlet temperature of 350°F converts to approximately 176.67°C, "
            f"which exceeds the 150°C threshold (PASS).\n"
            f"The inlet pressure of 50 psi converts to approximately 344.74 kPa, "
            f"which is close to but below the 350 kPa threshold (FAIL).\n"
            f"The reactor length of 10 ft converts to approximately 3.048 m, "
            f"which is below the 5 m threshold (FAIL)."
        )

        return AgentResult(solution=solution, rounds=rounds)


async def main():
    # Server config for Unit Converter
    server_configs = [
        {
            "name": "Unit Converter",
            "command": ["python3", "-m", "unit_converter_mcp.server"],
            "env": {},
            "cwd": "mcp_servers/unit-converter-mcp",
        }
    ]

    task = (
        "Convert the following sensor readings and check against thresholds:\n"
        "1. Inlet temperature: 350°F → threshold 150°C\n"
        "2. Inlet pressure: 50 psi → threshold 350 kPa\n"
        "3. Reactor length: 10 ft → threshold 5 m\n"
        "First list all supported unit types, then perform the conversions."
    )

    print("=" * 70)
    print("MCP-Bench OpenClaw Integration Demo")
    print("=" * 70)
    print(f"\nTask: {task}\n")

    # Step 1: Connect to MCP server
    print("Step 1: Connecting to Unit Converter MCP server...")
    async with ConnectionManager(server_configs) as conn_mgr:
        print(f"  Connected! Discovered {len(conn_mgr.all_tools)} tools:")
        for tool_name in conn_mgr.all_tools:
            print(f"    - {tool_name}")

        # Step 2: Create OpenClaw executor with demo agent
        print("\nStep 2: Creating OpenClaw executor with demo agent...")
        agent = DemoAgent()
        executor = OpenClawExecutor(
            server_manager=conn_mgr.server_manager,
            openclaw_agent=agent,
            trajectory_dir="trajectories",
            save_trajectories=True,
        )

        # Step 3: Execute
        print("\nStep 3: Executing task through demo agent...")
        print("-" * 70)
        result = await executor.execute(task)
        print("-" * 70)

        # Step 4: Show results in MCP-Bench format
        print("\n" + "=" * 70)
        print("RESULTS (MCP-Bench Evaluation Format)")
        print("=" * 70)

        print(f"\nTotal Rounds: {result['total_rounds']}")
        print(f"Total Tool Calls: {len(result['execution_results'])}")

        print("\n--- Execution Results ---")
        for i, er in enumerate(result["execution_results"]):
            status = "SUCCESS" if er["success"] else "FAILED"
            print(f"  [{status}] Round {er['round_num']}: {er['tool']}")
            print(f"           Params: {json.dumps(er['parameters'], indent=2)[:200]}")
            if er["success"]:
                preview = er["result"][:200]
                print(f"           Result: {preview}...")
            else:
                print(f"           Error: {er.get('error', '')[:200]}")

        print("\n--- Final Solution ---")
        print(result["solution"])

        print("\n--- Accumulated Information (for Evaluator) ---")
        print(result["accumulated_information"][:500])

        # Step 5: Show trajectory file
        import glob
        trajectory_files = sorted(glob.glob("trajectories/trajectory_*.json"))
        if trajectory_files:
            latest = trajectory_files[-1]
            print(f"\n{'=' * 70}")
            print(f"TRAJECTORY SAVED: {latest}")
            print(f"{'=' * 70}")
            with open(latest, "r") as f:
                traj = json.load(f)
            print(f"  Task: {traj['task'][:80]}...")
            print(f"  Tools available: {len(traj['available_tools'])}")
            print(f"  Rounds: {traj['total_rounds']}")
            print(f"  Total tool calls: {traj['total_tool_calls']}")
            print(f"\n  Round details:")
            for rnd in traj["rounds"]:
                print(f"    Round {rnd['round_num']}:")
                print(f"      Reasoning: {rnd['reasoning'][:100]}...")
                print(f"      Tool calls: {len(rnd['tool_calls'])}")
                for tc in rnd["tool_calls"]:
                    status = "OK" if tc["success"] else "FAIL"
                    print(f"        [{status}] {tc['tool_name']}")

        print(f"\n{'=' * 70}")
        print("Demo complete!")
        print(f"{'=' * 70}")
        print("\nTo use with your real OpenClaw agent:")
        print("  1. Replace DemoAgent with your agent in run_openclaw_benchmark.py")
        print("  2. Set OPENROUTER_API_KEY or AZURE_OPENAI_API_KEY for evaluation")
        print("  3. Run: python run_openclaw_benchmark.py --models <model_name>")
        print(f"\nTrajectory files for SFT training: trajectories/")


if __name__ == "__main__":
    asyncio.run(main())
