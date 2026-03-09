#!/usr/bin/env python3
"""
Filter and score collected trajectories for SFT training quality.

This is Step 2 of the data pipeline:
  Step 1: Collect trajectories (run_openclaw_benchmark.py)
  Step 2: Filter & score (this script)   ← YOU ARE HERE
  Step 3: Convert good ones to SFT format (convert_to_sft.py)

Two-stage filtering:
  Stage 1 (Rule-based, free): Hard metrics from execution results
  Stage 2 (LLM Judge, costs tokens): Soft quality scoring

Usage:
    # Stage 1 only (free, fast)
    python tools/filter_trajectories.py --input-dir trajectories_openclaw --rules-only

    # Stage 1 + Stage 2 (needs API key, thorough)
    python tools/filter_trajectories.py --input-dir trajectories_openclaw

    # Custom thresholds
    python tools/filter_trajectories.py --min-success-rate 0.9 --min-judge-score 5
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Stage 1: Rule-based filtering (free, no LLM needed)
# ============================================================================

def rule_based_check(traj_path: str) -> Dict[str, Any]:
    """Apply rule-based quality checks to a trajectory.

    Returns a report dict with pass/fail for each rule.
    """
    with open(traj_path, "r", encoding="utf-8") as f:
        traj = json.load(f)

    messages = traj.get("messages", [])
    tools = traj.get("tools", [])
    metadata = traj.get("metadata", {})

    # Extract execution info from messages
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    tool_call_msgs = [m for m in assistant_msgs if m.get("tool_calls")]

    # Count tool calls and results
    total_tool_calls = sum(len(m.get("tool_calls", [])) for m in tool_call_msgs)
    total_tool_results = len(tool_msgs)

    # Check for errors in tool results
    error_results = [m for m in tool_msgs if m.get("content", "").startswith("Error:")]
    success_results = total_tool_results - len(error_results)

    # Final answer check
    final_msgs = [m for m in assistant_msgs if not m.get("tool_calls") and m.get("content")]
    has_final_answer = len(final_msgs) > 0
    final_answer_length = len(final_msgs[-1].get("content", "")) if final_msgs else 0

    # Tool result lengths
    tool_result_lengths = [len(m.get("content", "")) for m in tool_msgs]
    max_tool_result_len = max(tool_result_lengths) if tool_result_lengths else 0
    avg_tool_result_len = sum(tool_result_lengths) / len(tool_result_lengths) if tool_result_lengths else 0

    # Rounds
    total_rounds = len(tool_call_msgs) + (1 if has_final_answer else 0)

    # Compute metrics
    success_rate = success_results / total_tool_results if total_tool_results > 0 else 0
    has_tools_def = len(tools) > 0

    report = {
        "file": os.path.basename(traj_path),
        "task_id": traj.get("task_id", "unknown"),
        "metrics": {
            "total_rounds": total_rounds,
            "total_tool_calls": total_tool_calls,
            "total_tool_results": total_tool_results,
            "success_results": success_results,
            "error_results": len(error_results),
            "success_rate": round(success_rate, 3),
            "has_final_answer": has_final_answer,
            "final_answer_length": final_answer_length,
            "has_tools_definition": has_tools_def,
            "tools_count": len(tools),
            "max_tool_result_length": max_tool_result_len,
            "avg_tool_result_length": round(avg_tool_result_len),
            "total_messages": len(messages),
            "token_usage": metadata.get("token_usage", {}),
        },
        "checks": {},
    }

    # Apply checks
    checks = {}

    # C1: At least one tool was called
    checks["has_tool_calls"] = {
        "pass": total_tool_calls > 0,
        "value": total_tool_calls,
        "rule": "total_tool_calls > 0",
    }

    # C2: Success rate above threshold
    checks["success_rate"] = {
        "pass": success_rate >= 0.7,
        "value": round(success_rate, 3),
        "rule": "success_rate >= 0.7",
    }

    # C3: Has final answer
    checks["has_final_answer"] = {
        "pass": has_final_answer,
        "value": has_final_answer,
        "rule": "has_final_answer == True",
    }

    # C4: Final answer is substantive (not just a few words)
    checks["final_answer_quality"] = {
        "pass": final_answer_length >= 100,
        "value": final_answer_length,
        "rule": "final_answer_length >= 100",
    }

    # C5: Not stuck in a loop (reasonable round count)
    checks["not_looping"] = {
        "pass": total_rounds <= 20,
        "value": total_rounds,
        "rule": "total_rounds <= 20",
    }

    # C6: Has tools definition (needed for SFT format)
    checks["has_tools_def"] = {
        "pass": has_tools_def,
        "value": len(tools),
        "rule": "tools definition present in trajectory",
    }

    # C7: Tool calls match tool results (no orphaned calls)
    checks["calls_results_match"] = {
        "pass": abs(total_tool_calls - total_tool_results) <= 2,
        "value": f"{total_tool_calls} calls, {total_tool_results} results",
        "rule": "tool_calls ≈ tool_results (within 2)",
    }

    report["checks"] = checks
    report["all_passed"] = all(c["pass"] for c in checks.values())
    report["pass_count"] = sum(1 for c in checks.values() if c["pass"])
    report["total_checks"] = len(checks)

    return report


# ============================================================================
# Stage 2: LLM Judge scoring (costs tokens)
# ============================================================================

async def llm_judge_score(traj_path: str, llm_provider) -> Dict[str, Any]:
    """Use LLM judge to score trajectory quality.

    This evaluates the final solution against the task description.
    """
    from benchmark.evaluator import TaskEvaluator

    with open(traj_path, "r", encoding="utf-8") as f:
        traj = json.load(f)

    messages = traj.get("messages", [])

    # Extract task (user message)
    user_msgs = [m for m in messages if m["role"] == "user"]
    task = user_msgs[0]["content"] if user_msgs else ""

    # Extract final solution
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    final_msgs = [m for m in assistant_msgs if not m.get("tool_calls") and m.get("content")]
    solution = final_msgs[-1]["content"] if final_msgs else ""

    # Extract execution results from tool messages
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    tool_call_msgs = [m for m in assistant_msgs if m.get("tool_calls")]

    execution_results = []
    tool_idx = 0
    for round_num, am in enumerate(tool_call_msgs, 1):
        for tc in am.get("tool_calls", []):
            fn_name = tc["function"]["name"]
            # Convert back to MCP format
            mcp_name = fn_name.replace("__", ":", 1)
            try:
                params = json.loads(tc["function"]["arguments"])
            except:
                params = {}

            result_content = ""
            is_error = False
            if tool_idx < len(tool_msgs):
                result_content = tool_msgs[tool_idx].get("content", "")
                is_error = result_content.startswith("Error:")
                tool_idx += 1

            server = mcp_name.split(":")[0] if ":" in mcp_name else "unknown"
            er = {
                "tool": mcp_name,
                "server": server,
                "parameters": params,
                "round_num": round_num,
                "success": not is_error,
            }
            if is_error:
                er["error"] = result_content
            else:
                er["result"] = result_content
            execution_results.append(er)

    # Build accumulated info
    accumulated = ""
    for er in execution_results:
        tool = er["tool"]
        params = er.get("parameters", {})
        server = er.get("server", "unknown")
        if er["success"]:
            accumulated += f"Tool `{tool}` with Parameter {params} on {server} succeeded. Result: {er.get('result', '')}\n"
        else:
            accumulated += f"Tool `{tool}` with Parameter {params} on {server} failed. Error: {er.get('error', '')}\n"

    # Build available_tools from trajectory tools field
    available_tools = {}
    for t in traj.get("tools", []):
        fn = t["function"]
        mcp_name = fn["name"].replace("__", ":", 1)
        server = mcp_name.split(":")[0] if ":" in mcp_name else "unknown"
        available_tools[mcp_name] = {
            "name": fn["name"],
            "server": server,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {}),
        }

    total_rounds = len(tool_call_msgs)

    evaluator = TaskEvaluator(llm_provider, enable_judge_stability=False)
    evaluation = await evaluator.evaluate(
        task=task,
        execution_results=execution_results,
        final_solution=solution,
        total_rounds=total_rounds,
        available_tools=available_tools,
        planning_json_compliance=1.0,
        accumulated_information=accumulated,
    )

    return evaluation


# ============================================================================
# Main pipeline
# ============================================================================

def print_report(reports: List[Dict], show_details: bool = True):
    """Print filtering results in a readable table."""
    print(f"\n{'='*90}")
    print(f"{'Task ID':<45} {'Checks':>8} {'Success%':>9} {'Calls':>6} {'Answer':>7} {'Tools':>6} {'Result'}")
    print(f"{'='*90}")

    passed = []
    failed = []

    for r in reports:
        task_id = r["task_id"]
        m = r["metrics"]
        pass_count = r["pass_count"]
        total_checks = r["total_checks"]
        all_ok = r["all_passed"]
        status = "✅ PASS" if all_ok else "❌ FAIL"

        print(f"{task_id:<45} {pass_count}/{total_checks:>5} {m['success_rate']:>8.0%} {m['total_tool_calls']:>6} {m['final_answer_length']:>6}c {'✅' if m['has_tools_definition'] else '❌':>5}  {status}")

        if all_ok:
            passed.append(r)
        else:
            failed.append(r)

        if show_details and not all_ok:
            for name, check in r["checks"].items():
                if not check["pass"]:
                    print(f"  └─ ❌ {name}: {check['value']} (rule: {check['rule']})")

    print(f"{'='*90}")
    print(f"Total: {len(reports)} | Passed: {len(passed)} | Failed: {len(failed)}")
    return passed, failed


def print_judge_scores(scores: Dict[str, Dict]):
    """Print LLM judge scores."""
    print(f"\n{'='*90}")
    print(f"{'Task ID':<45} {'Fulfill':>8} {'Ground':>7} {'ToolApp':>8} {'ParamAc':>8} {'DepAwr':>7} {'Effic':>6}")
    print(f"{'='*90}")

    for task_id, s in scores.items():
        if s is None:
            print(f"{task_id:<45} {'FAILED':>8}")
            continue
        print(
            f"{task_id:<45} "
            f"{s.get('task_fulfillment', 'N/A'):>8} "
            f"{s.get('grounding', 'N/A'):>7} "
            f"{s.get('tool_appropriateness', 'N/A'):>8} "
            f"{s.get('parameter_accuracy', 'N/A'):>8} "
            f"{s.get('dependency_awareness', 'N/A'):>7} "
            f"{s.get('parallelism_and_efficiency', 'N/A'):>6}"
        )

    print(f"{'='*90}")


async def run_pipeline(args):
    traj_files = sorted(glob.glob(os.path.join(args.input_dir, "traj_*.json")))
    if not traj_files:
        print(f"No trajectory files found in {args.input_dir}")
        return

    print(f"Found {len(traj_files)} trajectories in {args.input_dir}")

    # Stage 1: Rule-based filtering
    print(f"\n{'#'*90}")
    print(f"# Stage 1: Rule-based filtering (free, no LLM)")
    print(f"{'#'*90}")

    reports = []
    for traj_path in traj_files:
        report = rule_based_check(traj_path)
        reports.append(report)

    passed, failed = print_report(reports, show_details=True)

    # Save filtering results
    os.makedirs(args.output_dir, exist_ok=True)
    filter_result_path = os.path.join(args.output_dir, "filter_results.json")
    with open(filter_result_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(reports),
            "passed": len(passed),
            "failed": len(failed),
            "reports": reports,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFilter results saved to {filter_result_path}")

    if args.rules_only:
        # Output list of passed files
        passed_files = [os.path.join(args.input_dir, r["file"]) for r in passed]
        print(f"\n✅ {len(passed_files)} trajectories passed rule-based filtering:")
        for f in passed_files:
            print(f"  {f}")
        return passed

    # Stage 2: LLM Judge scoring (only on passed trajectories)
    if not passed:
        print("\nNo trajectories passed Stage 1. Skipping Stage 2.")
        return passed

    print(f"\n{'#'*90}")
    print(f"# Stage 2: LLM Judge scoring ({len(passed)} trajectories)")
    print(f"{'#'*90}")

    api_key = os.environ.get("KIMI_API_KEY")
    if not api_key:
        print("WARNING: KIMI_API_KEY not set. Skipping LLM judge scoring.")
        return passed

    from llm.factory import LLMFactory
    configs = LLMFactory.get_model_configs()
    judge_config = configs.get("kimi-k2.5")
    if not judge_config:
        print("WARNING: kimi-k2.5 model not configured. Skipping LLM judge scoring.")
        return passed

    judge_provider = await LLMFactory.create_llm_provider(judge_config)

    judge_scores = {}
    for r in passed:
        traj_path = os.path.join(args.input_dir, r["file"])
        task_id = r["task_id"]
        print(f"\n  Scoring {task_id}...")
        try:
            score = await llm_judge_score(traj_path, judge_provider)
            judge_scores[task_id] = score
            if score:
                avg = (
                    score.get("task_fulfillment", 0)
                    + score.get("grounding", 0)
                    + score.get("tool_appropriateness", 0)
                    + score.get("parameter_accuracy", 0)
                ) / 4
                print(f"    Average score: {avg:.1f}/10")
        except Exception as e:
            logger.error(f"    Judge scoring failed: {e}")
            judge_scores[task_id] = None

    print_judge_scores(judge_scores)

    # Filter by judge score
    min_score = args.min_judge_score
    final_passed = []
    for r in passed:
        task_id = r["task_id"]
        score = judge_scores.get(task_id)
        if score is None:
            continue
        avg = (
            score.get("task_fulfillment", 0)
            + score.get("grounding", 0)
            + score.get("tool_appropriateness", 0)
            + score.get("parameter_accuracy", 0)
        ) / 4
        if avg >= min_score:
            final_passed.append(r)
            r["judge_scores"] = score
            r["judge_avg"] = round(avg, 2)

    print(f"\n✅ {len(final_passed)}/{len(passed)} passed LLM judge (min avg score: {min_score})")

    # Save final results
    final_result_path = os.path.join(args.output_dir, "final_filter_results.json")
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": len(reports),
            "stage1_passed": len(passed),
            "stage2_passed": len(final_passed),
            "min_judge_score": min_score,
            "passed_trajectories": [
                {"file": r["file"], "task_id": r["task_id"], "judge_avg": r.get("judge_avg")}
                for r in final_passed
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"Final results saved to {final_result_path}")

    return final_passed


def main():
    parser = argparse.ArgumentParser(description="Filter trajectories for SFT quality")
    parser.add_argument("--input-dir", default="trajectories_openclaw")
    parser.add_argument("--output-dir", default="training_data")
    parser.add_argument("--rules-only", action="store_true", help="Only run rule-based filtering (no LLM)")
    parser.add_argument("--min-success-rate", type=float, default=0.7)
    parser.add_argument("--min-judge-score", type=float, default=4.0, help="Min average judge score (1-10)")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
