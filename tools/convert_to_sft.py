#!/usr/bin/env python3
"""
Convert OpenClaw trajectory files to standard OpenAI SFT training format.

Input:  trajectories_openclaw/traj_*.json (OpenClaw raw format)
Output: training_data/sft_data.jsonl (OpenAI fine-tuning format)

OpenAI SFT format (per line in JSONL):
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_0", "type": "function", "function": {"name": "...", "arguments": "..."}}
    ]},
    {"role": "tool", "tool_call_id": "call_0", "content": "..."},
    {"role": "assistant", "content": "final answer"}
  ]
}

Options:
  --keep-reasoning: Keep reasoning_content as a separate field (for reasoning distillation)
  --strip-reasoning: Remove reasoning_content entirely (pure OpenAI format)
  --reasoning-to-content: Merge reasoning into content field (for non-thinking model training)
"""

import argparse
import glob
import json
import os
import sys


def clean_message_for_sft(msg: dict, reasoning_mode: str = "strip") -> dict:
    """Clean a single message to conform to OpenAI SFT format.

    Args:
        msg: Raw message dict from OpenClaw trajectory
        reasoning_mode:
            "strip" - Remove reasoning_content entirely (standard OpenAI format)
            "keep" - Keep reasoning_content as-is (for Kimi/thinking model training)
            "to_content" - Prepend reasoning to content as <think>...</think> block
    """
    cleaned = {}
    cleaned["role"] = msg["role"]

    if msg["role"] == "system":
        cleaned["content"] = msg.get("content", "")

    elif msg["role"] == "user":
        cleaned["content"] = msg.get("content", "")

    elif msg["role"] == "assistant":
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")

        if reasoning_mode == "to_content" and reasoning:
            content = f"<think>\n{reasoning}\n</think>\n{content}" if content else f"<think>\n{reasoning}\n</think>"
            cleaned["content"] = content
        elif reasoning_mode == "keep" and reasoning:
            cleaned["content"] = content
            cleaned["reasoning_content"] = reasoning
        else:
            cleaned["content"] = content

        if tool_calls:
            # When tool_calls present, content should be null per OpenAI spec
            # (unless there's actual content alongside)
            if not content or content.strip() == "":
                cleaned["content"] = None
            cleaned["tool_calls"] = tool_calls

    elif msg["role"] == "tool":
        cleaned["role"] = "tool"
        cleaned["tool_call_id"] = msg.get("tool_call_id", "")
        cleaned["content"] = msg.get("content", "")

    return cleaned


def convert_trajectory(traj_path: str, reasoning_mode: str = "strip") -> dict:
    """Convert a single trajectory file to OpenAI SFT format."""
    with open(traj_path, "r", encoding="utf-8") as f:
        traj = json.load(f)

    messages = traj.get("messages", [])
    cleaned_messages = [clean_message_for_sft(m, reasoning_mode) for m in messages]

    return {"messages": cleaned_messages}


def convert_all(
    input_dir: str,
    output_file: str,
    reasoning_mode: str = "strip",
) -> int:
    """Convert all trajectory files to a single JSONL file."""
    traj_files = sorted(glob.glob(os.path.join(input_dir, "traj_*.json")))
    if not traj_files:
        print(f"No trajectory files found in {input_dir}")
        return 0

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    count = 0
    with open(output_file, "w", encoding="utf-8") as out:
        for traj_path in traj_files:
            try:
                sft_sample = convert_trajectory(traj_path, reasoning_mode)
                n_msgs = len(sft_sample["messages"])
                n_tool_calls = sum(
                    len(m.get("tool_calls", []))
                    for m in sft_sample["messages"]
                    if m["role"] == "assistant"
                )
                out.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")
                count += 1
                basename = os.path.basename(traj_path)
                print(f"  [{count}] {basename} -> {n_msgs} messages, {n_tool_calls} tool_calls")
            except Exception as e:
                print(f"  ERROR processing {traj_path}: {e}")

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert OpenClaw trajectories to OpenAI SFT format")
    parser.add_argument("--input-dir", default="trajectories_openclaw", help="Directory with traj_*.json files")
    parser.add_argument("--output", default="training_data/sft_data.jsonl", help="Output JSONL file")
    parser.add_argument(
        "--reasoning",
        choices=["strip", "keep", "to_content"],
        default="strip",
        help="How to handle reasoning_content: strip (remove), keep (as-is), to_content (merge into <think> block)",
    )
    args = parser.parse_args()

    print(f"Converting trajectories from: {args.input_dir}")
    print(f"Output: {args.output}")
    print(f"Reasoning mode: {args.reasoning}")
    print()

    count = convert_all(args.input_dir, args.output, args.reasoning)

    print(f"\nConverted {count} trajectories to {args.output}")

    # Also generate a reasoning version if stripping
    if args.reasoning == "strip":
        reasoning_output = args.output.replace(".jsonl", "_with_reasoning.jsonl")
        print(f"\nAlso generating reasoning version: {reasoning_output}")
        count2 = convert_all(args.input_dir, reasoning_output, "to_content")
        print(f"Converted {count2} trajectories to {reasoning_output}")


if __name__ == "__main__":
    main()
