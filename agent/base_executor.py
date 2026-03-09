"""Base Agent Executor Interface.

Defines the abstract interface that all agent executors must implement.
This allows MCP-Bench to work with different agent frameworks (built-in TaskExecutor,
OpenClaw, or any other agent system) through a unified interface.

The key contract: any executor receives a task string and returns a standardized
result dictionary that MCP-Bench's evaluator can process.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseAgentExecutor(ABC):
    """Abstract base class for agent executors.

    Any agent framework that wants to integrate with MCP-Bench's evaluation
    pipeline must implement this interface. The executor is responsible for:
    1. Receiving a natural language task
    2. Planning which tools to call
    3. Executing tool calls (possibly over multiple rounds)
    4. Synthesizing a final solution

    The returned result must conform to the schema expected by TaskEvaluator.
    """

    @abstractmethod
    async def execute(self, task: str) -> Dict[str, Any]:
        """Execute a task and return standardized results.

        Args:
            task: Natural language description of the task to execute.

        Returns:
            A dictionary with the following required keys:

            - "solution" (str): The final synthesized answer to the task.

            - "total_rounds" (int): Number of execution rounds performed.

            - "execution_results" (List[Dict]): List of tool call records.
              Each record must contain:
                - "tool" (str): Full tool name in "ServerName:tool_name" format.
                - "server" (str): The MCP server name.
                - "parameters" (dict): Parameters passed to the tool.
                - "round_num" (int): Which round this call was made in.
                - "result" (str): Tool output text (present if success=True).
                - "error" (str): Error message (present if success=False).
                - "success" (bool): Whether the tool call succeeded.

            - "planning_json_compliance" (float): Ratio of valid planned tools
              to total planned tools (0.0 to 1.0). Set to 1.0 if not applicable.

            - "accumulated_information" (str): Summary of all gathered information
              across rounds. Used by the evaluator for scoring.

            - "accumulated_information_uncompressed" (str): Full uncompressed version
              of accumulated information. Preferred by the LLM judge for evaluation.

            - "available_tools" (Dict[str, Any]): Dictionary of all tools that were
              available during execution. Keys are "ServerName:tool_name", values
              contain "name", "server", "description", "input_schema".

            Optional keys:
            - "total_output_tokens" (int): Total output tokens consumed.
            - "total_prompt_tokens" (int): Total prompt tokens consumed.
            - "total_tokens" (int): Total tokens consumed.
        """
        pass


# Example result for reference:
#
# {
#     "solution": "Based on the analysis, the temperature in Tokyo is 22°C...",
#     "total_rounds": 3,
#     "execution_results": [
#         {
#             "tool": "Weather Data:get_weather",
#             "server": "Weather Data",
#             "parameters": {"city": "Tokyo"},
#             "round_num": 1,
#             "result": '{"temperature": 22, "unit": "celsius"}',
#             "success": True
#         },
#         {
#             "tool": "Unit Converter:convert_temperature",
#             "server": "Unit Converter",
#             "parameters": {"value": 22, "from_unit": "celsius", "to_unit": "fahrenheit"},
#             "round_num": 2,
#             "result": '{"converted_value": 71.6}',
#             "success": True
#         }
#     ],
#     "planning_json_compliance": 1.0,
#     "accumulated_information": "Round 1: Got Tokyo weather (22°C)\nRound 2: Converted to 71.6°F",
#     "accumulated_information_uncompressed": "...(same as above for short runs)...",
#     "available_tools": {
#         "Weather Data:get_weather": {
#             "name": "get_weather",
#             "server": "Weather Data",
#             "description": "Get current weather for a city",
#             "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
#         },
#         ...
#     }
# }
