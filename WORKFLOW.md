# MCP-Bench 数据采集工作流

## 概述

使用强模型（Kimi K2.5）通过 OpenClaw agent 执行 MCP-Bench 任务，采集高质量的工具调用轨迹，经过质检过滤后转为标准 OpenAI SFT 格式，用于训练自己的模型。

```
采集 (collect)  →  质检 (filter)  →  转格式 (convert)  →  训练 (train)
```

## 环境准备

### 1. 设置 API Key

```bash
export KIMI_API_KEY="sk-..."
export KIMI_BASE_URL="https://api.moonshot.cn/v1"
```

### 2. 安装 MCP 服务器

```bash
# 一键安装全部 28 个服务器
bash mcp_servers/install.sh

# 或手动安装单个 Python 服务器（示例）
cd mcp_servers/unit-converter-mcp && pip3 install -e .
cd mcp_servers/time-mcp && pip3 install -e .

# Node.js 服务器
cd mcp_servers/metmuseum-mcp && npm install
```

23 个服务器不需要额外 API key，5 个需要（Google Maps、BioMCP、Hugging Face、National Parks、NASA Data）。

### 3. 安装 Python 依赖

```bash
pip3 install openai aiohttp mcp json-repair jsonschema pydantic pyyaml
```

## Step 1: 采集数据

### 方式 A：跑单个任务

```bash
python3 run_openclaw_benchmark.py \
  --task-id unit_converter_000 \
  --tasks-file tasks/mcpbench_tasks_single_runner_format.json \
  --trajectory-dir trajectories_openclaw \
  --max-rounds 30 \
  --no-eval
```

### 方式 B：批量跑多个任务

```bash
# 跑单服务器任务（前 10 个）
python3 run_openclaw_benchmark.py \
  --tasks-file tasks/mcpbench_tasks_single_runner_format.json \
  --trajectory-dir trajectories_openclaw \
  --max-rounds 30 \
  --max-tasks 10 \
  --no-eval

# 跑双服务器任务
python3 run_openclaw_benchmark.py \
  --tasks-file tasks/mcpbench_tasks_multi_2server_runner_format.json \
  --trajectory-dir trajectories_openclaw \
  --max-rounds 30 \
  --max-tasks 5 \
  --no-eval
```

### 方式 C：使用 pipeline 脚本

```bash
bash pipeline.sh collect-single unit_converter_000 tasks/mcpbench_tasks_single_runner_format.json
bash pipeline.sh collect-batch tasks/mcpbench_tasks_single_runner_format.json 10
```

### 采集输出

每个任务生成一个轨迹文件 `trajectories_openclaw/traj_<task_id>_<timestamp>.json`，包含：

```json
{
  "task_id": "unit_converter_000",
  "model": "kimi-k2.5",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "任务描述..."},
    {"role": "assistant", "content": null, "tool_calls": [...], "reasoning_content": "思考过程..."},
    {"role": "tool", "tool_call_id": "...", "content": "工具返回结果..."},
    ...
    {"role": "assistant", "content": "最终回答..."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
  ],
  "reasoning_traces": [...],
  "metadata": {...}
}
```

### Kimi K2.5 限制注意事项

- TPD：150 万 token/天
- RPM：20 请求/分钟
- 并发：3
- 必须设置 temperature=1
- 是 thinking model，返回 reasoning_content

agent 内置了限速逻辑（每次 LLM 调用间隔 ≥ 3.5 秒），不需要额外处理。

## Step 2: 质检过滤

### Stage 1：规则检查（免费，秒级完成）

```bash
python3 tools/filter_trajectories.py \
  --input-dir trajectories_openclaw \
  --output-dir training_data \
  --rules-only
```

检查 7 项硬指标：

| 检查项 | 规则 | 说明 |
|---|---|---|
| has_tool_calls | total_tool_calls > 0 | 至少调用了一次工具 |
| success_rate | success_rate >= 0.7 | 工具调用成功率 ≥ 70% |
| has_final_answer | has_final_answer == True | 有最终回答（不是中途断了） |
| final_answer_quality | final_answer_length >= 100 | 回答够长（不是空的或太短） |
| not_looping | total_rounds <= 20 | 没有死循环 |
| has_tools_def | tools definition present | 有工具定义（SFT 格式需要） |
| calls_results_match | tool_calls ≈ tool_results | 调用数和结果数匹配 |

### Stage 2：LLM Judge 评分（花 token）

```bash
python3 tools/filter_trajectories.py \
  --input-dir trajectories_openclaw \
  --output-dir training_data \
  --min-judge-score 4.0
```

只对通过 Stage 1 的轨迹，用 LLM 打分：

| 维度 | 评估什么 |
|---|---|
| Task Fulfillment | 任务完成了多少 |
| Grounding | 回答是否基于工具返回的真实数据 |
| Tool Appropriateness | 选的工具对不对 |
| Parameter Accuracy | 参数传得准不准 |
| Dependency Awareness | 有没有理解工具间的依赖 |
| Parallelism/Efficiency | 是否高效 |

4 项核心分数（Fulfillment + Grounding + ToolApp + ParamAcc）的平均值 ≥ `min-judge-score` 才通过。

### 质检输出

```
training_data/filter_results.json        # Stage 1 结果
training_data/final_filter_results.json  # Stage 1 + Stage 2 结果
```

## Step 3: 转为 SFT 训练格式

```bash
python3 tools/convert_to_sft.py \
  --input-dir trajectories_openclaw \
  --output training_data/sft_data.jsonl \
  --reasoning strip
```

三种 reasoning 处理模式：

| 模式 | 参数 | 说明 |
|---|---|---|
| strip | `--reasoning strip` | 移除 reasoning_content（纯 OpenAI 格式） |
| keep | `--reasoning keep` | 保留 reasoning_content 字段（Kimi 格式） |
| to_content | `--reasoning to_content` | 合并为 `<think>...</think>` 块 |

### 输出格式

`training_data/sft_data.jsonl`（每行一个样本）：

```json
{
  "messages": [
    {"role": "system", "content": "You are a capable AI assistant..."},
    {"role": "user", "content": "帮我转换14个传感器..."},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_0", "type": "function", "function": {"name": "Unit_Converter__convert_temperature", "arguments": "{\"value\":350,\"from_unit\":\"fahrenheit\",\"to_unit\":\"celsius\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_0", "content": "{\"converted_value\":176.67}"},
    {"role": "assistant", "content": "最终报告..."}
  ],
  "tools": [
    {"type": "function", "function": {"name": "Unit_Converter__convert_temperature", "description": "[Unit Converter] Convert temperature between units.", "parameters": {"type": "object", "properties": {...}}}}
  ]
}
```

`training_data/sft_data_with_reasoning.jsonl`（带 thinking）：

```json
{"role": "assistant", "content": "<think>\n用户有14个传感器读数需要转换...\n</think>", "tool_calls": [...]}
```

## 一键执行

```bash
# 全部三步
bash pipeline.sh all tasks/mcpbench_tasks_single_runner_format.json 10
```

## 任务文件

| 文件 | 类型 | 任务数 | 可直接跑 |
|---|---|---|---|
| `tasks/mcpbench_tasks_single_runner_format.json` | 单服务器 | 56 | 46 |
| `tasks/mcpbench_tasks_multi_2server_runner_format.json` | 双服务器 | 30 | 18 |
| `tasks/mcpbench_tasks_multi_3server_runner_format.json` | 三服务器 | 18 | 6 |
| **合计** | | **104** | **70** |

## OpenClaw vs TaskExecutor

本项目提供两种 agent 执行方式：

| | TaskExecutor（MCP-Bench 内置） | OpenClaw（本项目新增） |
|---|---|---|
| 工具调用 | 文本 prompt → JSON → 解析 | 原生 function_calling API |
| 多轮上下文 | 每轮重新拼 prompt（无状态） | 连续对话（有状态） |
| 训练数据格式 | 自定义，需后处理 | 标准 OpenAI messages，直接可训练 |
| 适用场景 | 评测 benchmark | 采数据训练模型 |

## 文件结构

```
├── run_openclaw_benchmark.py   # OpenClaw 采集入口
├── pipeline.sh                 # 一键工作流脚本
├── agent/
│   ├── openclaw_agent.py       # OpenClaw agent（function_calling）
│   ├── base_executor.py        # 抽象执行器接口
│   └── executor.py             # 原 TaskExecutor
├── tools/
│   ├── filter_trajectories.py  # 两阶段质检
│   └── convert_to_sft.py       # 转 OpenAI SFT 格式
├── trajectories_openclaw/      # 采集的原始轨迹
├── training_data/
│   ├── sft_data.jsonl          # 标准 OpenAI SFT 格式
│   ├── sft_data_with_reasoning.jsonl  # 带 <think> 块
│   ├── filter_results.json     # Stage 1 质检结果
│   └── final_filter_results.json      # Stage 1+2 质检结果
├── tasks/                      # MCP-Bench 任务定义
└── mcp_servers/                # 28 个 MCP 服务器
```
