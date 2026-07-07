# TracePilot

基于 LangGraph 构建的个人AI agent学习项目。TracePilot 是一个本地 AI Agent 工作台，围绕长上下文膨胀、记忆丢失与工具调用黑箱等问题，提供上下文压缩、记忆持久化、安全执行和核心链路审计能力。


## 核心能力

- 上下文压缩与记忆管理：实现大工具结果落盘、旧回合删除、历史工具输出压缩三层流水线，并提供 fallback transcript 持久化兜底，控制长对话 token 膨胀。
- 短期记忆：使用 SQLite 持久化多轮对话状态，支持会话恢复与连续推理。
- 长期记忆：通过 Markdown 用户画像维护跨对话记忆，沉淀用户偏好和稳定约束。
- 安全执行：通过系统提示、路径校验、危险命令拦截和超时熔断限制本地文件与命令执行范围，工具调用时采用两阶段调用
- 审计监控：使用 JSONL 记录模型输入、工具调用、工具结果和系统动作，并通过 Rich 终端展示 Agent 行为链路。
- 后台任务：基于 asyncio 和任务队列支持单次任务以及周期任务，到期后向 Agent 注入系统提醒。
- 技能与 MCP 扩展：扫描 SKILL.md 动态注册技能，支持懒加载、缓存热更新，并可将外部 MCP Server 工具转换为 Agent 可调用工具。

## 项目结构

```text
TracePilot/
├── tracepilot/            # 核心包
├── entry/                 # CLI、主程序和监控入口
├── tests/                 # 核心测试
├── examples/              # 示例脚本
├── docs/                  # 补充设计文档
├── setup.py
└── requirements.txt
```

## 核心模块

- Agent 循环：基于 LangGraph 组织模型推理、工具调用和状态回写。
- 上下文管理：负责工具结果落盘、消息压缩、旧消息删除和摘要兜底。
- 记忆系统：长期画像存放在 workspace/memory/user_profile.md，短期状态存放在 workspace/state.sqlite3。
- 沙盒工具：将文件读写和 Shell 执行限制在 workspace/office 内。
- 审计日志：记录 llm_input、tool_call、tool_result、ai_message、system_action 等事件。
- 任务系统：持久化任务队列并由心跳循环检查触发。
- 技能加载：扫描 workspace/office/skills 中的 SKILL.md，并按需加载技能说明和执行入口。

## 快速开始

请先激活你选择的虚拟环境，确认安装位置后再安装依赖，避免写入全局 Python。

```cmd
python -m pip install -e .
tracepilot config
tracepilot run
```

监控终端：

```cmd
tracepilot monitor
```

## 配置

复制 .env.example 为 .env 后配置模型提供商、模型名和 API Key。工作区默认位于项目根目录的 workspace 目录，也可以通过 TRACEPILOT_WORKSPACE 覆盖。

MCP 默认关闭。需要接入外部 MCP Server 时，设置 TRACEPILOT_MCP_ENABLED=true，并配置 TRACEPILOT_MCP_SERVER_COMMAND、TRACEPILOT_MCP_SERVER_ARGS 等参数。

## 测试

```cmd
python -m compileall tracepilot entry tests examples
python -m pytest tests -q
```


