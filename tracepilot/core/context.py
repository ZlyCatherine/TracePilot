import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages

from .config import WORKSPACE_DIR


CONTEXT_DIR = os.path.join(WORKSPACE_DIR, "context")
TOOL_RESULTS_DIR = os.path.join(CONTEXT_DIR, "tool-results")
TRANSCRIPTS_DIR = os.path.join(CONTEXT_DIR, "transcripts")
DEFAULT_CONTEXT_MAX_CHARS = 50000
KEEP_RECENT_TOOL_RESULTS = 3
FALLBACK_RECENT_MESSAGES = 5
PERSISTED_OUTPUT_MARKER = "<persisted-output>"
SHORTENED_TOOL_OUTPUT = "早期工具输出已压缩，如需详情请重新运行相关工具或查看落盘文件。"

# 上下文压缩总览：
# 1. 大工具结果落盘：把超长工具消息写入本地文件，只保留路径和预览。
# 2. 旧回合删除：按完整用户回合删除最早对话，避免破坏工具调用链。
# 3. 历史工具输出压缩：保留最近工具结果，将更早的工具输出替换为短提示。
# 兜底策略：三层处理后仍超限时，将完整历史写成转录文件，再只保留最近消息。
# 维护指引：新增压缩策略时，优先保持消息编号、工具调用编号和回合边界稳定。


@dataclass
class ContextCompressionResult:
    final_messages: list[BaseMessage]
    discarded_messages: list[BaseMessage] = field(default_factory=list)
    replacement_messages: list[BaseMessage] = field(default_factory=list)
    delete_messages: list[BaseMessage] = field(default_factory=list)
    transcript_path: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


# 构建agent节点
class AgentState(TypedDict):
    # 存储对话历史
    messages: Annotated[list[BaseMessage], add_messages]

    # 摘要压缩
    summary: str


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return safe[:80] or "default"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", type(message).__name__),
        "id": getattr(message, "id", None),
        "name": getattr(message, "name", None),
        "content": getattr(message, "content", ""),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
    }
    if hasattr(message, "tool_calls"):
        payload["tool_calls"] = getattr(message, "tool_calls", None)
    if hasattr(message, "invalid_tool_calls"):
        payload["invalid_tool_calls"] = getattr(message, "invalid_tool_calls", None)
    if isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        payload["status"] = message.status
        payload["artifact"] = message.artifact
    return payload


def estimate_context_chars(messages: list[BaseMessage]) -> int:
    total = 0
    for message in messages:
        try:
            total += len(json.dumps(_message_payload(message), ensure_ascii=False, default=str))
        except Exception:
            total += len(str(message))
    return total


def _has_persisted_marker(message: BaseMessage) -> bool:
    return PERSISTED_OUTPUT_MARKER in _content_to_text(getattr(message, "content", ""))


def _persist_tool_result(message: ToolMessage, thread_id: str, content: str) -> str:
    os.makedirs(TOOL_RESULTS_DIR, exist_ok=True)
    msg_key = getattr(message, "id", None) or message.tool_call_id or "tool"
    filename = f"{int(time.time() * 1000)}_{_safe_filename(thread_id)}_{_safe_filename(str(msg_key))}.txt"
    path = os.path.join(TOOL_RESULTS_DIR, filename)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return path


def _replace_message_content(message: BaseMessage, content: str) -> BaseMessage:
    return message.model_copy(update={"content": content})


def _apply_l3_persist_large_results(
        messages: list[BaseMessage],
        thread_id: str,
        max_chars: int,
        stats: dict[str, Any]
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    # 第三层：大工具结果落盘。
    # 适用场景：工具返回日志、文件内容、搜索结果等超长文本，直接塞回模型会快速撑大上下文。
    # 处理方式：把完整输出写入工具结果目录，只在消息中保留文件路径、原长度和预览。
    # 维护指引：替换消息时必须保留原消息编号和工具调用编号，否则状态回写时无法正确删除/更新消息。
    if estimate_context_chars(messages) <= max_chars:
        return messages, []

    compacted = list(messages)
    candidates: list[tuple[int, ToolMessage, int, str]] = []
    for idx, message in enumerate(compacted):
        if not isinstance(message, ToolMessage) or _has_persisted_marker(message):
            continue
        content = _content_to_text(message.content)
        # 只落盘足够长的输出，确保保留 2000 字预览后仍能明显降低上下文体积。
        if len(content) > 2000:
            candidates.append((idx, message, len(content), content))

    candidates.sort(key=lambda item: item[2], reverse=True)
    replacements: list[BaseMessage] = []

    for idx, message, content_len, content in candidates:
        if estimate_context_chars(compacted) <= max_chars:
            break
        path = _persist_tool_result(message, thread_id, content)
        preview = content[:2000]
        replacement_content = (
            f"{PERSISTED_OUTPUT_MARKER}\n"
            f"Full output: {path}\n"
            f"Original chars: {content_len}\n"
            f"Preview:\n{preview}\n"
            f"</persisted-output>"
        )
        replacement = _replace_message_content(message, replacement_content)
        compacted[idx] = replacement
        replacements.append(replacement)

    stats["l3_persisted"] = len(replacements)
    return compacted, replacements


def _split_turns(messages: list[BaseMessage]) -> tuple[SystemMessage | None, list[list[BaseMessage]]]:
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    for message in non_system_msgs:
        if isinstance(message, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
        else:
            if current_turn:
                current_turn.append(message)
            else:
                turns.append([message])

    if current_turn:
        turns.append(current_turn)

    return first_system, turns


def _flatten_turns(first_system: SystemMessage | None, turns: list[list[BaseMessage]]) -> list[BaseMessage]:
    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    for turn in turns:
        final_messages.extend(turn)
    return final_messages


def _apply_l1_delete_old_turns(
        messages: list[BaseMessage],
        max_chars: int,
        stats: dict[str, Any]
) -> tuple[list[BaseMessage], list[BaseMessage], list[BaseMessage]]:
    # 第一层：旧回合删除。
    # 适用场景：大工具结果落盘后仍超过预算，说明历史对话本身已经过长。
    # 处理方式：以用户消息开始的一整个用户回合为单位，从最早回合开始删除。
    # 维护指引：不要按单条消息随意裁剪，否则可能留下缺少前置工具调用的工具结果。
    if estimate_context_chars(messages) <= max_chars:
        return messages, [], []

    first_system, turns = _split_turns(messages)
    if not turns:
        return messages, [], []

    remaining_turns = list(turns)
    discarded_turns: list[list[BaseMessage]] = []
    while len(remaining_turns) > 1:
        candidate_remaining = remaining_turns[1:]
        candidate_messages = _flatten_turns(first_system, candidate_remaining)
        discarded_turns.append(remaining_turns[0])
        remaining_turns = candidate_remaining
        if estimate_context_chars(candidate_messages) <= max_chars:
            break

    if not discarded_turns:
        return messages, [], []

    final_messages = _flatten_turns(first_system, remaining_turns)
    discarded_messages = [message for turn in discarded_turns for message in turn]
    delete_messages = [message for message in discarded_messages if getattr(message, "id", None)]
    stats["l1_deleted"] = len(discarded_messages)
    return final_messages, discarded_messages, delete_messages


def _apply_l2_shorten_old_tool_outputs(
        messages: list[BaseMessage],
        max_chars: int,
        stats: dict[str, Any]
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    # 第二层：历史工具输出压缩。
    # 适用场景：旧回合删除后仍超限，且上下文中存在较早的工具结果。
    # 处理方式：保留最近几个工具结果，将更早的长工具输出替换为固定短提示。
    # 维护指引：最近工具结果保留数量由常量控制；调小会节省上下文，调大会保留更多现场信息。
    if estimate_context_chars(messages) <= max_chars:
        return messages, []

    compacted = list(messages)
    tool_indices = [
        idx for idx, message in enumerate(compacted)
        if isinstance(message, ToolMessage) and not _has_persisted_marker(message)
    ]
    if len(tool_indices) <= KEEP_RECENT_TOOL_RESULTS:
        return messages, []

    preserve = set(tool_indices[-KEEP_RECENT_TOOL_RESULTS:])
    replacements: list[BaseMessage] = []
    for idx in tool_indices:
        if idx in preserve:
            continue
        message = compacted[idx]
        if len(_content_to_text(message.content)) <= 120:
            continue
        replacement = _replace_message_content(message, SHORTENED_TOOL_OUTPUT)
        compacted[idx] = replacement
        replacements.append(replacement)
        if estimate_context_chars(compacted) <= max_chars:
            break

    stats["l2_shortened"] = len(replacements)
    return compacted, replacements


def _ai_message_has_tool_call(ai_message: BaseMessage, tool_call_id: str) -> bool:
    tool_calls = getattr(ai_message, "tool_calls", None) or []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and tool_call.get("id") == tool_call_id:
            return True

    raw_tool_calls = getattr(ai_message, "additional_kwargs", {}).get("tool_calls", [])
    for tool_call in raw_tool_calls:
        if isinstance(tool_call, dict) and tool_call.get("id") == tool_call_id:
            return True
    return False


def _select_recent_messages_with_tool_context(
        messages: list[BaseMessage],
        limit: int = FALLBACK_RECENT_MESSAGES
) -> list[BaseMessage]:
    # 兜底保留策略：默认保留最近若干条消息，同时补齐工具结果对应的模型工具调用。
    # 维护指引：如果最近窗口内存在孤立工具结果，优先回找它的调用来源；找不到来源就丢弃该工具结果。
    if len(messages) <= limit:
        return list(messages)

    selected_indices = set(range(max(0, len(messages) - limit), len(messages)))
    unsupported_tool_indices: set[int] = set()

    for idx in sorted(selected_indices):
        message = messages[idx]
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = message.tool_call_id
        has_support = any(
            earlier_idx < idx and isinstance(messages[earlier_idx], AIMessage)
            and _ai_message_has_tool_call(messages[earlier_idx], tool_call_id)
            for earlier_idx in selected_indices
        )
        if has_support:
            continue
        support_idx = None
        for earlier_idx in range(idx - 1, -1, -1):
            earlier = messages[earlier_idx]
            if isinstance(earlier, AIMessage) and _ai_message_has_tool_call(earlier, tool_call_id):
                support_idx = earlier_idx
                break
        if support_idx is not None:
            selected_indices.add(support_idx)
        else:
            unsupported_tool_indices.add(idx)

    selected_indices.difference_update(unsupported_tool_indices)
    return [messages[idx] for idx in sorted(selected_indices)]


def _write_transcript(messages: list[BaseMessage], thread_id: str) -> str:
    # 兜底历史转录持久化：把完整原始消息序列逐行写入文件，供审计、排障或人工恢复使用。
    # 维护指引：这里写的是原始消息列表，保留压缩前全量信息；不要改成压缩中的临时消息列表。
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    filename = f"transcript_{int(time.time() * 1000)}_{_safe_filename(thread_id)}.jsonl"
    path = os.path.join(TRANSCRIPTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for message in messages:
            f.write(json.dumps(_message_payload(message), ensure_ascii=False, default=str) + "\n")
    return path


def _apply_fallback_compact(
        original_messages: list[BaseMessage],
        current_messages: list[BaseMessage],
        thread_id: str,
        stats: dict[str, Any]
) -> tuple[list[BaseMessage], list[BaseMessage], list[BaseMessage], str]:
    # 兜底压缩：三层压缩后仍超预算时触发。
    # 处理方式：先把完整历史写入转录文件，再保留最近消息与必要的工具调用上下文。
    # 维护指引：返回的转录文件路径会写入日志和摘要提示，后续排障时可从该路径找回完整历史。
    transcript_path = _write_transcript(original_messages, thread_id)
    final_messages = _select_recent_messages_with_tool_context(current_messages)
    final_ids = {message.id for message in final_messages if getattr(message, "id", None)}
    discarded_messages = [
        message for message in original_messages
        if not getattr(message, "id", None) or message.id not in final_ids
    ]
    delete_messages = [message for message in discarded_messages if getattr(message, "id", None)]
    stats["fallback_compact"] = 1
    stats["fallback_kept"] = len(final_messages)
    return final_messages, discarded_messages, delete_messages, transcript_path


def compact_context_messages(
        messages: list[BaseMessage],
        thread_id: str = "default",
        max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
) -> ContextCompressionResult:
    # 三层上下文压缩流水线入口。
    # 执行顺序：大工具结果落盘 -> 旧回合删除 -> 历史工具输出压缩 -> 兜底历史转录持久化。
    # 返回值同时包含最终消息、待替换消息、待删除消息和统计信息，智能体节点会据此回写图状态。
    stats: dict[str, Any] = {
        "max_chars": max_chars,
        "original_chars": estimate_context_chars(messages),
        "l3_persisted": 0,
        "l1_deleted": 0,
        "l2_shortened": 0,
        "fallback_compact": 0,
    }
    final_messages = list(messages)
    replacement_messages: list[BaseMessage] = []
    discarded_messages: list[BaseMessage] = []
    delete_messages: list[BaseMessage] = []
    transcript_path: str | None = None

    final_messages, l3_replacements = _apply_l3_persist_large_results(
        final_messages, thread_id, max_chars, stats
    )
    replacement_messages.extend(l3_replacements)

    final_messages, l1_discarded, l1_delete = _apply_l1_delete_old_turns(
        final_messages, max_chars, stats
    )
    discarded_messages.extend(l1_discarded)
    delete_messages.extend(l1_delete)

    final_messages, l2_replacements = _apply_l2_shorten_old_tool_outputs(
        final_messages, max_chars, stats
    )
    replacement_messages.extend(l2_replacements)

    if estimate_context_chars(final_messages) > max_chars:
        final_messages, fallback_discarded, fallback_delete, transcript_path = _apply_fallback_compact(
            list(messages), final_messages, thread_id, stats
        )
        discarded_messages = fallback_discarded
        delete_messages = fallback_delete
        final_ids = {message.id for message in final_messages if getattr(message, "id", None)}
        replacement_messages = [
            message for message in replacement_messages
            if getattr(message, "id", None) in final_ids
        ]

    stats["final_chars"] = estimate_context_chars(final_messages)
    stats["replacement_messages"] = len(replacement_messages)
    stats["delete_messages"] = len(delete_messages)
    stats["discarded_messages"] = len(discarded_messages)

    return ContextCompressionResult(
        final_messages=final_messages,
        discarded_messages=discarded_messages,
        replacement_messages=replacement_messages,
        delete_messages=delete_messages,
        transcript_path=transcript_path,
        stats=stats
    )


def trim_context_messages(messages: list[BaseMessage], trigger_turns: int = 8, keep_turns: int = 4) -> tuple[
    list[BaseMessage], list[BaseMessage]]:
    # 按完整用户回合裁剪上下文：一个回合从用户消息开始，到下一个用户消息前结束，会一并保留模型回复、工具调用和工具结果。
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    # 遍历非系统信息，按回合进行分组
    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)

    # 最后回合还在构建，此时先把它追加进去
    if current_turn:
        turns.append(current_turn)

    total_turns = len(turns)

    if total_turns < trigger_turns:
        final_messages = ([first_system] if first_system else []) + non_system_msgs
        return final_messages, []

    recent_turns = turns[-keep_turns:]
    discarded_turns = turns[:-keep_turns]

    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    for turn in recent_turns:
        final_messages.extend(turn)

    discarded_messages: list[BaseMessage] = []
    for turn in discarded_turns:
        discarded_messages.extend(turn)

    return final_messages, discarded_messages
