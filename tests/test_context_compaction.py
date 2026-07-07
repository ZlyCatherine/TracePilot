import os
import re
import shutil
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import tracepilot.core.context as context
from tracepilot.core.context import (
    PERSISTED_OUTPUT_MARKER,
    SHORTENED_TOOL_OUTPUT,
    compact_context_messages,
    estimate_context_chars,
)


class ContextRuntimeDirsMixin:
    def setUp(self):
        self.tmpdir = os.path.join(os.getcwd(), "workspace", "context-test-runtime")
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.makedirs(self.tmpdir, exist_ok=True)
        self.old_tool_results = context.TOOL_RESULTS_DIR
        self.old_transcripts = context.TRANSCRIPTS_DIR
        context.TOOL_RESULTS_DIR = os.path.join(self.tmpdir, "tool-results")
        context.TRANSCRIPTS_DIR = os.path.join(self.tmpdir, "transcripts")

    def tearDown(self):
        context.TOOL_RESULTS_DIR = self.old_tool_results
        context.TRANSCRIPTS_DIR = self.old_transcripts
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestContextCompaction(ContextRuntimeDirsMixin, unittest.TestCase):

    def test_under_limit_does_not_compact(self):
        messages = [
            HumanMessage(content="你好", id="h1"),
            AIMessage(content="你好，有什么可以帮你？", id="a1"),
        ]

        result = compact_context_messages(messages, thread_id="test", max_chars=50000)

        self.assertEqual(result.final_messages, messages)
        self.assertEqual(result.discarded_messages, [])
        self.assertEqual(result.replacement_messages, [])
        self.assertEqual(result.delete_messages, [])
        self.assertEqual(result.stats["l3_persisted"], 0)
        self.assertEqual(result.stats["l1_deleted"], 0)
        self.assertEqual(result.stats["l2_shortened"], 0)
        self.assertEqual(result.stats["fallback_compact"], 0)

    def test_l3_persists_large_tool_result_with_same_id(self):
        messages = [
            HumanMessage(content="运行工具", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{"name": "reader", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="X" * 10000, tool_call_id="tc1", id="t1", name="reader"),
        ]

        result = compact_context_messages(messages, thread_id="thread 1", max_chars=5000)

        self.assertEqual(len(result.replacement_messages), 1)
        replacement = result.replacement_messages[0]
        self.assertEqual(replacement.id, "t1")
        self.assertEqual(replacement.tool_call_id, "tc1")
        self.assertEqual(replacement.name, "reader")
        self.assertIn(PERSISTED_OUTPUT_MARKER, replacement.content)
        match = re.search(r"Full output: (.+)", replacement.content)
        self.assertIsNotNone(match)
        self.assertTrue(os.path.exists(match.group(1).strip()))
        self.assertLessEqual(estimate_context_chars(result.final_messages), 5000)

    def test_l1_deletes_complete_old_turns(self):
        messages = [
            HumanMessage(content="旧问题1" + "A" * 1200, id="h1"),
            AIMessage(content="旧回答1" + "B" * 1200, id="a1"),
            HumanMessage(content="旧问题2" + "C" * 1200, id="h2"),
            AIMessage(content="旧回答2" + "D" * 1200, id="a2"),
            HumanMessage(content="新问题", id="h3"),
            AIMessage(content="新回答", id="a3"),
        ]

        result = compact_context_messages(messages, thread_id="test", max_chars=1800)

        final_ids = [msg.id for msg in result.final_messages]
        discarded_ids = [msg.id for msg in result.discarded_messages]
        delete_ids = [msg.id for msg in result.delete_messages]

        self.assertEqual(final_ids, ["h3", "a3"])
        self.assertEqual(discarded_ids, ["h1", "a1", "h2", "a2"])
        self.assertEqual(delete_ids, ["h1", "a1", "h2", "a2"])
        self.assertEqual(result.stats["l1_deleted"], 4)

    def test_l2_shortens_old_tool_outputs_and_keeps_recent_three(self):
        messages = [
            HumanMessage(content="同一回合内多个工具输出", id="h1"),
            AIMessage(content="处理中", id="a1"),
            ToolMessage(content="tool0-" + "x" * 900, tool_call_id="tc0", id="t0"),
            ToolMessage(content="tool1-" + "x" * 900, tool_call_id="tc1", id="t1"),
            ToolMessage(content="tool2-" + "x" * 900, tool_call_id="tc2", id="t2"),
            ToolMessage(content="tool3-" + "x" * 900, tool_call_id="tc3", id="t3"),
            ToolMessage(content="tool4-" + "x" * 900, tool_call_id="tc4", id="t4"),
        ]

        result = compact_context_messages(messages, thread_id="test", max_chars=4500)

        contents = {msg.id: msg.content for msg in result.final_messages if isinstance(msg, ToolMessage)}
        self.assertEqual(contents["t0"], SHORTENED_TOOL_OUTPUT)
        self.assertEqual(contents["t1"], SHORTENED_TOOL_OUTPUT)
        self.assertTrue(contents["t2"].startswith("tool2-"))
        self.assertTrue(contents["t3"].startswith("tool3-"))
        self.assertTrue(contents["t4"].startswith("tool4-"))
        self.assertEqual(result.stats["l2_shortened"], 2)

    def test_fallback_saves_transcript_and_keeps_tool_call_support(self):
        messages = [
            HumanMessage(content="无法再删除的超长当前回合" + "A" * 60000, id="h1"),
            AIMessage(
                content="需要工具",
                id="a1",
                tool_calls=[{"name": "reader", "args": {}, "id": "tc1"}],
            ),
            ToolMessage(content="短工具结果", tool_call_id="tc1", id="t1", name="reader"),
            AIMessage(content="后续2", id="a2"),
            AIMessage(content="后续3", id="a3"),
            AIMessage(content="后续4", id="a4"),
            AIMessage(content="后续5", id="a5"),
        ]

        result = compact_context_messages(messages, thread_id="fallback", max_chars=1000)

        self.assertEqual(result.stats["fallback_compact"], 1)
        self.assertIsNotNone(result.transcript_path)
        self.assertTrue(os.path.exists(result.transcript_path))
        self.assertEqual([msg.id for msg in result.final_messages], ["a1", "t1", "a2", "a3", "a4", "a5"])
        self.assertIn("h1", [msg.id for msg in result.delete_messages])

    def test_delete_messages_only_include_messages_with_ids(self):
        messages = [
            HumanMessage(content="无 id 的旧消息" + "A" * 3000),
            AIMessage(content="无 id 的旧回答" + "B" * 3000),
            HumanMessage(content="新消息", id="h2"),
            AIMessage(content="新回答", id="a2"),
        ]

        result = compact_context_messages(messages, thread_id="no-id", max_chars=1000)

        self.assertTrue(result.discarded_messages)
        self.assertTrue(all(msg.id for msg in result.delete_messages))


if __name__ == "__main__":
    unittest.main()
