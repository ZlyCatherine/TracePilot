import builtins
import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracepilot.core.mcp_client import (
    GENERIC_ARGUMENTS_FIELD,
    MCPServerConfig,
    MCPStdioClient,
    _select_process_errlog,
    create_args_model,
    create_mcp_tool,
    format_mcp_call_result,
    load_mcp_tools,
    normalize_mcp_name,
    shutdown_mcp_runtime,
)


class FakeMCPToolSpec:
    name = "search"
    description = "Search the web"
    inputSchema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Maximum results"},
        },
        "required": ["query"],
    }


class FakeMCPManager:
    def __init__(self):
        self.calls = []

    def call_tool(self, server_name, tool_name, arguments):
        self.calls.append((server_name, tool_name, arguments))
        return "ok"


class FakeValidErrlog:
    def fileno(self):
        return 2


class FakeInvalidErrlog:
    def fileno(self):
        raise OSError("fileno unavailable")


class TestMCPClient(unittest.TestCase):
    def tearDown(self):
        shutdown_mcp_runtime()

    def test_load_mcp_tools_disabled_by_default(self):
        with patch.dict(os.environ, {"TRACEPILOT_MCP_ENABLED": "false"}, clear=False):
            self.assertEqual(load_mcp_tools(), [])

    def test_load_mcp_tools_enabled_without_command_returns_empty(self):
        with patch.dict(
            os.environ,
            {
                "TRACEPILOT_MCP_ENABLED": "true",
                "TRACEPILOT_MCP_SERVER_COMMAND": "",
            },
            clear=False,
        ):
            self.assertEqual(load_mcp_tools(), [])

    def test_sdk_missing_degrades_to_empty_tools(self):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mcp":
                raise ImportError("missing mcp")
            return real_import(name, *args, **kwargs)

        with patch.dict(
            os.environ,
            {
                "TRACEPILOT_MCP_ENABLED": "true",
                "TRACEPILOT_MCP_SERVER_COMMAND": "fake-command",
            },
            clear=False,
        ), patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(load_mcp_tools(), [])

    def test_normalize_mcp_name(self):
        self.assertEqual(normalize_mcp_name("web search"), "web_search")
        self.assertEqual(normalize_mcp_name("docs/search"), "docs_search")
        self.assertEqual(normalize_mcp_name(""), "mcp")

    def test_select_process_errlog_prefers_real_stderr(self):
        valid_errlog = FakeValidErrlog()
        with patch.object(sys, "__stderr__", valid_errlog), patch.object(sys, "stderr", FakeInvalidErrlog()):
            self.assertIs(_select_process_errlog(), valid_errlog)

    def test_select_process_errlog_returns_none_when_fileno_unusable(self):
        with patch.object(sys, "__stderr__", FakeInvalidErrlog()), patch.object(sys, "stderr", io.StringIO()):
            self.assertIsNone(_select_process_errlog())

    def test_mcp_stdio_client_does_not_use_stringio_errlog(self):
        config = MCPServerConfig(
            name="tavily",
            command="npx",
            args=["-y", "tavily-mcp@latest"],
            env={},
        )

        client = MCPStdioClient(config)

        self.assertFalse(isinstance(client._errlog, io.StringIO))

    def test_simple_json_schema_to_args_model(self):
        model = create_args_model("web_search", "search", FakeMCPToolSpec.inputSchema)

        instance = model(query="tracepilot", max_results=3)

        self.assertEqual(instance.query, "tracepilot")
        self.assertEqual(instance.max_results, 3)
        self.assertIn("query", model.model_fields)
        self.assertTrue(model.model_fields["query"].is_required())

    def test_complex_json_schema_falls_back_to_arguments_dict(self):
        schema = {
            "type": "object",
            "properties": {
                "max-results": {"type": "integer"},
            },
        }

        model = create_args_model("web_search", "search", schema)

        self.assertIn(GENERIC_ARGUMENTS_FIELD, model.model_fields)
        instance = model(arguments={"max-results": 3})
        self.assertEqual(instance.arguments, {"max-results": 3})

    def test_mcp_tool_wrapper_forwards_arguments(self):
        manager = FakeMCPManager()
        tool = create_mcp_tool(manager, "web search", FakeMCPToolSpec())

        result = tool.invoke({"query": "tracepilot", "max_results": 2})

        self.assertEqual(result, "ok")
        self.assertEqual(tool.name, "mcp__web_search__search")
        self.assertEqual(
            manager.calls,
            [("web_search", "search", {"query": "tracepilot", "max_results": 2})],
        )

    def test_generic_mcp_tool_wrapper_forwards_raw_arguments(self):
        class ComplexToolSpec:
            name = "search"
            description = "Search"
            inputSchema = {
                "type": "object",
                "properties": {
                    "max-results": {"type": "integer"},
                },
            }

        manager = FakeMCPManager()
        tool = create_mcp_tool(manager, "web_search", ComplexToolSpec())

        result = tool.invoke({"arguments": {"max-results": 2}})

        self.assertEqual(result, "ok")
        self.assertEqual(manager.calls, [("web_search", "search", {"max-results": 2})])

    def test_format_mcp_call_result_text_and_structured_content(self):
        from mcp.types import CallToolResult, TextContent

        result = CallToolResult(
            content=[TextContent(type="text", text="hello")],
            structuredContent={"url": "https://example.com"},
        )

        formatted = format_mcp_call_result(result)

        self.assertIn("hello", formatted)
        self.assertIn("https://example.com", formatted)


if __name__ == "__main__":
    unittest.main()
