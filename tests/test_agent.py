import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tracepilot.core.context import AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class TestAgent(unittest.TestCase):

    def test_agent_state_initialization(self):
        """测试 AgentState 的初始化"""
        from tracepilot.core.context import AgentState

        initial_state = AgentState(
            messages=[],
            summary=""
        )

        self.assertEqual(initial_state["messages"], [])
        self.assertEqual(initial_state["summary"], "")

    @patch('tracepilot.core.provider.get_provider')
    @patch('tracepilot.core.skill_loader.load_dynamic_skills')
    @patch('tracepilot.core.tools.builtins.BUILTIN_TOOLS', [])
    @patch.dict(os.environ, {"TRACEPILOT_MCP_ENABLED": "false"}, clear=False)
    def test_create_agent_app_basic(self, mock_load_skills, mock_get_provider):
        """测试创建基础代理应用（带 Mock）"""
        from tracepilot.core.agent import create_agent_app

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        try:
            app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
            self.assertIsNotNone(app)
        except Exception as e:
            # 即使出现其他错误也记录
            print(f"Unexpected error: {e}")
            raise

    @patch('tracepilot.core.provider.get_provider')
    @patch('tracepilot.core.skill_loader.load_dynamic_skills')
    @patch('tracepilot.core.tools.builtins.BUILTIN_TOOLS', [])
    @patch.dict(os.environ, {"TRACEPILOT_MCP_ENABLED": "false"}, clear=False)
    def test_create_agent_app_with_custom_tools(self, mock_load_skills, mock_get_provider):
        """测试创建带有自定义工具的代理应用（带 Mock）"""
        from tracepilot.core.agent import create_agent_app
        from langchain_core.tools import tool

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        # 创建一个真正的 mock 工具（使用@tool 装饰器）
        @tool
        def mock_tool(test_param: str) -> str:
            """A mock tool for testing"""
            return f"mock result: {test_param}"

        try:
            with patch('tracepilot.core.agent.load_mcp_tools') as mock_load_mcp_tools:
                app = create_agent_app(
                    provider_name="openai",
                    model_name="gpt-4o-mini",
                    tools=[mock_tool]
                )
                mock_load_mcp_tools.assert_not_called()
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @patch('tracepilot.core.provider.get_provider')
    @patch('tracepilot.core.skill_loader.load_dynamic_skills')
    @patch('tracepilot.core.tools.builtins.BUILTIN_TOOLS', [])
    @patch.dict(os.environ, {"TRACEPILOT_MCP_ENABLED": "false"}, clear=False)
    def test_create_agent_app_with_checkpointer(self, mock_load_skills, mock_get_provider):
        """测试创建带有检查点的代理应用（带 Mock）"""
        from tracepilot.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        memory_saver = MemorySaver()
        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                checkpointer=memory_saver
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


if __name__ == '__main__':
    unittest.main()
