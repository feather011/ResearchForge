"""ReAct Agent 测试"""

import pytest
from researchforge.core.react_engine import (
    ReActAgent, BaseTool, LLMProvider, Thought, Observation, AgentResult
)
from researchforge.tools import get_all_tools


class MockLLM(LLMProvider):
    """模拟LLM，用于测试"""

    def generate(self, prompt: str) -> str:
        if "历史步骤" in prompt and "无" in prompt:
            return '{"reasoning": "需要搜索信息", "action": "web_search", "action_input": {"query": "test"}}'
        else:
            return '{"reasoning": "信息已足够", "action": "finish", "action_input": {"answer": "测试答案"}}'


class FailingLLM(LLMProvider):
    """总是失败的LLM，用于测试降级"""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM调用失败")


class TestReActAgent:
    """测试 ReAct Agent"""

    def test_agent_creation(self):
        """测试创建Agent"""
        tools = get_all_tools()
        llm = MockLLM()
        agent = ReActAgent(tools=tools, llm=llm, max_steps=3)

        assert agent.max_steps == 3
        assert len(agent.tools) == len(tools)

    def test_agent_run_success(self):
        """测试Agent成功执行"""
        tools = get_all_tools()
        llm = MockLLM()
        agent = ReActAgent(tools=tools, llm=llm, max_steps=3)

        result = agent.run("测试查询")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert len(result.steps) > 0
        assert result.answer is not None
        assert result.message != ""  # 有用户友好提示

    def test_tool_not_found_graceful(self):
        """测试工具不存在时的优雅处理"""
        tools = get_all_tools()
        llm = MockLLM()
        agent = ReActAgent(tools=tools, llm=llm, max_steps=3)

        thought = Thought(
            reasoning="测试",
            action="nonexistent_tool",
            action_input={}
        )
        observation = agent.act(thought)

        # 工具不存在时应该返回失败，但不报错
        assert observation.success is False

    def test_finish_action(self):
        """测试完成动作"""
        tools = get_all_tools()
        llm = MockLLM()
        agent = ReActAgent(tools=tools, llm=llm)

        thought = Thought(
            reasoning="完成",
            action="finish",
            action_input={"answer": "测试答案"}
        )
        observation = agent.act(thought)

        assert observation.success is True
        assert observation.result == "测试答案"

    def test_llm_failure_fallback(self):
        """测试LLM失败时的降级处理"""
        tools = get_all_tools()
        llm = FailingLLM()  # 总是失败的LLM
        agent = ReActAgent(tools=tools, llm=llm, max_steps=3)

        result = agent.run("测试查询")

        # LLM失败时应该使用降级方案，不报错
        assert isinstance(result, AgentResult)
        assert result.answer is not None  # 有答案（降级方案）
        assert result.message != ""  # 有用户友好提示

    def test_error_handling_stats(self):
        """测试错误处理统计"""
        tools = get_all_tools()
        llm = MockLLM()
        agent = ReActAgent(tools=tools, llm=llm, max_steps=3)

        result = agent.run("测试查询")

        # 验证统计信息
        assert "llm_calls" in result.stats
        assert "tool_calls" in result.stats
        assert "errors" in result.stats
        assert "fallbacks" in result.stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
