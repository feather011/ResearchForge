"""
PlanNode — 研究计划节点

调用 Planner 将主题拆解为多个子问题
"""

from ..core.planner import Planner, LLMProvider


def run_plan_node(topic: str, llm: LLMProvider) -> list:
    """
    制定研究计划，返回子问题列表

    原理：
    1. 用 LLM 把用户主题拆成 3-5 个子问题
    2. 每个子问题后续会被 SearchNode 单独搜索
    """
    planner = Planner(llm=llm, max_steps=5)
    plan = planner.plan(topic)

    questions = [step.description for step in plan.steps]
    return questions
