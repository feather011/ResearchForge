"""
Planner - 任务规划器

职责：
1. 接收复杂任务
2. 分解为多个可执行步骤
3. 返回步骤列表

与ReAct的关系：
- Planner是战略层（制定计划）
- ReAct是战术层（执行单个步骤）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
import logging

from .react_engine import LLMProvider

logger = logging.getLogger("Planner")


@dataclass
class Step:
    """执行步骤"""
    step_number: int
    description: str        # 步骤描述
    action: str             # 要执行的行动
    action_input: Dict      # 行动参数
    status: str = "pending" # pending/running/completed/failed
    result: Any = None      # 执行结果
    error: Optional[str] = None  # 错误信息


@dataclass
class Plan:
    """执行计划"""
    query: str              # 原始查询
    steps: List[Step]       # 步骤列表
    current_step: int = 0   # 当前执行到第几步
    status: str = "pending" # pending/running/completed/failed


class Planner:
    """
    任务规划器

    职责：
    1. 分析复杂任务
    2. 分解为多个可执行步骤
    3. 返回执行计划
    """

    def __init__(self, llm: LLMProvider, max_steps: int = 5):
        self.llm = llm
        self.max_steps = max_steps

    def _build_prompt(self, query: str) -> str:
        """构建Prompt"""
        return f"""你是一个任务规划器。根据用户查询，分解为多个独立的搜索查询。

## 用户查询
{query}

## 输出要求
请以JSON格式返回，包含以下字段：
- reasoning: 你的推理过程
- steps: 步骤列表，每个步骤包含：
  - description: 可以直接输入搜索引擎的搜索短语（不要带"搜索"、"分析"、"研究"等动词前缀）
  - action: 要执行的行动（固定为 web_search）
  - action_input: 行动参数（JSON对象，query字段放搜索短语）

注意：
1. 最多{self.max_steps}个步骤
2. 每个步骤是独立的搜索查询
3. description 必须是纯搜索短语，例如：用户问"文艺复兴" → description="文艺复兴 起源 背景 时间"

请直接返回JSON，不要添加其他内容："""

    def _parse_llm_output(self, output: str) -> Optional[List[Step]]:
        """解析LLM输出为步骤列表"""
        try:
            # 提取JSON
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                logger.warning("未找到JSON格式的输出")
                return None

            json_str = output[json_start:json_end]
            data = json.loads(json_str)

            # 提取步骤
            steps_data = data.get("steps", [])
            if not steps_data:
                logger.warning("未找到步骤列表")
                return None

            steps = []
            for i, step_data in enumerate(steps_data[:self.max_steps], 1):
                step = Step(
                    step_number=i,
                    description=step_data.get("description", ""),
                    action=step_data.get("action", ""),
                    action_input=step_data.get("action_input", {})
                )
                steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"解析LLM输出失败: {e}")
            return None

    def _generate_fallback_plan(self, query: str) -> List[Step]:
        """生成降级计划（当LLM失败时）"""
        return [
            Step(
                step_number=1,
                description=f"搜索: {query}",
                action="web_search",
                action_input={"query": query}
            ),
            Step(
                step_number=2,
                description="生成答案",
                action="finish",
                action_input={"answer": f"关于'{query}'的研究结果"}
            )
        ]

    def plan(self, query: str) -> Plan:
        """
        制定执行计划

        流程：
        1. 构建Prompt
        2. 调用LLM
        3. 解析输出为步骤列表
        4. 返回Plan
        """
        logger.info(f"制定计划: {query}")

        # 构建Prompt
        prompt = self._build_prompt(query)

        # 调用LLM（带重试）
        for attempt in range(3):
            try:
                llm_output = self.llm.generate(prompt)

                # 解析输出
                steps = self._parse_llm_output(llm_output)
                if steps:
                    plan = Plan(query=query, steps=steps)
                    logger.info(f"计划制定成功: {len(steps)}个步骤")
                    return plan

            except Exception as e:
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/3): {e}")

        # 全部失败，使用降级方案
        logger.info("使用降级方案")
        steps = self._generate_fallback_plan(query)
        return Plan(query=query, steps=steps)
