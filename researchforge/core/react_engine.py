"""
ReAct Agent 标准版

产品级错误处理：
- 系统自动处理所有错误
- 用户只看到正常结果或友好提示
- 错误日志只在后台记录
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import os
import logging
from collections import deque


# ==================== 日志配置 ====================

def setup_logger(name: str = "ReActAgent", log_file: str = "logs/agent.log") -> logging.Logger:
    """配置日志系统：控制台 + 文件输出"""
    import os

    # 创建日志目录
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler（支持轮转，最多保留 5MB × 3 个备份）
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file, encoding='utf-8',
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,  # 保留 3 个备份
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


# ==================== 错误类型 ====================

class ErrorType(Enum):
    """错误类型枚举"""
    LLM_ERROR = "llm_error"
    PARSE_ERROR = "parse_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    DEADLOCK = "deadlock"
    MAX_STEPS = "max_steps"


@dataclass
class AgentError:
    """Agent错误（内部使用）"""
    error_type: ErrorType
    message: str
    details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


# ==================== 用户友好提示 ====================

class UserMessages:
    """用户友好提示"""

    # 通用提示
    THINKING = "正在思考中..."
    SEARCHING = "正在搜索信息..."
    PROCESSING = "正在处理..."

    # 成功提示
    SUCCESS = "已完成研究"
    PARTIAL_SUCCESS = "已完成部分研究，以下是最新的结果"

    # 错误提示（用户看到的）
    SERVICE_UNAVAILABLE = "服务暂时不可用，请稍后再试"
    SEARCH_FAILED = "搜索暂时不可用，使用已有信息回答"
    PROCESSING_ERROR = "处理遇到问题，已使用备选方案"
    TIMEOUT = "处理超时，已使用已有结果"
    DEADLOCK = "检测到循环，已自动停止"
    MAX_STEPS = "已完成最大步骤，以下是当前结果"

    # 降级提示
    FALLBACK = "部分信息可能不完整，仅供参考"


# ==================== LLM Provider ====================

class LLMProvider(ABC):
    """LLM提供者基类"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class BailianProvider(LLMProvider):
    """百炼API提供者（DashScope）"""

    def __init__(self, api_key: str = None, model: str = "kimi-k2.6", timeout: float = 600.0):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("未找到环境变量 DASHSCOPE_API_KEY，请传入 api_key")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=self.timeout
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama 本地模型提供者"""

    def __init__(self, model: str = "qwen3.5:9b", base_url: str = "http://localhost:11434", timeout: float = 600.0):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        self.client = OpenAI(
            api_key="ollama",
            base_url=f"{base_url}/v1",
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=self.timeout,
        )
        return response.choices[0].message.content


# ==================== 数据结构 ====================

@dataclass
class Thought:
    """思考结果"""
    reasoning: str
    action: str
    action_input: Dict


@dataclass
class Observation:
    """观察结果"""
    result: Any
    success: bool
    error: Optional[str] = None
    duration: float = 0.0


@dataclass
class AgentStep:
    """执行步骤"""
    step_number: int
    thought: Thought
    observation: Observation


@dataclass
class AgentResult:
    """Agent执行结果（用户看到的）"""
    answer: str              # 最终答案
    message: str             # 用户友好提示
    success: bool            # 是否成功
    steps: List[AgentStep]   # 执行步骤
    stats: Dict[str, Any]    # 统计信息（后台使用）


# ==================== 工具基类 ====================

class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass


# ==================== ReAct Agent ====================

class ReActAgent:
    """
    ReAct Agent 核心引擎

    产品级错误处理：
    - 系统自动处理所有错误
    - 用户只看到正常结果或友好提示
    - 错误日志只在后台记录
    """

    def __init__(
        self,
        tools: List[BaseTool],
        llm: LLMProvider,
        max_steps: int = 5,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = llm
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.timeout = timeout

        # 死循环检测
        self.recent_actions: deque = deque(maxlen=3)

        # 内部统计（后台记录）
        self._stats = {
            "llm_calls": 0,
            "tool_calls": 0,
            "errors": 0,
            "retries": 0,
            "fallbacks": 0
        }

    def _build_prompt(self, query: str, history: List[AgentStep]) -> str:
        """构建Prompt"""

        tools_desc = "\n".join([
            f"- {tool.name}: {tool.description}"
            for tool in self.tools.values()
        ])

        history_desc = ""
        if history:
            history_desc = "\n".join([
                f"Step {s.step_number}: {s.thought.action}({s.thought.action_input}) -> {'成功' if s.observation.success else '失败'}"
                for s in history
            ])
            # 追加观察结果（工具返回的实际内容）
            obs_parts = []
            for s in history:
                if s.observation.success and s.observation.result:
                    text = str(s.observation.result)
                    obs_parts.append(f"Step {s.step_number} 结果: {text[:300]}")
            if obs_parts:
                history_desc += "\n" + "\n".join(obs_parts)
        else:
            history_desc = "无"

        return f"""你是一个智能助手。根据用户查询和历史信息，决定下一步行动。

## 可用工具
{tools_desc}

## 历史步骤
{history_desc}

## 用户查询
{query}

## 输出要求
请以JSON格式返回，包含以下字段：
- reasoning: 你的推理过程
- action: 工具名称（必须是上面列出的工具，或"finish"表示完成）
- action_input: 工具参数（JSON对象）

如果信息已经足够，请使用"finish"工具并给出答案。

请直接返回JSON，不要添加其他内容："""

    def _parse_llm_output(self, output: str) -> Optional[Thought]:
        """解析LLM输出，失败返回None"""
        try:
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return None

            json_str = output[json_start:json_end]
            data = json.loads(json_str)

            required_fields = ['reasoning', 'action', 'action_input']
            for field in required_fields:
                if field not in data:
                    return None

            return Thought(
                reasoning=data.get("reasoning", ""),
                action=data.get("action", ""),
                action_input=data.get("action_input", {})
            )
        except Exception:
            return None

    def _generate_fallback_thought(self, query: str, history: List[AgentStep]) -> Thought:
        """生成降级Thought（当LLM失败时）"""
        # 如果有历史步骤，尝试完成任务
        if history:
            # 收集所有成功的结果
            results = []
            for step in history:
                if step.observation.success and step.observation.result:
                    results.append(str(step.observation.result))

            if results:
                answer = f"根据已有信息：{'; '.join(results[:3])}"
            else:
                answer = f"关于'{query}'的研究暂时无法完成，请稍后再试"

            return Thought(
                reasoning="使用已有信息完成任务",
                action="finish",
                action_input={"answer": answer}
            )
        else:
            # 没有历史，尝试直接搜索
            return Thought(
                reasoning="尝试搜索信息",
                action="web_search",
                action_input={"query": query}
            )

    def think(self, query: str, history: List[AgentStep]) -> Thought:
        """
        思考阶段

        错误处理策略：
        1. 正常情况：调用LLM，解析输出
        2. LLM失败：自动重试
        3. 解析失败：自动重试
        4. 全部失败：使用降级方案
        """
        prompt = self._build_prompt(query, history)

        # 尝试调用LLM（带重试）
        for attempt in range(self.max_retries):
            try:
                self._stats["llm_calls"] += 1
                llm_output = self.llm.generate(prompt)

                # 尝试解析
                thought = self._parse_llm_output(llm_output)
                if thought:
                    return thought

                # 解析失败，记录日志，继续重试
                self._stats["errors"] += 1
                logger.warning(f"LLM输出解析失败 (尝试 {attempt + 1}/{self.max_retries})")

            except Exception as e:
                self._stats["errors"] += 1
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                self._stats["retries"] += 1

        # 全部失败，使用降级方案
        self._stats["fallbacks"] += 1
        logger.info("使用降级方案")
        return self._generate_fallback_thought(query, history)

    def act(self, thought: Thought) -> Observation:
        """
        执行阶段

        错误处理策略：
        1. 正常情况：执行工具
        2. 工具不存在：跳过，返回友好提示
        3. 工具失败：返回友好提示
        """
        start_time = time.time()

        # 完成指令
        if thought.action == "finish":
            return Observation(
                result=thought.action_input.get("answer", "完成"),
                success=True,
                duration=time.time() - start_time
            )

        # 查找工具
        tool = self.tools.get(thought.action)
        if not tool:
            # 工具不存在，跳过（不报错）
            self._stats["errors"] += 1
            logger.warning(f"工具不存在: {thought.action}，跳过")
            return Observation(
                result=None,
                success=False,
                duration=time.time() - start_time
            )

        # 执行工具
        try:
            self._stats["tool_calls"] += 1
            result = tool.run(**thought.action_input)
            return Observation(
                result=result,
                success=True,
                duration=time.time() - start_time
            )
        except Exception as e:
            # 工具失败，记录日志，返回失败（不报错）
            self._stats["errors"] += 1
            logger.warning(f"工具执行失败: {thought.action}, 错误: {e}")
            return Observation(
                result=None,
                success=False,
                duration=time.time() - start_time
            )

    def observe(self, observation: Observation, thought: Thought) -> bool:
        """
        观察阶段

        错误处理策略：
        1. 完成任务：停止
        2. 工具失败：继续（尝试其他方案）
        3. 死循环：停止（使用已有结果）
        """
        # 完成任务
        if thought.action == "finish":
            return False

        # 死循环检测
        action_key = f"{thought.action}:{json.dumps(thought.action_input)}"
        self.recent_actions.append(action_key)
        if len(self.recent_actions) >= 3:
            recent = list(self.recent_actions)
            if recent[-1] == recent[-2] == recent[-3]:
                logger.warning("检测到死循环，停止执行")
                return False

        # 工具失败也继续（尝试其他方案）
        return True

    def _extract_answer(self, history: List[AgentStep]) -> str:
        """从历史中提取答案"""
        # 优先使用finish的答案
        for step in reversed(history):
            if step.thought.action == "finish":
                return step.thought.action_input.get("answer", "")

        # 没有finish，收集成功的结果
        results = []
        for step in history:
            if step.observation.success and step.observation.result:
                results.append(str(step.observation.result))

        if results:
            return f"根据研究：{'; '.join(results[:3])}"

        return None

    def _get_user_message(self, history: List[AgentStep], has_answer: bool) -> str:
        """获取用户友好提示"""
        if has_answer:
            # 检查是否有失败的步骤
            failed_steps = [s for s in history if not s.observation.success]
            if failed_steps:
                return UserMessages.PARTIAL_SUCCESS
            return UserMessages.SUCCESS

        # 没有答案
        if len(history) >= self.max_steps:
            return UserMessages.MAX_STEPS

        # 检查是否有死循环
        if len(self.recent_actions) >= 3:
            recent = list(self.recent_actions)
            if recent[-1] == recent[-2] == recent[-3]:
                return UserMessages.DEADLOCK

        return UserMessages.FALLBACK

    def run(self, query: str, tracer: Optional["TraceCollector"] = None) -> AgentResult:
        """
        主循环

        产品级错误处理：
        - 系统自动处理所有错误
        - 用户只看到正常结果或友好提示
        - 错误日志只在后台记录

        tracer: 可选 TraceCollector，记录 thought/action/observation
        """
        logger.info(f"开始执行: {query}")

        history = []
        answer = None

        # 主循环
        for step_num in range(1, self.max_steps + 1):
            logger.info(f"Step {step_num}/{self.max_steps}")

            # Think（自动处理LLM错误）
            thought = self.think(query, history)
            logger.info(f"思考: {thought.reasoning}")
            logger.info(f"行动: {thought.action}")

            if tracer:
                tracer.record(
                    agent_name="ReActAgent",
                    stage="think",
                    thought=thought.reasoning,
                    action=thought.action,
                    input=str(thought.action_input)[:500],
                )

            # Act（自动处理工具错误）
            observation = self.act(thought)
            logger.info(f"结果: {'成功' if observation.success else '失败'}")

            if tracer:
                obs_duration = int((time.time() - (observation.duration if hasattr(observation, 'duration') else time.time())) * 1000)
                tracer.record(
                    agent_name="ReActAgent",
                    stage="action",
                    action=thought.action,
                    tool_name=thought.action,
                    input=str(thought.action_input)[:500],
                    observation=str(observation.result)[:500] if observation.result else "失败",
                    duration_ms=observation.duration * 1000,
                )

            # 记录步骤
            step = AgentStep(step_num, thought, observation)
            history.append(step)

            # Observe（自动处理死循环）
            if not self.observe(observation, thought):
                if tracer and thought.action == "finish":
                    tracer.record(
                        agent_name="ReActAgent",
                        stage="observation",
                        action="finish",
                        result=str(observation.result)[:500],
                    )
                break

        # 提取答案
        answer = self._extract_answer(history)

        # 获取用户友好提示
        message = self._get_user_message(history, answer is not None)

        # 构建统计信息
        stats = {
            **self._stats,
            "total_steps": len(history),
            "total_duration": sum(s.observation.duration for s in history)
        }

        logger.info(f"执行完成: steps={len(history)}, has_answer={answer is not None}")

        # 返回结果（用户看到的）
        return AgentResult(
            answer=answer or "暂时无法获取完整信息，请稍后再试",
            message=message,
            success=answer is not None,
            steps=history,
            stats=stats
        )
