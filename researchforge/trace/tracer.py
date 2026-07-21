"""
TraceCollector — 轻量 Agent Trace 可观测

记录 Agent 执行过程中的 Thought/Action/Observation 和 Workflow 节点执行。
每条事件通过回调推送到 SSE、写入 TraceStore 和保存内存，不阻塞主流程。

数据结构:
  TraceEvent(timestamp, run_id, agent_name, stage,
             thought, action, tool_name, input,
             observation, result, duration_ms)
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Callable, TYPE_CHECKING


@dataclass
class TraceEvent:
    """Agent Trace 事件"""
    timestamp: float           # 事件时间（秒级时间戳）
    run_id: str                # 研究运行的 ID
    agent_name: str            # 哪个 Agent：ResearchGraph | ReActAgent | GapAgent | etc.
    stage: str                 # think | action | observation | node_start | node_end
    thought: str = ""          # 推理过程（think 阶段）
    action: str = ""           # 执行的动作：工具名 | 节点名 | "finish"
    tool_name: str = ""        # 工具名（action 阶段的工具调用）
    input: str = ""            # 输入摘要
    observation: str = ""      # 观察结果摘要（observation 阶段）
    result: str = ""           # 结果摘要（node_end / finish）
    duration_ms: float = 0.0   # 耗时（毫秒）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TraceCollector:
    """
    Trace 收集器

    不修改任何 Agent 的决策逻辑。
    通过 record() 方法在关键节点插入，自动回调到 SSE + 持久化。

    用法:
        tracer = TraceCollector(run_id="abc123", callback=my_sse_push)
        tracer.record(stage="node_start", agent_name="ResearchGraph",
                      action="PLANNING", input=topic)
        ... 执行 ...
        tracer.record(stage="node_end", agent_name="ResearchGraph",
                      action="PLANNING", result=questions)
    """

    def __init__(
        self,
        run_id: str,
        callback: Optional[Callable[[TraceEvent], None]] = None,
        store: Optional["TraceStore"] = None,
    ):
        self.run_id = run_id
        self.callback = callback
        self._store = store
        self._events: list = []
        self._start_times: Dict[str, float] = {}  # action → start_time

    def record(
        self,
        agent_name: str,
        stage: str,
        thought: str = "",
        action: str = "",
        tool_name: str = "",
        input: str = "",
        observation: str = "",
        result: str = "",
        duration_ms: float = 0.0,
    ) -> TraceEvent:
        """记录一条 Trace 事件（自动计算 node_start → node_end 耗时）"""
        _now = time.time()

        # 自动计时：node_start 记录开始时间，node_end 计算耗时
        if stage == "node_start" and action:
            self._start_times[action] = _now
        elif stage == "node_end" and action:
            started = self._start_times.pop(action, None)
            if started is not None and duration_ms == 0.0:
                duration_ms = round((_now - started) * 1000, 1)

        event = TraceEvent(
            timestamp=time.time(),
            run_id=self.run_id,
            agent_name=agent_name,
            stage=stage,
            thought=thought[:500],
            action=action[:200],
            tool_name=tool_name[:100],
            input=str(input)[:500],
            observation=str(observation)[:500],
            result=str(result)[:500],
            duration_ms=duration_ms,
        )

        event_dict = event.to_dict()
        self._events.append(event_dict)

        # TraceStore 持久化（可选）
        if self._store is not None:
            try:
                self._store.append(self.run_id, event_dict)
            except Exception:
                pass  # 存储失败不阻断主流程

        # SSE 回调（可选）
        if self.callback:
            try:
                self.callback(event)
            except Exception:
                pass  # 回调失败不阻断主流程

        return event

    def get_all(self) -> list:
        """获取所有记录的事件"""
        return self._events
