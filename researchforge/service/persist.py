"""
轻量文件持久化

把已完成/失败的研究记录到 JSON 文件，重启不丢
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _safe_path(research_id: str) -> Path:
    """研究记录文件路径"""
    return DATA_DIR / f"{research_id}.json"


def save_task(research_id: str, task: dict):
    """保存研究记录到文件（排除 agent 对象等不可序列化字段）"""
    record = {
        "research_id": research_id,
        "topic": task.get("topic", ""),
        "state": task.get("state", ""),
        "intervention_id": task.get("intervention_id"),
        "report": task.get("report"),
        "agent": None,  # 不序列化 agent 对象
        "events": task.get("events", []),
    }

    # 已有的报告/结果
    agent = task.get("agent")
    if agent and hasattr(agent, "graph") and agent.graph and agent.graph.context:
        ctx = agent.graph.context
        record["results"] = ctx.results
        record["final_output"] = ctx.final_output
    else:
        record["results"] = task.get("results", [])
        record["final_output"] = task.get("report", "")

    filepath = _safe_path(research_id)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_task(research_id: str) -> Optional[dict]:
    """从文件加载研究记录"""
    filepath = _safe_path(research_id)
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_tasks(limit: int = 20) -> List[dict]:
    """列出所有已持久化的研究记录，按修改时间倒序"""
    files = sorted(DATA_DIR.glob("*.json"), key=os.path.getmtime, reverse=True)
    results = []
    for f in files[:limit]:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                record = json.load(fp)
                results.append({
                    "research_id": record.get("research_id", f.stem),
                    "topic": record.get("topic", ""),
                    "state": record.get("state", ""),
                    "time": os.path.getmtime(f),
                })
        except:
            pass
    return results
