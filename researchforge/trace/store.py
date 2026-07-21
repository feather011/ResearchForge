"""
TraceStore — JSONL 持久化的 Trace 存储

以 JSONL 格式（每行一个 JSON 对象）追加写入 trace 事件。
线程安全，支持多 Worker 并发记录。
"""

import json
import os
import threading
from pathlib import Path
from typing import List, Optional

# 存储根目录（相对 researchforge/trace/store.py → service/data/traces/）
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "service" / "data" / "traces"


class TraceStore:
    """
    Trace 存储：JSONL 文件的追加写入和读取。

    用法:
        store = TraceStore()
        store.append(event_dict)       # 追加一条
        events = store.load(task_id)   # 读取全部（损坏行跳过）
        store.delete(task_id)          # 删除
    """

    def __init__(self, store_dir: Optional[Path] = None):
        self._dir = Path(store_dir) if store_dir else _DEFAULT_DIR
        self._lock = threading.Lock()

    def _path(self, task_id: str) -> Path:
        """Trace 文件路径"""
        if not task_id:
            raise ValueError("task_id 不能为空")
        # 防止路径遍历
        safe = Path(task_id).name
        return self._dir / f"{safe}.jsonl"

    def append(self, task_id: str, event_dict: dict):
        """追加写入一条 Trace 事件（线程安全）"""
        if not task_id:
            return
        with self._lock:
            p = self._path(task_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(event_dict, ensure_ascii=False) + "\n"
            # 追加写入（UTF-8）
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)

    def load(self, task_id: str) -> List[dict]:
        """
        读取指定 task_id 的全部 Trace 事件。

        损坏的行会被跳过，不影响其他事件读取。
        返回按写入顺序排列的事件列表。
        """
        p = self._path(task_id)
        if not p.exists():
            return []
        events: List[dict] = []
        with self._lock:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # 损坏行跳过
                        continue
        return events

    def delete(self, task_id: str) -> bool:
        """删除指定 task_id 的 Trace 文件"""
        p = self._path(task_id)
        if p.exists():
            with self._lock:
                p.unlink()
            return True
        return False

    def count(self, task_id: str) -> int:
        """返回指定 task_id 的 Trace 事件数量"""
        return len(self.load(task_id))
