"""
CheckpointStore — 研究任务检查点存储

提供 ResearchState 的持久化保存与恢复，每次保存为原子写入。
存储路径: service/data/checkpoints/{task_id}.json

用法:
    store = CheckpointStore()
    store.save(state)          # 原子写入
    state = store.load(tid)    # 读取（不存在返回 None）
    ok = store.exists(tid)     # 是否存在
    store.delete(tid)          # 删除
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .research_state import ResearchState

# 检查点根目录（相对项目根 researchforge/../service/data/checkpoints）
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "service" / "data" / "checkpoints"


class CheckpointStore:
    """检查点存储 —— 原子写入的 ResearchState 持久化"""

    def __init__(self, store_dir: Optional[Path] = None):
        self._dir = Path(store_dir) if store_dir else _DEFAULT_DIR

    # ── 路径 ──

    def _path(self, task_id: str) -> Path:
        """返回 {task_id}.json 的完整路径（防止路径穿越）"""
        # 替换文件系统不允许或危险的字符
        safe = task_id.replace("/", "_").replace("\\", "_")
        # resolve() 后检查是否仍在 store_dir 内（防穿越）
        dest = (self._dir / f"{safe}.json").resolve()
        if not str(dest).startswith(str(self._dir.resolve())):
            raise ValueError(f"路径穿越被阻止: {task_id}")
        return dest

    def _ensure_dir(self):
        """确保存储目录存在"""
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── 核心方法 ──

    def save(self, state: ResearchState) -> str:
        """
        原子保存检查点。

        1. 序列化 state 为 JSON
        2. 写入临时文件（与目标文件同目录）
        3. 临时文件 → 正式文件（原子替换）
        4. 返回 task_id

        写入中途崩溃 → 临时文件残留，正式文件不变。
        """
        task_id = state.task_id
        if not task_id:
            raise ValueError("ResearchState.task_id 为空，无法保存")

        self._ensure_dir()
        dest = self._path(task_id)

        # 写入临时文件
        tmp_path = None
        try:
            # 临时文件的 prefix 也需要安全化（不含路径特殊字符）
            safe_prefix = task_id.replace("/", "_").replace("\\", "_").replace("..", "_")
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix=f"{safe_prefix}_",
                dir=str(self._dir),
            )
            os.close(fd)

            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(state.to_json(ensure_ascii=False, indent=2))

            # 原子替换（Windows 上 replace 是原子的，目标存在时覆盖）
            shutil.move(tmp_path, str(dest))
        except Exception:
            # 清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            raise

        return task_id

    def load(self, task_id: str) -> Optional[ResearchState]:
        """
        加载检查点。

        返回 ResearchState，如果文件不存在或损坏返回 None。
        """
        dest = self._path(task_id)
        if not dest.exists():
            return None

        try:
            return ResearchState.from_file(str(dest))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, Exception):
            # 损坏文件 → 返回 None
            return None

    def exists(self, task_id: str) -> bool:
        """检查检查点是否存在"""
        return self._path(task_id).exists()

    def delete(self, task_id: str) -> bool:
        """
        删除检查点。

        返回: True 已删除, False 文件不存在
        """
        dest = self._path(task_id)
        if not dest.exists():
            return False
        dest.unlink()
        return True

    # ── 管理 ──

    def list_ids(self) -> list:
        """列出所有检查点 task_id（按文件名排序）"""
        if not self._dir.exists():
            return []
        return sorted(
            p.stem for p in self._dir.iterdir()
            if p.suffix == ".json" and not p.name.startswith(".")
        )

    def count(self) -> int:
        """检查点数量"""
        return len(self.list_ids())

    def clear(self):
        """清空全部检查点（慎用）"""
        if self._dir.exists():
            shutil.rmtree(str(self._dir))
