"""
限流器 - 基于内存的滑动窗口
（可选替换为 Redis）
"""

import time
from collections import defaultdict, deque
from typing import Dict


class RateLimiter:
    """滑动窗口限流器"""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.records: Dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> bool:
        """检查是否允许请求"""
        now = time.time()
        window_start = now - self.window_seconds

        # 清理过期记录
        while self.records[key] and self.records[key][0] < window_start:
            self.records[key].popleft()

        # 检查是否超限
        if len(self.records[key]) >= self.max_requests:
            return False

        # 记录新请求
        self.records[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """获取剩余配额"""
        now = time.time()
        window_start = now - self.window_seconds

        while self.records[key] and self.records[key][0] < window_start:
            self.records[key].popleft()

        return self.max_requests - len(self.records[key])
