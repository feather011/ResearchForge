"""
RetryPolicy — 节点失败重试策略

统一判断每个节点失败后是否应该重试，以及等待多久后重试。

预置节点 Policy:
    searching_policy — 搜索节点，重试 TimeoutError / ConnectionError / OSError
    fetching_policy — 抓取节点，重试 TimeoutError / OSError
    llm_policy     — LLM 调用节点，额外重试 RuntimeError （LLM 偶发返回空可重试）
"""

from typing import Dict, Optional, Tuple, Type


class RetryPolicy:
    """
    节点失败重试策略

    用法:
        policy = RetryPolicy()
        for attempt in range(1, 10):
            if policy.should_retry("searching", exc, attempt):
                delay = policy.get_delay(attempt)
                time.sleep(delay)
                continue
            break

    规则:
        - 只重试网络/瞬态类异常：TimeoutError, ConnectionError, OSError
        - ValueError, TypeError, NameError → 不重试（配置/编码错误）
        - RuntimeError → 不重试（业务逻辑错误，不自动兜底）
        - 其他未知异常 → 不重试（白名单模式）
        - 超过最大重试次数 → 不重试
        - 节点可独立配置重试次数（如 synthesis 重试 3 次）
        - 可通过 extra_retryable 额外添加可重试异常类型
    """

    # 默认可重试的异常类型（白名单）
    RETRYABLE_EXCEPTIONS: tuple = (TimeoutError, ConnectionError, OSError)

    # 默认不可重试的异常类型
    NON_RETRYABLE_EXCEPTIONS: tuple = (
        ValueError, TypeError, NameError, AssertionError, SyntaxError, RuntimeError
    )

    def __init__(
        self,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        node_config: Optional[Dict[str, int]] = None,
        extra_retryable: Optional[Tuple[Type[BaseException], ...]] = None,
    ):
        """
        Args:
            max_retries: 全局默认最大重试次数
            base_delay: 指数退避的基数（秒）
            max_delay: 最大延迟（秒）
            node_config: 节点独立重试配置，如 {"synthesizing": 3, "searching": 1}
            extra_retryable: 额外可重试的异常类型，如 (IOError,)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._node_config = node_config or {}
        self._extra_retryable = extra_retryable or ()

    def _is_retryable_exception(self, exception: BaseException) -> bool:
        """判断异常是否可重试（白名单模式）"""
        if isinstance(exception, self.NON_RETRYABLE_EXCEPTIONS):
            return False
        if isinstance(exception, self.RETRYABLE_EXCEPTIONS):
            return True
        if isinstance(exception, self._extra_retryable):
            return True
        return False

    def should_retry(self, node_name: str, exception: BaseException, attempt: int) -> bool:
        """
        判断节点在指定失败次数后是否应重试

        Args:
            node_name: 节点名称（如 "searching", "synthesizing"）
            exception: 抛出的异常
            attempt: 当前是第几次尝试（从 1 开始）

        Returns:
            True 表示应该重试，False 表示放弃
        """
        max_retries = self._node_config.get(node_name, self.max_retries)
        if attempt > max_retries:
            return False
        return self._is_retryable_exception(exception)

    def get_delay(self, attempt: int) -> float:
        """
        计算第 attempt 次重试前应等待的秒数（指数退避）

        Args:
            attempt: 当前是第几次尝试（从 1 开始）

        Returns:
            延迟秒数
        """
        delay = self.base_delay * (2 ** (attempt - 1))
        return min(delay, self.max_delay)


# ==================== 预置节点 Policy ====================

# 搜索节点：重试网络超时/连接/OS 错误，最多 2 次
searching_policy = RetryPolicy(
    max_retries=2,
    base_delay=1.0,
    max_delay=60.0,
)

# 抓取节点：重试网络超时/OS 错误，最多 2 次
fetching_policy = RetryPolicy(
    max_retries=2,
    base_delay=1.0,
    max_delay=60.0,
)

# LLM 调用节点：额外重试 RuntimeError（LLM 偶发返回空可重试），最多 3 次
llm_policy = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    extra_retryable=(RuntimeError,),
)
# 注意：NON_RETRYABLE_EXCEPTIONS 包含 RuntimeError，
# extra_retryable 无法覆盖 NON_RETRYABLE 的判断（白名单模式下 NON_RETRYABLE 优先级更高）。
# 如果业务上需要 LLM 重试 RuntimeError，需将 RuntimeError 移出 NON_RETRYABLE_EXCEPTIONS。
