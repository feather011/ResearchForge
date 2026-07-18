"""
RetryPolicy 单元测试
"""

import pytest
from researchforge.orchestration.retry_policy import RetryPolicy


class TestRetryPolicy:
    """测试基础配置"""

    def test_default_values(self):
        """默认值：max_retries=2, base_delay=1.0"""
        p = RetryPolicy()
        assert p.max_retries == 2
        assert p.base_delay == 1.0

    def test_custom_max_retries(self):
        """自定义全局最大重试次数"""
        p = RetryPolicy(max_retries=5)
        assert p.max_retries == 5


class TestShouldRetry:
    """测试 should_retry 判断逻辑"""

    # ── 可重试异常（白名单） ──

    def test_timeout_error_is_retryable(self):
        """TimeoutError → 可重试"""
        p = RetryPolicy()
        assert p.should_retry("searching", TimeoutError("连接超时"), attempt=1)

    def test_connection_error_is_retryable(self):
        """ConnectionError → 可重试"""
        p = RetryPolicy()
        assert p.should_retry("fetching", ConnectionError("连接被拒绝"), attempt=1)

    def test_os_error_is_retryable(self):
        """OSError → 可重试（网络类系统错误）"""
        p = RetryPolicy()
        assert p.should_retry("fetching", OSError("远端断开"), attempt=1)

    # ── 不可重试异常（白名单模式） ──

    def test_runtime_error_not_retryable(self):
        """RuntimeError → 不重试（业务逻辑错误，不自动兜底）"""
        p = RetryPolicy()
        assert not p.should_retry("synthesizing", RuntimeError("LLM返回空"), attempt=1)

    def test_value_error_not_retryable(self):
        """ValueError → 不重试"""
        p = RetryPolicy()
        assert not p.should_retry("planning", ValueError("无效的配置项"), attempt=1)

    def test_type_error_not_retryable(self):
        """TypeError → 不重试"""
        p = RetryPolicy()
        assert not p.should_retry("synthesizing", TypeError("类型不匹配"), attempt=1)

    def test_name_error_not_retryable(self):
        """NameError → 不重试"""
        p = RetryPolicy()
        assert not p.should_retry("writing", NameError("未定义的变量"), attempt=1)

    def test_syntax_error_not_retryable(self):
        """SyntaxError → 不重试"""
        p = RetryPolicy()
        assert not p.should_retry("extracting", SyntaxError("语法错误"), attempt=1)

    def test_assertion_error_not_retryable(self):
        """AssertionError → 不重试"""
        p = RetryPolicy()
        assert not p.should_retry("searching", AssertionError("断言失败"), attempt=1)

    def test_generic_exception_not_retryable(self):
        """普通 Exception → 不重试（白名单外）"""
        p = RetryPolicy()
        assert not p.should_retry("writing", Exception("未知错误"), attempt=1)

    # ── 超过最大次数 ──

    def test_exceed_max_retries(self):
        """超过全局最大重试次数 → 不重试"""
        p = RetryPolicy(max_retries=2)
        assert p.should_retry("searching", TimeoutError(""), attempt=1)
        assert p.should_retry("searching", TimeoutError(""), attempt=2)
        assert not p.should_retry("searching", TimeoutError(""), attempt=3)

    def test_exceed_with_custom_retries(self):
        """超过节点单独配置的最大重试次数 → 不重试"""
        p = RetryPolicy(max_retries=2, node_config={"synthesizing": 3})
        # RuntimeError 不可重试，换成 TimeoutError
        assert p.should_retry("synthesizing", TimeoutError(""), attempt=1)
        assert p.should_retry("synthesizing", TimeoutError(""), attempt=2)
        assert p.should_retry("synthesizing", TimeoutError(""), attempt=3)
        assert not p.should_retry("synthesizing", TimeoutError(""), attempt=4)


class TestExtraRetryable:
    """测试额外可重试异常配置"""

    def test_extra_retryable_works(self):
        """通过 extra_retryable 额外添加可重试类型"""
        class MyCustomError(Exception):
            pass
        p = RetryPolicy(extra_retryable=(MyCustomError,))
        assert p.should_retry("searching", MyCustomError("自定义错误"), attempt=1)

    def test_extra_retryable_does_not_affect_unlisted(self):
        """extra_retryable 添加后，其他未列出的异常仍不重试"""
        class ListedError(Exception):
            pass
        class UnlistedError(Exception):
            pass
        p = RetryPolicy(extra_retryable=(ListedError,))
        assert p.should_retry("searching", ListedError(""), attempt=1)
        assert not p.should_retry("searching", UnlistedError(""), attempt=1)

    def test_extra_retryable_cannot_override_non_retryable(self):
        """extra_retryable 添加的异常如果也在 NON_RETRYABLE 中 → 不重试"""
        p = RetryPolicy(extra_retryable=(ValueError,))
        # ValueError 同时在 NON_RETRYABLE 和 extra_retryable 中 → NON_RETRYABLE 优先级高
        assert not p.should_retry("planning", ValueError(""), attempt=1)


class TestNodeConfig:
    """测试节点独立配置"""

    def test_node_config_override(self):
        """节点配置覆盖全局默认值"""
        p = RetryPolicy(max_retries=2, node_config={"searching": 1, "synthesizing": 3})
        # searching: 最多 1 次重试
        assert p.should_retry("searching", TimeoutError(""), attempt=1)
        assert not p.should_retry("searching", TimeoutError(""), attempt=2)
        # synthesizing: 最多 3 次重试
        assert p.should_retry("synthesizing", TimeoutError(""), attempt=1)
        assert p.should_retry("synthesizing", TimeoutError(""), attempt=2)
        assert p.should_retry("synthesizing", TimeoutError(""), attempt=3)
        assert not p.should_retry("synthesizing", TimeoutError(""), attempt=4)

    def test_unconfigured_node_uses_global(self):
        """未单独配置的节点使用全局默认值"""
        p = RetryPolicy(max_retries=3, node_config={"searching": 5})
        assert p.should_retry("writing", TimeoutError(""), attempt=3)
        assert not p.should_retry("writing", TimeoutError(""), attempt=4)

    def test_empty_node_config(self):
        """空节点配置不影响全局"""
        p = RetryPolicy(max_retries=2)
        assert p.should_retry("any_node", TimeoutError(""), attempt=2)
        assert not p.should_retry("any_node", TimeoutError(""), attempt=3)


class TestGetDelay:
    """测试退避时间计算"""

    def test_exponential_backoff_attempt_1(self):
        """第1次重试等待 base_delay 秒"""
        p = RetryPolicy(base_delay=1.0)
        assert p.get_delay(1) == 1.0

    def test_exponential_backoff_attempt_2(self):
        """第2次重试等待 base * 2 = 2 秒"""
        p = RetryPolicy(base_delay=1.0)
        assert p.get_delay(2) == 2.0

    def test_exponential_backoff_attempt_3(self):
        """第3次重试等待 base * 4 = 4 秒"""
        p = RetryPolicy(base_delay=1.0)
        assert p.get_delay(3) == 4.0

    def test_exponential_backoff_custom_base(self):
        """自定义基数"""
        p = RetryPolicy(base_delay=0.5)
        assert p.get_delay(1) == 0.5
        assert p.get_delay(2) == 1.0
        assert p.get_delay(3) == 2.0

    def test_delay_capped_by_max_delay(self):
        """延迟时间受 max_delay 上限限制"""
        p = RetryPolicy(base_delay=10.0, max_delay=30.0)
        # 10 * 2^(4-1) = 80 → 限制为 30
        assert p.get_delay(4) == 30.0


class TestCombined:
    """测试综合场景"""

    def test_transient_failure_then_retry(self):
        """瞬态故障 → 在限制次数内重试"""
        p = RetryPolicy(max_retries=3)
        for attempt in range(1, 5):
            if p.should_retry("searching", TimeoutError(""), attempt):
                delay = p.get_delay(attempt)
                assert delay > 0
            else:
                assert attempt == 4  # 第4次才放弃
                return
        pytest.fail("应第4次放弃")

    def test_config_error_never_retried(self):
        """配置错误 → 即使次数未用完也不重试"""
        p = RetryPolicy(max_retries=5)
        assert not p.should_retry("planning", ValueError("错误参数"), attempt=1)

    def test_node_combines_with_exception_check(self):
        """节点配置 + 异常类型 同时生效"""
        p = RetryPolicy(max_retries=2, node_config={"searching": 3})
        # ValueError 即使在 searching 上也不重试
        assert not p.should_retry("searching", ValueError("错误"), attempt=1)
        # TimeoutError 在 searching 上可以重试 3 次
        assert p.should_retry("searching", TimeoutError(""), attempt=3)
        assert not p.should_retry("searching", TimeoutError(""), attempt=4)

    def test_custom_retries_no_degradation(self):
        """配置自定义节点后全局默认值不受影响"""
        p = RetryPolicy(max_retries=2, base_delay=1.0, node_config={"searching": 5})
        assert p.should_retry("writing", TimeoutError(""), attempt=2)
        assert not p.should_retry("writing", TimeoutError(""), attempt=3)
        assert p.get_delay(2) == 2.0
