"""
FaultInjector — 可控故障注入，用于演示 Retry / Resume / Degradation

仅在 Mock Demo 中启用，默认关闭，不影响正式运行。
"""

import threading

# 全局故障规则表
_fault_rules = {}
_lock = threading.Lock()


class FaultInjector:
    """
    故障注入器。

    用法:
        injector = FaultInjector()
        injector.add_rule("searching", fail_count=1)
        injector.add_rule("deep_worker", worker_id="W2", fail_count=1)

    激活后，searching 节点第一次调用会抛出 TimeoutError，
    Deep Worker W2 第一次运行会抛出 RuntimeError。
    """

    def __init__(self):
        self._rules: dict = {}
        self._call_count: dict = {}
        self._active = False

    def reset(self):
        """清除所有规则和计数"""
        self._rules.clear()
        self._call_count.clear()
        self._active = False

    def activate(self):
        """启用故障注入"""
        self._active = True

    def deactivate(self):
        """停用故障注入"""
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def add_rule(self, node_name: str, fail_count: int = 1, worker_id: str = None):
        """
        添加故障规则。

        Args:
            node_name: 节点名称，如 "searching", "writing", "deep_worker"
            fail_count: 前几次调用失败
            worker_id: 仅对 deep_worker 生效，指定哪个 worker 失败
        """
        key = (node_name, worker_id or "*")
        self._rules[key] = fail_count
        self._call_count[key] = 0

    def should_fail(self, node_name: str, worker_id: str = None) -> bool:
        """检查当前调用是否应该失败"""
        if not self._active:
            return False

        # 先尝试精确匹配 (node, worker_id)
        for key in [(node_name, worker_id), (node_name, None), (node_name, "*")]:
            if key in self._rules:
                with _lock:
                    self._call_count[key] = self._call_count.get(key, 0) + 1
                    count = self._call_count[key]
                return count <= self._rules[key]

        return False


# 全局单例
_injector = FaultInjector()


def get_fault_injector() -> FaultInjector:
    """获取全局故障注入器"""
    return _injector


def with_fault_tolerance(node_name: str, worker_id: str = None):
    """
    在节点调用前检查是否应注入故障。
    如果应注入，抛出 TimeoutError。
    """
    inj = get_fault_injector()
    if inj.should_fail(node_name, worker_id):
        import time
        time.sleep(0.1)  # 模拟卡顿
        raise TimeoutError(f"[FaultInjector] 模拟 {node_name} 故障")
