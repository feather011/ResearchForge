"""
前端 Trace 查看器结构测试
验证 index.html 中包含执行轨迹弹窗所需的元素和逻辑
"""

from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent.parent / "service" / "static" / "index.html"


class TestFrontendTraceViewer:
    """前端执行轨迹查看器结构检查"""

    @classmethod
    def setup_class(cls):
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            cls.html = f.read()

    def test_index_html_exists(self):
        assert INDEX_PATH.exists(), f"index.html not found: {INDEX_PATH}"

    def test_trace_button_in_render_history(self):
        """历史项模板中包含 trace-btn"""
        assert "trace-btn" in self.html
        assert "showTraceModal" in self.html

    def test_modal_overlay_exists(self):
        """弹窗覆盖层定义"""
        assert "trace-modal-overlay" in self.html

    def test_modal_structure(self):
        """弹窗包含 header/body/close"""
        assert "trace-modal-header" in self.html
        assert "trace-modal-body" in self.html
        assert "trace-modal-empty" in self.html

    def test_render_traces_function_exists(self):
        """renderTracesInModal 函数存在"""
        assert "function renderTracesInModal" in self.html

    def test_summary_stats_display(self):
        """概要统计显示区域"""
        assert "tm-summary" in self.html
        assert "tm-summary-label" in self.html
        assert "tm-summary-value" in self.html

    def test_fail_badge_exists(self):
        """失败文本标识"""
        assert "tm-badge fail" in self.html

    def test_retry_badge_exists(self):
        """重试文本标识"""
        assert "tm-badge retry" in self.html

    def test_degraded_badge_exists(self):
        """降级文本标识"""
        assert "tm-badge degraded" in self.html

    def test_worker_grouping_exists(self):
        """Worker 分组标题"""
        assert "tm-worker-header" in self.html

    def test_skip_badge_exists(self):
        """跳过文本标识"""
        assert "tm-badge skip" in self.html

    def test_retry_exhausted_uses_fail_badge(self):
        """重试耗尽使用失败标识"""
        assert "retry_exhausted" in self.html

    def test_resume_skip_stage_handled(self):
        """resume_skip 阶段被识别"""
        assert "resume_skip" in self.html

    def test_resume_started_stage_handled(self):
        """resume_started 阶段被识别"""
        assert "resume_started" in self.html

    def test_resume_completed_stage_handled(self):
        """resume_completed 阶段被识别"""
        assert "resume_completed" in self.html

    def test_benchmark_button_exists(self):
        """Benchmark 按钮存在"""
        assert "btnBenchmark" in self.html
        assert "showBenchmarkModal" in self.html

    def test_quality_panel_exists(self):
        """质量面板存在"""
        assert "buildQualityPanel" in self.html
        assert "quality-panel" in self.html
        assert "quality_score" in self.html
        assert "supported_rate" in self.html
        assert "fast" in self.html or "当前模式未执行覆盖率评估" in self.html

    def test_degraded_state_in_history(self):
        """历史项支持 degraded 状态标记"""
        assert "degraded" in self.html

    def test_hidden_reasoning_skipped(self):
        """模型隐藏推理内容不展示（think 阶段跳过）"""
        # think 阶段被 return 跳过（不展示模型隐藏推理）
        assert "跳过 think" in self.html

    def test_event_types_have_css_styles(self):
        """事件类型有对应 CSS 类"""
        for cls in [".start", ".end", ".fail", ".retry", ".degraded", ".retry_exhausted", ".resume_skip"]:
            assert cls in self.html, f"Missing CSS class: {cls}"

    def test_trace_modal_css(self):
        """弹窗 CSS 定义完整"""
        assert ".trace-modal-overlay" in self.html
        assert ".trace-modal" in self.html
        assert "@keyframes modal-in" in self.html
