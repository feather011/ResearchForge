"""
Research Graph - 研究状态机（三种模式通用）

状态流转（完整）：
CREATED → PLANNING → SEARCHING → FETCHING → EXTRACTING
  → EVALUATING → GAP_SEARCHING → SYNTHESIZING
  → WRITING → AUDITING → HUMAN_REVIEW → COMPLETED

Fast 模式跳过：EVALUATING, GAP_SEARCHING, AUDITING, HUMAN_REVIEW
Standard 模式：全部经过，GAP_SEARCHING 最多 1 轮
Deep 模式：全部经过，GAP_SEARCHING 最多 2 轮，多 Worker 并行
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import logging

from .mode_policy import ModePolicy, ResearchMode
from .research_state import ResearchState
from .checkpoint_store import CheckpointStore
from .retry_policy import searching_policy, fetching_policy, llm_policy

logger = logging.getLogger("ResearchGraph")


class State(Enum):
    """研究状态（13 个状态，通用）"""
    CREATED = "created"
    PLANNING = "planning"
    SEARCHING = "searching"
    FETCHING = "fetching"
    EXTRACTING = "extracting"
    EVALUATING = "evaluating"
    GAP_SEARCHING = "gap_searching"
    SYNTHESIZING = "synthesizing"
    WRITING = "writing"
    AUDITING = "auditing"
    HUMAN_REVIEW = "human_review"
    COMPLETED = "completed"
    FAILED = "failed"


# 不同模式的状态执行路径
MODE_FLOW = {
    ResearchMode.FAST: [
        State.PLANNING, State.SEARCHING, State.FETCHING,
        State.EXTRACTING, State.WRITING, State.COMPLETED,
    ],
    ResearchMode.STANDARD: [
        State.PLANNING, State.SEARCHING, State.FETCHING,
        State.EXTRACTING, State.EVALUATING,
        State.WRITING, State.AUDITING, State.HUMAN_REVIEW, State.COMPLETED,
    ],
    ResearchMode.DEEP: [
        State.PLANNING, State.SEARCHING, State.FETCHING,
        State.EXTRACTING, State.EVALUATING,
        State.SYNTHESIZING, State.WRITING, State.AUDITING,
        State.HUMAN_REVIEW, State.COMPLETED,
    ],
}


class ResearchGraph:
    """
    研究状态机 — 驱动节点执行（三种模式通用）

    用法:
        graph = ResearchGraph(mode=ResearchMode.FAST)
        result = graph.execute("RLHF技术", llm=llm, progress_callback=cb)

    ResearchService 只负责创建 Graph 并调用 execute()，
    所有编排逻辑和状态推进都在 Graph 内部完成。
    """

    def __init__(self, mode: ResearchMode = ResearchMode.STANDARD, checkpoint_store: Optional[CheckpointStore] = None, task_id: Optional[str] = None):
        self.mode = mode
        self.policy = ModePolicy.for_mode(mode)
        self.state = State.CREATED
        self.rs: Optional[ResearchState] = None  # 研究数据
        self._ck = checkpoint_store  # 检查点存储（None=不启用）
        self._task_id = task_id  # API 层传入的 task_id（与检查点 ID 一致）

    def start(self, topic: str, task_id: Optional[str] = None) -> "ResearchGraph":
        """开始研究"""
        import uuid as _uuid
        tid = task_id or self._task_id or _uuid.uuid4().hex[:8]
        self.rs = ResearchState(mode=self.mode, topic=topic, task_id=tid)
        self.state = State.CREATED
        logger.info(f"研究开始: mode={self.mode.value}, topic={topic}, id={tid}")
        return self

    def _flow(self) -> List[State]:
        """获取当前模式的状态流"""
        return MODE_FLOW.get(self.mode, MODE_FLOW[ResearchMode.STANDARD])

    def get_next_state(self) -> Optional[State]:
        """获取当前状态的下一个状态（考虑模式差异）"""
        flow = self._flow()
        if self.state not in flow:
            return flow[0]  # 当前状态不在模式流中，从头开始
        for i, s in enumerate(flow):
            if s == self.state and i + 1 < len(flow):
                return flow[i + 1]
        return None

    def advance(self) -> Optional[State]:
        """推进到下一个状态"""
        next_state = self.get_next_state()
        if next_state:
            self.state = next_state
            logger.info(f"状态推进: {self.state.value}")
        return next_state

    def complete(self):
        """COMPLETED: 完成"""
        self.state = State.COMPLETED
        if self.rs:
            self.rs.status = "completed"
            self._save_checkpoint()
        logger.info("研究完成")

    # ==================== 检查点与恢复 ====================

    def _save_checkpoint(self):
        """保存检查点（如果启用了 CheckpointStore）"""
        if self._ck and self.rs and self.rs.task_id:
            try:
                self._ck.save(self.rs)
            except Exception:
                logger.warning(f"检查点保存失败: {self.rs.task_id}", exc_info=True)

    def _node_start(self, node_name: str):
        """标记节点开始并保存检查点"""
        self.state = node_name
        self.rs.mark_node_start(node_name.value)
        self._save_checkpoint()

    def _node_end(self, node_name: str):
        """标记节点完成并保存检查点"""
        self.rs.mark_node_end(node_name.value)
        self._save_checkpoint()

    def _step_start(self, step_name: str):
        """标记一个唯一步骤开始并保存检查点"""
        self.rs.mark_step_start(step_name)
        self._save_checkpoint()

    def _step_end(self, step_name: str):
        """标记一个唯一步骤完成并保存检查点"""
        self.rs.mark_step_end(step_name)
        self._save_checkpoint()

    def _should_run(self, step_name: str) -> bool:
        """判断某个步骤是否尚未执行（未在 completed_steps 或 completed_nodes 中）"""
        if not self.rs:
            return True
        if step_name in self.rs.completed_steps:
            return False
        if step_name in self.rs.completed_nodes:
            return False
        return True

    def _run_with_retry(self, node_name: str, policy, operation):
        """
        以重试方式执行一个操作。

        Args:
            node_name: 节点名称（如 "searching"）
            policy: RetryPolicy 实例，决定是否重试和退避时间
            operation: 无参可调用对象，执行操作的具体逻辑

        Returns:
            operation 的返回值

        Raises:
            最后一次异常（重试耗尽或不可重试时抛出原异常）
        """
        import time as _time
        last_exc = None
        for _attempt in range(1, 10):  # 循环由 policy.should_retry 控制退出
            try:
                return operation()
            except Exception as _e:
                last_exc = _e
                if policy.should_retry(node_name, _e, _attempt):
                    _delay = policy.get_delay(_attempt)
                    logger.warning(
                        f"{node_name} 节点失败 (attempt {_attempt}/{policy._node_config.get(node_name, policy.max_retries)}): {_e}, "
                        f"等待 {_delay:.1f}s 后重试"
                    )
                    _time.sleep(_delay)
                else:
                    break
        raise last_exc  # 重试耗尽或不可重试，由调用方处理标记失败
        # 重试耗尽或不可重试 → 标记失败并抛异常
        logger.error(f"{node_name} 节点最终失败 (共尝试 {_attempt} 次): {last_exc}")
        self.rs.mark_node_failed(node_name)
        self._save_checkpoint()
        raise last_exc

    # ==================== 后提取阶段公共入口 ====================

    def continue_from_state(
        self,
        state: "ResearchState",
        llm: "LLMProvider",
        progress_callback: Optional[Callable] = None,
        tracer: Optional["TraceCollector"] = None,
    ) -> Dict[str, Any]:
        """
        从已有 ResearchState 继续执行（跳过 Planning/Searching/Fetching/Extracting）
        Deep 模式在 Workers 完成后调用此方法进入公共后半段流程。
        """
        self.rs = state
        for _n in ("planning", "searching", "fetching", "extracting"):
            if _n not in self.rs.completed_nodes:
                self.rs.completed_nodes.append(_n)
        return self._run_post_harvest(llm, progress_callback, tracer)

    # ──────────────────────────────────────────────
    # 公共后半段（从 Synthesis 开始到 Complete）
    # 与 execute() 中 #5 SYNTHESIS 至 #9 COMPLETE 完全一致
    # 当前请先保持两处代码一致，后续会迁移 execute() 到此
    # ──────────────────────────────────────────────

    def resume(self, task_id: str, llm: "LLMProvider",
               progress_callback: Optional[Callable] = None,
               tracer: Optional["TraceCollector"] = None) -> Dict[str, Any]:
        """
        从检查点恢复研究

        1. 从 CheckpointStore 加载 ResearchState
        2. 清空失败状态
        3. 复用已有中间结果，跳过已完成步骤
        4. 从第一个未完成或失败的步骤继续

        返回: execute() 同样的结果 dict
        """
        if not self._ck:
            raise RuntimeError("恢复失败：未配置 CheckpointStore")

        state = self._ck.load(task_id)
        if state is None:
            raise ValueError(f"检查点不存在或已损坏: {task_id}")

        if state.status == "completed":
            raise ValueError(f"任务已完成，无需恢复: {task_id}")

        # 从保存的状态恢复 Graph
        mode_val = state.mode
        if isinstance(mode_val, str):
            mode_val = ResearchMode(mode_val)
        self.mode = mode_val
        self.policy = ModePolicy.for_mode(self.mode)
        self.state = State.CREATED
        self.rs = state

        # 清空失败状态，标记恢复运行
        self.rs.failed_node = ""
        self.rs.current_node = ""
        self.rs.status = "running"

        logger.info(f"恢复研究: task_id={task_id}, mode={self.mode.value}, "
                    f"已完成 {len(self.rs.completed_steps)} 步")

        return self.execute(
            topic=state.topic,
            llm=llm,
            progress_callback=progress_callback,
            tracer=tracer,
            _resumed_state=state,
        )

    # ==================== 主执行引擎 ====================

    def execute(
        self,
        topic: str,
        llm: "LLMProvider",
        progress_callback: Optional[Callable] = None,
        tracer: Optional["TraceCollector"] = None,
        _resumed_state: Optional[ResearchState] = None,
        _task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行一次研究的完整流程（Fast / Standard 模式）

        如果传入 _resumed_state，从该状态后续的第一个未完成步骤继续执行。
        已完成的步骤会被跳过，中间结果（sources/documents/evidences）复用不重新获取。

        _task_id: 外部传入的 task_id（使 API 层和检查点 ID 一致）。"""
        import time as _time
        if False:
            from ..trace import TraceCollector  # noqa

        _start_time = _time.time()
        _audit_passed = True
        _rewritten = 0
        _audit_issues = []

        # 延迟导入避免循环依赖
        from ..nodes.plan_node import run_plan_node
        from ..nodes.search_node import run_search_node
        from ..nodes.fetch_node import run_fetch_node
        from ..nodes.extract_node import run_extract_node
        from ..nodes.synthesis_node import run_synthesis_node
        from ..nodes.claim_verification_node import run_claim_verification_node
        from ..nodes.coverage_node import run_coverage_node
        from ..nodes.gap_agent import run_evidence_gap_agent
        from ..nodes.write_node import run_write_node

        # ── 恢复模式 ──
        is_resume = _resumed_state is not None
        if is_resume:
            self.rs = _resumed_state
            done = set(self.rs.completed_steps)
            done_nodes = set(self.rs.completed_nodes)
            has_claims = bool(self.rs.claims)
            has_sources = bool(self.rs.sources)
            has_docs = bool(self.rs.documents)
            has_evidences = bool(self.rs.evidences)
            has_report = bool(self.rs.report)
            logger.info(f"恢复模式: topic={self.rs.topic}, "
                        f"已完成 {len(done)} 步, 已有 sources={has_sources}, "
                        f"evidences={has_evidences}, claims={has_claims}")
        else:
            self.start(topic, task_id=_task_id)
            done = set()
            done_nodes = set()
            has_claims = False
            has_sources = False
            has_docs = False
            has_evidences = False
            has_report = False

        N = self._flow  # 别名

        # ── 辅助: 条件推导（恢复时根据已完成的步骤推导当年判断结果） ──
        def _cov_was_complete():
            """恢复模式下：coverage 当时是否完整（无缺口）"""
            if not is_resume:
                return None  # 走正常逻辑
            return "gap_searching" not in done_nodes

        def _cov_gap_was_filled():
            """恢复模式下：gap 是否被填了"""
            if not is_resume:
                return None
            return "synthesis_after_gap" in done

        def _audit_was_passed():
            """恢复模式下：审计当时是否通过"""
            if not is_resume:
                return None
            return "REWRITE" not in done_nodes

        # ── 1. PLANNING ──
        if self._should_run(N()[0].value):
            self._node_start(N()[0])
            try:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="PLANNING", input=topic)
                if progress_callback:
                    progress_callback("Planner", f"正在为「{topic}」制定研究计划...")

                def _plan_op():
                    return run_plan_node(topic, llm)

                self.rs.questions = self._run_with_retry(
                    N()[0].value, llm_policy, _plan_op
                )
                logger.info(f"规划完成: {len(self.rs.questions)} 个问题")
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="PLANNING", result=f"{len(self.rs.questions)}个问题")
                if progress_callback:
                    progress_callback("Planner", "计划完成，开始搜索...")
            except Exception:
                self.rs.mark_node_failed(N()[0].value)
                self._save_checkpoint()
                raise
            self._node_end(N()[0])
        else:
            logger.info(f"[恢复] 跳过已完成步骤: {N()[0].value}")

        # ── 2. SEARCHING ──
        if self._should_run(N()[1].value):
            self._node_start(N()[1])
            try:
                def _search_op():
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_start",
                                       action="SEARCHING", input=f"{len(self.rs.questions)}个问题")
                    self.rs.sources = run_search_node(
                        self.rs.questions,
                        sources_per_question=self.policy.search_sources,
                        use_real_search=True,
                    )
                    logger.info(f"搜索完成: {len(self.rs.sources)} 个来源")
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_end",
                                       action="SEARCHING", result=f"{len(self.rs.sources)}个来源")
                    if progress_callback:
                        progress_callback(
                            "SearchAgent",
                            f"找到 {len(self.rs.sources)} 个来源，抓取正文...",
                            extra_data={
                                "sources": [
                                    {"id": s.id, "title": s.title, "snippet": s.snippet}
                                    for s in self.rs.sources
                                ]
                            },
                        )
                    return self.rs.sources

                self._run_with_retry(N()[1].value, searching_policy, _search_op)
            except Exception:
                self.rs.mark_node_failed(N()[1].value)
                self._save_checkpoint()
                raise
            self._node_end(N()[1])
        else:
            logger.info(f"[恢复] 跳过已完成步骤: {N()[1].value}")

        # ── 3. FETCHING ──
        if self._should_run(N()[2].value):
            self._node_start(N()[2])
            try:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="FETCHING", input=f"{len(self.rs.sources)}个来源")

                # 逐个对来源抓取，每页独立重试
                _fetch_docs: List[Document] = []
                _fetch_fail_count = 0
                for _src in self.rs.sources[:self.policy.search_sources]:
                    def _make_fetch_op(src):
                        def _op():
                            from ..nodes.fetch_node import _fetch_single
                            return _fetch_single(src)
                        return _op
                    try:
                        doc = self._run_with_retry(N()[2].value, fetching_policy,
                                                    _make_fetch_op(_src))
                        _fetch_docs.append(doc)
                    except Exception as _fe:
                        _fetch_fail_count += 1
                        logger.warning(f"抓取失败 (跳过): {_src.url} — {_fe}")
                        if progress_callback:
                            progress_callback("System", f"抓取失败 (跳过): {_src.title or _src.url}")

                self.rs.documents = _fetch_docs

                # 全部失败 → 节点失败
                if _fetch_fail_count >= len(self.rs.sources[:self.policy.search_sources]):
                    raise RuntimeError(
                        f"全部 {_fetch_fail_count}/{len(self.rs.sources[:self.policy.search_sources])} "
                        f"个来源抓取失败"
                    )

                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="FETCHING", result=f"{len(self.rs.documents)}篇文档")
                if progress_callback:
                    progress_callback("System", "提取证据片段...")
            except Exception:
                self.rs.mark_node_failed(N()[2].value)
                self._save_checkpoint()
                raise
            self._node_end(N()[2])
        else:
            logger.info(f"[恢复] 跳过已完成步骤: {N()[2].value}")

        # ── 4. EXTRACTING ──
        if self._should_run(N()[3].value):
            self._node_start(N()[3])
            try:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="EXTRACTING", input=f"{len(self.rs.documents)}篇文档")
                self.rs.evidences = run_extract_node(self.rs.documents, self.rs.questions)
                logger.info(f"提取完成: {len(self.rs.evidences)} 条证据")
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="EXTRACTING", result=f"{len(self.rs.evidences)}条证据")
            except Exception:
                self.rs.mark_node_failed(N()[3].value)
                self._save_checkpoint()
                raise
            self._node_end(N()[3])
        else:
            logger.info(f"[恢复] 跳过已完成步骤: {N()[3].value}")

        # ── 5-9. 委托给公共后半段 _run_post_harvest ──
        return self._run_post_harvest(llm, progress_callback, tracer)

    def _run_post_harvest(
        self,
        llm: "LLMProvider",
        progress_callback: Optional[Callable] = None,
        tracer: Optional["TraceCollector"] = None,
    ) -> Dict[str, Any]:
        """
        公共后半段：Synthesis → Claim Verify → Coverage → Gap → Writing → Audit → Complete
        """
        import time as _time
        _start_time = _time.time()
        _audit_passed = True
        _rewritten = 0
        _audit_issues = []

        from ..nodes.synthesis_node import run_synthesis_node
        from ..nodes.claim_verification_node import run_claim_verification_node
        from ..nodes.coverage_node import run_coverage_node
        from ..nodes.gap_agent import run_evidence_gap_agent
        from ..nodes.write_node import run_write_node

        is_resume = bool(self.rs.completed_steps)
        done = set(self.rs.completed_steps)
        done_nodes = set(self.rs.completed_nodes)

        # ── 5. SYNTHESIS（所有模式都做，首次执行） ──
        if self._should_run("synthesis_initial"):
            self._step_start("synthesis_initial")
            self._node_start(State.SYNTHESIZING)
            try:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="SYNTHESIS", input=f"{len(self.rs.evidences)}条证据")
                if progress_callback:
                    progress_callback("AnalystAgent", "综合分析所有证据...")

                def _synth_op():
                    return run_synthesis_node(self.rs, llm)

                self.rs.claims = self._run_with_retry(
                    State.SYNTHESIZING.value, llm_policy, _synth_op
                )
                logger.info(f"分析完成: {len(self.rs.claims)} 条核心结论")
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="SYNTHESIS", result=f"{len(self.rs.claims)}条结论")
            except Exception:
                self.rs.mark_node_failed(State.SYNTHESIZING.value)
                self._save_checkpoint()
                raise
            self._node_end(State.SYNTHESIZING)
            self._step_end("synthesis_initial")
        else:
            logger.info("[恢复] 跳过已完成步骤: synthesis_initial")

        # ── 5b. CLAIM_VERIFICATION（所有模式都做，首次执行） ──
        if self.rs.claims and self._should_run("claim_verification_initial"):
            self._step_start("claim_verification_initial")
            self.rs.mark_node_start("CLAIM_VERIFICATION")
            self._save_checkpoint()
            try:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="CLAIM_VERIFICATION",
                                   input=f"验证{len(self.rs.claims)}条结论的证据支持")
                if progress_callback:
                    progress_callback("AnalystAgent", "验证结论的证据支撑...")

                def _cv_op():
                    return run_claim_verification_node(self.rs, llm)

                try:
                    verified = self._run_with_retry("CLAIM_VERIFICATION", llm_policy, _cv_op)
                except Exception as _cv_e:
                    # 重试耗尽或不可重试 → 降级，不终止任务
                    logger.error(f"Claim 验证彻底失败: {_cv_e}，进入降级模式")
                    from ..nodes.claim_verification_node import ClaimStatus as _CS, VerifiedClaim
                    verified = [
                        VerifiedClaim(
                            claim_index=i,
                            status=_CS.UNVERIFIED,
                            explanation="LLM验证失败，标记为未验证",
                        )
                        for i in range(len(self.rs.claims))
                    ]
                    self.rs.metadata = getattr(self.rs, "metadata", {}) or {}
                    self.rs.metadata["claim_verify_degraded"] = True
                    self.rs.metadata["claim_verify_error"] = str(_cv_e)
                    if progress_callback:
                        progress_callback("AnalystAgent", "结论验证降级：LLM 调用失败，所有结论标记为未验证")

                # 写回 confidence（含降级路径：全部设为 0.0）
                from ..nodes.claim_verification_node import ClaimStatus as _CS2
                for vc in verified:
                    idx = vc.claim_index
                    if 0 <= idx < len(self.rs.claims):
                        if vc.status == _CS2.SUPPORTED:
                            self.rs.claims[idx].confidence = 1.0
                        elif vc.status == _CS2.PARTIALLY_SUPPORTED:
                            self.rs.claims[idx].confidence = 0.5
                        else:
                            self.rs.claims[idx].confidence = 0.0

                unsupported = [v for v in verified if v.status.value == "unsupported"]
                if unsupported:
                    logger.warning(f"发现 {len(unsupported)} 条无证据支持的结论")
                    for v in unsupported:
                        logger.warning(f"  Claim{v.claim_index}: {self.rs.claims[v.claim_index].text[:60]}")
                if tracer:
                    statuses = ", ".join(f"{v.claim_index}:{v.status.value}" for v in verified)
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="CLAIM_VERIFICATION", result=statuses)
            except Exception:
                self.rs.mark_node_failed("CLAIM_VERIFICATION")
                self._save_checkpoint()
                raise
            self.rs.mark_node_end("CLAIM_VERIFICATION")
            self._save_checkpoint()
            self._step_end("claim_verification_initial")
        else:
            logger.info("[恢复] 跳过 claim_verification_initial（无 claims 或已完成）")

        # ── 6. COVERAGE（仅在 Standard/Deep 启用） ──
        if self.policy.enable_coverage_check:
            if self._should_run(State.EVALUATING.value):
                self._node_start(State.EVALUATING)
                try:
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_start",
                                       action="COVERAGE_CHECK",
                                       input=f"{len(self.rs.evidences)}条证据, {len(self.rs.questions)}个问题")
                    if progress_callback:
                        progress_callback("System", "检查证据完整性...")
                    complete, gaps = run_coverage_node(self.rs)
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_end",
                                       action="COVERAGE_CHECK",
                                       result=f"{'完整' if complete else f'发现{len(gaps)}个缺口'}")
                except Exception:
                    self.rs.mark_node_failed(State.EVALUATING.value)
                    self._save_checkpoint()
                    raise
                self._node_end(State.EVALUATING)
            else:
                logger.info("[恢复] 跳过已完成步骤: evaluating")
                # 恢复模式下从 completed_steps 推导当时的条件
                complete = "gap_searching" not in (set(self.rs.completed_nodes) if is_resume else set())
                gaps = []

            if not complete and self.policy.max_gap_search_rounds > 0:
                if self._should_run(State.GAP_SEARCHING.value):
                    if progress_callback:
                        progress_callback(
                            "System",
                            f"发现 {len(gaps)} 个缺口，启动补搜 Agent...",
                        )

                    self._node_start(State.GAP_SEARCHING)
                    try:
                        filled, filled_gaps = run_evidence_gap_agent(
                            self.rs,
                            gaps,
                            llm,
                            max_rounds=self.policy.max_gap_search_rounds,
                            progress_callback=progress_callback,
                            tracer=tracer,
                        )
                    except Exception:
                        self.rs.mark_node_failed(State.GAP_SEARCHING.value)
                        self._save_checkpoint()
                        raise
                    self._node_end(State.GAP_SEARCHING)
                else:
                    logger.info("[恢复] 跳过已完成步骤: gap_searching")
                    filled = "synthesis_after_gap" in (set(self.rs.completed_steps) if is_resume else set())
                    filled_gaps = []

                if filled:
                    logger.info(
                        f"补搜完成: 填补了 {len(filled_gaps)}/{len(gaps)} 个缺口"
                    )
                    # 补搜后重新 Synthesis
                    if self._should_run("synthesis_after_gap"):
                        self._step_start("synthesis_after_gap")
                        self._node_start(State.SYNTHESIZING)
                        try:
                            if tracer:
                                tracer.record(agent_name="ResearchGraph", stage="node_start",
                                               action="RE_SYNTHESIS",
                                               input=f"补搜后重新分析")
                            if progress_callback:
                                progress_callback("AnalystAgent", "重新综合分析...")

                            def _re_synth_op():
                                return run_synthesis_node(self.rs, llm)

                            self.rs.claims = self._run_with_retry(
                                State.SYNTHESIZING.value, llm_policy, _re_synth_op
                            )
                            # 补搜后重新验证 Claim
                            if self.rs.claims:
                                run_claim_verification_node(self.rs, llm)
                            # 重新覆盖检查（确认缺口已填）
                            complete, gaps = run_coverage_node(self.rs)
                            if tracer:
                                tracer.record(agent_name="ResearchGraph", stage="node_end",
                                               action="RE_SYNTHESIS",
                                               result=f"{'缺口已填' if complete else '仍有缺'}")
                        except Exception:
                            self.rs.mark_node_failed(State.SYNTHESIZING.value)
                            self._save_checkpoint()
                            raise
                        self._node_end(State.SYNTHESIZING)
                        self._step_end("synthesis_after_gap")
                    else:
                        logger.info("[恢复] 跳过已完成步骤: synthesis_after_gap")
                else:
                    logger.info("补搜未填补任何缺口")
            if progress_callback:
                progress_callback("System", "证据检查完成")

        # ── 7. WRITING ──
        if self._should_run(State.WRITING.value):
            self._node_start(State.WRITING)
            try:
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="WRITING",
                                   input=f"{len(self.rs.evidences)}条证据, {len(self.rs.claims)}条结论")
                if progress_callback:
                    progress_callback("WriterAgent", "撰写报告...")
                self.state = State.WRITING

                def _write_op():
                    return run_write_node(self.rs, llm, mode=self.mode.value)

                self.rs.report = self._run_with_retry(
                    State.WRITING.value, llm_policy, _write_op
                )
                logger.info(f"写作完成: {len(self.rs.report)} 字")
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="WRITING", result=f"{len(self.rs.report)}字")
            except Exception:
                self.rs.mark_node_failed(State.WRITING.value)
                self._save_checkpoint()
                raise
            self._node_end(State.WRITING)
        else:
            logger.info(f"[恢复] 跳过已完成步骤: writing (已有 report={len(self.rs.report)}字)")

        # ── 8. AUDIT（仅在启用审计的模式做） ──
        if self.policy.enable_report_audit:
            from ..nodes.audit_node import run_audit_node

            if self._should_run("audit_initial"):
                self._step_start("audit_initial")
                self._node_start(State.AUDITING)
                try:
                    if tracer:
                        tracer.record(agent_name="ResearchGraph", stage="node_start",
                                       action="AUDIT", input=f"{len(self.rs.report)}字报告")
                    if progress_callback:
                        progress_callback("System", "审计报告质量...")

                    def _audit_op():
                        return run_audit_node(self.rs, llm)

                    try:
                        audit = self._run_with_retry(State.AUDITING.value, llm_policy, _audit_op)
                        _audit_passed = audit.passed
                        _audit_issues = audit.issues[:5]
                        self.rs.audit_passed = audit.passed
                    except Exception as _audit_e:
                        # 重试耗尽或不可重试 → 降级，不终止任务
                        logger.error(f"报告审计彻底失败: {_audit_e}，进入降级模式")
                        from ..nodes.audit_node import AuditResult
                        audit = AuditResult(passed=False, issues=[],
                                            suggestions="【质量审核未完成】LLM调用失败，未完成审核")
                        _audit_passed = False
                        _audit_issues = ["质量审核未完成：LLM调用失败"]
                        self.rs.audit_passed = False
                        self.rs.metadata = getattr(self.rs, "metadata", {}) or {}
                        self.rs.metadata["audit_degraded"] = True
                        self.rs.metadata["audit_error"] = str(_audit_e)
                        if progress_callback:
                            progress_callback("System",
                                "⚠️ 质量审核未完成（LLM 调用失败），当前报告已保留")
                except Exception:
                    self.rs.mark_node_failed(State.AUDITING.value)
                    self._save_checkpoint()
                    raise
                self._node_end(State.AUDITING)
                self._step_end("audit_initial")
            else:
                logger.info("[恢复] 跳过已完成步骤: audit_initial")
                audit = None
                _audit_passed = True
                _audit_issues = []

            # 审计降级时（audit_degraded）不触发 Rewrite
            _audit_degraded = bool(
                getattr(self.rs, "metadata", {}).get("audit_degraded")
            )

            need_rewrite = (
                not _audit_degraded
                and audit is not None and not audit.passed
                and self.policy.max_rewrite_rounds > 0 and audit.suggestions
            )

            # 恢复模式：audit_initial 已完成但 REWRITE 未完成 → 根据 audit_passed 判断是否需要重写
            if not need_rewrite and is_resume and "audit_initial" in set(self.rs.completed_steps) and "REWRITE" not in set(self.rs.completed_nodes):
                # 如果原始审计已降级，不触发 Rewrite
                if getattr(self.rs, "metadata", {}).get("audit_degraded"):
                    need_rewrite = False
                elif not self.rs.audit_passed:
                    need_rewrite = True
                    audit = __import__("researchforge.nodes.audit_node", fromlist=["AuditResult"]).AuditResult(
                        passed=False, issues=["恢复：审计未完成"], suggestions="请根据审计意见修改报告"
                    )

            # 恢复模式：如果 REWRITE 已完成，跳过 rewrite 块
            if not need_rewrite and is_resume and "REWRITE" in set(self.rs.completed_nodes):
                need_rewrite = False
                _rewritten = 1

            if need_rewrite:
                if progress_callback:
                    progress_callback("WriterAgent",
                        f"审计发现 {len(audit.issues)} 个问题，重写报告...")
                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_start",
                                   action="REWRITE", observation=audit.suggestions[:300])

                self.rs.mark_node_start("REWRITE")
                self._save_checkpoint()
                try:
                    def _rewrite_op():
                        return run_write_node(
                            self.rs, llm, extra_instructions=audit.suggestions, mode=self.mode.value
                        )
                    self.rs.report = self._run_with_retry("REWRITE", llm_policy, _rewrite_op)
                    logger.info(f"重写完成: {len(self.rs.report)} 字")
                except Exception:
                    self.rs.mark_node_failed("REWRITE")
                    self._save_checkpoint()
                    raise
                self.rs.mark_node_end("REWRITE")
                self._save_checkpoint()

                if tracer:
                    tracer.record(agent_name="ResearchGraph", stage="node_end",
                                   action="REWRITE", result=f"{len(self.rs.report)}字")
                _rewritten += 1
                # 重写后重新审计
                if self._should_run("audit_after_rewrite"):
                    self._step_start("audit_after_rewrite")
                    self._node_start(State.AUDITING)
                    try:
                        audit = run_audit_node(self.rs, llm)
                    except Exception:
                        self.rs.mark_node_failed(State.AUDITING.value)
                        self._save_checkpoint()
                        raise
                    self._node_end(State.AUDITING)
                    self._step_end("audit_after_rewrite")
                    _audit_passed = audit.passed
                    _audit_issues = audit.issues[:5]
                    if audit.passed:
                        logger.info("重写后审计通过")
                    else:
                        logger.warning(f"重写后审计仍发现 {len(audit.issues)} 个问题")
                    if progress_callback:
                        progress_callback("WriterAgent", "报告重写完成")
                else:
                    logger.info("[恢复] 跳过已完成步骤: audit_after_rewrite")

        # ── 9. COMPLETE ──
        self.complete()
        if progress_callback:
            progress_callback("System", "研究完成")

        # 统计 Claim 验证结果
        supported = sum(1 for c in self.rs.claims if c.confidence >= 0.9)
        unchecked = sum(1 for c in self.rs.claims if c.confidence == 0.0)

        return {
            "topic": self.rs.topic,
            "report": self.rs.report,
            "mode": self.mode.value,
            "stats": {
                "sources": len(self.rs.sources),
                "documents": len(self.rs.documents),
                "evidences": len(self.rs.evidences),
                "report_length": len(self.rs.report),
            },
            "tracer": tracer.get_all() if tracer else [],
            "claim_verification": {
                "total": len(self.rs.claims),
                "supported": supported,
                "unverified": unchecked,
            },
            "audit": {
                "passed": _audit_passed,
                "rewritten": _rewritten,
                "issues": _audit_issues,
            },
            "_duration_s": round(_time.time() - _start_time, 2),
        }

    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "mode": self.mode.value,
            "state": self.state.value,
            "topic": self.rs.topic if self.rs else "",
            "sources": len(self.rs.sources) if self.rs else 0,
            "documents": len(self.rs.documents) if self.rs else 0,
            "evidences": len(self.rs.evidences) if self.rs else 0,
            "max_gap_rounds": self.policy.max_gap_search_rounds,
        }
