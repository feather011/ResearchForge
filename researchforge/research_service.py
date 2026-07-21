"""
ResearchService — 研究服务
把 ResearchGraph + 节点管线 + LLM 绑定成可调用接口
"""

import logging
from typing import Dict, Any, Optional, Callable

from .orchestration import ResearchGraph, ResearchMode, ResearchState, State
from .orchestration.research_state import DeepWorkerState
from .core.react_engine import LLMProvider
from .nodes.deep_research import run_deep_research
from .nodes.write_node import run_write_node

logger = logging.getLogger("ResearchService")


class ResearchService:
    """
    研究服务 — 统一研究入口

    用法:
        service = ResearchService(llm=llm)
        result = service.run("RLHF技术", mode=ResearchMode.FAST)

    ResearchService 只做两件事：
      1. 模式路由（Fast/Standard → ResearchGraph.execute, Deep → run_deep_research）
      2. 结果组装
    """

    def __init__(self, llm: LLMProvider, checkpoint_store=None):
        self.llm = llm
        self._ck = checkpoint_store

    def run(
        self,
        topic: str,
        mode: ResearchMode = ResearchMode.FAST,
        progress_callback: Optional[Callable] = None,
        tracer=None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行一次研究"""
        if mode == ResearchMode.DEEP:
            return self._run_deep(topic, progress_callback, tracer=tracer, task_id=task_id)
        self._explicit_task_id = task_id  # 供 _run_standard 使用
        return self._run_standard(topic, mode, progress_callback, tracer=tracer)

    def _resume_deep(self, task_id: str, progress_callback=None, tracer=None) -> Dict[str, Any]:
        """Deep 模式断点恢复：恢复 Planning 和 Worker 阶段"""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .nodes.deep_research import LeadResearcher, ResearchWorker

        if not self._ck:
            raise RuntimeError("恢复失败：未配置 CheckpointStore")

        rs = self._ck.load(task_id)
        if rs is None:
            raise ValueError(f"检查点不存在或已损坏: {task_id}")

        _deep_lock = threading.Lock()
        lead = LeadResearcher(llm=self.llm)

        # ── 1. Planning 阶段 ──
        if rs.deep_workers:
            logger.info(f"Deep 恢复: 跳过 Planning（已有 {len(rs.deep_workers)} 个 Worker 记录）")
        else:
            raise RuntimeError(f"Deep 检查点无 Worker 记录，无法恢复: {task_id}")

        if progress_callback:
            progress_callback("LeadResearcher", f"恢复研究... 共 {len(rs.deep_workers)} 个 Worker")

        # ── 2. Workers 恢复 ──
        worker_results_raw = []
        for dw in rs.deep_workers:
            if dw.sources or dw.documents or dw.evidences:
                # 已有中间产物 → 作为已有结果
                pass  # 已 completed 的 worker 在 _run_if_needed 中直接返回

        def _run_worker_if_needed(worker, idx):
            """如果 Worker 未完成则执行，否则复用已有结果"""
            dw = rs.deep_workers[idx]
            if dw.status == "completed":
                logger.info(f"Worker {worker.worker_id} 已完成，跳过重执行")
                return {
                    "worker_id": dw.worker_id,
                    "task": dw.task,
                    "sources": dw.sources,
                    "documents": dw.documents,
                    "evidences": dw.evidences,
                    "claims": [],
                }

            # failed / pending / running → 重新执行
            if progress_callback:
                progress_callback(f"Worker {worker.worker_id}",
                    f"{'重试' if dw.status == 'failed' else '继续'}执行: {dw.task[:40]}")
            try:
                with _deep_lock:
                    rs.deep_workers[idx].status = "running"
                    if self._ck:
                        self._ck.save(rs)

                result = worker.run()

                with _deep_lock:
                    rs.deep_workers[idx].status = "completed"
                    rs.deep_workers[idx].sources = result.get("sources", [])
                    rs.deep_workers[idx].documents = result.get("documents", [])
                    rs.deep_workers[idx].evidences = result.get("evidences", [])
                    if self._ck:
                        self._ck.save(rs)
                return result
            except Exception as e:
                logger.warning(f"Worker {worker.worker_id} 恢复后失败: {e}")
                with _deep_lock:
                    rs.deep_workers[idx].status = "failed"
                    rs.deep_workers[idx].error = str(e)
                    if self._ck:
                        self._ck.save(rs)
                return None

        # 构建 Worker 列表（只重建未完成的）
        workers = []
        worker_indices = []
        for i, dw in enumerate(rs.deep_workers):
            if dw.status == "completed":
                # 直接复用已有结果
                worker_results_raw.append({
                    "worker_id": dw.worker_id,
                    "task": dw.task,
                    "sources": dw.sources,
                    "documents": dw.documents,
                    "evidences": dw.evidences,
                    "claims": [],
                })
            else:
                workers.append(ResearchWorker(f"W{i+1}", dw.task, self.llm, tracer=tracer))
                worker_indices.append(i)

        # 并行执行未完成的 Worker
        if workers:
            with ThreadPoolExecutor(max_workers=min(len(workers), 5)) as pool:
                futures = {pool.submit(_run_worker_if_needed, w, idx): w
                           for w, idx in zip(workers, worker_indices)}
                for f in as_completed(futures):
                    r = f.result()
                    if r is not None:
                        worker_results_raw.append(r)

        # checkpoint: all workers done
        rs.deep_workers_completed = True
        if self._ck:
            self._ck.save(rs)

        # ── 3. 检查是否全部失败 ──
        all_failed = all(
            dw.status == "failed"
            for dw in rs.deep_workers
        )
        if all_failed or not worker_results_raw:
            raise RuntimeError(f"所有 Worker 均失败，无法继续 Deep 研究 task_id={task_id}")

        # ── 4. Merge ──
        merged = lead.merge_results(worker_results_raw)
        rs.sources = merged["sources"]
        rs.documents = merged["documents"]
        rs.evidences = merged["evidences"]

        if progress_callback and rs.sources:
            progress_callback("SearchAgent",
                f"所有Worker共找到 {len(rs.sources)} 个来源",
                extra_data={"sources": [
                    {"id": s.id, "title": s.title, "snippet": s.snippet}
                    for s in rs.sources[:10]]})

        for wr in worker_results_raw:
            if progress_callback:
                progress_callback(f"Worker {wr['worker_id']}",
                    f"完成: {len(wr['evidences'])}条证据")

        if progress_callback:
            progress_callback("AnalystAgent", "综合分析所有Worker结果...")
            progress_callback("LeadResearcher", "合并结果, 生成报告...")

        # ── 4. Conflicts ──
        conflicts = lead.analyze_conflicts(rs.evidences, self.llm)
        rs.conflicts = conflicts

        _dg = ResearchGraph(mode=ResearchMode.DEEP, checkpoint_store=self._ck)
        _base = _dg.continue_from_state(rs, self.llm, progress_callback, tracer)
        # 补充 Deep 模式特有字段
        _base["stats"]["workers"] = len(worker_results_raw)
        _base["stats"]["conflicts"] = len(conflicts)
        _base["mode"] = "deep"
        _base["require_human_review"] = True
        if _base["report"] and conflicts:
            _ct = "\n".join(f"- {c.claim}" for c in conflicts)
            _base["report"] += f"\n\n---\n### 来源冲突\n{_ct}"
        return _base

    def _run_deep(self, topic: str, progress_callback=None, tracer=None, task_id=None) -> Dict[str, Any]:
        """Deep 模式：多 Worker 并行研究（含 Planning/Worker Checkpoint）"""
        import uuid, threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from .nodes.deep_research import LeadResearcher, ResearchWorker

        ck = self._ck
        rs = ResearchState(
            mode=ResearchMode.DEEP, topic=topic,
            task_id=task_id or uuid.uuid4().hex[:8],
        )
        _deep_lock = threading.Lock()

        # ── 1. Planning ──
        if progress_callback:
            progress_callback("LeadResearcher", f"制定深度研究计划...")

        lead = LeadResearcher(llm=self.llm)
        tasks = lead.make_plan(topic, num_workers=3)
        rs.questions = tasks
        rs.deep_workers = [
            DeepWorkerState(worker_id=f"W{i+1}", task=t)
            for i, t in enumerate(tasks)
        ]
        if ck:
            ck.save(rs)  # checkpoint 1: after plan
            logger.info(f"Deep检查点(规划后): task_id={rs.task_id}, workers={len(tasks)}")

        if progress_callback:
            progress_callback("LeadResearcher",
                f"规划完成: {len(tasks)}个子任务, 启动Worker...")

        # ── 2. Workers（并行，线程安全保存） ──
        raw_worker_results: list = []

        def _run_worker(worker, idx):
            try:
                with _deep_lock:
                    rs.deep_workers[idx].status = "running"
                    if ck:
                        ck.save(rs)  # checkpoint: worker starting

                result = worker.run()

                with _deep_lock:
                    rs.deep_workers[idx].status = "completed"
                    rs.deep_workers[idx].sources = result.get("sources", [])
                    rs.deep_workers[idx].documents = result.get("documents", [])
                    rs.deep_workers[idx].evidences = result.get("evidences", [])
                    if ck:
                        ck.save(rs)  # checkpoint: worker completed
                return result
            except Exception as e:
                logger.warning(f"Worker {worker.worker_id} 失败: {e}")
                with _deep_lock:
                    rs.deep_workers[idx].status = "failed"
                    rs.deep_workers[idx].error = str(e)
                    if ck:
                        ck.save(rs)  # checkpoint: worker failed
                return None

        workers = [
            ResearchWorker(f"W{i+1}", task, self.llm, tracer=tracer)
            for i, task in enumerate(tasks)
        ]

        with ThreadPoolExecutor(max_workers=min(len(workers), 5)) as pool:
            futures = {pool.submit(_run_worker, w, i): w for i, w in enumerate(workers)}
            for f in as_completed(futures):
                r = f.result()
                if r is not None:
                    raw_worker_results.append(r)

        # checkpoint 3: all workers done
        rs.deep_workers_completed = True
        if ck:
            ck.save(rs)
            logger.info(f"Deep检查点(Worker结束): {len(raw_worker_results)}/{len(workers)} 成功")

        # ── 3. Merge ──
        merged = lead.merge_results(raw_worker_results)
        rs.sources = merged["sources"]
        rs.documents = merged["documents"]
        rs.evidences = merged["evidences"]

        if progress_callback and rs.sources:
            progress_callback("SearchAgent",
                f"所有Worker共找到 {len(rs.sources)} 个来源",
                extra_data={"sources": [
                    {"id": s.id, "title": s.title, "snippet": s.snippet}
                    for s in rs.sources[:10]]})

        for wr in raw_worker_results:
            if progress_callback:
                progress_callback(f"Worker {wr['worker_id']}",
                    f"完成: {len(wr['evidences'])}条证据")

        if progress_callback:
            progress_callback("AnalystAgent", "综合分析所有Worker结果...")
            progress_callback("LeadResearcher", "合并结果, 生成报告...")

        # ── 4. Conflicts ──
        conflicts = lead.analyze_conflicts(rs.evidences, self.llm)
        rs.conflicts = conflicts

        _dg = ResearchGraph(mode=ResearchMode.DEEP, checkpoint_store=self._ck)
        _base = _dg.continue_from_state(rs, self.llm, progress_callback, tracer)
        _base["stats"]["workers"] = len(raw_worker_results)
        _base["stats"]["conflicts"] = len(conflicts)
        _base["mode"] = "deep"
        _base["require_human_review"] = True
        if _base["report"] and conflicts:
            _ct = "\n".join(f"- {c.claim}" for c in conflicts)
            _base["report"] += f"\n\n---\n### 来源冲突\n{_ct}"
        return _base

    def _run_standard(
        self,
        topic: str,
        mode: ResearchMode,
        progress_callback: Optional[Callable] = None,
        tracer=None,
    ) -> Dict[str, Any]:
        """Fast / Standard 模式 — 委托给 ResearchGraph.execute()"""
        task_id = getattr(self, '_explicit_task_id', None)
        graph = ResearchGraph(mode=mode, checkpoint_store=self._ck, task_id=task_id)
        result = graph.execute(topic, self.llm, progress_callback, tracer=tracer, _task_id=task_id)
        # 是否进入人工审核阶段
        result["require_human_review"] = (
            mode != ResearchMode.FAST
            and graph.policy.require_human_review
        )
        return result

    def resume(
        self,
        task_id: str,
        progress_callback: Optional[Callable] = None,
        tracer=None,
    ) -> Dict[str, Any]:
        """从检查点恢复一项研究"""
        if not self._ck:
            raise RuntimeError("恢复失败：未配置 CheckpointStore")

        state = self._ck.load(task_id)
        if state is None:
            raise ValueError(f"检查点不存在或已损坏: {task_id}")
        if state.status == "completed":
            raise ValueError(f"任务已完成，无需恢复: {task_id}")

        mode = state.mode
        if isinstance(mode, str):
            mode = ResearchMode(mode)

        if mode == ResearchMode.DEEP:
            return self._resume_deep(task_id, progress_callback, tracer=tracer)

        graph = ResearchGraph(mode=mode, checkpoint_store=self._ck)
        result = graph.resume(task_id, self.llm, progress_callback, tracer=tracer)
        result["require_human_review"] = (
            mode != ResearchMode.FAST
            and graph.policy.require_human_review
        )
        return result
