"""
LeadResearcher & ResearchWorker — Deep 模式多 Worker 并行研究（增强版）

架构：
  LeadResearcher Agent:
    - 分解研究任务（LLM 规划）
    - 分配 Worker
    - 合并结果（LLM 语义去重 + 冲突检测）

  Worker Agent（每个 Worker 是独立 ResearchAgent）:
    - 独立 Plan（LLM 子任务分解）
    - Search → Fetch → Extract
    - Analyze（LLM 综合分析）
    - 独立 Trace 记录
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from ..orchestration.research_state import ResearchState, Source, Document, Evidence, Claim, Conflict
from ..core.react_engine import LLMProvider
from ..nodes.search_node import run_search_node
from ..nodes.fetch_node import run_fetch_node
from ..nodes.extract_node import run_extract_node

logger = logging.getLogger("DeepMode")


class ResearchWorker:
    """
    Worker Agent — 独立研究代理

    每个 Worker 拥有：
    - 独立子任务规划
    - 独立搜索/抓取/提取能力
    - 独立 Analyzer 分析
    """

    def __init__(self, worker_id: str, task: str, llm: LLMProvider, tracer=None):
        self.worker_id = worker_id
        self.task = task
        self.llm = llm
        self.tracer = tracer
        self.sources: List[Source] = []
        self.documents: List[Document] = []
        self.evidences: List[Evidence] = []
        self.claims: List[str] = []
        self.sub_plan: List[str] = []

    def run(self) -> dict:
        """执行完整子任务：Plan → Search → Fetch → Extract → Analyze"""
        logger.info(f"Worker {self.worker_id} 开始: {self.task}")

        # ── 1. Plan ──
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="think", action="plan", input=self.task)
        self.sub_plan = self._make_sub_plan()
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_end", action="plan",
                                result=f"{len(self.sub_plan)}个子任务")

        # ── 2. Search ──
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_start", action="search",
                                input=f"{len(self.sub_plan)}个方向")
        self.sources = run_search_node(self.sub_plan, sources_per_question=3)
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_end", action="search",
                                result=f"{len(self.sources)}个来源")
        if not self.sources:
            return self._result()

        # ── 3. Fetch ──
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_start", action="fetch",
                                input=f"{len(self.sources)}个来源")
        self.documents = run_fetch_node(self.sources, max_pages=3)
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_end", action="fetch",
                                result=f"{len(self.documents)}篇文档")

        # ── 4. Extract ──
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_start", action="extract",
                                input=f"{len(self.documents)}篇文档")
        self.evidences = run_extract_node(self.documents, self.sub_plan)
        if self.tracer:
            self.tracer.record(agent_name=f"Worker{self.worker_id}",
                                stage="node_end", action="extract",
                                result=f"{len(self.evidences)}条证据")

        # ── 5. Analyze ──
        self.claims = self._analyze()

        logger.info(f"Worker {self.worker_id} 完成: {len(self.sources)}来源, {len(self.evidences)}证据")
        return self._result()

    def _make_sub_plan(self) -> List[str]:
        """LLM 将子任务拆为 2-3 个搜索方向"""
        prompt = f"""你是一个研究助手。需要研究以下课题：{self.task}
请将研究课题拆解为 2-3 个具体的搜索方向，每个方向是独立可搜索的短语。
每行一个，不要编号："""
        try:
            result = self.llm.generate(prompt)
            lines = [l.strip() for l in result.split("\n") if l.strip() and not l.startswith("你")]
            if lines:
                return lines[:3]
        except Exception:
            pass
        return [self.task]

    def _analyze(self) -> List[str]:
        """LLM 分析提取的证据，生成结论"""
        if not self.evidences:
            return []
        ev_text = "\n".join(e.text[:300] for e in self.evidences[:5])
        prompt = f"""你是一个分析师。基于以下资料对「{self.task}」进行综合分析：
{ev_text}
请提供核心发现（2-3条），每行一条："""
        try:
            result = self.llm.generate(prompt)
            return [l.strip() for l in result.split("\n") if l.strip() and len(l.strip()) > 10][:3]
        except Exception:
            return []

    def _result(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "task": self.task,
            "sub_plan": self.sub_plan,
            "sources": self.sources,
            "documents": self.documents,
            "evidences": self.evidences,
            "claims": self.claims,
        }


class LeadResearcher:
    """首席研究员 — 负责总体规划、分配 Worker、合并结果"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def make_plan(self, topic: str, num_workers: int = 5) -> List[str]:
        """将研究主题拆分为多个子任务"""
        prompt = f"""你是一个首席研究员。研究主题：{topic}
请将研究主题拆分为 {num_workers} 个子任务，每个子任务是一个明确的研究方向。
直接返回子任务列表，每行一个：
1. 子任务描述
2. ..."""
        result = self.llm.generate(prompt)
        tasks = []
        for line in result.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                task = line.split(".", 1)[-1].strip()
                task = task.lstrip("-").strip()
                if task:
                    tasks.append(task)
        tasks = tasks[:num_workers]
        if not tasks:
            tasks = [f"搜索{topic}的基本信息", f"分析{topic}的核心内容", f"研究{topic}的最新发展"]
        return tasks

    def merge_results(self, worker_results: List[dict]) -> dict:
        """
        Merge Agent — 合并所有 Worker 的结果

        LLM 语义去重：同来源同内容的 evidence 只保留一条。
        不去重的证据保留冲突信息供后续冲突检测。
        """
        all_sources = []
        all_documents = []
        all_evidences = []
        all_claims = []

        for wr in worker_results:
            all_sources.extend(wr["sources"])
            all_documents.extend(wr["documents"])
            all_evidences.extend(wr["evidences"])
            all_claims.extend(wr.get("claims", []))

        # 语义去重：相同 evidence 文本只保留一条
        seen_texts = set()
        deduped_evidences = []
        for ev in all_evidences:
            key = ev.text[:100]
            if key not in seen_texts:
                seen_texts.add(key)
                deduped_evidences.append(ev)

        if len(all_evidences) - len(deduped_evidences) > 0:
            logger.info(f"Merge: 去重 {len(all_evidences) - len(deduped_evidences)} 条重复证据")

        merged = {
            "sources": all_sources,
            "documents": all_documents,
            "evidences": deduped_evidences,
            "claims": all_claims,
        }
        return merged

    def analyze_conflicts(self, evidences: List[Evidence], llm: LLMProvider) -> List[Conflict]:
        """分析不同来源之间的信息冲突"""
        if len(evidences) < 3:
            return []

        from collections import defaultdict
        by_source = defaultdict(list)
        for ev in evidences:
            by_source[ev.source_id].append(ev.text[:100])

        if len(by_source) < 2:
            return []

        prompt = "以下是多份来源的摘要，请找出它们之间的矛盾之处：\n\n"
        for sid, texts in list(by_source.items())[:5]:
            prompt += f"来源{sid}: {'; '.join(texts[:2])}\n"
        prompt += "\n如果存在矛盾，请列出；否则返回'无冲突'。"
        result = llm.generate(prompt)

        if "无冲突" in result:
            return []

        conflicts = []
        for line in result.strip().split("\n")[:3]:
            if line.strip():
                conflicts.append(Conflict(
                    claim=line.strip()[:100], source_a="", source_b="",
                ))
        return conflicts


def run_deep_research(topic: str, llm: LLMProvider, num_workers: int = 5, tracer=None) -> dict:
    """
    Deep 模式入口：LeadResearcher 分配任务 → Workers 并行研究 → Merge Agent
    """
    lead = LeadResearcher(llm=llm)

    # 1. 制定总计划
    tasks = lead.make_plan(topic, num_workers=num_workers)
    logger.info(f"Deep研究计划: {len(tasks)}个子任务")

    # 2. 并行执行 Worker（每个带独立 Trace）
    workers = [
        ResearchWorker(f"W{i+1}", task, llm, tracer=tracer)
        for i, task in enumerate(tasks)
    ]
    worker_results = []

    with ThreadPoolExecutor(max_workers=min(len(workers), 5)) as pool:
        future_map = {pool.submit(w.run): w for w in workers}
        for f in as_completed(future_map):
            worker_results.append(f.result())

    # 3. Merge Agent — 语义去重合并
    merged = lead.merge_results(worker_results)

    # 4. 分析冲突
    conflicts = lead.analyze_conflicts(merged["evidences"], llm)

    return {
        "tasks": tasks,
        "worker_results": worker_results,
        "merged": merged,
        "conflicts": conflicts,
    }
