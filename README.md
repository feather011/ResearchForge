# ResearchForge — 智能研究工坊

> 一个基于 LLM Agent 的自动化研究系统。给定一个主题，自动完成搜索、分析、写作、审计全流程。
>
> 面向 AI Agent Engineer 求职展示。

---

## Problem

传统 LLM 有两个局限：

1. **单次对话无法完成长流程研究任务** — Plan → Search → Fetch → Extract → Synthesis → Write → Audit
2. **没有质量保证机制** — LLM 直接生成的内容不可验证

ResearchForge 把研究拆成 7 个阶段、每个阶段用专用 Agent 执行、末尾加审计闭环，让输出可追溯、可验证。

---

## Agent Architecture

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    ResearchGraph Workflow Engine              │
│                                                              │
│  Plan → Search → Fetch → Extract → Synthesis → Write → Audit │
│                                    ↘            ↙           │
│                               ClaimVerification              │
│                                    ↘            ↙           │
│                              Coverage → GapAgent             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  Agent Capabilities
    │
    ├── Planner            LLM 将主题拆解为可搜索的子问题
    ├── ReActAgent         GapAgent：Think → Act → Observe 循环补搜
    ├── GapSearchTool      Bing/Google/DuckDuckGo 三源搜索
    ├── GapFetchTool       网页抓取
    ├── GapEvidenceSearchTool  证据检索
    ├── ClaimVerification  LLM 验证每条结论是否被证据支持
    ├── ReportAudit        LLM 检查覆盖、引用、幻觉
    └── Rewrite            审计失败 → 带建议重写
```

### 三种模式

| 模式 | 流程 | 适用场景 |
|------|------|---------|
| **Fast** | Plan → Search → Fetch → Extract → Synthesis → Write | 快速概览 |
| **Standard** | 上面 + Coverage → GapAgent → Audit → Rewrite | 标准研究 |
| **Deep** | 3 Worker 并行（每个独立 Plan → 搜索 → 分析）→ Merge Agent → Coverage → GapAgent → ClaimVerification → Audit → Rewrite | 深度研究 |

### Deep 模式 Multi-Agent

```
LeadResearcher Agent
    │
    ├── Worker 1: 独立 Plan → Search → Fetch → Extract → Analyze
    ├── Worker 2: 独立 Plan → Search → Fetch → Extract → Analyze
    └── Worker 3: 独立 Plan → Search → Fetch → Extract → Analyze
    │
    ▼
Merge Agent（LLM 语义去重 + 冲突检测）
    │
    ▼
Coverage → GapAgent → ClaimVerification → Write → Audit
```

每个 Worker 是独立 ResearchAgent：拥有独立子任务规划、独立搜索/抓取/提取、独立 Analyzer、独立 Trace 记录。

---

## Data Flow

```
Source(id="来源1", url="...")
  → Document(source_id="来源1", content="...")
    → Evidence(id="ev_0", source_id="来源1", text="...")
      → Claim(text="RLHF 依赖人类反馈", evidence_ids=["ev_0"], confidence=1.0)
        → Report("...RLHF依赖人类反馈[来源1]...")

每条结论可追溯到具体证据片段和原始搜索来源。
```

---

## Quality Loop

```
Write
  │
  ▼
ReportAudit
  │
  ├── 检查1: 研究问题是否被报告覆盖（关键词匹配）
  ├── 检查2: 引用 [来源X] 是否对应真实来源
  ├── 检查3: LLM 检测无依据内容、结构完整性
  │
  ├── 全部通过 → Complete
  │
  └── 发现问题 + 有修改建议 → Rewrite → 重新 Audit
```

---

## Technical Details

```
技术栈:
  Python 3.14+
  FastAPI + SSE 实时推送
  Ollama / DashScope (OpenAI-compatible API)
  Google / Bing / DuckDuckGo 搜索

项目结构:
  researchforge/
    ├── core/              ReAct引擎 + LLM Provider
    ├── trace/             Agent Trace 可观测
    ├── nodes/             8 个研究节点
    ├── orchestration/     状态机 + 数据中心 + 模式策略
    ├── service/           FastAPI + 前端
    ├── tools/             搜索/抓取工具
    ├── research_service.py  统一入口
    └── evals/             自动化评测
```

---

## Evaluation

10 个 AI 研究主题 × 3 种模式的自动化评测：

```
                  Fast        Standard    Deep
耗时              30-60s      60-120s     120-180s
来源数            3-4          5-8          8-15
证据数            3-4          6-12        10-20
结论数            3-4          3-5          4-6
结论可信率        60-80%       60-80%       60-80%
质量审计通过率    N/A          ~70%         ~80%
```

运行 `python evals/runner.py` 在本机复现。

---

## Quick Start

```bash
# 1. 确保 Ollama 运行
ollama pull qwen3.5:9b

# 2. 启动服务
cd ai_research_workshop
python -m uvicorn researchforge.service.app:app --port 8002 --reload

# 3. 打开浏览器
# http://localhost:8002
```

---

## Screenshots

### Agent Trace 时间线
```
THINK    GapAgent → 需要搜索财务数据
         💭 需要查找最新财报数据来验证这个缺口
ACT      GapAgent → gap_search("xxx")
         📋 找到 5 个相关结果
OBSERVE  GapAgent → 结果中有官方财报数据
         ✓ 缺口已填补
```

### 搜索结果展示
```
搜索完成: 3 个来源
[来源1] 数据库在人工智能中的作用与选型
[来源2] AI Agent 的 ReAct 模式原理
[来源3] 大模型评估基准与方法综述
```

---

## License

MIT
