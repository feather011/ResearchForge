# ResearchForge Evaluation Runner

对三种运行模式（Fast / Standard / Deep）进行定量评估。

## 使用方法

```bash
python evals/runner.py
```

输出文件：`evals/evaluation_report.json`

## 数据集

`dataset.json` 包含 10 个固定的 AI 研究主题，覆盖：
- LLM 基础技术（RLHF、RAG、CoT、LoRA）
- Agent 技术（ReAct 模式）
- 质量保障（幻觉、对齐、评估）
- 基础设施（向量数据库、多模态）

## 指标

### Performance（性能）
| 指标 | 说明 |
|------|------|
| total_time_seconds | 单次研究总耗时（秒） |
| llm_calls | LLM 调用次数 |
| tool_calls | 工具调用次数 |

### Research Quality（研究质量）
| 指标 | 说明 |
|------|------|
| question_coverage_rate | 研究问题被报告覆盖的比例 |
| source_count | 搜索到的来源数 |
| evidence_count | 提取的证据片段数 |
| claim_count | 核心结论数 |
| claim_supported_rate | 结论中证据支持的占比 |
| audit_pass_rate | 质量审计通过率 |
| rewrite_count | 审计触发的重写次数 |
| report_length | 报告字数 |

## 设计原则

- **不引入 RAGAS**：所有指标从现有 events / trace / audit 数据计算
- **不需要人工标注**：指标全是过程可量化的
- **可对比**：同一 topic 跑三种模式，直接对比指标差异
- **可重复**：每次运行覆盖 10 个 topic × 3 模式 = 30 次研究
