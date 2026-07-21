# ResearchForge Demo

这套 Demo 用于展示 ResearchForge 的三种运行模式、故障恢复能力和质量评估。

---

## Demo 目录

```
demo/
├── README.md              ← 本文档
├── demo_cases.json        ← 三个固定演示案例
├── mock_provider.py       ← 稳定 Mock 数据（不调外部 API）
├── expected_outputs/      ← 脱敏的预期输出示例
├── outputs/               ← 运行输出（已 gitignore）
└── scripts/
    ├── run_demo.py        ← 统一运行脚本
    └── fault_injector.py  ← 可控故障注入（仅 Mock 模式）
```

## 三个演示案例

| Case ID | 模式 | 主题 | 展示重点 |
|---------|------|------|---------|
| `fast_simple` | Fast | Python 装饰器原理与应用 | 低延迟完整流程 |
| `standard_multi_source` | Standard | Transformer 自注意力机制加速方法 | Coverage / Claim Verify / Audit |
| `deep_complex` | Deep | 大语言模型推理优化技术综述 | LeadResearcher + 并行 Worker + 结果合并 |

主题选择原则：
- 稳定的技术概念，不依赖时效新闻
- 无政治、医疗或敏感内容
- 中文主题，展示系统对中文搜索和写作的支持

---

## Mock Demo（无需 API Key）

无需任何外部服务，使用稳定 Mock 数据，输出固定且可复现。

```bash
# 运行全部三个案例
python demo/scripts/run_demo.py --all --mock

# 运行单个案例
python demo/scripts/run_demo.py --case fast_simple --mock
python demo/scripts/run_demo.py --case standard_multi_source --mock
python demo/scripts/run_demo.py --case deep_complex --mock

# 故障恢复演示（运行 mock + 注入故障）
python demo/scripts/run_demo.py --case fast_simple --mock --inject-fault
```

`--inject-fault` 会在搜索节点第一次调用时注入 TimeoutError，
触发 RetryPolicy 重试。查看 `demo/outputs/` 下的 summary 可以看到 `retry_count > 0`。

---

## 真实 Demo（需要 API Key）

```bash
# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 运行
python demo/scripts/run_demo.py --case fast_simple
python demo/scripts/run_demo.py --case standard_multi_source
python demo/scripts/run_demo.py --case deep_complex
python demo/scripts/run_demo.py --all
```

---

## 前端演示

1. 启动服务：
   ```bash
   python -m uvicorn researchforge.service.app:app --host 0.0.0.0 --port 8002
   ```

2. 打开 http://localhost:8002/

3. 建议 5 分钟演示顺序：

   | 步骤 | 操作 | 观察点 |
   |------|------|--------|
   | 1 | 输入 "Python 装饰器" → 选 **Fast** → 启动 | 实时 SSE 事件流 |
   | 2 | 任务完成后点击 **📋** | Trace 时间线 + 耗时 |
   | 3 | 输入 "LLM 推理优化" → 选 **Deep** → 启动 | Worker 状态变化 |
   | 4 | 完成报告 → 查看 **质量评估面板** | Claims / Citation / Coverage / Quality Score |
   | 5 | 点击 **📊 Benchmark** → 运行 | 三模式对比结果 |

---

## 故障恢复演示

### Docker 运行

```bash
# 全部案例
docker compose run --rm researchforge python demo/scripts/run_demo.py --all --mock

# 故障注入
docker compose run --rm researchforge python demo/scripts/run_demo.py --case fast_simple --mock --inject-fault
```

### 通过脚本演示

```bash
# 搜索节点第一次失败 → 自动重试 → 最终成功
python demo/scripts/run_demo.py --case fast_simple --mock --inject-fault
```

### 通过前端手动演示

1. 用 Standard 或 Deep 模式启动一个任务
2. 在任务运行中停止服务器（Ctrl+C）
3. 重启服务器
4. 该任务出现在历史列表，状态为 "failed"
5. 点击 **↻** 恢复按钮
6. 观察恢复后跳过已完成节点，从断点继续

### 故障注入说明

`demo/scripts/fault_injector.py` 提供可控故障注入：

```python
from demo.scripts.fault_injector import FaultInjector

injector = FaultInjector()
injector.activate()
injector.add_rule("searching", fail_count=1)      # 搜索前 1 次失败
injector.add_rule("deep_worker", fail_count=1, worker_id="W2")  # Deep Worker W2 第 1 次失败
```

默认关闭，仅在 `--mock --inject-fault` 时启用。不影响正式运行。

---

## Demo 输出

每次运行保存在 `demo/outputs/{run_id}/`：

```
outputs/{run_id}/
├── report.md         ← 最终报告
├── result.json       ← 完整结果（含 stats）
├── traces.json       ← 全部 Trace 事件
└── summary.md        ← 运行摘要
```

`outputs/` 目录已在 `.gitignore` 中，不会提交到 Git。

---

## 预期输出

`expected_outputs/` 包含脱敏的预期输出示例，供验证 Demo 运行结果参考。
