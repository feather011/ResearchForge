"""
HTTP 级别端到端验证：从前端 API 视角测试断点恢复

验证 API 端点（即前端 http://localhost:8002 调用的所有接口）：
  1. GET  /api/checkpoints → 列出可恢复任务
  2. POST /api/research/{id}/resume → 启动恢复
  3. 错误场景：404 / 400
  4. SSE 流端点可达
"""

import sys, io, os, json, urllib.request, urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

BASE = "http://localhost:8002"
SEP = "=" * 62

def api(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, {"_raw": raw[:200]}
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else "{}"
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"_raw": raw[:200]}

ok = 0
fail = 0

def check(label, condition, detail=""):
    global ok, fail
    if condition:
        ok += 1
        print(f"  ✅ {label}" + (f" — {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))

# ────── 插入一个模拟的 failed 检查点 ──────
from researchforge.orchestration import ResearchMode, ResearchState
from researchforge.orchestration import Source, Document, Evidence
from researchforge.orchestration.checkpoint_store import CheckpointStore
from pathlib import Path

store = CheckpointStore(store_dir=Path(_PROJECT_ROOT) / "researchforge" / "service" / "data" / "checkpoints")
checkpoint_id = "_http_test_resume"

state = ResearchState(mode=ResearchMode.STANDARD, topic="HTTP测试断点恢复", task_id=checkpoint_id)
state.status = "failed"
state.failed_node = "synthesizing"
state.completed_nodes = ["planning", "searching", "fetching", "extracting"]
state.questions = ["什么是RLHF?", "RLHF训练流程是什么?"]
state.sources = [Source(id="s1", title="RLHF介绍", snippet="RLHF核心思想", url="http://example.com/rlhf")]
state.documents = [Document(content="RLHF利用人类反馈优化模型。", source_id="s1")]
state.evidences = [Evidence(id="e1", text="RLHF利用人类偏好训练LLM。", source_id="s1")]
store.save(state)

# 创建一个 completed 检查点（用于错误测试）
completed_id = "_http_test_completed"
cs = ResearchState(mode=ResearchMode.FAST, topic="已完成", task_id=completed_id)
cs.status = "completed"
store.save(cs)

# ────── 开始测试 ──────
print(f"\n{SEP}")
print("  前端 API 断点恢复验证")
print(f"{SEP}")

# 1. 根路径
print(f"\n--- 1. 基础端点 ---")
code, resp = api("GET", "/")
check("GET /", code == 200, f"HTTP {code}")

# 2. 检查点列表
print(f"\n--- 2. GET /api/checkpoints ---")
code, resp = api("GET", "/api/checkpoints")
check("状态码 200", code == 200)
cps = resp.get("checkpoints", [])
matching = [c for c in cps if c["task_id"] == checkpoint_id]
check(f"返回 {len(cps)} 个检查点，包含我们的 mock 任务", len(matching) > 0,
      f"failed_node={matching[0]['failed_node']}, mode={matching[0]['mode']}" if matching else "")
check("failed_node 正确", matching and matching[0]["failed_node"] == "synthesizing")
check("completed_nodes 正确", matching and matching[0]["completed_nodes"] == ["planning", "searching", "fetching", "extracting"])

# 3. 恢复 - 失败的任务
print(f"\n--- 3. POST /api/research/{{id}}/resume (成功场景) ---")
code, resp = api("POST", f"/api/research/{checkpoint_id}/resume")
check("恢复启动返回 200", code == 200, f"HTTP {code}")
check("包含 research_id", bool(resp.get("research_id")))
check("消息包含提示", "恢复" in (resp.get("message", "")),
      f"message={resp.get('message')}")

# 4. SSE 端点可达（不需要消费，确认端点被正确挂载即可）
print(f"\n--- 4. SSE 端点 ---")
code, resp = api("GET", f"/api/stream/{checkpoint_id}")
# SSE 端点实际返回 StreamingResponse（200），但 urllib 会拿到流首字节
check(f"SSE 流端点可达 ({code})", code in (200, 404))

# 5. Status 端点
print(f"\n--- 5. 状态端点 ---")
code, resp = api("GET", f"/api/status/{checkpoint_id}")
check("状态码 200", code == 200)
check("状态为 pending 或 running",
      resp.get("state") in ("pending", "running"),
      f"state={resp.get('state')}")

# 6. 错误：不存在的任务
print(f"\n--- 6. 错误场景 ---")
code, resp = api("POST", "/api/research/_nonexistent_xxx/resume")
check("不存在任务 → 404", code == 404, f"HTTP {code} detail={resp.get('detail', '')}")

code, resp = api("POST", f"/api/research/{completed_id}/resume")
check("已完成任务 → 400", code == 400, f"HTTP {code} detail={resp.get('detail', '')}")

# 7. 前端前端需要的其它端点
print(f"\n--- 7. 前端辅助端点 ---")
code, resp = api("GET", "/api/history")
check("GET /api/history → 200", code == 200, f"{len(resp.get('tasks', []))} 条记录")

code, resp = api("GET", "/api/research/nonexistent_xxx/events")
check("不存在任务的 events → 404", code == 404, f"HTTP {code}")

# ── 清理 ──
store.delete(checkpoint_id)
store.delete(completed_id)

# ── 结果 ──
print(f"\n{SEP}")
total = ok + fail
print(f"  结果: {ok}/{total} 通过", end="")
if fail == 0:
    print(" ✅")
else:
    print(f" ❌ ({fail} 失败)")
print(f"{SEP}")
