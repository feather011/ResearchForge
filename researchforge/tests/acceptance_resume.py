#!/usr/bin/env python3
"""
端到端断点恢复验收测试 — 最终版

方案：
  1. 备份 synthesis_node.py → 在函数头注入 raise RuntimeError
  2. 启动服务 → 创建 Standard 研究 → 等待 failed
  3. 验证检查点文件
  4. 关闭服务 → 还原 synthesis_node.py → 重启
  5. POST /api/research/{id}/resume → 等待 completed
  6. 验证最终状态
"""

import sys, io, os, json, time, shutil, urllib.request, urllib.error, subprocess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

BASE = "http://localhost:8002"
SEP = "=" * 62

CHECKS = []; PASS = 0; FAIL = 0

def check(name, ok, detail=""):
    global PASS, FAIL
    CHECKS.append((name, bool(ok)))
    if ok: PASS += 1; print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else: FAIL += 1; print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))

def http(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read().decode()
            try: return 200, json.loads(raw)
            except: return 200, {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode() if e.fp else "{}"
        try: return e.code, json.loads(raw)
        except: return e.code, {}

def poll_status(rid, timeout=300, interval=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c, d = http("GET", f"/api/status/{rid}")
            if c == 200:
                s = d.get("state", "")
                if s in ("failed", "completed", "error"):
                    return s, d.get("report", "")
        except:
            pass
        time.sleep(interval)
    return "timeout", ""

def start_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "researchforge.service.app:app",
         "--host", "0.0.0.0", "--port", "8002"],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for _ in range(30):
        try:
            if http("GET", "/")[0] == 200: return proc
        except:
            pass
        time.sleep(1)
    raise RuntimeError("服务启动超时")

def stop_server(proc):
    if not proc: return
    proc.terminate()
    try: proc.wait(timeout=10)
    except: proc.kill()
    time.sleep(2)

SYN_NODE = os.path.join(PROJECT_ROOT, "researchforge", "nodes", "synthesis_node.py")
SYN_BAK = SYN_NODE + ".bak"

# ────── 开始验收 ──────
print(f"\n{SEP}")
print("  ResearchForge 断点恢复 — 端到端验收测试")
print(f"  日期: 2026-07-17")
print(f"{SEP}")

try:
    # ═══ Step 0: 注入硬编码异常 ═══
    print(f"\n{SEP}")
    print("  Step 0: 注入硬编码异常 raise RuntimeError")
    print(f"{SEP}")

    shutil.copy2(SYN_NODE, SYN_BAK)
    with open(SYN_NODE, "r", encoding="utf-8") as f: orig = f.read()

    # 在 run_synthesis_node 函数的 def 与 docstring 之间插入 raise（可执行代码）
    injected = orig.replace(
        "def run_synthesis_node(state: ResearchState, llm: LLMProvider) -> List[Claim]:",
        "def run_synthesis_node(state: ResearchState, llm: LLMProvider) -> List[Claim]:\n    raise RuntimeError('ACCEPTANCE_TEST: synthesis 节点主动注入失败')"
    )
    with open(SYN_NODE, "w", encoding="utf-8") as f: f.write(injected)
    print("  ✅ 已注入 raise RuntimeError")
    check("异常注入完成", True)

    # ═══ Step 1: 启动服务 ═══
    print(f"\n{SEP}")
    print("  Step 1: 启动 FastAPI 服务（带异常）")
    print(f"{SEP}")

    proc = start_server()
    serv_proc = proc
    check("服务启动成功", True)

    # ═══ Step 2: 创建研究任务 ═══
    print(f"\n{SEP}")
    print("  Step 2: 创建 Standard 研究")
    print(f"{SEP}")

    c, d = http("POST", "/api/research",
                {"topic": "LLM Agent 的规划与记忆机制", "mode": "standard"})
    check("POST /api/research → 200", c == 200, f"HTTP {c}")
    rid = d.get("research_id", "")
    check("返回 research_id", bool(rid), f"id={rid}")
    print(f"  task_id: {rid}")

    # ═══ Step 3: 等待 synthesis 失败 ═══
    print(f"\n{SEP}")
    print("  Step 3: 等待 synthesis 失败（约 1-3 分钟）")
    print(f"{SEP}")

    state, _ = poll_status(rid, timeout=300, interval=15)
    check("任务状态为 failed", state == "failed", f"got='{state}'")

    if state == "failed":
        c, ev = http("GET", f"/api/research/{rid}/events")
        events = ev.get("events", [])
        progress = [e for e in events if e.get("type") == "progress"]
        errs = [e for e in events if e.get("type") == "error"]
        print(f"  事件日志: {len(events)} 条（progress:{len(progress)} error:{len(errs)}）")
        for e in errs:
            print(f"  ❌ {e.get('error','?')}")

    # ═══ Step 4: 验证检查点 ═══
    print(f"\n{SEP}")
    print("  Step 4: 验证 checkpoint 文件")
    print(f"{SEP}")

    ck_path = os.path.join(PROJECT_ROOT, "researchforge", "service", "data", "checkpoints", f"{rid}.json")
    ck_exists = os.path.exists(ck_path)
    check("检查点文件存在", ck_exists)

    if ck_exists:
        with open(ck_path, "r", encoding="utf-8") as f: ck = json.load(f)
        check("status == 'failed'", ck.get("status") == "failed", f"got='{ck.get('status')}'")
        check("failed_node == 'synthesizing'", ck.get("failed_node") == "synthesizing",
              f"got='{ck.get('failed_node')}'")
        nodes = ck.get("completed_nodes", [])
        check("planning 已完成", "planning" in nodes)
        check("searching 已完成", "searching" in nodes)
        check("fetching 已完成", "fetching" in nodes)
        check("extracting 已完成", "extracting" in nodes)
        check("synthesis_initial 未完成", "synthesis_initial" not in (ck.get("completed_steps") or []))
        check("已有 sources", len(ck.get("sources", [])) > 0, f"{len(ck.get('sources',[]))} 个")
        check("已有 documents", len(ck.get("documents", [])) > 0, f"{len(ck.get('documents',[]))} 篇")
        check("已有 evidences", len(ck.get("evidences", [])) > 0, f"{len(ck.get('evidences',[]))} 条")
        print(f"  completed_nodes: {nodes}")

    # ═══ Step 5: 关闭 → 还原代码 → 重启 ═══
    print(f"\n{SEP}")
    print("  Step 5: 关闭服务 → 还原 synthesis_node.py → 重启")
    print(f"{SEP}")

    stop_server(proc)
    serv_proc = None

    shutil.copy2(SYN_BAK, SYN_NODE); os.unlink(SYN_BAK)
    print("  ✅ synthesis_node.py 已还原")
    check("代码已还原", True)

    proc2 = start_server()
    serv_proc = proc2
    check("服务重启成功", True)

    # ═══ Step 6: 调用 resume ═══
    print(f"\n{SEP}")
    print("  Step 6: POST /api/research/{id}/resume")
    print(f"{SEP}")

    c, d = http("POST", f"/api/research/{rid}/resume")
    check("resume → 200", c == 200, f"HTTP {c}")
    check("返回 research_id", bool(d.get("research_id")))
    print(f"  message: {d.get('message')}")

    # ═══ Step 7: 等待恢复完成 ═══
    print(f"\n{SEP}")
    print("  Step 7: 等待恢复完成（约 1-3 分钟）")
    print(f"{SEP}")

    final_state, report = poll_status(rid, timeout=480, interval=15)
    check("最终状态为 completed", final_state == "completed", f"got='{final_state}'")

    # ═══ Step 8: 验证最终状态 ═══
    print(f"\n{SEP}")
    print("  Step 8: 验证最终报告和状态")
    print(f"{SEP}")

    check("报告已生成", bool(report) and len(report) > 100,
          f"报告长度: {len(report) if report else 0} 字")

    if ck_exists:
        with open(ck_path, "r", encoding="utf-8") as f: ck2 = json.load(f)
        check("检查点最终状态为 completed", ck2.get("status") == "completed",
              f"got='{ck2.get('status')}'")

    print(f"\n{SEP}")
    print(f"  {'🎉 验证全部通过!' if FAIL == 0 else f'⚠️ {FAIL} 项失败'}")
    print(f"  总计: {PASS}/{PASS + FAIL} 通过")
    print(f"{SEP}")

except Exception as e:
    print(f"\n❌ 异常: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()

finally:
    # ═══ Step 9: 清理 ═══
    print(f"\n{SEP}")
    print("  Step 9: 清理")
    print(f"{SEP}")
    if os.path.exists(SYN_BAK):
        shutil.copy2(SYN_BAK, SYN_NODE); os.unlink(SYN_BAK)
        print("  ✅ synthesis_node.py 已恢复")
    for pn in ['proc', 'proc2']:
        p = locals().get(pn)
        if p: stop_server(p)
    print("  ✅ 服务已停止")

    print(f"\n{SEP}")
    print(f"  验收结束: {PASS}/{PASS + FAIL} 通过{' ✅' if FAIL == 0 else ' ❌'}")
    print(f"{SEP}")
