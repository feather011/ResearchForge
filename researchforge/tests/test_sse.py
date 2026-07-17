"""
SSE 模块单元测试 - 验证完整事件生命周期

直接测试 sse.py 模块的函数，不依赖于研究管线
"""
import sys
import json
import asyncio
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


async def test_event_creation():
    """测试事件创建函数"""
    from researchforge.service.sse import (
        create_progress_event, create_review_event,
        create_complete_event, create_error_event
    )

    # 测试进度事件
    pe = create_progress_event(1, 3, "SearchAgent", "正在搜索...")
    assert pe["type"] == "progress", f"类型错误: {pe['type']}"
    assert pe["payload"]["step"] == 1
    assert pe["payload"]["total"] == 3
    assert pe["payload"]["agent"] == "SearchAgent"
    assert pe["payload"]["content"] == "正在搜索..."
    print("[PASS] create_progress_event")

    # 测试审核事件
    re = create_review_event("iv_123", "请审核")
    assert re["type"] == "review", f"类型错误: {re['type']}"
    assert re["payload"]["intervention_id"] == "iv_123"
    print("[PASS] create_review_event")

    # 测试完成事件
    ce = create_complete_event("最终报告内容")
    assert ce["type"] == "complete", f"类型错误: {ce['type']}"
    assert ce["payload"]["report"] == "最终报告内容"
    print("[PASS] create_complete_event")

    # 测试错误事件
    ee = create_error_event("API错误")
    assert ee["type"] == "error", f"类型错误: {ee['type']}"
    assert ee["payload"]["error"] == "API错误"
    print("[PASS] create_error_event")

    # 测试内容截断(超过200字符)
    long_content = "A" * 300
    le = create_progress_event(0, 0, "Agent", long_content)
    assert len(le["payload"]["content"]) == 200, f"截断错误: {len(le['payload']['content'])}"
    print("[PASS] 内容截断（200字符）")


async def test_event_stream():
    """测试 event_stream 生成器的完整生命周期"""
    from researchforge.service.sse import event_stream

    queue = asyncio.Queue()

    # 模拟事件序列
    async def producer(q):
        await asyncio.sleep(0.1)
        from researchforge.service.sse import create_progress_event, create_complete_event
        await q.put(create_progress_event(1, 2, "TestAgent", "步骤1"))
        await asyncio.sleep(0.1)
        await q.put(create_complete_event("完成报告"))

    async def consumer():
        events = []
        async for event_str in event_stream("test_123", queue):
            # 解析 SSE 格式
            lines = event_str.strip().split("\n")
            data = {}
            for line in lines:
                if line.startswith("event:"):
                    data["event"] = line[6:].strip()
                elif line.startswith("data:"):
                    data["data"] = json.loads(line[5:].strip())
            events.append(data)

            # 完成或错误事件后结束
            if data.get("event") in ("complete", "error"):
                break
        return events

    # 启动生产者和消费者
    producer_task = asyncio.create_task(producer(queue))
    events = await consumer()
    await producer_task

    # 验证事件流
    assert len(events) >= 3, f"事件数不足: {len(events)}"

    # 1. connected 事件
    assert events[0]["event"] == "connected", f"首个事件不是connected: {events[0]}"
    assert events[0]["data"]["research_id"] == "test_123"
    print(f"[PASS] connected事件: {events[0]['data']}")

    # 2. progress 事件
    assert events[1]["event"] == "progress", f"第二个事件不是progress: {events[1]}"
    assert events[1]["data"]["step"] == 1
    assert events[1]["data"]["agent"] == "TestAgent"
    print(f"[PASS] progress事件: step={events[1]['data']['step']}")

    # 3. complete 事件（终止）
    assert events[2]["event"] == "complete", f"第三个事件不是complete: {events[2]}"
    assert "完成报告" in events[2]["data"]["report"]
    print(f"[PASS] complete事件（流自动终止）")

    # 验证流在 complete 后终止
    assert len(events) == 3, f"complete后还有事件: {events[3:]}"
    print("[PASS] 流在complete事件后正确终止")


async def test_heartbeat():
    """测试心跳机制（30秒超时）"""
    from researchforge.service.sse import event_stream

    queue = asyncio.Queue()

    # 不放入任何事件，等待心跳触发
    start = asyncio.get_event_loop().time()

    async def consumer():
        events = []
        async for event_str in event_stream("hb_test", queue):
            lines = event_str.strip().split("\n")
            data = {}
            for line in lines:
                if line.startswith("event:"):
                    data["event"] = line[6:].strip()
                elif line.startswith("data:"):
                    data["data"] = json.loads(line[5:].strip())
            events.append(data)
            # 收到一个heartbeat就够
            if data.get("event") == "heartbeat":
                # 放一个complete事件让流结束
                from researchforge.service.sse import create_complete_event
                await queue.put(create_complete_event(""))
            if data.get("event") == "complete":
                break
        return events

    events = await consumer()
    elapsed = asyncio.get_event_loop().time() - start

    # 应该收到: connected(立即) -> heartbeat(~30s) -> complete
    heartbeat_events = [e for e in events if e.get("event") == "heartbeat"]
    assert len(heartbeat_events) == 1, f"心跳事件数: {len(heartbeat_events)}"

    # 心跳大约在30秒触发（允许误差5秒）
    assert 25 <= elapsed <= 40, f"心跳超时时间异常: {elapsed:.1f}s"
    print(f"[PASS] 心跳在 {elapsed:.1f}s 时触发")


async def test_queue_timeout_isolation():
    """测试不同 research_id 的队列隔离"""
    from researchforge.service.sse import event_stream

    q1 = asyncio.Queue()
    q2 = asyncio.Queue()

    from researchforge.service.sse import create_complete_event

    # 只向 q2 发完成事件
    async def producer():
        await asyncio.sleep(0.2)
        await q2.put(create_complete_event("q2完成"))

    async def consume(q, name):
        events = []
        async for event_str in event_stream(name, q):
            lines = event_str.strip().split("\n")
            data = {}
            for line in lines:
                if line.startswith("event:"):
                    data["event"] = line[6:].strip()
                elif line.startswith("data:"):
                    data["data"] = json.loads(line[5:].strip())
            events.append(data)
            if data.get("event") in ("complete", "error"):
                break
        return events

    producer_task = asyncio.create_task(producer())

    # q1 没有complete事件，会一直等，我们不能等30秒
    # 所以只验证它们能独立消费
    ev1_task = asyncio.create_task(consume(q1, "q1"))

    # 给 q1 放一个complete让它快速结束
    async def stop_q1():
        await asyncio.sleep(1)
        await q1.put(create_complete_event(""))

    stop_task = asyncio.create_task(stop_q1())

    ev2 = await consume(q2, "q2")
    ev1 = await ev1_task
    await producer_task
    await stop_task

    ev2_complete = [e for e in ev2 if e.get("event") == "complete"]
    assert len(ev2_complete) == 1
    assert ev2_complete[0]["data"]["report"] == "q2完成"
    print("[PASS] 队列隔离正常: q2独立收到complete事件")

    # 验证 q1 也正常
    ev1_complete = [e for e in ev1 if e.get("event") == "complete"]
    assert len(ev1_complete) == 1
    print("[PASS] 队列隔离正常: q1也独立终止")


async def main():
    print("=" * 50)
    print("SSE 模块单元测试")
    print("=" * 50)

    print("\n--- 1. 事件创建函数 ---")
    await test_event_creation()

    print("\n--- 2. 事件流生命周期 ---")
    await test_event_stream()

    print("\n--- 3. 心跳机制（约30秒，请耐心等待...）---")
    await test_heartbeat()

    print("\n--- 4. 队列隔离 ---")
    await test_queue_timeout_isolation()

    print("\n" + "=" * 50)
    print("全部测试通过! ✅")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
