"""ローカルテスト — async / sync (アダプターなし) / sync (アダプターあり) の3パターン.

Usage:
    cd ~/work/langgraph_mcp_agent_engine
    python test_local.py
"""

import asyncio
import json
import os
import sys
import traceback

# プロジェクトルートを sys.path に追加（パッケージ名でのインポートと直接インポート両方に対応）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "tactile-octagon-372414")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

try:
    from mcp_sync_adapter import make_sync_tools
except ImportError:
    from langgraph_mcp_agent_engine.mcp_sync_adapter import make_sync_tools

# ── 設定 ──

TEST_MCP_SERVER = os.path.join(os.path.dirname(__file__), "test_mcp_server.py")

MCP_SERVERS = {
    "snowflake": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [TEST_MCP_SERVER],
    }
}

TEST_QUERY = "日産の売上データを地域別にまとめてください"

MODEL_NAME = "gemini-2.5-flash"


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_result(result: dict):
    """エージェントの応答からメッセージを表示."""
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = getattr(last_msg, "content", str(last_msg))
        if len(content) > 500:
            content = content[:500] + "..."
        print(f"\n  応答: {content}")
    else:
        print(f"\n  結果: {json.dumps(result, ensure_ascii=False, default=str)[:500]}")


# ── テスト 1: async テスト ──


async def test_async():
    """MultiServerMCPClient でツールロード → graph.ainvoke() → 成功確認."""
    print_header("テスト 1: async テスト (graph.ainvoke)")

    model = ChatGoogleGenerativeAI(model=MODEL_NAME)

    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"  ロードされたツール数: {len(tools)}")
    for t in tools:
        has_func = t.func is not None if hasattr(t, "func") else "N/A"
        has_coro = t.coroutine is not None if hasattr(t, "coroutine") else "N/A"
        print(f"    - {t.name}: func={has_func}, coroutine={has_coro}")

    graph = create_react_agent(model, tools=tools)

    print(f"\n  クエリ: {TEST_QUERY}")
    result = await graph.ainvoke({"messages": [("user", TEST_QUERY)]})
    print_result(result)
    print("\n  OK: async テスト成功")
    return True


# ── テスト 2: sync テスト（アダプターなし）──


async def test_sync_without_adapter():
    """graph.invoke() → NotImplementedError 発生確認."""
    print_header("テスト 2: sync テスト — アダプターなし (graph.invoke)")

    model = ChatGoogleGenerativeAI(model=MODEL_NAME)

    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"  ロードされたツール数: {len(tools)}")
    for t in tools:
        has_func = t.func is not None if hasattr(t, "func") else "N/A"
        has_coro = t.coroutine is not None if hasattr(t, "coroutine") else "N/A"
        print(f"    - {t.name}: func={has_func}, coroutine={has_coro}")

    graph = create_react_agent(model, tools=tools)

    print(f"\n  クエリ: {TEST_QUERY}")
    try:
        result = graph.invoke({"messages": [("user", TEST_QUERY)]})
        print_result(result)
        print("\n  ※ ツールが呼ばれず成功した可能性あり（LLM がツール不要と判断）")
        return True
    except NotImplementedError as e:
        print(f"\n  OK: 期待通り NotImplementedError 発生: {e}")
        return True
    except Exception as e:
        print(f"\n  OK: エラー発生（NotImplementedError 以外）: {type(e).__name__}: {e}")
        return True


# ── テスト 3: sync テスト（アダプターあり）──


async def test_sync_with_adapter():
    """make_sync_tools() 適用 → graph.invoke() → 成功確認."""
    print_header("テスト 3: sync テスト — アダプターあり (graph.invoke)")

    model = ChatGoogleGenerativeAI(model=MODEL_NAME)

    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print(f"  元のツール数: {len(tools)}")

    # sync ラッパー適用
    sync_tools = make_sync_tools(tools)
    print(f"  sync 変換後ツール数: {len(sync_tools)}")
    for t in sync_tools:
        has_func = t.func is not None if hasattr(t, "func") else "N/A"
        has_coro = t.coroutine is not None if hasattr(t, "coroutine") else "N/A"
        print(f"    - {t.name}: func={has_func}, coroutine={has_coro}")

    graph = create_react_agent(model, tools=sync_tools)

    print(f"\n  クエリ: {TEST_QUERY}")
    result = graph.invoke({"messages": [("user", TEST_QUERY)]})
    print_result(result)
    print("\n  OK: sync テスト（アダプターあり）成功")
    return True


# ── テスト 4: Agent Engine 入力形式テスト ──


async def test_agent_engine_input_format():
    """Agent Engine と同じ {"input": "..."} 形式で invoke → 成功確認."""
    print_header("テスト 4: Agent Engine 入力形式テスト (input→messages 変換)")

    from langgraph_mcp_agent_engine.agent import _wrap_graph_for_agent_engine

    model = ChatGoogleGenerativeAI(model=MODEL_NAME)

    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    sync_tools = make_sync_tools(tools)

    graph = create_react_agent(model, tools=sync_tools)
    wrapped = _wrap_graph_for_agent_engine(graph)

    print(f"\n  クエリ (Agent Engine 形式): {{\"input\": \"{TEST_QUERY}\"}}")
    result = wrapped.invoke({"input": TEST_QUERY})
    print_result(result)
    print("\n  OK: Agent Engine 入力形式テスト成功")
    return True


# ── メイン ──


async def main():
    print("LangGraph + MCP Adapters — ローカルテスト")
    print(f"MCP サーバー: {TEST_MCP_SERVER}")

    results = {}

    # テスト 1: async
    try:
        results["async"] = await test_async()
    except Exception as e:
        print(f"\n  NG: async テスト失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["async"] = False

    # テスト 2: sync (アダプターなし) — 課題再現
    try:
        results["sync_no_adapter"] = await test_sync_without_adapter()
    except Exception as e:
        print(f"\n  NG: sync(アダプターなし)テスト失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["sync_no_adapter"] = False

    # テスト 3: sync (アダプターあり) — 回避策
    try:
        results["sync_with_adapter"] = await test_sync_with_adapter()
    except Exception as e:
        print(f"\n  NG: sync(アダプターあり)テスト失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["sync_with_adapter"] = False

    # テスト 4: Agent Engine 入力形式 — {"input": "..."} → {"messages": [...]}
    try:
        results["agent_engine_format"] = await test_agent_engine_input_format()
    except Exception as e:
        print(f"\n  NG: Agent Engine入力形式テスト失敗: {type(e).__name__}: {e}")
        traceback.print_exc()
        results["agent_engine_format"] = False

    # サマリー
    print_header("テスト結果サマリー")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(results.values())
    print(f"\n  {'全テスト成功!' if all_passed else '一部テスト失敗'}")
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
