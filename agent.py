"""LangGraph エージェント定義 — MCP ツール + Agent Engine 対応.

2つのモードを提供:
  1. build_without_adapter: MCP ツールをそのまま（async-only）使用 → 課題再現用
  2. build_with_adapter:    MCP ツールを sync ラッパーで包む → 回避策適用

Usage:
    from agent import create_agent_without_adapter, create_agent_with_adapter
"""

import asyncio
import concurrent.futures
import os
import sys
import threading
from typing import Any, Mapping, Optional, Sequence

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from langgraph_mcp_agent_engine.mcp_sync_adapter import _run_async_in_thread, make_sync_tools

# ── MCP サーバー設定 ──

# テスト用 stdio サーバーのパス
_TEST_MCP_SERVER = os.path.join(os.path.dirname(__file__), "test_mcp_server.py")

# デフォルト MCP サーバー設定（テスト用 stdio）
DEFAULT_MCP_SERVERS = {
    "snowflake": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [_TEST_MCP_SERVER],
    }
}


def _transform_input(input_dict: dict) -> dict:
    """Agent Engine の入力形式を create_react_agent の形式に変換する.

    LanggraphAgent.query() は文字列入力を {"input": "..."} に変換するが、
    create_react_agent は {"messages": [("user", "...")]} を期待する。
    """
    if "messages" in input_dict:
        return input_dict
    user_input = input_dict.get("input", "")
    return {"messages": [("user", user_input)]}


def _wrap_graph_for_agent_engine(graph):
    """create_react_agent のグラフを Agent Engine の入力形式に対応させる."""
    return RunnableLambda(_transform_input) | graph


def _load_mcp_tools(mcp_servers: dict[str, dict]) -> list[BaseTool]:
    """MCP サーバーからツールをロードする（同期）.

    MultiServerMCPClient 自体が async なので、
    _run_async_in_thread で同期化して呼び出す。
    """

    async def _async_load():
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(mcp_servers)
        tools = await client.get_tools()
        return tools

    return _run_async_in_thread(_async_load, timeout=30)


# ── runnable_builder: アダプターなし（課題再現用）──


def build_without_adapter(
    *,
    model: BaseLanguageModel,
    tools: Optional[Sequence[BaseTool]] = None,
    checkpointer: Any = None,
    model_tool_kwargs: Optional[Mapping[str, Any]] = None,
    runnable_kwargs: Optional[Mapping[str, Any]] = None,
    mcp_servers: Optional[dict[str, dict]] = None,
):
    """MCP ツールをそのまま（async-only）で使用する runnable_builder.

    Agent Engine の query() (同期 invoke) でツール呼び出し時に
    NotImplementedError が発生する — 課題再現用。
    """
    servers = mcp_servers or DEFAULT_MCP_SERVERS
    mcp_tools = _load_mcp_tools(servers)

    all_tools = list(tools or []) + mcp_tools
    graph = create_react_agent(model, tools=all_tools, checkpointer=checkpointer)
    return _wrap_graph_for_agent_engine(graph)


# ── runnable_builder: アダプターあり（回避策）──


def build_with_adapter(
    *,
    model: BaseLanguageModel,
    tools: Optional[Sequence[BaseTool]] = None,
    checkpointer: Any = None,
    model_tool_kwargs: Optional[Mapping[str, Any]] = None,
    runnable_kwargs: Optional[Mapping[str, Any]] = None,
    mcp_servers: Optional[dict[str, dict]] = None,
):
    """MCP ツールを sync ラッパーで包んで使用する runnable_builder.

    make_sync_tools() で各ツールに同期 func を追加するため、
    Agent Engine の query() (同期 invoke) でも正常動作する。
    """
    servers = mcp_servers or DEFAULT_MCP_SERVERS
    mcp_tools = _load_mcp_tools(servers)

    # async-only ツールに sync func を追加
    sync_mcp_tools = make_sync_tools(mcp_tools)

    all_tools = list(tools or []) + sync_mcp_tools
    graph = create_react_agent(model, tools=all_tools, checkpointer=checkpointer)
    return _wrap_graph_for_agent_engine(graph)


# ── ファクトリ関数 ──


def create_agent_without_adapter(
    model: str = "gemini-2.5-flash",
    mcp_servers: Optional[dict[str, dict]] = None,
):
    """課題再現用エージェントを作成（アダプターなし）."""
    from vertexai.agent_engines import LanggraphAgent

    servers = mcp_servers or DEFAULT_MCP_SERVERS

    def _builder(*, model, tools, checkpointer, model_tool_kwargs, runnable_kwargs):
        return build_without_adapter(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
            model_tool_kwargs=model_tool_kwargs,
            runnable_kwargs=runnable_kwargs,
            mcp_servers=servers,
        )

    return LanggraphAgent(
        model=model,
        runnable_builder=_builder,
    )


def create_agent_with_adapter(
    model: str = "gemini-2.5-flash",
    mcp_servers: Optional[dict[str, dict]] = None,
):
    """回避策適用エージェントを作成（アダプターあり）."""
    from vertexai.agent_engines import LanggraphAgent

    servers = mcp_servers or DEFAULT_MCP_SERVERS

    def _builder(*, model, tools, checkpointer, model_tool_kwargs, runnable_kwargs):
        return build_with_adapter(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
            model_tool_kwargs=model_tool_kwargs,
            runnable_kwargs=runnable_kwargs,
            mcp_servers=servers,
        )

    return LanggraphAgent(
        model=model,
        runnable_builder=_builder,
    )
