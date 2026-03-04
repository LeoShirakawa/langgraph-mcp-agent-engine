"""MCP Sync Adapter — async-only LangChain ツールに同期 func を追加する.

langchain_mcp_adapters が生成する StructuredTool は coroutine のみを持ち、
func=None のため、Agent Engine の LanggraphAgent.query() (同期 invoke) で
NotImplementedError が発生する。

本モジュールは ADK Runner が採用する「別スレッド + asyncio.run()」パターンで
async→sync ブリッジを実装し、この問題を解消する。
"""

import asyncio
import concurrent.futures
import threading
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


def _run_async_in_thread(coro_func, *args, timeout: float = 120, **kwargs) -> Any:
    """別スレッドで新しいイベントループを作成し、async 関数を同期的に実行する.

    Agent Engine 内部の既存イベントループと衝突しないよう、
    専用スレッドで新規ループを作成して実行する。

    Args:
        coro_func: 実行する async 関数
        *args: 位置引数
        timeout: タイムアウト秒数
        **kwargs: キーワード引数

    Returns:
        async 関数の戻り値

    Raises:
        TimeoutError: タイムアウト
        Exception: async 関数内で発生した例外
    """
    future: concurrent.futures.Future = concurrent.futures.Future()

    def _thread_target():
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(coro_func(*args, **kwargs))
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            loop.close()

    thread = threading.Thread(target=_thread_target, daemon=True)
    thread.start()
    return future.result(timeout=timeout)


def make_sync_tool(async_tool: BaseTool, timeout: float = 60) -> StructuredTool:
    """async-only ツールに同期 func を追加した StructuredTool を返す.

    Args:
        async_tool: langchain_mcp_adapters が生成した async-only ツール
        timeout: 同期実行時のタイムアウト秒数

    Returns:
        func (同期) と coroutine (非同期) の両方を持つ StructuredTool
    """
    original_coroutine = async_tool.coroutine
    if original_coroutine is None:
        # 既に sync func を持っている場合はそのまま返す
        return async_tool

    def sync_func(**kwargs):
        return _run_async_in_thread(original_coroutine, timeout=timeout, **kwargs)

    return StructuredTool(
        name=async_tool.name,
        description=async_tool.description,
        args_schema=async_tool.args_schema,
        func=sync_func,
        coroutine=original_coroutine,
        response_format=getattr(async_tool, "response_format", "content"),
        metadata=async_tool.metadata,
    )


def make_sync_tools(
    async_tools: list[BaseTool], timeout: float = 60
) -> list[StructuredTool]:
    """ツールリスト全体を同期対応に変換する.

    Args:
        async_tools: langchain_mcp_adapters が生成したツールリスト
        timeout: 各ツールのタイムアウト秒数

    Returns:
        同期 func 付きのツールリスト
    """
    return [make_sync_tool(tool, timeout=timeout) for tool in async_tools]
