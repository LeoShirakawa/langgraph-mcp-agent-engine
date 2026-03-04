"""Agent Engine デプロイスクリプト.

課題再現（アダプターなし）と回避策確認（アダプターあり）の2パターンをデプロイ・テスト。

Usage:
    cd ~/work
    python langgraph_mcp_agent_engine/deploy.py --mode with-adapter --action deploy
    python langgraph_mcp_agent_engine/deploy.py --mode with-adapter --action query --query "地域別の売上を教えて"
    python langgraph_mcp_agent_engine/deploy.py --mode without-adapter --action deploy
    python langgraph_mcp_agent_engine/deploy.py --action list
    python langgraph_mcp_agent_engine/deploy.py --action delete --resource-name <name>
"""

import argparse
import json
import os
import sys

# ~/work をパスに追加（langgraph_mcp_agent_engine パッケージを認識させる）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "tactile-octagon-372414")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = "us-central1"

REQUIREMENTS = [
    # langchain-mcp-adapters 0.1.14: requires langchain-core>=0.3.36,<2.0.0
    # langchain 0.3.x has langchain.load module needed by vertexai's LanggraphAgent.query()
    "langchain-mcp-adapters==0.1.14",
    "langchain>=0.3.25,<1.0",
    "langchain-google-genai",
    "langchain-google-vertexai",
    "langgraph",
    "google-cloud-aiplatform>=1.56.0",
    "mcp>=1.9.2",
]

DISPLAY_NAME_PREFIX = "LangGraph MCP Agent"

# ── MCP サーバー設定（Agent Engine 上で使用）──
# Agent Engine ではリモート MCP サーバー（SSE/Streamable HTTP）を使用する。
# テスト用には stdio も可能だが、デプロイ先で Python パスが異なる場合あり。
# ここでは Cloud Run の Snowflake MCP サーバーを想定。
REMOTE_MCP_SERVERS = {
    "snowflake": {
        "url": "https://snowflake-mcp-server-781890406104.us-central1.run.app/mcp/",
        "transport": "streamable_http",
    }
}


def deploy(mode: str, mcp_servers: dict | None = None):
    """Agent Engine にデプロイ."""
    import vertexai
    from vertexai import agent_engines

    vertexai.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=f"gs://{PROJECT_ID}",
    )

    display_name = f"{DISPLAY_NAME_PREFIX} ({mode})"
    print(f"Agent Engine へのデプロイを開始します...")
    print(f"  Project : {PROJECT_ID}")
    print(f"  Location: {LOCATION}")
    print(f"  Display : {display_name}")
    print(f"  Mode    : {mode}")

    from langgraph_mcp_agent_engine.agent import create_agent_with_adapter, create_agent_without_adapter

    servers = mcp_servers or REMOTE_MCP_SERVERS

    if mode == "with-adapter":
        agent = create_agent_with_adapter(mcp_servers=servers)
    else:
        agent = create_agent_without_adapter(mcp_servers=servers)

    # 既存アプリを検索
    remote_app = None
    for item in agent_engines.list():
        if hasattr(item, "display_name") and item.display_name == display_name:
            remote_app = item
            print(f"  既存アプリを発見: {item.resource_name}")
            break

    if not remote_app:
        print("  新規アプリを作成します...")
        remote_app = agent_engines.create(
            agent_engine=agent,
            display_name=display_name,
            description=f"LangGraph + MCP Agent Engine test ({mode})",
            requirements=REQUIREMENTS,
            extra_packages=["./langgraph_mcp_agent_engine"],
        )
        print(f"  作成完了: {remote_app.resource_name}")
    else:
        print("  既存アプリを更新します...")
        remote_app = agent_engines.update(
            agent_engine=agent,
            resource_name=remote_app.resource_name,
            requirements=REQUIREMENTS,
            extra_packages=["./langgraph_mcp_agent_engine"],
        )
        print(f"  更新完了: {remote_app.resource_name}")

    return remote_app


def query(resource_name: str, query_text: str):
    """デプロイ済みエージェントにクエリ."""
    import vertexai
    from vertexai import agent_engines

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print(f"クエリを実行します...")
    print(f"  Resource: {resource_name}")
    print(f"  Query   : {query_text}")

    remote_app = agent_engines.get(resource_name)

    try:
        result = remote_app.query(input=query_text)
        print(f"\n  結果:")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str)[:5000])
    except Exception as e:
        print(f"\n  エラー: {type(e).__name__}: {e}")
        raise


def list_agents():
    """デプロイ済みエージェント一覧."""
    import vertexai
    from vertexai import agent_engines

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    agents = list(agent_engines.list())
    print(f"\nデプロイ済みエージェント ({len(agents)} 件):")
    for a in agents:
        name = getattr(a, "display_name", "N/A")
        rn = getattr(a, "resource_name", "N/A")
        print(f"  - {name}: {rn}")


def delete(resource_name: str):
    """エージェントを削除."""
    import vertexai
    from vertexai import agent_engines

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print(f"削除: {resource_name}")
    agent_engines.delete(resource_name)
    print("  削除完了")


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph MCP Agent Engine デプロイツール"
    )
    parser.add_argument(
        "--action",
        choices=["deploy", "query", "list", "delete"],
        default="deploy",
    )
    parser.add_argument(
        "--mode",
        choices=["with-adapter", "without-adapter"],
        default="with-adapter",
        help="アダプターの有無 (deploy/query 時)",
    )
    parser.add_argument("--query", default="地域別の売上を教えて")
    parser.add_argument("--resource-name", help="query/delete 時のリソース名")
    args = parser.parse_args()

    if args.action == "deploy":
        remote_app = deploy(args.mode)
        print(f"\nデプロイ完了!")
        print(f"Resource Name: {remote_app.resource_name}")
        print(f"\nクエリテスト:")
        print(f"  python deploy.py --action query --resource-name {remote_app.resource_name}")

    elif args.action == "query":
        if not args.resource_name:
            # display_name から検索
            import vertexai
            from vertexai import agent_engines

            vertexai.init(project=PROJECT_ID, location=LOCATION)
            display_name = f"{DISPLAY_NAME_PREFIX} ({args.mode})"
            for item in agent_engines.list():
                if hasattr(item, "display_name") and item.display_name == display_name:
                    args.resource_name = item.resource_name
                    break
            if not args.resource_name:
                print(f"エラー: '{display_name}' が見つかりません。先に deploy してください。")
                sys.exit(1)
        query(args.resource_name, args.query)

    elif args.action == "list":
        list_agents()

    elif args.action == "delete":
        if not args.resource_name:
            print("エラー: --resource-name が必要です")
            sys.exit(1)
        delete(args.resource_name)


if __name__ == "__main__":
    main()
