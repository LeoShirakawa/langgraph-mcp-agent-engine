# LangGraph + MCP on Agent Engine — 課題と Workaround

## 課題: async/sync の不整合

Agent Engine の `LanggraphAgent.query()` は **同期 invoke** のみだが、`langchain_mcp_adapters` は **async-only** のツールを生成するため、MCP ツール呼び出し時に `NotImplementedError` が発生する。

```mermaid
graph LR
    A["langchain_mcp_adapters<br/>StructuredTool<br/>func = None<br/>coroutine = async"] -- "同期呼び出し" --> B["StructuredTool._run()"]
    B -- "func が None" --> C["NotImplementedError"]

    style A fill:#fff8f0,stroke:#ffaa44
    style C fill:#ff6666,color:#fff
```

## 根本原因: 3つのコンポーネントの不整合

```mermaid
graph TB
    subgraph C3["Agent Engine (LanggraphAgent)"]
        T3["query() は runnable.invoke() のみ\n= 同期呼び出し"]
    end

    subgraph C1["langchain_mcp_adapters"]
        T1["StructuredTool を生成\ncoroutine = call_tool\nfunc = None"]
    end

    subgraph C2["langchain_core (StructuredTool)"]
        T2["_run() で func が None\n→ NotImplementedError"]
    end

    T3 -- "同期 invoke" --> T2
    T1 -- "func=None のツール" --> T2

    style C1 fill:#fff8f0,stroke:#ffaa44
    style C2 fill:#fff0f0,stroke:#ff6666
    style C3 fill:#f0f0ff,stroke:#6666ff
```

## Workaround 適用前 → 適用後

```mermaid
flowchart TB
    subgraph BEFORE["適用前: NotImplementedError"]
        direction TB
        B1["LanggraphAgent.query()"] --> B2["runnable.invoke()"]
        B2 --> B3["StructuredTool._run()"]
        B3 --> B4["func = None"]
        B4 --> B5["NotImplementedError"]
    end

    subgraph AFTER["適用後: 正常動作"]
        direction TB
        A1["LanggraphAgent.query()"] --> A2["runnable.invoke()"]
        A2 --> A3["入力形式変換<br/>{'input':'...'} → {'messages':[...]}"]
        A3 --> A4["StructuredTool._run()"]
        A4 --> A5["func = sync_func"]
        A5 --> A6["別スレッド +<br/>new_event_loop()"]
        A6 --> A7["await coroutine()"]
        A7 --> A8["MCP サーバー"]
        A8 --> A9["正常応答"]
    end

    style BEFORE fill:#fff0f0,stroke:#ff6666
    style AFTER fill:#f0fff0,stroke:#66cc66
    style B5 fill:#ff6666,color:#fff
    style A9 fill:#66cc66,color:#fff
```

## Workaround の仕組み: async→sync ブリッジ

```mermaid
sequenceDiagram
    participant AE as Agent Engine<br/>(同期環境)
    participant Tool as StructuredTool<br/>(sync_func 追加済み)
    participant Thread as 別スレッド
    participant EvLoop as 新規 event loop
    participant MCP as MCP サーバー

    AE->>Tool: tool._run() [同期]
    Tool->>Thread: threading.Thread 起動
    Thread->>EvLoop: asyncio.new_event_loop()
    EvLoop->>MCP: await coroutine(**kwargs)
    MCP-->>EvLoop: 結果
    EvLoop-->>Thread: run_until_complete 完了
    Thread-->>Tool: Future.result() で取得
    Tool-->>AE: 結果を返却
```

## コア実装

### make_sync_tool: async ツールに同期 func を追加

```python
def make_sync_tool(async_tool, timeout=60):
    original_coroutine = async_tool.coroutine

    def sync_func(**kwargs):
        # 別スレッドで新規イベントループを作成し async を実行
        return _run_async_in_thread(original_coroutine, timeout=timeout, **kwargs)

    return StructuredTool(
        func=sync_func,               # ← 同期 func を追加
        coroutine=original_coroutine,  # ← async も保持
        ...
    )
```

### ツール属性の変化

```mermaid
graph LR
    subgraph BEFORE["変換前"]
        BF["func = None"]
        BC["coroutine = async"]
    end

    subgraph AFTER["変換後"]
        AF["func = sync_func"]
        AC["coroutine = async"]
    end

    BEFORE -- "make_sync_tool()" --> AFTER

    style BF fill:#ff6666,color:#fff
    style AF fill:#66cc66,color:#fff
    style BC fill:#66cc66,color:#fff
    style AC fill:#66cc66,color:#fff
```

## 検証結果

| テスト | 結果 |
|--------|------|
| ローカル: async (`ainvoke`) | **PASS** |
| ローカル: sync アダプターなし (`invoke`) | **PASS** — `NotImplementedError` 再現 |
| ローカル: sync アダプターあり (`invoke`) | **PASS** — 正常動作 |
| Agent Engine: アダプターなし | **`NotImplementedError`** 発生 |
| Agent Engine: アダプターあり | **正常応答** — Snowflake 実データ取得成功 |
