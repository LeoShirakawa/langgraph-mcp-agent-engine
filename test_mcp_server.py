"""テスト用 FastMCP サーバー (stdio).

Snowflake MCP サーバーと同等のツールをモックデータで提供する。
ローカルテストおよび課題再現に使用。

Usage:
    python test_mcp_server.py          # stdio モードで起動
"""

import json
from datetime import date, timedelta
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test-snowflake-mcp")

# ── モックデータ ──

MOCK_TABLES = [
    {"schema": "RAW_DATA", "name": "SALES_TRANSACTIONS", "type": "TABLE"},
    {"schema": "RAW_DATA", "name": "MODELS", "type": "TABLE"},
    {"schema": "RAW_DATA", "name": "DEALERS", "type": "TABLE"},
    {"schema": "RAW_DATA", "name": "SALES_SUMMARY_VIEW", "type": "VIEW"},
]

MOCK_SCHEMA = {
    "SALES_TRANSACTIONS": [
        {"column_name": "TRANSACTION_ID", "data_type": "VARCHAR", "nullable": False},
        {"column_name": "SALE_DATE", "data_type": "DATE", "nullable": False},
        {"column_name": "REGION", "data_type": "VARCHAR", "nullable": False},
        {"column_name": "COUNTRY_CODE", "data_type": "VARCHAR", "nullable": False},
        {"column_name": "MODEL_NAME", "data_type": "VARCHAR", "nullable": False},
        {"column_name": "CATEGORY", "data_type": "VARCHAR", "nullable": True},
        {"column_name": "ENGINE_TYPE", "data_type": "VARCHAR", "nullable": True},
        {"column_name": "UNIT_PRICE_USD", "data_type": "NUMBER", "nullable": False},
        {"column_name": "QUANTITY", "data_type": "NUMBER", "nullable": False},
        {"column_name": "TOTAL_AMOUNT_USD", "data_type": "NUMBER", "nullable": False},
    ],
}

MOCK_SALES_DATA = [
    {
        "TRANSACTION_ID": "TXN-001",
        "SALE_DATE": "2025-01-15",
        "REGION": "North America",
        "COUNTRY_CODE": "US",
        "MODEL_NAME": "Ariya",
        "CATEGORY": "SUV",
        "ENGINE_TYPE": "EV",
        "UNIT_PRICE_USD": 45000.00,
        "QUANTITY": 1,
        "TOTAL_AMOUNT_USD": 45000.00,
    },
    {
        "TRANSACTION_ID": "TXN-002",
        "SALE_DATE": "2025-01-20",
        "REGION": "Europe",
        "COUNTRY_CODE": "DE",
        "MODEL_NAME": "Qashqai",
        "CATEGORY": "SUV",
        "ENGINE_TYPE": "Hybrid",
        "UNIT_PRICE_USD": 35000.00,
        "QUANTITY": 2,
        "TOTAL_AMOUNT_USD": 70000.00,
    },
    {
        "TRANSACTION_ID": "TXN-003",
        "SALE_DATE": "2025-02-10",
        "REGION": "Asia",
        "COUNTRY_CODE": "JP",
        "MODEL_NAME": "Note",
        "CATEGORY": "Compact",
        "ENGINE_TYPE": "e-POWER",
        "UNIT_PRICE_USD": 22000.00,
        "QUANTITY": 3,
        "TOTAL_AMOUNT_USD": 66000.00,
    },
    {
        "TRANSACTION_ID": "TXN-004",
        "SALE_DATE": "2025-03-05",
        "REGION": "North America",
        "COUNTRY_CODE": "US",
        "MODEL_NAME": "Rogue",
        "CATEGORY": "SUV",
        "ENGINE_TYPE": "ICE",
        "UNIT_PRICE_USD": 32000.00,
        "QUANTITY": 1,
        "TOTAL_AMOUNT_USD": 32000.00,
    },
    {
        "TRANSACTION_ID": "TXN-005",
        "SALE_DATE": "2025-03-15",
        "REGION": "Europe",
        "COUNTRY_CODE": "FR",
        "MODEL_NAME": "Juke",
        "CATEGORY": "Compact SUV",
        "ENGINE_TYPE": "Hybrid",
        "UNIT_PRICE_USD": 28000.00,
        "QUANTITY": 2,
        "TOTAL_AMOUNT_USD": 56000.00,
    },
]


@mcp.tool()
async def execute_query(query: str, limit: int = 100) -> str:
    """Execute a SELECT query on Snowflake and return results. Only SELECT statements are allowed."""
    query_upper = query.upper().strip()
    if not query_upper.startswith("SELECT"):
        return json.dumps({"error": "Only SELECT statements are allowed."}, ensure_ascii=False)

    # モック: 全データを返す（実際の SQL パースはしない）
    data = MOCK_SALES_DATA[:limit]
    result = {
        "columns": list(MOCK_SALES_DATA[0].keys()),
        "row_count": len(data),
        "data": data,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_schema(table_name: str, schema_name: str = "RAW_DATA") -> str:
    """Get the schema (column definitions) of a table."""
    columns = MOCK_SCHEMA.get(table_name.upper(), [])
    if not columns:
        return json.dumps(
            {"error": f"Table {schema_name}.{table_name} not found."},
            ensure_ascii=False,
        )
    result = {"table": f"{schema_name}.{table_name}", "columns": columns}
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def list_tables(schema_name: str = "") -> str:
    """List all tables and views in the database."""
    objects = MOCK_TABLES
    if schema_name:
        objects = [o for o in objects if o["schema"] == schema_name.upper()]
    return json.dumps({"objects": objects}, ensure_ascii=False)


@mcp.tool()
async def get_sample_data(table_name: str, limit: int = 5) -> str:
    """Get sample data from a table."""
    data = MOCK_SALES_DATA[:limit]
    result = {
        "columns": list(MOCK_SALES_DATA[0].keys()),
        "row_count": len(data),
        "data": data,
    }
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_sales_summary(
    group_by: str = "REGION",
    year: int = 0,
    region: str = "",
) -> str:
    """Get sales summary grouped by a specified column."""
    valid_columns = ["REGION", "COUNTRY_CODE", "MODEL_NAME", "CATEGORY", "ENGINE_TYPE"]
    group_by_upper = group_by.upper()
    if group_by_upper not in valid_columns:
        return json.dumps(
            {"error": f"Invalid group_by. Use: {valid_columns}"},
            ensure_ascii=False,
        )

    filtered = MOCK_SALES_DATA
    if year:
        filtered = [r for r in filtered if r["SALE_DATE"].startswith(str(year))]
    if region:
        filtered = [r for r in filtered if r["REGION"] == region]

    # 集計
    groups: dict = {}
    for row in filtered:
        key = row.get(group_by_upper, "Unknown")
        if key not in groups:
            groups[key] = {"units_sold": 0, "total_revenue": 0.0, "prices": []}
        groups[key]["units_sold"] += row["QUANTITY"]
        groups[key]["total_revenue"] += row["TOTAL_AMOUNT_USD"]
        groups[key]["prices"].append(row["UNIT_PRICE_USD"])

    summary = []
    for key, vals in sorted(groups.items(), key=lambda x: -x[1]["units_sold"]):
        summary.append({
            group_by_upper: key,
            "UNITS_SOLD": vals["units_sold"],
            "TOTAL_REVENUE_USD": round(vals["total_revenue"], 2),
            "AVG_UNIT_PRICE": round(sum(vals["prices"]) / len(vals["prices"]), 2),
        })

    result = {
        "columns": [group_by_upper, "UNITS_SOLD", "TOTAL_REVENUE_USD", "AVG_UNIT_PRICE"],
        "row_count": len(summary),
        "data": summary,
    }
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
