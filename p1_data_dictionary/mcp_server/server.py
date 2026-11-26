"""Clinical Data Dictionary MCP server — hosted as a Databricks App.

Exposes Unity Catalog metadata + steward-approved LLM docs as MCP tools.
Transport: streamable HTTP at https://<app-url>/mcp (required for Databricks Apps).

Security model
- Every query is parameterised (named markers + IDENTIFIER()) — no string-built SQL from user/LLM input.
- Table names are validated against an allow-list regex and the configured catalog.
- get_sample_rows() masks columns tagged as PHI before returning them.
- The app's service principal needs: USE CATALOG/USE SCHEMA/SELECT on the catalog, CAN USE on the warehouse.
"""
import os
import re
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem, StatementState
from mcp.server.fastmcp import FastMCP

CATALOG = os.getenv("CATALOG", "hc_agentic")
WAREHOUSE_ID = os.environ["DATABRICKS_WAREHOUSE_ID"]
DOCS = f"{CATALOG}.ai.column_docs"
TABLE_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")

w = WorkspaceClient()
mcp = FastMCP(
    "hc-data-dictionary",
    host="0.0.0.0",
    port=int(os.getenv("DATABRICKS_APP_PORT", "8000")),
    stateless_http=True,
)


def run_sql(statement: str, params: dict | None = None, max_wait_s: int = 120) -> list[dict]:
    resp = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE_ID,
        statement=statement,
        catalog=CATALOG,
        parameters=[StatementParameterListItem(name=k, value=str(v)) for k, v in (params or {}).items()],
        wait_timeout="30s",
    )
    waited = 0
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING) and waited < max_wait_s:
        time.sleep(2); waited += 2
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state != StatementState.SUCCEEDED:
        msg = resp.status.error.message if resp.status.error else str(resp.status.state)
        raise RuntimeError(f"SQL failed: {msg}")
    cols = [c.name for c in resp.manifest.schema.columns]
    return [dict(zip(cols, row)) for row in (resp.result.data_array or [])]


def _check_table(schema_dot_table: str) -> tuple[str, str]:
    t = schema_dot_table.strip().lower().removeprefix(f"{CATALOG}.")
    if not TABLE_RE.match(t):
        raise ValueError("Use the form '<schema>.<table>', e.g. 'clinical.observation'")
    s, n = t.split(".")
    return s, n


@mcp.tool()
def list_tables(schema: str = "") -> list[dict]:
    """List documented tables (optionally in one schema: clinical, claims, trials, prior_auth) with their descriptions."""
    return run_sql(
        f"""SELECT table_schema, table_name, comment AS description
            FROM {CATALOG}.information_schema.tables
            WHERE table_schema NOT IN ('information_schema')
              AND (:schema = '' OR table_schema = :schema)
            ORDER BY 1, 2""",
        {"schema": schema.lower()},
    )


@mcp.tool()
def describe_table(table: str) -> list[dict]:
    """Column-level dictionary for one table ('schema.table'): type, description, business definition,
    PHI classification, sensitivity, code system and FHIR mapping."""
    s, n = _check_table(table)
    return run_sql(
        f"""SELECT column_name, data_type, description, business_definition, phi_identifier, sensitivity,
                   code_system, fhir_mapping, review_status
            FROM {DOCS} WHERE table_schema = :s AND table_name = :n""",
        {"s": s, "n": n},
    )


@mcp.tool()
def search_data_dictionary(keyword: str, limit: int = 20) -> list[dict]:
    """Find columns whose name, description, business definition or code system mention a keyword
    (e.g. 'HbA1c', 'denial', 'LOINC', 'birth'). Use this when the user does not know which table holds something."""
    return run_sql(
        f"""SELECT table_schema, table_name, column_name, description, code_system, sensitivity
            FROM {DOCS}
            WHERE lower(concat_ws(' ', column_name, description, business_definition, code_system, example_question))
                  LIKE concat('%', lower(:kw), '%')
            LIMIT {max(1, min(int(limit), 50))}""",
        {"kw": keyword},
    )


@mcp.tool()
def list_phi_columns(schema: str = "") -> list[dict]:
    """List every column classified as PHI / a HIPAA identifier, optionally filtered to one schema."""
    return run_sql(
        f"""SELECT table_schema, table_name, column_name, phi_identifier, sensitivity
            FROM {DOCS}
            WHERE (phi_identifier <> 'NONE' OR sensitivity = 'PHI') AND (:schema = '' OR table_schema = :schema)
            ORDER BY 1, 2, 3""",
        {"schema": schema.lower()},
    )


@mcp.tool()
def get_sample_rows(table: str, limit: int = 5) -> list[dict]:
    """Return a few example rows from a table. Columns classified as direct identifiers are masked as '***'."""
    s, n = _check_table(table)
    cols = run_sql(f"SELECT column_name, phi_identifier FROM {DOCS} WHERE table_schema=:s AND table_name=:n", {"s": s, "n": n})
    if not cols:
        raise ValueError("Unknown or undocumented table")
    select = ", ".join(
        f"'***' AS `{c['column_name']}`" if c["phi_identifier"] not in ("NONE", None) else f"`{c['column_name']}`"
        for c in cols
    )
    return run_sql(f"SELECT {select} FROM IDENTIFIER(:tbl) LIMIT {max(1, min(int(limit), 10))}", {"tbl": f"{CATALOG}.{s}.{n}"})


@mcp.tool()
def get_table_lineage(table: str) -> list[dict]:
    """Upstream and downstream tables for 'schema.table' from Unity Catalog lineage (system.access.table_lineage)."""
    s, n = _check_table(table)
    full = f"{CATALOG}.{s}.{n}"
    try:
        return run_sql(
            """SELECT DISTINCT
                 CASE WHEN target_table_full_name = :t THEN 'upstream' ELSE 'downstream' END AS direction,
                 CASE WHEN target_table_full_name = :t THEN source_table_full_name ELSE target_table_full_name END AS related_table,
                 entity_type
               FROM system.access.table_lineage
               WHERE (target_table_full_name = :t OR source_table_full_name = :t)
                 AND source_table_full_name IS NOT NULL AND target_table_full_name IS NOT NULL
                 AND event_date >= current_date() - INTERVAL 90 DAYS""",
            {"t": full},
        )
    except RuntimeError as e:
        return [{"note": f"Lineage unavailable (system tables may not be enabled or granted): {e}"}]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
