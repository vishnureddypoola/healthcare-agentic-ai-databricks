"""Shared configuration for all four projects.

Change values here once; every notebook, agent and app reads from this file
(apps read the same names from environment variables in their app.yaml).
"""
import os

# ---- Unity Catalog layout --------------------------------------------------
CATALOG = os.getenv("CATALOG", "hc_agentic")
SCHEMA_CLINICAL = "clinical"      # FHIR-shaped: patient, encounter, condition, observation, medication_request
SCHEMA_CLAIMS = "claims"          # X12-837-shaped claims, claim lines, denials (CARC), provider inquiries
SCHEMA_TRIALS = "trials"          # clinical trials + parsed eligibility criteria
SCHEMA_PA = "prior_auth"          # PA requests, policies, review queue, determinations, audit
SCHEMA_AI = "ai"                  # column docs, UC functions, eval sets, agent logs

# ---- Models (check Serving > Foundation models in YOUR workspace; names change) --
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-5-5")
LLM_ENDPOINT_CHEAP = os.getenv("LLM_ENDPOINT_CHEAP", "databricks-claude-haiku-5-5")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "databricks-gte-large-en")

# ---- Vector Search / AI Search (renamed "Databricks AI Search" in June 2026; same APIs) --
VS_ENDPOINT = os.getenv("VS_ENDPOINT", "hc-agentic-vs")

# ---- SQL warehouse used by MCP servers and apps -----------------------------
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "815f20eca2cdb0db")

# ---- MLflow ------------------------------------------------------------------
EXPERIMENT_ROOT = os.getenv("EXPERIMENT_ROOT", "/Shared/hc-agentic")


def fq(schema: str, obj: str) -> str:
    """Fully-qualified UC name: catalog.schema.object"""
    return f"{CATALOG}.{schema}.{obj}"
