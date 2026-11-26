# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Workspace setup (run once)
# MAGIC Creates the Unity Catalog catalog, schemas and volume used by all four projects.
# MAGIC Run on **serverless** compute. You need `CREATE CATALOG` on the metastore (you have it as workspace admin on a trial).

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))
from common.config import *

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG} COMMENT 'Healthcare Agentic AI portfolio - SYNTHETIC DATA ONLY'")
for s, c in [
    (SCHEMA_CLINICAL, "FHIR R4-shaped clinical tables (synthetic)"),
    (SCHEMA_CLAIMS, "X12 837/835-shaped claims, CARC denials, provider inquiries (synthetic)"),
    (SCHEMA_TRIALS, "Clinical trials and LLM-parsed eligibility criteria"),
    (SCHEMA_PA, "Prior authorization requests, policies, review queue, audit"),
    (SCHEMA_AI, "AI assets: column docs, UC function tools, eval sets, agent logs"),
]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{s} COMMENT '{c}'")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_AI}.raw_files")
display(spark.sql(f"SHOW SCHEMAS IN {CATALOG}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the Vector Search (AI Search) endpoint
# MAGIC One STANDARD endpoint is shared by Projects 2, 3 and 4.
# MAGIC **Cost warning:** an endpoint bills every hour it exists. Delete it (last cell) when you pause for more than a day.
# MAGIC **Project 1 does not need it** — leave `CREATE_VS_ENDPOINT = False` until you start Project 2.

# COMMAND ----------

# MAGIC %pip install -qU databricks-vectorsearch
# MAGIC %restart_python

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))
from common.config import *
from databricks.vector_search.client import VectorSearchClient

CREATE_VS_ENDPOINT = False  # set True when you start Project 2 (bills hourly while it exists)
if not CREATE_VS_ENDPOINT:
    dbutils.notebook.exit("Skipped AI Search endpoint (not needed for Project 1)")

vsc = VectorSearchClient(disable_notice=True)
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if VS_ENDPOINT not in existing:
    vsc.create_endpoint(name=VS_ENDPOINT, endpoint_type="STANDARD")

import time
while (state := vsc.get_endpoint(VS_ENDPOINT)["endpoint_status"]["state"]) != "ONLINE":
    print("endpoint state:", state); time.sleep(30)
print("VS endpoint ONLINE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke-test the foundation models
# MAGIC If this fails, open **Serving** and copy the exact endpoint names available in your region into `common/config.py`.

# COMMAND ----------

display(spark.sql(f"""
SELECT ai_query('{LLM_ENDPOINT}', 'Reply with exactly: OK') AS llm_check,
       size(ai_query('{EMBEDDING_ENDPOINT}', 'hello')) AS embedding_dim
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Pause cell) delete the VS endpoint to stop billing — indexes must be re-created afterwards
# MAGIC ```python
# MAGIC # vsc.delete_endpoint(VS_ENDPOINT)
# MAGIC ```
