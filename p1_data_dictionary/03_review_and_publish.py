# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # P1 · Step 3 — Human review, then publish to Unity Catalog (comments + tags)
# MAGIC Pattern: **LLM drafts → steward approves → system of record updated.** The LLM never writes to UC directly.
# MAGIC
# MAGIC 1. Auto-approve only rows with `confidence >= 0.85` AND no PHI disagreement.
# MAGIC 2. Review the rest yourself (edit the UPDATE cell) — this is your "human-in-the-loop".
# MAGIC 3. Publish: `COMMENT` on every column, and governed tags `phi` / `sensitivity` / `code_system`.

# COMMAND ----------

# DBTITLE 1,Auto-approve high-confidence docs
import sys, os
sys.path.append(os.path.abspath(".."))
from common.config import *
DOCS = fq(SCHEMA_AI, "column_docs")
ME = spark.sql("SELECT current_user()").first()[0]
PARSE_OK_FILTER = " AND parse_ok" if "parse_ok" in spark.table(DOCS).columns else ""

spark.sql(f"""
UPDATE {DOCS}
SET review_status = 'APPROVED', reviewer = 'auto:confidence>=0.85'
WHERE review_status = 'PENDING_REVIEW' AND confidence >= 0.85 AND NOT phi_disagreement{PARSE_OK_FILTER}
""")
display(spark.sql(f"SELECT review_status, count(*) n FROM {DOCS} GROUP BY 1"))

# COMMAND ----------

# MAGIC %md ### Manual review — look at each pending row, fix fields if needed, then approve

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {DOCS} WHERE review_status = 'PENDING_REVIEW' ORDER BY confidence"))

# COMMAND ----------

# Example manual corrections (edit to match what you see):
spark.sql(f"""
UPDATE {DOCS}
SET phi_identifier = 'DATE', sensitivity = 'PHI', review_status = 'APPROVED', reviewer = '{ME}'
WHERE column_name IN ('birth_date', 'note_date', 'performed_date') AND review_status = 'PENDING_REVIEW'
""")
# Approve the rest after you have read them:
# spark.sql(f"UPDATE {DOCS} SET review_status='APPROVED', reviewer='{ME}' WHERE review_status='PENDING_REVIEW'")

# COMMAND ----------

# MAGIC %md ### Publish approved docs to Unity Catalog

# COMMAND ----------

def q(s: str) -> str:
    return (s or "").replace("'", "\\'")[:1000]

approved = spark.sql(f"SELECT * FROM {DOCS} WHERE review_status = 'APPROVED'").collect()
for r in approved:
    tbl = f"{CATALOG}.{r.table_schema}.{r.table_name}"
    spark.sql(f"ALTER TABLE {tbl} ALTER COLUMN `{r.column_name}` COMMENT '{q(r.description)}'")
    tags = {"sensitivity": r.sensitivity, "phi": r.phi_identifier, "code_system": r.code_system}
    tag_sql = ", ".join(f"'{k}' = '{q(v)}'" for k, v in tags.items() if v)
    spark.sql(f"ALTER TABLE {tbl} ALTER COLUMN `{r.column_name}` SET TAGS ({tag_sql})")
print(f"Published {len(approved)} column docs")

# COMMAND ----------

# MAGIC %md ### Table-level descriptions (LLM summarises the approved column docs)

# COMMAND ----------

tables = spark.sql(f"""
SELECT table_schema, table_name,
  ai_query('{LLM_ENDPOINT}', concat(
    'Write a 2-sentence description of this healthcare table for a data catalog. Mention grain (one row per ...) and main use. Columns: ',
    array_join(collect_list(concat(column_name, ': ', description)), '; '))) AS table_description
FROM {DOCS} WHERE review_status = 'APPROVED'
GROUP BY table_schema, table_name
""").collect()
for t in tables:
    spark.sql(f"COMMENT ON TABLE {CATALOG}.{t.table_schema}.{t.table_name} IS '{q(t.table_description)}'")
    spark.sql(f"ALTER TABLE {CATALOG}.{t.table_schema}.{t.table_name} SET TAGS ('documented_by' = 'llm+steward')")

# COMMAND ----------

# MAGIC %md
# MAGIC ✅ Open **Catalog Explorer** on `hc_agentic.clinical.patient` — you should see column comments and `phi` tags.
# MAGIC
# MAGIC **Stretch (strong interview story):** add an ABAC policy / column mask so any column tagged `phi=NAME` is masked for
# MAGIC users outside a `phi_readers` group. Then the chat app (running on-behalf-of the user) automatically respects it.