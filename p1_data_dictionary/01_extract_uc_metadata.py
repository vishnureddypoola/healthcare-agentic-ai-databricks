# Databricks notebook source
# MAGIC %md
# MAGIC # P1 · Step 1 — Extract Unity Catalog metadata + profile every column
# MAGIC Reads `information_schema` for the whole catalog, then profiles each column (null %, distinct count, sample values).
# MAGIC
# MAGIC **Governance design choice (talk about this in interviews):** sample values from columns that *look* like
# MAGIC direct identifiers are **masked before they ever reach the LLM** — the HIPAA "minimum necessary" principle applied to
# MAGIC prompt construction. The rule-based PHI guess is later cross-checked against the LLM's classification.

# COMMAND ----------

import sys, os, re
sys.path.append(os.path.abspath(".."))
from common.config import *
from pyspark.sql import functions as F

TARGET_SCHEMAS = [SCHEMA_CLINICAL, SCHEMA_CLAIMS, SCHEMA_TRIALS, SCHEMA_PA]

cols = spark.sql(f"""
  SELECT c.table_schema, c.table_name, c.column_name, c.data_type, c.is_nullable, c.ordinal_position,
         c.comment AS existing_column_comment, t.comment AS table_comment
  FROM {CATALOG}.information_schema.columns c
  JOIN {CATALOG}.information_schema.tables t
    ON c.table_schema = t.table_schema AND c.table_name = t.table_name
  WHERE c.table_schema IN ({",".join(f"'{s}'" for s in TARGET_SCHEMAS)})
  ORDER BY c.table_schema, c.table_name, c.ordinal_position
""")
display(cols)

# COMMAND ----------

# Rule-based first pass: HIPAA Safe Harbor-style identifier patterns on column NAMES
PHI_NAME_RULES = {
    "NAME": r"(first|last|full)_?name",
    "DATE": r"birth|dob|_date$|_ts$|authored_on|period_start",
    "GEOGRAPHIC": r"address|postal|zip|city",
    "PHONE": r"phone|fax",
    "MEDICAL_RECORD_NUMBER": r"\bmrn\b|^mrn",
    "HEALTH_PLAN_BENEFICIARY_NUMBER": r"member_id|subscriber",
    "FREE_TEXT_MAY_CONTAIN_PHI": r"note_text|body|provider_statement",
}

def rule_phi(col: str) -> str:
    for label, pat in PHI_NAME_RULES.items():
        if re.search(pat, col.lower()):
            return label
    return "NONE"

def mask(v: str, phi: str) -> str:
    if v is None or phi == "NONE":
        return v
    if phi == "DATE":
        return v[:4] + "-XX-XX"  # keep year only
    return re.sub(r"[A-Za-z0-9]", "x", v[:3]) + "…"

# COMMAND ----------

profiles = []
for (schema, table), group in cols.toPandas().groupby(["table_schema", "table_name"]):
    df = spark.table(f"{CATALOG}.{schema}.{table}")
    n = df.count()
    for _, r in group.iterrows():
        c = r["column_name"]
        stats = df.agg(F.countDistinct(c).alias("d"), F.sum(F.col(c).isNull().cast("int")).alias("nulls")).first()
        samples = [str(x[0]) for x in df.select(F.col(c).cast("string")).where(F.col(c).isNotNull()).distinct().limit(5).collect()]
        phi = rule_phi(c)
        profiles.append(dict(
            table_schema=schema, table_name=table, column_name=c, data_type=r["data_type"],
            ordinal_position=int(r["ordinal_position"]), table_comment=r["table_comment"],
            row_count=n, distinct_count=int(stats["d"]), null_pct=round(100 * (stats["nulls"] or 0) / max(n, 1), 2),
            sample_values_masked=[mask(s[:60], phi) for s in samples], rule_phi_guess=phi,
            sibling_columns=", ".join(group["column_name"].tolist()),
        ))

prof_df = spark.createDataFrame(profiles)
prof_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(SCHEMA_AI, "column_profile"))
display(spark.table(fq(SCHEMA_AI, "column_profile")))
