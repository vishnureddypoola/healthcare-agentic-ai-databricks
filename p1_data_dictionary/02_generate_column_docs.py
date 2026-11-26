# Databricks notebook source
# MAGIC %md
# MAGIC # P1 · Step 2 — LLM-generated column documentation (batch, with `ai_query`)
# MAGIC One SQL statement documents every column in parallel, and the output is parsed into typed columns.
# MAGIC
# MAGIC **Model-aware output contract.** Newer Claude models reject `ai_query`'s `responseFormat` (strict structured output
# MAGIC is implemented with forced tool use, which they refuse). So:
# MAGIC * **Claude models** → the JSON contract (same fields, same enums) goes in the prompt, then `from_json` parses it.
# MAGIC * **Models that support structured output** (e.g. `databricks-gpt-oss-120b`) → strict `responseFormat`.
# MAGIC * Either way, rows that fail to parse get `parse_ok = false`, confidence 0, and go to human review — never silently dropped.
# MAGIC
# MAGIC Every row lands as `PENDING_REVIEW`. Nothing is written to Unity Catalog comments/tags until a human approves (Step 3).

# COMMAND ----------

import sys, os, json
sys.path.append(os.path.abspath(".."))
from common.config import *

# Default: GPT-OSS 120B — supports strict responseFormat, so the JSON contract is enforced by the platform.
# Claude (LLM_ENDPOINT) also works here via the JSON-in-prompt path; Claude stays the model for agent reasoning.
dbutils.widgets.text("doc_model", "databricks-gpt-oss-120b")
DOC_MODEL = dbutils.widgets.get("doc_model")
USE_RESPONSE_FORMAT = "claude" not in DOC_MODEL.lower()
print(f"Model: {DOC_MODEL} | output contract: {'strict responseFormat' if USE_RESPONSE_FORMAT else 'JSON-in-prompt + from_json validation'}")

PHI_ENUM = ["NONE", "NAME", "DATE", "GEOGRAPHIC", "PHONE", "EMAIL", "SSN", "MEDICAL_RECORD_NUMBER",
            "HEALTH_PLAN_BENEFICIARY_NUMBER", "ACCOUNT_NUMBER", "DEVICE_ID", "FREE_TEXT_MAY_CONTAIN_PHI", "OTHER_UNIQUE_ID"]
SENS_ENUM = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "PHI"]

SCHEMA = {
    "type": "object",
    "properties": {
        "description":         {"type": "string", "description": "One clear sentence a non-technical analyst understands."},
        "business_definition": {"type": "string", "description": "How the business or clinic uses this field, incl. caveats."},
        "phi_identifier":      {"type": "string", "enum": PHI_ENUM},
        "sensitivity":         {"type": "string", "enum": SENS_ENUM},
        "code_system":         {"type": "string", "description": "ICD-10-CM, SNOMED CT, LOINC, RxNorm, CPT, HCPCS, CARC, NPI, or NONE"},
        "fhir_mapping":        {"type": "string", "description": "Closest FHIR R4 element path, e.g. Observation.valueQuantity.value, or NONE"},
        "example_question":    {"type": "string", "description": "A natural-language question this column helps answer."},
        "confidence":          {"type": "number", "description": "0-1 self-rated confidence"},
    },
    "required": ["description", "business_definition", "phi_identifier", "sensitivity", "code_system",
                 "fhir_mapping", "example_question", "confidence"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = json.dumps({"type": "json_schema", "json_schema": {"name": "column_doc", "strict": True, "schema": SCHEMA}})
DDL = ("description STRING, business_definition STRING, phi_identifier STRING, sensitivity STRING, code_system STRING, "
       "fhir_mapping STRING, example_question STRING, confidence DOUBLE")

JSON_CONTRACT = (
    "Return ONLY one JSON object, no markdown fences, no text before or after, with exactly these keys: "
    "description (one clear sentence for a non-technical analyst), business_definition (how the business uses it, incl. caveats), "
    f"phi_identifier (one of: {', '.join(PHI_ENUM)}), sensitivity (one of: {', '.join(SENS_ENUM)}), "
    "code_system (ICD-10-CM, SNOMED CT, LOINC, RxNorm, CPT, HCPCS, CARC, NPI or NONE), "
    "fhir_mapping (closest FHIR R4 element path or NONE), example_question (a question this column answers), "
    "confidence (number between 0 and 1)."
)

def sql_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")

# COMMAND ----------

contract_suffix = "" if USE_RESPONSE_FORMAT else ", '\\n\\n" + sql_str(JSON_CONTRACT) + "'"

prompt_sql = f"""
concat(
  'You are a senior healthcare data steward documenting a Unity Catalog lakehouse built on FHIR R4 and X12 claims. ',
  'Document ONE column. Be precise; never invent business rules you cannot infer. If unsure, say so and lower confidence. ',
  'Classify PHI using the HIPAA Safe Harbor identifier categories. Clinical codes alone (e.g. ICD-10) are NOT direct identifiers, ',
  'but any column that links to an individual in a clinical table is sensitivity=PHI.\\n\\n',
  'Table: {CATALOG}.', table_schema, '.', table_name, '\\n',
  'Table comment: ', coalesce(table_comment, 'n/a'), '\\n',
  'Other columns in table: ', sibling_columns, '\\n',
  'Column: ', column_name, ' (', data_type, ')\\n',
  'Profile: rows=', row_count, ', distinct=', distinct_count, ', null%=', null_pct, '\\n',
  'Sample values (masked): ', array_join(sample_values_masked, ' | ')
  {contract_suffix}
)"""

llm_call = (f"ai_query('{DOC_MODEL}', prompt, responseFormat => '{sql_str(RESPONSE_FORMAT)}')" if USE_RESPONSE_FORMAT
            else f"ai_query('{DOC_MODEL}', prompt)")

docs = spark.sql(f"""
WITH prompts AS (SELECT *, {prompt_sql} AS prompt FROM {fq(SCHEMA_AI, 'column_profile')}),
raw AS (SELECT table_schema, table_name, column_name, data_type, rule_phi_guess, {llm_call} AS raw_response FROM prompts),
cleaned AS (
  SELECT *, trim(regexp_replace(regexp_replace(raw_response, '^\\\\s*```(json)?', ''), '```\\\\s*$', '')) AS json_text FROM raw
)
SELECT table_schema, table_name, column_name, data_type, rule_phi_guess, raw_response,
       from_json(json_text, '{DDL}') AS doc
FROM cleaned
""").selectExpr("* EXCEPT (doc)", "doc.*")

from pyspark.sql import functions as F
docs = (docs
  .withColumn("parse_ok", F.col("description").isNotNull()
                          & F.col("phi_identifier").isin(PHI_ENUM) & F.col("sensitivity").isin(SENS_ENUM))
  .withColumn("confidence", F.when(F.col("parse_ok"), F.col("confidence")).otherwise(F.lit(0.0)))
  .withColumn("phi_disagreement", (F.col("rule_phi_guess") != "NONE") != (F.coalesce(F.col("phi_identifier"), F.lit("NONE")) != "NONE"))
  .withColumn("review_status", F.lit("PENDING_REVIEW"))
  .withColumn("reviewer", F.lit(None).cast("string"))
  .withColumn("generated_by", F.lit(DOC_MODEL))
  .withColumn("generated_at", F.current_timestamp()))

docs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(SCHEMA_AI, "column_docs"))
display(spark.sql(f"""SELECT count(*) AS columns, sum(CASE WHEN parse_ok THEN 1 ELSE 0 END) AS parsed_ok,
                             sum(CASE WHEN NOT parse_ok THEN 1 ELSE 0 END) AS parse_failures
                      FROM {fq(SCHEMA_AI, 'column_docs')}"""))

# COMMAND ----------

display(spark.table(fq(SCHEMA_AI, "column_docs")).drop("raw_response").orderBy(F.desc("phi_disagreement"), "confidence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quality check — the human review queue
# MAGIC Rows where the rule engine and the LLM disagree on PHI, low confidence, or output that failed validation.
# MAGIC Screenshot this table for your README.

# COMMAND ----------

display(spark.sql(f"""
SELECT table_schema, table_name, column_name, rule_phi_guess, phi_identifier, sensitivity, confidence, parse_ok
FROM {fq(SCHEMA_AI, 'column_docs')}
WHERE phi_disagreement OR confidence < 0.7 OR NOT parse_ok
ORDER BY parse_ok, confidence
"""))
