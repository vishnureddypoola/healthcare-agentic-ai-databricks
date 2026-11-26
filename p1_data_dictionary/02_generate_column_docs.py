# Databricks notebook source
# MAGIC %md
# MAGIC # P1 · Step 2 — LLM-generated column documentation (batch, with `ai_query` + JSON schema)
# MAGIC One SQL statement documents every column in parallel. `responseFormat` forces valid, typed JSON so the output can
# MAGIC be parsed into columns — no regex-parsing of LLM prose.
# MAGIC
# MAGIC Every row lands as `PENDING_REVIEW`. Nothing is written to Unity Catalog comments/tags until a human approves (Step 3).

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))
from common.config import *

RESPONSE_FORMAT = """{
  "type": "json_schema",
  "json_schema": {
    "name": "column_doc",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {
        "description":         {"type": "string", "description": "One clear sentence a non-technical analyst understands."},
        "business_definition": {"type": "string", "description": "How the business/clinic uses this field, incl. caveats."},
        "phi_identifier":      {"type": "string", "enum": ["NONE","NAME","DATE","GEOGRAPHIC","PHONE","EMAIL","SSN","MEDICAL_RECORD_NUMBER","HEALTH_PLAN_BENEFICIARY_NUMBER","ACCOUNT_NUMBER","DEVICE_ID","FREE_TEXT_MAY_CONTAIN_PHI","OTHER_UNIQUE_ID"]},
        "sensitivity":         {"type": "string", "enum": ["PUBLIC","INTERNAL","CONFIDENTIAL","PHI"]},
        "code_system":         {"type": "string", "description": "ICD-10-CM, SNOMED CT, LOINC, RxNorm, CPT, HCPCS, CARC, NPI, or NONE"},
        "fhir_mapping":        {"type": "string", "description": "Closest FHIR R4 element path, e.g. Observation.valueQuantity.value, or NONE"},
        "example_question":    {"type": "string", "description": "A natural-language question this column helps answer."},
        "confidence":          {"type": "number", "description": "0-1 self-rated confidence"}
      },
      "required": ["description","business_definition","phi_identifier","sensitivity","code_system","fhir_mapping","example_question","confidence"],
      "additionalProperties": false
    }
  }
}"""

# COMMAND ----------

docs = spark.sql(f"""
WITH prompts AS (
  SELECT *,
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
    ) AS prompt
  FROM {fq(SCHEMA_AI, 'column_profile')}
)
SELECT table_schema, table_name, column_name, data_type, rule_phi_guess,
       from_json(
         ai_query('{LLM_ENDPOINT}', prompt, responseFormat => '{RESPONSE_FORMAT.replace(chr(10), ' ')}'),
         'description STRING, business_definition STRING, phi_identifier STRING, sensitivity STRING, code_system STRING, fhir_mapping STRING, example_question STRING, confidence DOUBLE'
       ) AS doc
FROM prompts
""").selectExpr("*", "doc.*").drop("doc")

from pyspark.sql import functions as F
docs = (docs
  .withColumn("phi_disagreement", (F.col("rule_phi_guess") != "NONE") != (F.col("phi_identifier") != "NONE"))
  .withColumn("review_status", F.lit("PENDING_REVIEW"))
  .withColumn("reviewer", F.lit(None).cast("string"))
  .withColumn("generated_by", F.lit(LLM_ENDPOINT))
  .withColumn("generated_at", F.current_timestamp()))

docs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq(SCHEMA_AI, "column_docs"))
display(spark.table(fq(SCHEMA_AI, "column_docs")).orderBy(F.desc("phi_disagreement"), "confidence"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quality check — where do the rule engine and the LLM disagree on PHI?
# MAGIC These rows are your most important human-review queue. Screenshot this table for your README.

# COMMAND ----------

display(spark.sql(f"""
SELECT table_schema, table_name, column_name, rule_phi_guess, phi_identifier, sensitivity, confidence
FROM {fq(SCHEMA_AI, 'column_docs')}
WHERE phi_disagreement OR confidence < 0.7
ORDER BY confidence
"""))
