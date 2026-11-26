# Databricks notebook source
# MAGIC %md
# MAGIC # Business value KPIs for all four projects (gap closer)
# MAGIC Gartner's reason #1 for cancelled agent projects is **unclear business value**. Each query below turns a project's
# MAGIC operational tables into a value number. Assumptions (minutes per manual task, loaded cost per hour) are widgets —
# MAGIC change them, state them in your README, and never present them as measured savings.

# COMMAND ----------

import sys, os
sys.path.append(os.path.abspath(".."))
from common.config import *
C = CATALOG

dbutils.widgets.text("nurse_minutes_per_pa_review", "25")
dbutils.widgets.text("nurse_cost_per_hour", "65")
dbutils.widgets.text("agent_minutes_per_inquiry", "12")
dbutils.widgets.text("csr_cost_per_hour", "35")
dbutils.widgets.text("coordinator_minutes_per_chart_screen", "20")
W = {k: float(dbutils.widgets.get(k)) for k in ["nurse_minutes_per_pa_review", "nurse_cost_per_hour", "agent_minutes_per_inquiry",
                                                 "csr_cost_per_hour", "coordinator_minutes_per_chart_screen"]}

# COMMAND ----------

# MAGIC %md ## P3 Prior auth — automation rate, nurse hours avoided, SLA compliance

# COMMAND ----------

display(spark.sql(f"""
WITH agent AS (SELECT * FROM {C}.prior_auth.determination WHERE decided_by = 'agent')
SELECT count(*)                                                     AS requests_processed,
       round(avg(CASE WHEN decision = 'APPROVED_AUTO' THEN 1 ELSE 0 END), 3) AS automation_rate,
       round(sum(CASE WHEN decision = 'APPROVED_AUTO' THEN 1 ELSE 0 END) * {W['nurse_minutes_per_pa_review']} / 60, 1) AS nurse_hours_avoided,
       round(sum(CASE WHEN decision = 'APPROVED_AUTO' THEN 1 ELSE 0 END) * {W['nurse_minutes_per_pa_review']} / 60
             * {W['nurse_cost_per_hour']}, 0)                       AS est_cost_avoided_usd,
       count(DISTINCT policy_version)                               AS policy_versions_used,
       sum(CASE WHEN mlflow_trace_id IS NULL OR mlflow_trace_id = '' THEN 1 ELSE 0 END) AS decisions_missing_lineage
FROM agent
"""))
display(spark.sql(f"""
SELECT priority, count(*) AS open_items,
       sum(CASE WHEN sla_due_ts < current_timestamp() THEN 1 ELSE 0 END) AS sla_breached,
       sum(CASE WHEN sla_due_ts BETWEEN current_timestamp() AND current_timestamp() + INTERVAL 24 HOURS THEN 1 ELSE 0 END) AS due_24h
FROM {C}.prior_auth.review_queue WHERE status = 'OPEN' GROUP BY priority
"""))

# COMMAND ----------

# MAGIC %md ## P4 Triage — routing accuracy proxy, human-touch rate, handling time avoided
# MAGIC Reads the serving endpoint's inference table. Find its name on the endpoint page (Inference tables) and set it below.

# COMMAND ----------

dbutils.widgets.text("triage_inference_table", f"{C}.ai.claims_triage_agent_payload")
INF = dbutils.widgets.get("triage_inference_table")
try:
    display(spark.sql(f"""
    WITH r AS (SELECT from_json(response, 'custom_outputs STRUCT<category STRING, priority STRING, needs_human_review BOOLEAN>').custom_outputs AS co
               FROM {INF} WHERE status_code = 200)
    SELECT count(*) AS inquiries,
           round(avg(CASE WHEN co.needs_human_review THEN 1 ELSE 0 END), 3) AS human_touch_rate,
           round(sum(CASE WHEN NOT co.needs_human_review THEN 1 ELSE 0 END) * {W['agent_minutes_per_inquiry']} / 60, 1) AS csr_hours_avoided,
           round(sum(CASE WHEN NOT co.needs_human_review THEN 1 ELSE 0 END) * {W['agent_minutes_per_inquiry']} / 60 * {W['csr_cost_per_hour']}, 0) AS est_cost_avoided_usd
    FROM r
    """))
except Exception as e:
    print("Inference table not found yet — enable it on the endpoint (AI Gateway → Inference tables).", e)

# COMMAND ----------

# MAGIC %md ## P2 Trial matcher — pre-screen effort avoided
# MAGIC Every hard-criterion FAIL is a chart a coordinator no longer opens.

# COMMAND ----------

display(spark.sql(f"""
WITH pairs AS (
  SELECT {C}.ai.compute_match_score({C}.ai.check_trial_eligibility(p.patient_id, t.trial_id), 0.5) AS s
  FROM (SELECT patient_id FROM {C}.clinical.patient_profile LIMIT 500) p
  CROSS JOIN (SELECT trial_id FROM {C}.trials.trial_criteria WHERE overall_status = 'RECRUITING' LIMIT 5) t)
SELECT count(*) AS candidate_pairs,
       sum(CASE WHEN get_json_object(s, '$.verdict') = 'INELIGIBLE' THEN 1 ELSE 0 END) AS auto_excluded,
       round(sum(CASE WHEN get_json_object(s, '$.verdict') = 'INELIGIBLE' THEN 1 ELSE 0 END)
             * {W['coordinator_minutes_per_chart_screen']} / 60, 1) AS coordinator_hours_avoided
FROM pairs
"""))

# COMMAND ----------

# MAGIC %md ## P1 Data dictionary — documentation coverage and PHI inventory

# COMMAND ----------

display(spark.sql(f"""
SELECT count(*) AS columns_total,
       sum(CASE WHEN review_status = 'APPROVED' THEN 1 ELSE 0 END) AS columns_documented,
       round(avg(CASE WHEN review_status = 'APPROVED' THEN 1 ELSE 0 END), 3) AS coverage,
       sum(CASE WHEN phi_identifier <> 'NONE' THEN 1 ELSE 0 END) AS phi_columns_inventoried,
       sum(CASE WHEN phi_disagreement THEN 1 ELSE 0 END) AS phi_disputes_caught
FROM {C}.ai.column_docs
"""))
# Pair this with the Genie benchmark accuracy before vs. after publishing comments (P1 Part 7).
