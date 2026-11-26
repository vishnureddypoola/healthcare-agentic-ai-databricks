# Healthcare Agentic AI on Databricks — 4 production-pattern projects

Built by **Vishnuvardhan Reddy Poola** · Azure Databricks · Unity Catalog · MCP · LangGraph · AI Search (Vector Search) · MLflow 3 · Databricks Apps · Model Serving
**All data is synthetic. Policies are illustrative, not real coverage criteria.**

| # | Project | Business problem | Databricks capabilities |
|---|---|---|---|
| 1 | **Clinical Data Dictionary** | Undocumented FHIR/claims schemas; PHI nobody has catalogued | UC `information_schema`, `ai_query` + JSON schema, column tags, custom **MCP server on Apps**, Streamlit chat app |
| 2 | **Patient ↔ Trial Matcher** | Trials miss enrollment because eligibility is free text | LLM criteria extraction, 2 Vector Search indexes, **UC Functions** (SQL + Python), LangGraph, ResponsesAgent, extraction eval + fairness check |
| 3 | **Prior-Authorization Agent** | CMS-0057-F decision deadlines (72 h / 7 days), denial-reason rules | Streamlit App + MCP server + LangGraph, evidence-verification guardrail, policy logic in code, human reviewer queue, audit log |
| 4 | **Provider-Inquiry Triage** | Mis-routed appeals miss deadlines | Retrieval over resolved history, rules + LLM, **MLflow 3 evaluation with release gates**, labeling sessions, production monitoring, **Model Serving** |

## Architecture principle used in every project
> **LLMs read and write language. Code makes decisions. Unity Catalog governs everything. Humans own the irreversible actions.**

## Repo layout
```
common/            config.py · 00_setup_catalog · 01_generate_synthetic_data   (run first)
p1_data_dictionary/ 01_extract · 02_generate_docs · 03_review_publish · mcp_server/ · chat_app/
p2_trial_matcher/   01_parse_trials · 02_profiles_indexes · 03_uc_functions · agent.py · 04_test_eval_deploy
p3_prior_auth/      01_setup · 02_offline_evaluation · mcp_server/ · app/ (app.py + pa_graph.py)
p4_claims_triage/   01_prepare · agent.py · 02_evaluate · 03_log_register_deploy · 04_human_review_monitoring
databricks.yml      Declarative Automation Bundle (jobs + apps) — deploy after the manual run-through
```

## Quick start
1. Azure Databricks **Premium** workspace with Unity Catalog + serverless enabled.
2. Workspace → **Git folder** → clone this repo.
3. Edit `common/config.py` (model endpoint names in *your* region, SQL warehouse id).
4. Run `common/00_setup_catalog`, then `common/01_generate_synthetic_data`.
5. Follow each project folder in order (step-by-step guide in the accompanying doc).

## Gaps closed against the Databricks HLS agentic-AI guidance
| Gap | Where it is closed |
|---|---|
| Decision lineage (trace any decision to policy + evidence + LLM calls) | P3 `determination.policy_id / policy_version / mlflow_trace_id`; P2 `extracted_by / extraction_prompt_version`; P4 `custom_outputs.mlflow_trace_id` |
| Self-assessment before a response is sent | P4 `self_check` node (judge → 1 rewrite → force human review) |
| Agent Bricks (no-code) vs. coded pipeline | P2 `05_compare_agent_bricks_extraction` (same gold set) |
| Business value, not just accuracy | `common/02_business_value_kpis` |
| Governance at the model boundary | AI Gateway guardrails + usage tracking on the P4 endpoint (runbook) |
| Operational state for apps | Lakebase for the P3 review queue / LangGraph checkpoints (runbook, optional) |
| Domain-expert self-service | Genie Agent benchmark before vs. after P1 documentation (runbook) |
| Multi-agent orchestration | Capstone: Agent Bricks Supervisor Agent over Genie + Knowledge Assistant + P3/P4 MCP (runbook) |

## Results (fill in with your numbers + screenshots)
| Project | Metric | Result |
|---|---|---|
| P1 | columns documented / PHI disagreements caught | |
| P2 | criteria-extraction lab-rule F1 · age/sex exact | |
| P3 | unsafe auto-approval rate · automation rate on complete requests | |
| P4 | category acc · team acc · P1 recall · judge pass rates | |

## Cost hygiene
Delete the AI Search endpoint and stop Apps when you pause; serving endpoints use `scale_to_zero`.
