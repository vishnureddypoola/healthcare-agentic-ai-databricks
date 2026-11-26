# P1 · Clinical Data Dictionary Agent - Results Scorecard

Synthetic healthcare lakehouse (`hc_agentic`, 13 tables, 103 columns) on Azure Databricks (serverless, Unity Catalog).
Pipeline: LLM drafts column docs → rule-vs-LLM PHI cross-check → human steward review → publish to Unity Catalog
(comments + governed tags) → MCP server (Databricks App) → chat agent (Databricks App) → evaluation.

## 1. Governance (measured, from `hc_agentic.ai.column_docs` and `information_schema`)

| KPI | Value |
|---|---|
| Documentation coverage in Unity Catalog | **103 / 103 columns (100%)** |
| Auto-approved (confidence ≥ 0.85, no PHI disagreement) | 63 (61%) |
| Routed to human steward | 40 (39%) |
| Steward changed the LLM output when routed | **33 of 40 (83%)**: routing found real issues |
| Rule-based vs LLM PHI agreement | 67%: neither alone is trustworthy |
| HIPAA Safe Harbor identifier columns | 41 |
| PHI-sensitive columns (incl. non-identifiers) | 66 (conservative; possible over-tagging) |
| Code systems mapped | 7 (ICD-10-CM, SNOMED CT, LOINC, RxNorm, CPT, CARC, NPI) |

## 2. Agent quality (measured, 8-question eval × 3 runs per round)

| Round | Accuracy | Avg tool calls | Median latency | Notes |
|---|---|---|---|---|
| 1 (raw scorer) | 17/24 (71%) | 2.9 | — | 6 of 7 failures were **scorer bugs** (Unicode non-breaking hyphens, over-strict masking check) |
| 2 (fixed scorer) | 23/24 (96%) | 2.5 | 7.6 s | 1 real failure: early-stopping retrieval (wrong table) |
| 3 (after tool fix) | **24/24 (100%)** | 2.75 | 7.6 s | "No match ≠ absent" note in `search_data_dictionary`; denial question now costs more calls (avg 9.0) |

- Governance questions (PHI identifiers, masked samples): 100% stable, 1 tool call every run.
- Real failure modes found: **false negative** ("HbA1c not in catalog": keyword search missed the abbreviation) and
  **wrong-but-confident table**. Both happened when the agent stopped after 2 calls.
- Open gap: **groundedness**. The agent sometimes adds facts from model knowledge (e.g. LOINC 4549-2) that keyword
  checks cannot catch. Next step: LLM-judge groundedness scoring (MLflow).
- 24/24 is evidence, not proof: baseline failure rate was ~1 in 24; more runs are needed for a statistical claim.
- Latency tail (p95 27–62 s) is attributed to shared pay-per-token endpoint load, not agent logic.

## 3. Business value (ESTIMATE: assumptions stated)

Assumptions: 10 min to document a column manually; 3 min to review a routed column; 0.5 min to spot-check an
auto-approved column; steward cost $60/h.

| Metric | Value |
|---|---|
| Manual effort | 17.2 h |
| AI-assisted effort | 2.5 h |
| **Steward effort reduction** | **~85%** |
| Estimated saving (this catalog) | 14.6 h ≈ $879 |

The percentage is the scalable claim: an enterprise warehouse has 20k–100k columns; the dollar figure here is for 103.

## 4. Engineering lessons (real issues hit during the build)

1. Foundation models served through Unity AI Gateway; Claude structured output (forced tool use) rejected → GPT-OSS 120B for strict-schema extraction and tool calling.
2. Claude tool calling blocked on this account in Playground/apps → GPT-OSS chosen; model is a config value, not code.
3. Databricks Apps pre-install packages; loose `>=` pins mixed old `langgraph` with new `langgraph-prebuilt` → ImportError. Fixed with exact pins.
4. A chat-app `app.yaml` pasted into `mcp_server/` crashed the server on restart. Lesson: check the breadcrumb; identical file names across folders are a trap.
5. Over-tightened system prompt → empty answers. Rolled back; added a retry-on-empty safety net. Lesson: change one thing at a time, and only with an eval running.
6. Dependent services: start the MCP server first, stop it last.
7. Genie before/after docs: accuracy unchanged (4/5); docs surfaced metric-definition ambiguity. Metric definitions belong in Genie instructions / metric views, not column comments.