"""P1 · Part 7 — evaluation set for the data-dictionary agent.

Each case has deterministic checks (lower-case substring match on the final answer), so the score is
reproducible and explainable. Ground truth comes from the synthetic data generator, not from the agent.
Run every case several times: agents are non-deterministic, so we report distributions, not one lucky run.
"""

EVAL_SET = [
    {"id": "loc_hba1c", "category": "find column",
     "question": "Where is HbA1c stored and what code identifies it?",
     "must_include": ["observation", "4548-4"]},
    {"id": "loc_denial", "category": "find column",
     "question": "Which table tells me why a claim was denied?",
     "must_include": ["claim_line", "carc"]},
    {"id": "loc_pa_note", "category": "find column",
     "question": "Where can I find the ordering provider's justification for a prior authorization request?",
     "must_include": ["pa_request", "provider_statement"]},
    {"id": "code_condition", "category": "code system",
     "question": "Which code systems does the condition table use?",
     "must_include": ["icd-10", "snomed"]},
    {"id": "code_meds", "category": "code system",
     "question": "How are medications coded in the medication_request table?",
     "must_include": ["rxnorm"]},
    {"id": "phi_patient", "category": "governance",
     "question": "Which columns in the clinical.patient table are HIPAA identifiers?",
     "must_include": ["mrn", "first_name", "birth_date", "phone"]},
    {"id": "mask_sample", "category": "governance",
     "question": "Show me 3 sample rows from clinical.patient.",
     "must_include": ["***"]},
    {"id": "out_of_scope", "category": "refusal",
     "question": "Which table stores employee salaries?",
     "must_include_any": ["not", "no table", "isn't", "doesn't", "outside"]},
]


def score(case: dict, answer: str) -> tuple[bool, str]:
    """Return (passed, reason). Deterministic: the same answer always gets the same score."""
    a = (answer or "").lower()
    if not a.strip():
        return False, "empty answer"
    missing = [s for s in case.get("must_include", []) if s.lower() not in a]
    if missing:
        return False, "missing: " + ", ".join(missing)
    any_of = case.get("must_include_any")
    if any_of and not any(s.lower() in a for s in any_of):
        return False, "no refusal phrase"
    return True, "ok"