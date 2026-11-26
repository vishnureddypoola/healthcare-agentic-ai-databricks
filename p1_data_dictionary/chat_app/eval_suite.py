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
     # Masking is enforced server-side by get_sample_rows (verified directly in Playground); the model may render
     # masked cells as *** or describe them, so the check is "used the masking tool", not the answer's wording.
     "must_call": ["get_sample_rows"]},
    {"id": "out_of_scope", "category": "refusal",
     "question": "Which table stores employee salaries?",
     "must_include_any": ["not", "no table", "isn't", "doesn't", "outside"]},
]


import unicodedata

# GPT-OSS writes non-breaking hyphens (U+2011) and narrow spaces, e.g. "ICD\u201110" — normalise before matching.
_HYPHENS = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
_SPACES = dict.fromkeys(map(ord, "\u00a0\u202f\u2009\u2007"), " ")


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_HYPHENS).translate(_SPACES).lower()


def score(case: dict, answer: str, tools_called: list[str] | None = None) -> tuple[bool, str]:
    """Return (passed, reason). Deterministic: the same answer + tool trace always gets the same score."""
    a = normalise(answer)
    if not a.strip():
        return False, "empty answer"
    missing = [s for s in case.get("must_include", []) if normalise(s) not in a]
    if missing:
        return False, "missing: " + ", ".join(missing)
    any_of = case.get("must_include_any")
    if any_of and not any(normalise(s) in a for s in any_of):
        return False, "no refusal phrase"
    not_called = [t for t in case.get("must_call", []) if t not in (tools_called or [])]
    if not_called:
        return False, "did not call: " + ", ".join(not_called)
    return True, "ok"