# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate the shared synthetic healthcare dataset
# MAGIC One coherent, **100% synthetic** dataset that all four projects use, so the portfolio tells one story:
# MAGIC
# MAGIC | Schema | Tables | Shape |
# MAGIC |---|---|---|
# MAGIC | `clinical` | patient, encounter, condition, observation, medication_request, procedure_history, clinical_note | FHIR R4 resources flattened (ICD-10, SNOMED, LOINC, RxNorm, CPT) |
# MAGIC | `claims` | claim, claim_line, provider_inquiry | X12 837 / 835-like with CARC denial codes |
# MAGIC | `trials` | trial_raw | free-text eligibility criteria (ClinicalTrials.gov style) |
# MAGIC | `prior_auth` | policy, pa_request | medical policies + incoming PA requests |
# MAGIC
# MAGIC **Optional swap:** if your Fabric FHIR/X12 project already has Gold tables, read them over OneLake
# MAGIC (`abfss://<workspace>@onelake.dfs.fabric.microsoft.com/<lakehouse>.Lakehouse/Tables/...`) and map them to these
# MAGIC column names — that makes the Fabric project and this one a single end-to-end story.
# MAGIC
# MAGIC Policies and criteria below are *illustrative*, inspired by common utilization-management logic. They are **not** real payer coverage criteria.

# COMMAND ----------

import sys, os, re, random, uuid
from datetime import date, datetime, timedelta
import pandas as pd
sys.path.append(os.path.abspath(".."))
from common.config import *

random.seed(42)
N_PATIENTS = 2000
TODAY = date(2026, 11, 1)

def rdate(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))

def save(rows, schema, table, comment):
    df = spark.createDataFrame(pd.DataFrame(rows))
    name = fq(schema, table)
    (df.write.mode("overwrite").option("overwriteSchema", "true")
       .option("delta.enableChangeDataFeed", "true").saveAsTable(name))
    spark.sql(f"COMMENT ON TABLE {name} IS '{comment}'")
    print(f"{name}: {df.count():,} rows")

# COMMAND ----------

# MAGIC %md ## Reference vocabularies

# COMMAND ----------

CONDITIONS = {  # key: (icd10, snomed, description, prevalence)
    "t2d":       ("E11.9",   "44054006",  "Type 2 diabetes mellitus without complications", 0.18),
    "htn":       ("I10",     "38341003",  "Essential (primary) hypertension", 0.35),
    "ckd3":      ("N18.30",  "433144002", "Chronic kidney disease, stage 3 unspecified", 0.08),
    "asthma":    ("J45.909", "195967001", "Asthma, unspecified, uncomplicated", 0.09),
    "hf":        ("I50.9",   "84114007",  "Heart failure, unspecified", 0.05),
    "obesity":   ("E66.9",   "414916001", "Obesity, unspecified", 0.30),
    "depression":("F32.9",   "35489007",  "Major depressive disorder, single episode, unspecified", 0.12),
    "knee_oa":   ("M17.11",  "239873007", "Unilateral primary osteoarthritis, right knee", 0.10),
    "migraine":  ("G43.909", "37796009",  "Migraine, unspecified, not intractable", 0.08),
    "osa":       ("G47.33",  "78275009",  "Obstructive sleep apnea", 0.07),
    "breast_ca": ("C50.911", "254837009", "Malignant neoplasm of unspecified site of right female breast", 0.02),
}
LABS = {  # key: (loinc, display, unit)
    "hba1c": ("4548-4",  "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "egfr":  ("98979-8", "Glomerular filtration rate/1.73 sq M.predicted (CKD-EPI 2021)", "mL/min/1.73m2"),
    "bmi":   ("39156-5", "Body mass index", "kg/m2"),
    "sbp":   ("8480-6",  "Systolic blood pressure", "mm[Hg]"),
    "ldl":   ("13457-7", "LDL cholesterol (calculated)", "mg/dL"),
}
MEDS = {  # condition -> (rxnorm, name)
    "t2d": ("860975", "metformin 500 MG Extended Release Oral Tablet"),
    "htn": ("314076", "lisinopril 10 MG Oral Tablet"),
    "asthma": ("745679", "albuterol 0.09 MG/ACTUAT Metered Dose Inhaler"),
    "depression": ("312940", "sertraline 50 MG Oral Tablet"),
    "hf": ("200031", "carvedilol 6.25 MG Oral Tablet"),
    "migraine": ("313782", "sumatriptan 50 MG Oral Tablet"),
}
FIRST = ["James","Mary","Robert","Patricia","Michael","Jennifer","David","Linda","Priya","Wei","Carlos","Aisha","Ravi","Sofia","Ahmed","Grace","Hiro","Fatima","Lucas","Ana"]
LAST = ["Smith","Johnson","Lee","Garcia","Patel","Nguyen","Brown","Kim","Reddy","Martinez","Chen","Okafor","Wilson","Singh","Lopez","Ali","Davis","Rao","Silva","Khan"]
STATES = ["WA","OR","CA","TX","NY","IL","FL","GA","AZ","MA"]
RACES = ["White","Black or African American","Asian","American Indian or Alaska Native","Native Hawaiian or Other Pacific Islander","Other"]
PLANS = [("MA","Medicare Advantage"),("MCD","Medicaid Managed Care"),("COM","Commercial PPO")]

# COMMAND ----------

# MAGIC %md ## Clinical (FHIR-shaped)

# COMMAND ----------

patients, conditions, observations, meds, encounters = [], [], [], [], []
pat_profile = {}  # in-memory helper for later scenario generation

for i in range(1, N_PATIENTS + 1):
    pid = f"P{i:05d}"
    sex = random.choice(["female", "male"])
    birth = rdate(date(1940, 1, 1), date(2006, 12, 31))
    age = (TODAY - birth).days // 365
    plan = random.choices(PLANS, weights=[0.35 if age >= 65 else 0.05, 0.25, 0.5])[0]
    patients.append(dict(
        patient_id=pid, mrn=f"MRN{random.randint(10_000_000, 99_999_999)}",
        first_name=random.choice(FIRST), last_name=random.choice(LAST),
        birth_date=birth, sex=sex, race=random.choices(RACES, weights=[55,14,12,2,1,16])[0],
        ethnicity=random.choices(["Hispanic or Latino","Not Hispanic or Latino"], weights=[19,81])[0],
        address_state=random.choice(STATES), postal_code_3=f"{random.randint(100,999)}",
        phone=f"({random.randint(200,989)}) 555-{random.randint(1000,9999)}",
        payer_plan_code=plan[0], payer_plan_name=plan[1], member_id=f"M{random.randint(100000000,999999999)}",
    ))
    has = set()
    for key, (_, _, _, prev) in CONDITIONS.items():
        p = prev * (1.6 if age > 60 and key in ("t2d","htn","ckd3","hf","knee_oa") else 1.0)
        if key == "breast_ca" and sex == "male":
            continue
        if random.random() < p:
            has.add(key)
    if "t2d" in has and random.random() < 0.3:
        has.add("ckd3")
    for key in sorted(has):  # sorted => reproducible across runs
        icd, snomed, desc, _ = CONDITIONS[key]
        conditions.append(dict(condition_id=str(uuid.uuid4()), patient_id=pid, icd10_code=icd, snomed_code=snomed,
                               description=desc, onset_date=rdate(date(2012,1,1), date(2026,6,1)), clinical_status="active"))
        if key in MEDS:
            rx, name = MEDS[key]
            meds.append(dict(medication_request_id=str(uuid.uuid4()), patient_id=pid, rxnorm_code=rx, medication=name,
                             authored_on=rdate(date(2018,1,1), date(2026,9,1)), status="active"))
    # labs (most recent value matters for eligibility)
    vals = {
        "hba1c": round(random.uniform(6.6, 11.0) if "t2d" in has else random.uniform(4.8, 6.2), 1),
        "egfr": round(random.uniform(22, 59) if "ckd3" in has else random.uniform(60, 115)),
        "bmi": round(random.uniform(30, 46) if "obesity" in has else random.uniform(19, 29.9), 1),
        "sbp": round(random.uniform(138, 178) if "htn" in has else random.uniform(105, 135)),
        "ldl": round(random.uniform(70, 190)),
    }
    for key, v in vals.items():
        loinc, disp, unit = LABS[key]
        observations.append(dict(observation_id=str(uuid.uuid4()), patient_id=pid, loinc_code=loinc, display=disp,
                                 value_numeric=float(v), unit=unit, effective_date=rdate(date(2026,3,1), date(2026,10,25))))
    for _ in range(random.randint(1, 6)):
        encounters.append(dict(encounter_id=str(uuid.uuid4()), patient_id=pid, encounter_class=random.choices(["AMB","EMER","IMP"],[85,10,5])[0],
                               period_start=rdate(date(2025,1,1), date(2026,10,30)), provider_npi=f"1{random.randint(100000000,999999999)}",
                               facility=random.choice(["Evergreen Medical Center","Cascade Family Clinic","Lakeside Ortho Group","Northwest Sleep Center"])))
    pat_profile[pid] = dict(age=age, sex=sex, has=has, vals=vals, plan=plan[0])

save(patients, SCHEMA_CLINICAL, "patient", "FHIR Patient (synthetic). Contains direct identifiers.")
save(encounters, SCHEMA_CLINICAL, "encounter", "FHIR Encounter (synthetic)")
save(conditions, SCHEMA_CLINICAL, "condition", "FHIR Condition with ICD-10-CM and SNOMED CT codes (synthetic)")
save(observations, SCHEMA_CLINICAL, "observation", "FHIR Observation, LOINC-coded labs and vitals (synthetic)")
save(meds, SCHEMA_CLINICAL, "medication_request", "FHIR MedicationRequest with RxNorm codes (synthetic)")

# COMMAND ----------

# MAGIC %md ## Prior-auth policies and requests (with deliberately mixed evidence quality)

# COMMAND ----------

POLICIES = [
    dict(policy_id="POL-IMG-001", cpt_code="73721", service_name="MRI lower extremity joint (knee) without contrast", version="2026.2",
         policy_text="""MEDICAL NECESSITY CRITERIA – MRI KNEE (CPT 73721)
MRI of the knee is considered medically necessary when ALL of the following are met:
C1. Knee pain or mechanical symptoms (locking, catching, instability) documented for at least 6 weeks.
C2. Failure of at least 6 weeks of conservative therapy (physical therapy, NSAIDs, activity modification) documented in the record.
C3. Plain radiographs (X-ray) of the knee performed within the prior 60 days.
RF. EXCEPTION: C1 and C2 are waived when red-flag findings are documented (acute trauma with suspected ligament tear, suspected infection, suspected tumor). Such requests are EXPEDITED.
Requests not meeting criteria require review by a licensed clinician; denials may only be issued by a physician reviewer."""),
    dict(policy_id="POL-ORTHO-004", cpt_code="27447", service_name="Total knee arthroplasty", version="2026.1",
         policy_text="""MEDICAL NECESSITY CRITERIA – TOTAL KNEE ARTHROPLASTY (CPT 27447)
ALL of the following must be met:
C1. Radiographic evidence of advanced osteoarthritis (Kellgren-Lawrence grade 3 or 4).
C2. Persistent pain and functional limitation interfering with activities of daily living, documented.
C3. Failure of at least 12 weeks of non-surgical management (physical therapy and/or injections and/or NSAIDs).
C4. Body mass index below 40 kg/m2 at the most recent measurement.
C5. If the patient has diabetes, most recent HbA1c below 8.0 percent.
Requests not meeting criteria require review by a licensed clinician; denials may only be issued by a physician reviewer."""),
    dict(policy_id="POL-SLEEP-002", cpt_code="95810", service_name="Attended in-lab polysomnography", version="2026.1",
         policy_text="""MEDICAL NECESSITY CRITERIA – IN-LAB POLYSOMNOGRAPHY (CPT 95810)
Medically necessary when C1 AND (C2 OR C3):
C1. Clinical suspicion of obstructive sleep apnea documented (e.g., STOP-BANG score 3 or more, witnessed apneas, excessive daytime sleepiness).
C2. A home sleep apnea test was technically inadequate or negative despite high clinical suspicion.
C3. A contraindication to home testing is documented (e.g., significant cardiopulmonary disease such as heart failure, neuromuscular disease, suspected non-OSA sleep disorder).
Requests not meeting criteria require review by a licensed clinician; denials may only be issued by a physician reviewer."""),
    dict(policy_id="POL-DME-007", cpt_code="E0601", service_name="Continuous positive airway pressure (CPAP) device", version="2026.1",
         policy_text="""MEDICAL NECESSITY CRITERIA – CPAP DEVICE (HCPCS E0601)
Medically necessary when EITHER:
C1. Apnea-hypopnea index (AHI) of 15 or more events per hour on a qualifying sleep test; OR
C2. AHI of 5 to 14 events per hour with documented excessive daytime sleepiness, hypertension, ischemic heart disease, or history of stroke.
Requests not meeting criteria require review by a licensed clinician; denials may only be issued by a physician reviewer."""),
    dict(policy_id="POL-IMG-010", cpt_code="70553", service_name="MRI brain with and without contrast", version="2026.3",
         policy_text="""MEDICAL NECESSITY CRITERIA – MRI BRAIN (CPT 70553)
Medically necessary when ANY of the following is documented:
C1. New focal neurological deficit.
C2. New-onset seizure.
C3. Headache with red flags: thunderclap onset, progressive pattern, new headache after age 50, or associated papilledema.
C4. Known intracranial lesion requiring surveillance per treating specialist.
Uncomplicated chronic migraine without red flags does NOT meet criteria.
Requests not meeting criteria require review by a licensed clinician; denials may only be issued by a physician reviewer."""),
]
# Machine-readable decision logic over criterion ids (evaluated deterministically by the agent, never by the LLM)
LOGIC = {"73721": ("(C1 and C2 and C3) or (RF and C3)", [], False),
         "27447": ("C1 and C2 and C3 and C4 and C5", ["C5"], True),   # high-cost surgery: clinician sign-off always
         "95810": ("C1 and (C2 or C3)", [], False),
         "E0601": ("C1 or C2", [], False),
         "70553": ("C1 or C2 or C3 or C4", [], False)}
for p in POLICIES:
    p["effective_date"] = date(2026, 1, 1)
    p["decision_logic"], p["conditional_criteria"], p["always_human_review"] = LOGIC[p["cpt_code"]]
save(POLICIES, SCHEMA_PA, "policy", "Illustrative medical policies (NOT real coverage criteria)")

# COMMAND ----------

notes, procs, pa_requests = [], [], []
scenarios = []  # (patient, cpt, quality)

def candidates(cond):
    return [p for p, prof in pat_profile.items() if cond in prof["has"]]

for pid in random.sample(candidates("knee_oa"), min(60, len(candidates("knee_oa")))):
    scenarios.append((pid, random.choice(["73721", "27447"])))
for pid in random.sample(candidates("osa"), min(40, len(candidates("osa")))):
    scenarios.append((pid, random.choice(["95810", "E0601"])))
for pid in random.sample(candidates("migraine"), min(30, len(candidates("migraine")))):
    scenarios.append((pid, "70553"))

def add_note(pid, d, kind, text):
    notes.append(dict(note_id=str(uuid.uuid4()), patient_id=pid, note_date=d, note_type=kind, note_text=text))

for n, (pid, cpt) in enumerate(scenarios, start=1):
    prof = pat_profile[pid]
    quality = random.choices(["complete", "missing", "ambiguous"], weights=[55, 30, 15])[0]
    d0 = rdate(date(2026, 9, 1), date(2026, 10, 28))
    urgency = "STANDARD"
    icds = []
    if cpt in ("73721", "27447"):
        icds = ["M17.11"]
        weeks = random.randint(8, 30) if quality != "missing" else random.randint(2, 5)
        pt_weeks = random.randint(6, 16) if quality == "complete" else (random.randint(1, 4) if quality == "missing" else None)
        if pt_weeks:
            for w in range(0, pt_weeks, 2):
                procs.append(dict(patient_id=pid, cpt_code="97110", description="Therapeutic exercise (physical therapy)",
                                  performed_date=d0 - timedelta(weeks=pt_weeks - w)))
        xr_days = random.randint(5, 45) if quality == "complete" else random.randint(90, 200)
        procs.append(dict(patient_id=pid, cpt_code="73562", description="X-ray knee, 3 views", performed_date=d0 - timedelta(days=xr_days)))
        kl = random.choice([3, 4]) if quality == "complete" else random.choice([2, 3])
        pt_line = (f"Completed {pt_weeks} weeks of PT without meaningful improvement." if pt_weeks and quality == "complete"
                   else ("Started PT recently." if pt_weeks else "Patient reports 'tried some exercises at home'."))
        add_note(pid, d0, "Orthopedic consult",
                 f"Right knee pain for {weeks} weeks. {pt_line} NSAIDs {'trialed x 8 weeks' if quality=='complete' else 'not documented'}. "
                 f"X-ray: Kellgren-Lawrence grade {kl}. Difficulty with stairs and walking >1 block. "
                 f"Pain 7/10. BMI {prof['vals']['bmi']}. "
                 + ("Possible meniscal tear; mechanical locking." if cpt == "73721" else "Discussed TKA risks/benefits."))
        if cpt == "73721" and random.random() < 0.1:
            urgency = "EXPEDITED"
            add_note(pid, d0, "ED note", "Acute twisting injury yesterday, effusion, positive Lachman, suspected ACL tear.")
    elif cpt in ("95810", "E0601"):
        icds = ["G47.33"]
        stop_bang = random.randint(4, 7) if quality != "missing" else random.randint(1, 2)
        ahi = random.randint(16, 45) if quality == "complete" else (random.randint(6, 13) if quality == "ambiguous" else random.randint(1, 4))
        hst = "Home sleep test technically inadequate (insufficient recording time)." if quality == "complete" else "No home sleep test on file."
        add_note(pid, d0, "Sleep medicine consult",
                 f"Loud snoring, witnessed apneas per spouse. STOP-BANG {stop_bang}. Epworth {random.randint(8,18)}. {hst} "
                 + (f"Qualifying study AHI {ahi} events/hour." if cpt == "E0601" else "")
                 + (" History of heart failure." if "hf" in prof["has"] else ""))
    else:  # 70553
        icds = ["G43.909"]
        if quality == "complete":
            add_note(pid, d0, "Neurology consult", "New left arm weakness and word-finding difficulty x 5 days. Headaches progressive over 3 weeks.")
            urgency = random.choice(["EXPEDITED", "STANDARD"])
        elif quality == "ambiguous":
            add_note(pid, d0, "Primary care note", "Headaches worse than usual this month, patient worried. Neuro exam documented as 'grossly normal'.")
        else:
            add_note(pid, d0, "Primary care note", "Long-standing migraine, typical pattern, responds to sumatriptan. Patient requests MRI for reassurance.")
    pa_requests.append(dict(
        request_id=f"PA-{2026}{n:05d}", patient_id=pid, cpt_code=cpt, icd10_codes=icds, urgency=urgency,
        requesting_npi=f"1{random.randint(100000000,999999999)}", submitted_ts=datetime.combine(d0, datetime.min.time()) + timedelta(hours=random.randint(8,17)),
        provider_statement=random.choice(["Medically necessary per attached notes.", "Please expedite, patient in significant pain.", "See clinical documentation.", ""]),
        status="RECEIVED", synthetic_expected_quality=quality,  # ground truth for evaluating Project 3 — hide from the agent!
    ))

save(procs, SCHEMA_CLINICAL, "procedure_history", "FHIR Procedure with CPT codes (synthetic)")
save(notes, SCHEMA_CLINICAL, "clinical_note", "FHIR DocumentReference-like free-text clinical notes (synthetic)")
save(pa_requests, SCHEMA_PA, "pa_request", "Incoming prior authorization requests (synthetic, X12 278 / Da Vinci PAS-like)")

# COMMAND ----------

# MAGIC %md ## Clinical trials (free-text eligibility, to be parsed by the LLM in Project 2)

# COMMAND ----------

TRIALS = [
    ("Type 2 Diabetes", "Phase 3", "Inclusion: Adults aged 18 to 75 years with type 2 diabetes mellitus. HbA1c between 7.5% and 10.5% at screening. On stable metformin for at least 3 months. Exclusion: eGFR below 30 mL/min/1.73m2. History of pancreatitis. Pregnancy."),
    ("Type 2 Diabetes", "Phase 2", "Inclusion: Men and women 40-80 years old with T2DM and BMI of 30 kg/m2 or greater. HbA1c 7.0-9.5%. Exclusion: Heart failure (NYHA class III-IV). Current insulin therapy."),
    ("Chronic Kidney Disease", "Phase 3", "Inclusion: Age 18 or older. CKD with eGFR between 25 and 60 mL/min/1.73m2. Exclusion: Kidney transplant. Active malignancy."),
    ("Chronic Kidney Disease", "Phase 2", "Inclusion: Adults 30-85 with CKD stage 3 AND type 2 diabetes. eGFR 30-59. Exclusion: Systolic blood pressure above 170 mmHg."),
    ("Hypertension", "Phase 3", "Inclusion: Adults 18-70 with uncontrolled essential hypertension, systolic blood pressure 140-180 mmHg despite at least one antihypertensive. Exclusion: eGFR below 45. Heart failure."),
    ("Obesity", "Phase 3", "Inclusion: Adults 18-65 with BMI 30 or greater, or BMI 27 or greater with hypertension. Exclusion: Type 1 diabetes. History of major depression requiring hospitalization."),
    ("Asthma", "Phase 2", "Inclusion: Patients 12 to 65 years with persistent asthma using inhaled therapy. Exclusion: Current smokers. COPD diagnosis."),
    ("Heart Failure", "Phase 3", "Inclusion: Adults 50 or older with chronic heart failure. eGFR 30 or above. Exclusion: Systolic blood pressure below 100 mmHg."),
    ("Major Depressive Disorder", "Phase 2", "Inclusion: Adults 18-65 with major depressive disorder, current episode, on at least one antidepressant. Exclusion: Bipolar disorder. Active substance use disorder."),
    ("Migraine", "Phase 3", "Inclusion: Women and men 18-60 with migraine of at least 1 year duration. Exclusion: Medication overuse headache. Uncontrolled hypertension (systolic above 160)."),
    ("Knee Osteoarthritis", "Phase 2", "Inclusion: Adults 45-80 with osteoarthritis of the knee and BMI below 40. Exclusion: Knee replacement in the study knee. Intra-articular steroid in the last 3 months."),
    ("Obstructive Sleep Apnea", "Phase 2", "Inclusion: Adults 25-70 with moderate to severe obstructive sleep apnea, BMI between 27 and 42. Exclusion: Heart failure. Central sleep apnea."),
    ("Breast Cancer", "Phase 3", "Inclusion: Women 18 or older with HR-positive, HER2-negative early breast cancer after surgery. Exclusion: Prior chemotherapy for this cancer. eGFR below 40."),
    ("Hypercholesterolemia", "Phase 3", "Inclusion: Adults 40-80 with LDL cholesterol of 100 mg/dL or higher despite statin therapy. Exclusion: Recent myocardial infarction within 3 months."),
]
trials = []
for n, (cond, phase, elig) in enumerate(TRIALS * 2, start=1):  # duplicate with different sites/status for retrieval realism
    trials.append(dict(trial_id=f"SYN-{n:04d}", title=f"A {phase} Study in {cond} (synthetic #{n})", condition=cond, phase=phase,
                       overall_status=random.choices(["RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"], [70, 20, 10])[0],
                       sponsor=random.choice(["Cascade Research Network", "Evergreen Academic Medical Center", "Pacific Pharma (synthetic)"]),
                       site_states=random.sample(STATES, 3), eligibility_text=elig))
save(trials, SCHEMA_TRIALS, "trial_raw", "Clinical trials with free-text eligibility criteria (synthetic; optional ClinicalTrials.gov load in Project 2)")

# COMMAND ----------

# MAGIC %md ## Claims and provider inquiries (labelled history for Project 4)

# COMMAND ----------

CPTS = [("99213","Office visit est. patient, low MDM",130),("99214","Office visit est. patient, moderate MDM",190),("73721","MRI knee w/o contrast",1100),
        ("27447","Total knee arthroplasty",28000),("95810","Polysomnography in-lab",2400),("80053","Comprehensive metabolic panel",45),("97110","Therapeutic exercise",95)]
CARC = [("CO-16","Claim lacks information needed for adjudication"),("CO-50","Not deemed a medical necessity"),("CO-197","Precertification/authorization absent"),
        ("CO-97","Benefit included in another service already adjudicated"),("CO-29","Time limit for filing has expired"),("CO-11","Diagnosis inconsistent with procedure"),
        ("CO-18","Exact duplicate claim/service"),("PR-1","Deductible amount")]
claims, lines = [], []
member_by_pid = {x["patient_id"]: x["member_id"] for x in patients}
for n in range(1, 6001):
    pid = random.choice(list(pat_profile))
    svc = rdate(date(2026, 1, 1), date(2026, 10, 25))
    status = random.choices(["PAID", "DENIED", "PENDED", "PARTIAL"], [70, 15, 8, 7])[0]
    cid = f"CLM{n:07d}"
    total, paid = 0.0, 0.0
    for ln in range(1, random.randint(1, 3) + 1):
        cpt, desc, price = random.choice(CPTS)
        charge = round(price * random.uniform(0.9, 1.3), 2)
        carc = random.choice(CARC)[0] if status in ("DENIED", "PARTIAL") else None
        p = 0.0 if status in ("DENIED", "PENDED") else round(charge * random.uniform(0.4, 0.8), 2)
        total += charge; paid += p
        lines.append(dict(claim_id=cid, line_number=ln, cpt_code=cpt, description=desc, charge_amount=charge, paid_amount=p, carc_code=carc))
    claims.append(dict(claim_id=cid, patient_id=pid, member_id=member_by_pid[pid],
                       billing_npi=f"1{random.randint(100000000,999999999)}", service_date=svc, received_date=svc + timedelta(days=random.randint(2, 120)),
                       claim_status=status, total_charge=round(total, 2), total_paid=round(paid, 2)))
save(claims, SCHEMA_CLAIMS, "claim", "Claim header (X12 837-like, synthetic)")
save(lines, SCHEMA_CLAIMS, "claim_line", "Claim service lines with CARC adjustment codes (X12 835-like, synthetic)")

# COMMAND ----------

CATEGORIES = {
    # category: (team, templates, base_priority)
    "CLAIM_STATUS": ("Claims Operations", [
        "Hi, checking on status of claim {cid} submitted {d} days ago. Still showing pended on our portal.",
        "Can you tell me why claim {cid} hasn't been adjudicated yet? It's been over {d} days.",
        "Following up on claim {cid}. Need an ETA on payment please."], "P3"),
    "DENIAL_APPEAL": ("Appeals & Grievances", [
        "Claim {cid} denied with {carc}. We disagree, the service was documented. How do we file an appeal? Appeal window closes in {d} days.",
        "We received a denial ({carc}) on {cid}. Attaching records, please reconsider.",
        "Requesting reconsideration for {cid}, denial reason {carc} is incorrect, we had authorization on file."], "P2"),
    "PRIOR_AUTH_STATUS": ("Utilization Management", [
        "Patient is scheduled for surgery tomorrow morning and the prior auth for CPT {cpt} is still pending. URGENT.",
        "What is the status of the authorization request for {cpt}? Submitted last week.",
        "Expedited request: patient in the ED, need MRI auth for {cpt} today."], "P2"),
    "ELIGIBILITY_BENEFITS": ("Eligibility & Benefits", [
        "Please verify eligibility and benefits for member before the visit next week. Does the plan cover {cpt}?",
        "Is a referral required for {cpt} under this member's plan? Need copay info."], "P4"),
    "PAYMENT_DISCREPANCY": ("Payment Integrity", [
        "Claim {cid} was paid ${amt} but contracted rate is much higher. Please review underpayment.",
        "We were overpaid on {cid}, how do we return ${amt}?",
        "Payment on {cid} doesn't match the remittance. Total difference ${amt}."], "P3"),
    "CREDENTIALING": ("Provider Network & Credentialing", [
        "New physician joined our group, how long does credentialing take? Claims are being denied as out-of-network.",
        "Need to update our practice address and TIN on file."], "P4"),
    "CODING_QUESTION": ("Coding & Billing Support", [
        "Claim {cid} denied {carc}, is modifier 25 required with {cpt} when billed with an E/M on the same day?",
        "Which diagnosis codes support medical necessity for {cpt}? We keep getting CO-11."], "P4"),
}
inquiries = []
for n in range(1, 1501):
    cat = random.choices(list(CATEGORIES), [22, 20, 14, 12, 14, 8, 10])[0]
    team, tmpls, prio = CATEGORIES[cat]
    clm = random.choice(claims)
    text = random.choice(tmpls).format(cid=clm["claim_id"], d=random.randint(3, 60), carc=random.choice(CARC)[0],
                                       cpt=random.choice(CPTS)[0], amt=random.randint(80, 25000))
    lowered = text.lower()
    days_left = re.search(r"closes in (\d+) days", lowered)
    amount = re.search(r"\$(\d+)", text)
    if any(k in lowered for k in ["urgent", "tomorrow", "expedited", " ed,", "today"]):
        prio = "P1"
    elif days_left and int(days_left.group(1)) <= 10:
        prio = "P1"
    elif amount and int(amount.group(1)) > 10000:
        prio = "P2"
    if random.random() < 0.05:  # realistic label noise from human triage
        team = random.choice([v[0] for v in CATEGORIES.values()])
    inquiries.append(dict(inquiry_id=f"INQ{n:06d}", created_ts=datetime(2026, 1, 1) + timedelta(minutes=random.randint(0, 430_000)),
                          channel=random.choice(["portal", "email", "phone_transcript"]), provider_npi=clm["billing_npi"], claim_id=clm["claim_id"],
                          body=text, category=cat, priority=prio, assigned_team=team,
                          resolution_days=round(random.uniform(0.2, 2) if prio == "P1" else random.uniform(1, 15), 1),
                          resolution_note=f"Resolved by {team}: " + random.choice(["reprocessed claim", "provided status and ETA", "appeal case opened",
                                                                                   "auth expedited to nurse reviewer", "benefits confirmed", "corrected coding guidance sent"])))
save(inquiries, SCHEMA_CLAIMS, "provider_inquiry", "Historical provider inquiries with human triage labels (synthetic)")

# COMMAND ----------

# MAGIC %md ✅ Done. Open **Catalog Explorer → hc_agentic** and look around before starting Project 1.
