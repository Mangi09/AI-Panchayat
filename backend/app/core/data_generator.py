import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure reproducibility
np.random.seed(42)

def generate_corporate_hiring(n: int = 1500) -> Dict[str, Any]:
    years_experience = np.random.randint(1, 16, size=n)
    tech_assessment_score = np.random.normal(70, 15, size=n).clip(0, 100).round()
    interview_score = np.random.normal(75, 12, size=n).clip(0, 100).round()
    gender = np.random.choice(["Male", "Female"], size=n)

    # Base probability calculation based on true merit metrics
    base_prob = (0.1 + 
                 (years_experience / 15) * 0.3 + 
                 (tech_assessment_score / 100) * 0.3 + 
                 (interview_score / 100) * 0.3)
    
    # The mathematically embedded bias: Female penalty of 15%
    penalty = np.where(gender == "Female", -0.15, 0.0)
    final_prob = np.clip(base_prob + penalty, 0, 1)

    hired = (np.random.rand(n) < final_prob).astype(int)

    df = pd.DataFrame({
        "years_experience": years_experience,
        "tech_assessment_score": tech_assessment_score,
        "interview_score": interview_score,
        "gender": gender,
        "hired": hired
    })
    
    return {"dataframe": df, "target_column": "hired", "sensitive_column": "gender"}

def generate_mortgage_approvals(n: int = 1500) -> Dict[str, Any]:
    annual_income = np.random.randint(40000, 200001, size=n)
    credit_score = np.random.normal(680, 80, size=n).clip(300, 850).round().astype(int)
    debt_to_income_ratio = np.random.uniform(0.1, 0.6, size=n).round(2)
    zip_code_zone = np.random.choice(["Zone_A", "Zone_B"], size=n)

    # Base odds based on financial strength
    base_prob = (0.5 + 
                 (annual_income / 200000) * 0.2 + 
                 ((credit_score - 300) / 550) * 0.4 - 
                 (debt_to_income_ratio / 0.6) * 0.4)

    # Severe penalty to approval odds if the applicant is from Zone_B
    penalty = np.where(zip_code_zone == "Zone_B", -0.35, 0.0)
    final_prob = np.clip(base_prob + penalty, 0, 1)
    
    approved = (np.random.rand(n) < final_prob).astype(int)

    df = pd.DataFrame({
        "annual_income": annual_income,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "zip_code_zone": zip_code_zone,
        "approved": approved
    })
    
    return {"dataframe": df, "target_column": "approved", "sensitive_column": "zip_code_zone"}

def generate_hospital_triage(n: int = 1500) -> Dict[str, Any]:
    age = np.random.randint(18, 91, size=n)
    symptom_severity = np.random.randint(1, 11, size=n)
    comorbidity_count = np.random.randint(0, 6, size=n)
    insurance_type = np.random.choice(["Private", "Medicaid", "Uninsured"], size=n)

    # Base patient need without considering insurance
    base_prob = (0.05 + 
                 (age / 90) * 0.25 + 
                 (symptom_severity / 10) * 0.45 + 
                 (comorbidity_count / 5) * 0.25)

    # Private (+20%), Uninsured (-15% implicit penalty)
    bonus_penalty = np.where(insurance_type == "Private", 0.20,
                    np.where(insurance_type == "Uninsured", -0.15, 0.0))
    
    final_prob = np.clip(base_prob + bonus_penalty, 0, 1)
    icu_admitted = (np.random.rand(n) < final_prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "symptom_severity": symptom_severity,
        "comorbidity_count": comorbidity_count,
        "insurance_type": insurance_type,
        "icu_admitted": icu_admitted
    })
    
    return {"dataframe": df, "target_column": "icu_admitted", "sensitive_column": "insurance_type"}

def generate_criminal_recidivism(n: int = 1500) -> Dict[str, Any]:
    prior_arrests = np.random.randint(0, 11, size=n)
    age_at_first_arrest = np.random.randint(14, 41, size=n)
    months_employed_last_year = np.random.randint(0, 13, size=n)
    demographic_group = np.random.choice(["Group_X", "Group_Y"], size=n)

    # Base model of true risk
    base_risk = (0.2 + 
                 (prior_arrests / 10) * 0.45 - 
                 (months_employed_last_year / 12) * 0.35 + 
                 ((40 - age_at_first_arrest) / 26) * 0.2)

    # Flat 25% artificial risk increase if Group_Y
    boost = np.where(demographic_group == "Group_Y", 0.25, 0.0)
    final_risk = np.clip(base_risk + boost, 0, 1)

    high_risk_flag = (np.random.rand(n) < final_risk).astype(int)

    df = pd.DataFrame({
        "prior_arrests": prior_arrests,
        "age_at_first_arrest": age_at_first_arrest,
        "months_employed_last_year": months_employed_last_year,
        "demographic_group": demographic_group,
        "high_risk_flag": high_risk_flag
    })
    
    return {"dataframe": df, "target_column": "high_risk_flag", "sensitive_column": "demographic_group"}

def generate_university_admissions(n: int = 1500) -> Dict[str, Any]:
    gpa = np.random.uniform(2.0, 4.0, size=n).round(2)
    sat_score = np.random.randint(800, 1601, size=n)
    extracurricular_hours = np.random.randint(0, 21, size=n)
    legacy_status = np.random.choice(["Yes", "No"], size=n, p=[0.15, 0.85])

    # Unbiased calculation of qualification
    base_prob = (((gpa - 2.0) / 2.0) * 0.4 + 
                 ((sat_score - 800) / 800) * 0.4 + 
                 (extracurricular_hours / 20) * 0.2)

    # Legacy massive boost
    boost = np.where(legacy_status == "Yes", 0.45, 0.0)
    final_prob = np.clip(base_prob + boost, 0, 1)

    admitted = (np.random.rand(n) < final_prob).astype(int)

    df = pd.DataFrame({
        "gpa": gpa,
        "sat_score": sat_score,
        "extracurricular_hours": extracurricular_hours,
        "legacy_status": legacy_status,
        "admitted": admitted
    })
    
    return {"dataframe": df, "target_column": "admitted", "sensitive_column": "legacy_status"}


# Unified Master Router
_REGISTRY = {
    "corporate_hiring": generate_corporate_hiring,
    "mortgage_approvals": generate_mortgage_approvals,
    "hospital_triage": generate_hospital_triage,
    "criminal_recidivism": generate_criminal_recidivism,
    "university_admissions": generate_university_admissions
}

AVAILABLE_DATASETS = list(_REGISTRY.keys())

def get_test_dataset(dataset_name: str) -> Dict[str, Any]:
    """Retrieves a mock dataframe and its associated truth columns given a key."""
    if dataset_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {AVAILABLE_DATASETS}")
    return _REGISTRY[dataset_name]()
